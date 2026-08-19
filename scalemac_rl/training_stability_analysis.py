from __future__ import annotations

import csv
import html
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .reward_study import RewardStudyPlan, read_csv_rows, safe_float


def _last_validation(case_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(case_dir / "validation.csv")
    if not rows:
        raise ValueError(f"missing validation.csv rows for {case_dir.name}")
    return rows[-1]


def _tail_training(case_dir: Path, tail_updates: int = 32) -> list[dict[str, str]]:
    rows = read_csv_rows(case_dir / "training.csv")
    if not rows:
        raise ValueError(f"missing training.csv rows for {case_dir.name}")
    return rows[-tail_updates:]


def _avg(rows: list[dict[str, str]], key: str) -> float:
    values = [safe_float(row, key) for row in rows]
    return mean(values) if values else 0.0


def _max(rows: list[dict[str, str]], key: str) -> float:
    values = [safe_float(row, key) for row in rows]
    return max(values) if values else 0.0


def _summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _recipe(common: dict[str, Any]) -> str:
    lr = float(common.get("learning_rate_start", 1.0e-4))
    epochs = int(common.get("update_epochs", 4))
    if abs(lr - 1.0e-4) < 1e-12 and epochs == 4:
        return "baseline"
    if abs(lr - 5.0e-5) < 1e-12 and epochs == 4:
        return "low_lr"
    if abs(lr - 1.0e-4) < 1e-12 and epochs == 2:
        return "epochs2"
    if abs(lr - 5.0e-5) < 1e-12 and epochs == 2:
        return "low_lr_epochs2"
    return f"lr={lr:g}_epochs={epochs}"


def build_training_stability_analysis(
    *, plan: RewardStudyPlan, round_dir: Path, output_path: Path
) -> Path:
    rows: list[dict[str, Any]] = []
    for case in plan.cases:
        case_dir = round_dir / case.case_id
        validation = _last_validation(case_dir)
        tail = _tail_training(case_dir)
        common = dict(plan.common)
        common.update(case.common_overrides)
        starvation = safe_float(validation, "max_starvation_rate")
        p99 = safe_float(validation, "max_p99_wait_slots")
        max_wait = safe_float(validation, "max_wait_slots")
        rows.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "recipe": _recipe(common),
                "seed": int(common.get("seed", 1701)),
                "learning_rate_start": float(common.get("learning_rate_start", 1.0e-4)),
                "learning_rate_end": float(common.get("learning_rate_end", 2.5e-5)),
                "update_epochs": int(common.get("update_epochs", 4)),
                "clip_coef": float(common.get("clip_coef", 0.1)),
                "target_kl": float(common.get("target_kl", 0.02)),
                "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                "spectral_efficiency_bps_hz": safe_float(validation, "mean_spectral_efficiency_bps_hz"),
                "jain_fairness": safe_float(validation, "final_jain_fairness"),
                "starvation_rate": starvation,
                "p99_wait_slots": p99,
                "max_wait_slots": max_wait,
                "observed_bler": safe_float(validation, "mean_observed_bler"),
                "harq_retx_fraction": safe_float(validation, "mean_harq_retransmission_fraction"),
                "tail_mean_approx_kl": _avg(tail, "approx_kl"),
                "tail_max_approx_kl": _max(tail, "approx_kl"),
                "tail_mean_clip_fraction": _avg(tail, "clip_fraction"),
                "tail_mean_training_jain": _avg(tail, "mean_jain_fairness"),
                "tail_mean_training_starvation": _avg(tail, "mean_starvation_rate"),
                "zero_starvation": int(starvation <= 1e-12),
                "full_collapse": int(starvation >= 0.5 or p99 >= 4999.0 or max_wait >= 4999.0),
            }
        )

    metrics_output = Path(str(plan.analysis.get("metrics_output", output_path.with_suffix(".csv"))))
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    recipes = ["baseline", "low_lr", "epochs2", "low_lr_epochs2"]
    for recipe in recipes:
        subset = [row for row in rows if row["recipe"] == recipe]
        if not subset:
            continue
        entry: dict[str, Any] = {
            "recipe": recipe,
            "seeds": len(subset),
            "learning_rate_start": subset[0]["learning_rate_start"],
            "update_epochs": subset[0]["update_epochs"],
            "zero_starvation_seeds": sum(int(row["zero_starvation"]) for row in subset),
            "full_collapse_seeds": sum(int(row["full_collapse"]) for row in subset),
        }
        for key in (
            "goodput_bits_per_slot",
            "spectral_efficiency_bps_hz",
            "jain_fairness",
            "starvation_rate",
            "p99_wait_slots",
            "max_wait_slots",
            "observed_bler",
            "tail_mean_approx_kl",
            "tail_mean_clip_fraction",
        ):
            avg, sd = _summary([float(row[key]) for row in subset])
            entry[f"mean_{key}"] = avg
            entry[f"std_{key}"] = sd
        summary_rows.append(entry)

    summary_output = Path(str(plan.analysis.get("summary_output", output_path.with_name("recipe_summary.csv"))))
    with summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Descriptive 2x2 main/interaction effects. These are not significance tests.
    lookup = {row["recipe"]: row for row in summary_rows}
    effect_rows: list[dict[str, Any]] = []
    if all(name in lookup for name in recipes):
        metrics = ("goodput_bits_per_slot", "jain_fairness", "starvation_rate", "p99_wait_slots", "tail_mean_approx_kl", "tail_mean_clip_fraction")
        for metric in metrics:
            b = float(lookup["baseline"][f"mean_{metric}"])
            l = float(lookup["low_lr"][f"mean_{metric}"])
            e = float(lookup["epochs2"][f"mean_{metric}"])
            le = float(lookup["low_lr_epochs2"][f"mean_{metric}"])
            effect_rows.append(
                {
                    "metric": metric,
                    "low_lr_effect_at_4_epochs": l - b,
                    "low_lr_effect_at_2_epochs": le - e,
                    "epochs2_effect_at_standard_lr": e - b,
                    "epochs2_effect_at_low_lr": le - l,
                    "interaction": le - l - e + b,
                }
            )
    effects_output = Path(str(plan.analysis.get("factor_effects_output", output_path.with_name("factor_effects.csv"))))
    if effect_rows:
        with effects_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(effect_rows[0]))
            writer.writeheader()
            writer.writerows(effect_rows)

    markdown_output = Path(str(plan.analysis.get("markdown_output", output_path.with_suffix(".md"))))
    lines = [
        "# Round 14A — PPO training stability",
        "",
        "A self-contained 2×2 factorial screen: standard/low learning rate × four/two update epochs. Environment, features, reward and feed-forward PPO architecture are fixed.",
        "",
        "| Recipe | Seeds | Zero-starvation | Full collapse | Goodput mean±sd | JFI mean±sd | Starvation mean±sd | P99 mean±sd | Tail KL | Tail clip fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['recipe']} | {row['seeds']} | {row['zero_starvation_seeds']}/{row['seeds']} | {row['full_collapse_seeds']}/{row['seeds']} | "
            f"{row['mean_goodput_bits_per_slot']:.0f}±{row['std_goodput_bits_per_slot']:.0f} | "
            f"{row['mean_jain_fairness']:.4f}±{row['std_jain_fairness']:.4f} | "
            f"{100*row['mean_starvation_rate']:.2f}%±{100*row['std_starvation_rate']:.2f}% | "
            f"{row['mean_p99_wait_slots']:.1f}±{row['std_p99_wait_slots']:.1f} | "
            f"{row['mean_tail_mean_approx_kl']:.4f} | {row['mean_tail_mean_clip_fraction']:.3f} |"
        )
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td><td>{row['seed']}</td><td>{html.escape(str(row['recipe']))}</td>"
            f"<td>{row['goodput_bits_per_slot']:.0f}</td><td>{row['spectral_efficiency_bps_hz']:.4f}</td>"
            f"<td>{row['jain_fairness']:.4f}</td><td>{100*row['starvation_rate']:.2f}%</td>"
            f"<td>{row['p99_wait_slots']:.0f}</td><td>{row['max_wait_slots']:.0f}</td>"
            f"<td>{row['tail_mean_approx_kl']:.4f}</td><td>{row['tail_mean_clip_fraction']:.3f}</td>"
            "</tr>"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 14A PPO Training Stability</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1250px;margin:32px auto;padding:0 16px;line-height:1.55}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f5f7}</style>"
        "</head><body><h1>Round 14A — PPO Training Stability</h1>"
        "<p>Only the learning-rate schedule and number of PPO update epochs change. Reward T–J–S, observation features, feed-forward architecture, Dynamic CQI, delayed CSI and CQI→MCS→BLER are fixed.</p>"
        "<table><thead><tr><th>Case</th><th>Seed</th><th>Recipe</th><th>Goodput</th><th>SE</th><th>JFI</th><th>Starv.</th><th>P99</th><th>Max wait</th><th>Tail KL</th><th>Tail clip</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return output_path
