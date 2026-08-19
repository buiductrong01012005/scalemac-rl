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


def _profile(common: dict[str, Any]) -> str:
    age = bool(common.get("observation_include_csi_age", False))
    trend = bool(common.get("observation_include_reported_cqi_trend", False))
    if age and trend:
        return "csi_age_plus_cqi_trend"
    if age:
        return "csi_age"
    if trend:
        return "cqi_trend"
    return "baseline_16"


def _summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def build_feature_ablation_analysis(
    *, plan: RewardStudyPlan, round_dir: Path, output_path: Path
) -> Path:
    rows: list[dict[str, Any]] = []
    for case in plan.cases:
        common = dict(plan.common)
        common.update(case.common_overrides)
        validation = _last_validation(round_dir / case.case_id)
        starvation = safe_float(validation, "max_starvation_rate")
        p99 = safe_float(validation, "max_p99_wait_slots")
        max_wait = safe_float(validation, "max_wait_slots")
        rows.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "profile": _profile(common),
                "seed": int(common.get("seed", 1701)),
                "observation_features_per_ue": 16
                + int(bool(common.get("observation_include_csi_age", False)))
                + int(bool(common.get("observation_include_reported_cqi_trend", False))),
                "include_csi_age": int(bool(common.get("observation_include_csi_age", False))),
                "include_cqi_trend": int(bool(common.get("observation_include_reported_cqi_trend", False))),
                "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                "spectral_efficiency_bps_hz": safe_float(validation, "mean_spectral_efficiency_bps_hz"),
                "jain_fairness": safe_float(validation, "final_jain_fairness"),
                "starvation_rate": starvation,
                "p99_wait_slots": p99,
                "max_wait_slots": max_wait,
                "observed_bler": safe_float(validation, "mean_observed_bler"),
                "harq_retx_fraction": safe_float(validation, "mean_harq_retransmission_fraction"),
                "mean_csi_abs_error": safe_float(validation, "mean_csi_abs_error"),
                "mean_csi_report_age_slots": safe_float(validation, "mean_csi_report_age_slots"),
                "mean_inference_us": safe_float(validation, "mean_inference_us"),
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

    profiles = ["baseline_16", "csi_age", "cqi_trend", "csi_age_plus_cqi_trend"]
    summary_rows: list[dict[str, Any]] = []
    for profile in profiles:
        subset = [row for row in rows if row["profile"] == profile]
        if not subset:
            continue
        entry: dict[str, Any] = {
            "profile": profile,
            "seeds": len(subset),
            "observation_features_per_ue": subset[0]["observation_features_per_ue"],
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
            "harq_retx_fraction",
            "mean_inference_us",
        ):
            avg, sd = _summary([float(row[key]) for row in subset])
            entry[f"mean_{key}"] = avg
            entry[f"std_{key}"] = sd
        summary_rows.append(entry)

    summary_output = Path(str(plan.analysis.get("summary_output", output_path.with_name("feature_summary.csv"))))
    with summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    baseline_by_seed = {
        int(row["seed"]): row for row in rows if row["profile"] == "baseline_16"
    }
    paired_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["profile"] == "baseline_16":
            continue
        base = baseline_by_seed.get(int(row["seed"]))
        if base is None:
            continue
        paired_rows.append(
            {
                "profile": row["profile"],
                "seed": row["seed"],
                "delta_goodput_bits_per_slot": float(row["goodput_bits_per_slot"]) - float(base["goodput_bits_per_slot"]),
                "delta_spectral_efficiency_bps_hz": float(row["spectral_efficiency_bps_hz"]) - float(base["spectral_efficiency_bps_hz"]),
                "delta_jain_fairness": float(row["jain_fairness"]) - float(base["jain_fairness"]),
                "delta_starvation_rate": float(row["starvation_rate"]) - float(base["starvation_rate"]),
                "delta_p99_wait_slots": float(row["p99_wait_slots"]) - float(base["p99_wait_slots"]),
                "delta_max_wait_slots": float(row["max_wait_slots"]) - float(base["max_wait_slots"]),
                "delta_observed_bler": float(row["observed_bler"]) - float(base["observed_bler"]),
                "delta_mean_inference_us": float(row["mean_inference_us"]) - float(base["mean_inference_us"]),
            }
        )
    paired_output = Path(str(plan.analysis.get("paired_output", output_path.with_name("feature_paired_comparison.csv"))))
    if paired_rows:
        with paired_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)

    markdown_output = Path(str(plan.analysis.get("markdown_output", output_path.with_suffix(".md"))))
    lines = [
        "# Round 14B — PPO observation feature ablation",
        "",
        "Feed-forward PPO, T–J–S reward and the realistic radio environment are fixed. Only CSI-age and reported-CQI-trend observation features change.",
        "",
        "| Profile | Features/UE | Stable zero-starvation | Full collapse | Goodput mean±sd | JFI mean±sd | P99 mean±sd | BLER mean±sd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['profile']} | {row['observation_features_per_ue']} | {row['zero_starvation_seeds']}/{row['seeds']} | "
            f"{row['full_collapse_seeds']}/{row['seeds']} | "
            f"{row['mean_goodput_bits_per_slot']:.0f}±{row['std_goodput_bits_per_slot']:.0f} | "
            f"{row['mean_jain_fairness']:.4f}±{row['std_jain_fairness']:.4f} | "
            f"{row['mean_p99_wait_slots']:.1f}±{row['std_p99_wait_slots']:.1f} | "
            f"{100*row['mean_observed_bler']:.2f}%±{100*row['std_observed_bler']:.2f}% |"
        )
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td><td>{row['seed']}</td><td>{row['observation_features_per_ue']}</td>"
            f"<td>{row['goodput_bits_per_slot']:.0f}</td><td>{row['spectral_efficiency_bps_hz']:.4f}</td>"
            f"<td>{row['jain_fairness']:.4f}</td><td>{100*row['starvation_rate']:.2f}%</td>"
            f"<td>{row['p99_wait_slots']:.0f}</td><td>{row['max_wait_slots']:.0f}</td>"
            f"<td>{100*row['observed_bler']:.2f}%</td><td>{row['mean_inference_us']:.1f}</td>"
            "</tr>"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 14B Feature Ablation</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1250px;margin:32px auto;padding:0 16px;line-height:1.55}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f5f7}</style>"
        "</head><body><h1>Round 14B — PPO Observation Feature Ablation</h1>"
        "<p>Only observation information changes: baseline 16 features, +CSI age, +reported-CQI trend, or both. PPO recipe, T–J–S reward, Dynamic CQI, delayed CSI and CQI→MCS→BLER stay fixed.</p>"
        "<table><thead><tr><th>Case</th><th>Seed</th><th>F/UE</th><th>Goodput</th><th>SE</th><th>JFI</th><th>Starv.</th><th>P99</th><th>Max wait</th><th>BLER</th><th>Inference µs</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return output_path
