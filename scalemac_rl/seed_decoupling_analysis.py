from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from scalemac_rl.reward_study import RewardStudyPlan


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _service_feasible(row: dict[str, str]) -> bool:
    return (
        _f(row, "final_target_starvation_excess") <= 1e-12
        and _f(row, "final_target_wait_excess") <= 1e-12
        and _f(row, "final_target_max_wait_excess") <= 1e-12
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_seed_decoupling_analysis(*, plan: RewardStudyPlan, round_dir: Path, output_path: Path) -> Path:
    metrics: list[dict[str, Any]] = []
    for case in plan.cases:
        validation = _read_csv(round_dir / case.case_id / "validation_summary.csv")
        training = _read_csv(round_dir / case.case_id / "training.csv")
        if not validation:
            continue
        latest = validation[-1]
        ever = any(_service_feasible(row) for row in validation)
        latest_feasible = _service_feasible(latest)
        overrides = case.common_overrides
        tail = training[-max(1, len(training) // 5):] if training else []
        metrics.append({
            "case_id": case.case_id,
            "training_seed": int(overrides.get("seed", 0)),
            "environment_seed": int(overrides.get("environment_seed", overrides.get("seed", 0))),
            "ever_service_feasible": int(ever),
            "latest_service_feasible": int(latest_feasible),
            "learn_then_drift": int(ever and not latest_feasible),
            "latest_goodput_bits_per_slot": _f(latest, "mean_goodput_bits_per_slot"),
            "latest_jain_fairness": _f(latest, "mean_jain_fairness"),
            "latest_starvation_rate": _f(latest, "worst_starvation_rate"),
            "latest_p99_wait_slots": _f(latest, "worst_p99_wait_slots"),
            "latest_max_wait_slots": _f(latest, "worst_max_wait_slots"),
            "tail_mean_approx_kl": sum(_f(r, "approx_kl") for r in tail) / max(len(tail), 1),
        })

    if len(metrics) != 9:
        raise ValueError(f"seed-decoupling analysis requires 9 completed cases; found {len(metrics)}")

    train_summary = []
    env_summary = []
    for seed in sorted({int(r["training_seed"]) for r in metrics}):
        rows = [r for r in metrics if int(r["training_seed"]) == seed]
        train_summary.append({
            "training_seed": seed,
            "latest_feasible_environments": sum(int(r["latest_service_feasible"]) for r in rows),
            "ever_feasible_environments": sum(int(r["ever_service_feasible"]) for r in rows),
            "mean_latest_starvation_rate": sum(float(r["latest_starvation_rate"]) for r in rows) / 3.0,
            "mean_latest_jain_fairness": sum(float(r["latest_jain_fairness"]) for r in rows) / 3.0,
        })
    for seed in sorted({int(r["environment_seed"]) for r in metrics}):
        rows = [r for r in metrics if int(r["environment_seed"]) == seed]
        env_summary.append({
            "environment_seed": seed,
            "latest_feasible_training_seeds": sum(int(r["latest_service_feasible"]) for r in rows),
            "ever_feasible_training_seeds": sum(int(r["ever_service_feasible"]) for r in rows),
            "mean_latest_starvation_rate": sum(float(r["latest_starvation_rate"]) for r in rows) / 3.0,
            "mean_latest_jain_fairness": sum(float(r["latest_jain_fairness"]) for r in rows) / 3.0,
        })

    matrix = []
    for r in sorted(metrics, key=lambda x: (x["training_seed"], x["environment_seed"])):
        matrix.append({
            "training_seed": r["training_seed"],
            "environment_seed": r["environment_seed"],
            "latest_service_feasible": r["latest_service_feasible"],
            "latest_starvation_rate": r["latest_starvation_rate"],
            "latest_jain_fairness": r["latest_jain_fairness"],
        })

    train_range = max(r["latest_feasible_environments"] for r in train_summary) - min(r["latest_feasible_environments"] for r in train_summary)
    env_range = max(r["latest_feasible_training_seeds"] for r in env_summary) - min(r["latest_feasible_training_seeds"] for r in env_summary)
    if train_range > env_range:
        driver = "training_rng_dominant"
    elif env_range > train_range:
        driver = "environment_seed_dominant"
    else:
        driver = "interaction_or_mixed"
    decision = {
        "primary_pattern": driver,
        "training_seed_feasibility_range": int(train_range),
        "environment_seed_feasibility_range": int(env_range),
        "latest_feasible_cases": int(sum(r["latest_service_feasible"] for r in metrics)),
        "ever_feasible_cases": int(sum(r["ever_service_feasible"] for r in metrics)),
        "interpretation": "Use the full matrix, not the label of a diagonal seed, to decide whether failures follow PPO RNG, environment realization, or their interaction.",
    }

    analysis = plan.analysis
    _write_csv(Path(str(analysis["metrics_output"])), metrics)
    _write_csv(Path(str(analysis["train_seed_output"])), train_summary)
    _write_csv(Path(str(analysis["environment_seed_output"])), env_summary)
    _write_csv(Path(str(analysis["matrix_output"])), matrix)
    decision_path = Path(str(analysis["decision_output"]))
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    rows_html = "".join(
        "<tr>" +
        f"<td>{r['training_seed']}</td><td>{r['environment_seed']}</td>" +
        f"<td>{'YES' if r['latest_service_feasible'] else 'NO'}</td>" +
        f"<td>{float(r['latest_goodput_bits_per_slot']):,.0f}</td>" +
        f"<td>{float(r['latest_jain_fairness']):.4f}</td>" +
        f"<td>{100*float(r['latest_starvation_rate']):.2f}%</td>" +
        f"<td>{float(r['latest_p99_wait_slots']):.0f}</td></tr>"
        for r in sorted(metrics, key=lambda x: (x["training_seed"], x["environment_seed"]))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 17A</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1100px;margin:32px auto;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:7px}th{background:#eee}</style>"
        "</head><body><h1>Round 17A — Seed Decoupling</h1>"
        f"<p><b>Primary pattern:</b> {html.escape(driver)}</p>"
        "<p>The table separates PPO/model RNG seed from the environment/channel/CSI seed.</p>"
        "<table><thead><tr><th>Training RNG</th><th>Environment seed</th><th>Latest service-feasible</th>"
        "<th>Goodput</th><th>JFI</th><th>Starvation</th><th>P99</th></tr></thead><tbody>"
        + rows_html + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    md_path = Path(str(analysis.get("markdown_output", output_path.with_suffix(".md"))))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        f"# Round 17A — Seed Decoupling\n\nPrimary pattern: **{driver}**.\n\n"
        "Interpret the full 3×3 matrix to separate PPO RNG effects from environment-seed effects.\n",
        encoding="utf-8",
    )
    return output_path
