from __future__ import annotations

import csv
import html
from pathlib import Path
from statistics import mean
from typing import Any

from .reward_study import RewardStudyPlan, read_csv_rows, safe_float


def _last_validation_row(case_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(case_dir / "validation.csv")
    if not rows:
        raise ValueError(f"missing validation.csv rows for {case_dir.name}")
    return rows[-1]


def build_dynamic_cqi_analysis(
    *, plan: RewardStudyPlan, round_dir: Path, output_path: Path
) -> Path:
    rows: list[dict[str, Any]] = []
    for case in plan.cases:
        validation = _last_validation_row(round_dir / case.case_id)
        common = dict(plan.common)
        common.update(case.common_overrides)
        rows.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "cqi_mode": common.get("cqi_mode", "static"),
                "cqi_temporal_correlation": float(common.get("cqi_temporal_correlation", 0.97)),
                "cqi_innovation_std": float(common.get("cqi_innovation_std", 0.0)),
                "cqi_max_delta_per_update": int(common.get("cqi_max_delta_per_update", 1)),
                "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                "jain_fairness": safe_float(validation, "final_jain_fairness"),
                "starvation_rate": safe_float(validation, "max_starvation_rate"),
                "p99_wait_slots": safe_float(validation, "max_p99_wait_slots"),
                "max_wait_slots": safe_float(validation, "max_wait_slots"),
                "mean_cqi": safe_float(validation, "mean_cqi"),
                "mean_cqi_std": safe_float(validation, "mean_cqi_std"),
                "mean_cqi_abs_change_per_slot": safe_float(validation, "mean_cqi_abs_change_per_slot"),
                "mean_cqi_changed_fraction": safe_float(validation, "mean_cqi_changed_fraction"),
            }
        )

    metrics_output = Path(str(plan.analysis.get("metrics_output", output_path.with_suffix(".csv"))))
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    static = next((row for row in rows if row["cqi_mode"] == "static"), rows[0])
    md_lines = [
        "# Round 10 — Dynamic CQI screen",
        "",
        "Reward/PPO/state/action remain fixed. Only CQI temporal dynamics change.",
        "",
        "| Case | CQI | Goodput | Jain | Starvation | P99 | Max wait | Mean |ΔCQI| | Changed UE fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['label']} | {row['cqi_mode']} | {row['goodput_bits_per_slot']:.0f} | "
            f"{row['jain_fairness']:.4f} | {100*row['starvation_rate']:.2f}% | "
            f"{row['p99_wait_slots']:.0f} | {row['max_wait_slots']:.0f} | "
            f"{row['mean_cqi_abs_change_per_slot']:.3f} | {100*row['mean_cqi_changed_fraction']:.1f}% |"
        )
    md_lines += [
        "",
        "## Interpretation rule",
        "Dynamic CQI is considered manageable only if the scheduler remains non-collapsed while preserving useful goodput/fairness/delay relative to the static baseline.",
        "BLER is still fixed and CQI-independent in this round; Link Adaptation is intentionally not changed yet.",
    ]
    markdown_output = Path(str(plan.analysis.get("markdown_output", output_path.with_suffix(".md"))))
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    body_rows = []
    for row in rows:
        dg = row["goodput_bits_per_slot"] - static["goodput_bits_per_slot"]
        dj = row["jain_fairness"] - static["jain_fairness"]
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['cqi_mode']))}</td>"
            f"<td>{row['cqi_temporal_correlation']:.2f}</td>"
            f"<td>{row['cqi_innovation_std']:.2f}</td>"
            f"<td>{row['goodput_bits_per_slot']:.0f} ({dg:+.0f})</td>"
            f"<td>{row['jain_fairness']:.4f} ({dj:+.4f})</td>"
            f"<td>{100*row['starvation_rate']:.2f}%</td>"
            f"<td>{row['p99_wait_slots']:.0f}</td>"
            f"<td>{row['max_wait_slots']:.0f}</td>"
            f"<td>{row['mean_cqi_abs_change_per_slot']:.3f}</td>"
            f"<td>{100*row['mean_cqi_changed_fraction']:.1f}%</td>"
            "</tr>"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 10 Dynamic CQI</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1150px;margin:32px auto;padding:0 16px;line-height:1.55}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f5f7}"
        ".note{padding:14px;background:#eef5ff;border-left:4px solid #3568a8}</style></head><body>"
        "<h1>Round 10 — Dynamic CQI screen</h1>"
        "<p>Static, slow-correlated and faster-correlated CQI are compared with the same T–J–S reward and PPO setup.</p>"
        "<div class='note'><b>Scope:</b> CQI changes over time, but BLER remains fixed and CQI-independent. "
        "This round tests channel non-stationarity only; it is not yet Link Adaptation.</div>"
        "<table><thead><tr><th>Case</th><th>Mode</th><th>ρ</th><th>σ</th><th>Goodput (Δ)</th><th>Jain (Δ)</th>"
        "<th>Starvation</th><th>P99</th><th>Max wait</th><th>Mean |ΔCQI|</th><th>UE changed</th></tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return output_path
