from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from .reward_study import RewardStudyPlan, read_csv_rows, safe_float


def _last_validation_row(case_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(case_dir / "validation.csv")
    if not rows:
        raise ValueError(f"missing validation.csv rows for {case_dir.name}")
    return rows[-1]


def build_csi_reporting_analysis(
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
                "csi_report_mode": str(common.get("csi_report_mode", "perfect")),
                "csi_report_period_slots": int(common.get("csi_report_period_slots", 1)),
                "csi_report_delay_slots": int(common.get("csi_report_delay_slots", 0)),
                "csi_report_error_std": float(common.get("csi_report_error_std", 0.0)),
                "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                "jain_fairness": safe_float(validation, "final_jain_fairness"),
                "starvation_rate": safe_float(validation, "max_starvation_rate"),
                "p99_wait_slots": safe_float(validation, "max_p99_wait_slots"),
                "max_wait_slots": safe_float(validation, "max_wait_slots"),
                "mean_true_cqi": safe_float(validation, "mean_cqi"),
                "mean_reported_cqi": safe_float(validation, "mean_reported_cqi"),
                "mean_csi_abs_error": safe_float(validation, "mean_csi_abs_error"),
                "max_p95_csi_abs_error": safe_float(validation, "max_p95_csi_abs_error"),
                "mean_csi_stale_fraction": safe_float(validation, "mean_csi_stale_fraction"),
                "mean_csi_report_age_slots": safe_float(validation, "mean_csi_report_age_slots"),
            }
        )

    metrics_output = Path(
        str(plan.analysis.get("metrics_output", output_path.with_suffix(".csv")))
    )
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = next(
        (row for row in rows if row["csi_report_mode"] == "perfect"), rows[0]
    )
    md_lines = [
        "# Round 11 — CSI reporting screen",
        "",
        "Slow Dynamic CQI, T–J–S reward, PPO, traffic and HARQ remain fixed. Only the CSI observation path changes.",
        "",
        "| Case | CSI | Period | Delay | Error σ | Goodput | Jain | Starvation | P99 | Max wait | Mean CSI |error| | Stale UE | Mean age |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['label']} | {row['csi_report_mode']} | {row['csi_report_period_slots']} | "
            f"{row['csi_report_delay_slots']} | {row['csi_report_error_std']:.2f} | "
            f"{row['goodput_bits_per_slot']:.0f} | {row['jain_fairness']:.4f} | "
            f"{100*row['starvation_rate']:.2f}% | {row['p99_wait_slots']:.0f} | "
            f"{row['max_wait_slots']:.0f} | {row['mean_csi_abs_error']:.3f} | "
            f"{100*row['mean_csi_stale_fraction']:.1f}% | {row['mean_csi_report_age_slots']:.2f} |"
        )
    md_lines += [
        "",
        "## Interpretation",
        "The true CQI still drives the PHY abstraction. The actor sees reported CQI only. This isolates CSI staleness/noise without changing the PPO architecture or reward.",
        "This round intentionally does not add MCS selection or CQI-dependent BLER yet.",
    ]
    markdown_output = Path(
        str(plan.analysis.get("markdown_output", output_path.with_suffix(".md")))
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    body_rows: list[str] = []
    for row in rows:
        dg = row["goodput_bits_per_slot"] - baseline["goodput_bits_per_slot"]
        dj = row["jain_fairness"] - baseline["jain_fairness"]
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['csi_report_mode']))}</td>"
            f"<td>{row['csi_report_period_slots']}</td>"
            f"<td>{row['csi_report_delay_slots']}</td>"
            f"<td>{row['csi_report_error_std']:.2f}</td>"
            f"<td>{row['goodput_bits_per_slot']:.0f} ({dg:+.0f})</td>"
            f"<td>{row['jain_fairness']:.4f} ({dj:+.4f})</td>"
            f"<td>{100*row['starvation_rate']:.2f}%</td>"
            f"<td>{row['p99_wait_slots']:.0f}</td>"
            f"<td>{row['max_wait_slots']:.0f}</td>"
            f"<td>{row['mean_csi_abs_error']:.3f}</td>"
            f"<td>{100*row['mean_csi_stale_fraction']:.1f}%</td>"
            f"<td>{row['mean_csi_report_age_slots']:.2f}</td>"
            "</tr>"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 11 CSI reporting</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1180px;margin:32px auto;padding:0 16px;line-height:1.55}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f5f7}"
        ".note{padding:14px;background:#eef5ff;border-left:4px solid #3568a8}</style></head><body>"
        "<h1>Round 11 — CSI reporting screen</h1>"
        "<p>All cases use Slow Dynamic CQI and the selected T–J–S reward. The only experimental variable is what CQI the scheduler receives through the CSI reporting path.</p>"
        "<div class='note'><b>Scope:</b> true CQI continues to determine the PHY efficiency; reported CQI drives the actor observation. MCS, CQI-dependent BLER and transmission layers are not changed in this round.</div>"
        "<table><thead><tr><th>Case</th><th>Mode</th><th>Period</th><th>Delay</th><th>Error σ</th>"
        "<th>Goodput (Δ)</th><th>Jain (Δ)</th><th>Starvation</th><th>P99</th><th>Max wait</th>"
        "<th>Mean |CSI error|</th><th>Stale UE</th><th>Mean report age</th></tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return output_path
