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


def build_link_adaptation_analysis(
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
                "link_adaptation_mode": str(
                    common.get("link_adaptation_mode", "legacy_fixed_bler")
                ),
                "csi_report_mode": str(common.get("csi_report_mode", "perfect")),
                "csi_period_slots": int(common.get("csi_report_period_slots", 1)),
                "csi_delay_slots": int(common.get("csi_report_delay_slots", 0)),
                "csi_error_std": float(common.get("csi_report_error_std", 0.0)),
                "cqi_backoff": int(common.get("link_adaptation_cqi_backoff", 0)),
                "bler_mismatch_slope": float(common.get("bler_mismatch_slope", 1.5)),
                "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                "spectral_efficiency_bps_hz": safe_float(
                    validation, "mean_spectral_efficiency_bps_hz"
                ),
                "attempted_spectral_efficiency_bps_hz": safe_float(
                    validation, "mean_attempted_spectral_efficiency_bps_hz"
                ),
                "jain_fairness": safe_float(validation, "final_jain_fairness"),
                "starvation_rate": safe_float(validation, "max_starvation_rate"),
                "p99_wait_slots": safe_float(validation, "max_p99_wait_slots"),
                "max_wait_slots": safe_float(validation, "max_wait_slots"),
                "mean_mcs_index": safe_float(validation, "mean_mcs_index", -1.0),
                "mean_modulation_order": safe_float(
                    validation, "mean_modulation_order"
                ),
                "mean_predicted_bler": safe_float(validation, "mean_predicted_bler"),
                "mean_observed_bler": safe_float(validation, "mean_observed_bler"),
                "harq_retx_fraction": safe_float(
                    validation, "mean_harq_retransmission_fraction"
                ),
                "mean_csi_abs_error": safe_float(validation, "mean_csi_abs_error"),
                "mean_csi_report_age_slots": safe_float(
                    validation, "mean_csi_report_age_slots"
                ),
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

    markdown_output = Path(
        str(plan.analysis.get("markdown_output", output_path.with_suffix(".md")))
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Round 12 — CQI/MCS/BLER Link Adaptation foundation",
        "",
        "T–J–S reward and PPO remain fixed. The screen adds an NR-inspired CQI→MCS mapping and a smooth true-CQI/MCS mismatch BLER abstraction.",
        "",
        "| Case | LA mode | CSI | Goodput | SE | Jain | BLER | HARQ retx | Starvation | P99 | Max wait | Mean MCS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['label']} | {row['link_adaptation_mode']} | "
            f"{row['csi_report_mode']} p={row['csi_period_slots']} d={row['csi_delay_slots']} σ={row['csi_error_std']:.2f} | "
            f"{row['goodput_bits_per_slot']:.0f} | {row['spectral_efficiency_bps_hz']:.4f} | "
            f"{row['jain_fairness']:.4f} | {100*row['mean_observed_bler']:.2f}% | "
            f"{100*row['harq_retx_fraction']:.2f}% | {100*row['starvation_rate']:.2f}% | "
            f"{row['p99_wait_slots']:.0f} | {row['max_wait_slots']:.0f} | {row['mean_mcs_index']:.2f} |"
        )
    md += [
        "",
        "## Scope",
        "The MCS table is taken from the 3GPP NR PDSCH MCS Table-1 structure, but the BLER function is an explicit simulator abstraction rather than a link-level 3GPP BLER curve. Exact BLER would require SINR/channel/coding/receiver modeling.",
    ]
    markdown_output.write_text("\n".join(md) + "\n", encoding="utf-8")

    body: list[str] = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['link_adaptation_mode']))}</td>"
            f"<td>{html.escape(str(row['csi_report_mode']))}; p={row['csi_period_slots']}; d={row['csi_delay_slots']}; σ={row['csi_error_std']:.2f}</td>"
            f"<td>{row['goodput_bits_per_slot']:.0f}</td>"
            f"<td>{row['spectral_efficiency_bps_hz']:.4f}</td>"
            f"<td>{row['jain_fairness']:.4f}</td>"
            f"<td>{100*row['mean_observed_bler']:.2f}%</td>"
            f"<td>{100*row['harq_retx_fraction']:.2f}%</td>"
            f"<td>{100*row['starvation_rate']:.2f}%</td>"
            f"<td>{row['p99_wait_slots']:.0f}</td>"
            f"<td>{row['max_wait_slots']:.0f}</td>"
            f"<td>{row['mean_mcs_index']:.2f}</td>"
            f"<td>{row['mean_csi_abs_error']:.3f}</td>"
            "</tr>"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 12 Link Adaptation</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1240px;margin:32px auto;padding:0 16px;line-height:1.55}"
        "table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{border:1px solid #ddd;padding:8px}"
        "th{background:#f3f5f7}.note{padding:14px;background:#fff7ed;border-left:4px solid #d97706}</style></head><body>"
        "<h1>Round 12 — CQI/MCS/BLER Link Adaptation foundation</h1>"
        "<p>T–J–S, PPO, traffic and scheduler action stay fixed. This round introduces a link-adaptation PHY abstraction so stale CSI can cause an MCS/channel mismatch and therefore a changing BLER.</p>"
        "<div class='note'><b>Important:</b> CQI and MCS table values are NR-inspired/tabulated; the smooth BLER-vs-mismatch curve is a simulator model, not a claim of exact 3GPP link-level BLER.</div>"
        "<table><thead><tr><th>Case</th><th>LA</th><th>CSI</th><th>Goodput</th><th>Spectral eff.</th><th>Jain</th><th>Observed BLER</th><th>HARQ retx</th><th>Starvation</th><th>P99</th><th>Max wait</th><th>Mean MCS</th><th>CSI |error|</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return output_path
