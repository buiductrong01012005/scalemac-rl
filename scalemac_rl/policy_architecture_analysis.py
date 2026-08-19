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


def _summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def build_policy_architecture_analysis(
    *, plan: RewardStudyPlan, round_dir: Path, output_path: Path
) -> Path:
    rows: list[dict[str, Any]] = []
    for case in plan.cases:
        validation = _last_validation(round_dir / case.case_id)
        common = dict(plan.common)
        common.update(case.common_overrides)
        rows.append(
            {
                "case_id": case.case_id,
                "label": case.label,
                "architecture": str(common.get("policy_architecture", "feedforward")),
                "seed": int(common.get("seed", 1701)),
                "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                "spectral_efficiency_bps_hz": safe_float(validation, "mean_spectral_efficiency_bps_hz"),
                "jain_fairness": safe_float(validation, "final_jain_fairness"),
                "starvation_rate": safe_float(validation, "max_starvation_rate"),
                "p99_wait_slots": safe_float(validation, "max_p99_wait_slots"),
                "max_wait_slots": safe_float(validation, "max_wait_slots"),
                "observed_bler": safe_float(validation, "mean_observed_bler"),
                "harq_retx_fraction": safe_float(validation, "mean_harq_retransmission_fraction"),
                "mean_inference_us": safe_float(validation, "mean_inference_us"),
                "p99_inference_us": safe_float(validation, "p99_inference_us"),
            }
        )

    metrics_output = Path(str(plan.analysis.get("metrics_output", output_path.with_suffix(".csv"))))
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    for architecture in sorted({str(row["architecture"]) for row in rows}):
        subset = [row for row in rows if row["architecture"] == architecture]
        entry: dict[str, Any] = {"architecture": architecture, "seeds": len(subset)}
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
            "p99_inference_us",
        ):
            avg, sd = _summary([float(row[key]) for row in subset])
            entry[f"mean_{key}"] = avg
            entry[f"std_{key}"] = sd
        entry["zero_starvation_seeds"] = sum(float(row["starvation_rate"]) <= 1e-12 for row in subset)
        summary_rows.append(entry)

    summary_output = Path(str(plan.analysis.get("summary_output", output_path.with_name("policy_architecture_summary.csv"))))
    with summary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_output = Path(str(plan.analysis.get("markdown_output", output_path.with_suffix(".md"))))
    lines = [
        "# Round 13 — PPO vs RPPO checkpoint",
        "",
        "Environment, T–J–S reward, CSI, MCS and BLER are fixed. Only policy architecture changes.",
        "",
        "| Architecture | Seeds | Goodput mean±sd | SE mean±sd | JFI mean±sd | Zero-starvation | P99 mean±sd | BLER mean±sd | P99 inference us |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['architecture']} | {row['seeds']} | "
            f"{row['mean_goodput_bits_per_slot']:.0f}±{row['std_goodput_bits_per_slot']:.0f} | "
            f"{row['mean_spectral_efficiency_bps_hz']:.4f}±{row['std_spectral_efficiency_bps_hz']:.4f} | "
            f"{row['mean_jain_fairness']:.4f}±{row['std_jain_fairness']:.4f} | "
            f"{row['zero_starvation_seeds']}/{row['seeds']} | "
            f"{row['mean_p99_wait_slots']:.1f}±{row['std_p99_wait_slots']:.1f} | "
            f"{100*row['mean_observed_bler']:.2f}%±{100*row['std_observed_bler']:.2f}% | "
            f"{row['mean_p99_inference_us']:.1f} |"
        )
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{row['seed']}</td>"
            f"<td>{html.escape(str(row['architecture']))}</td>"
            f"<td>{row['goodput_bits_per_slot']:.0f}</td>"
            f"<td>{row['spectral_efficiency_bps_hz']:.4f}</td>"
            f"<td>{row['jain_fairness']:.4f}</td>"
            f"<td>{100*row['starvation_rate']:.2f}%</td>"
            f"<td>{row['p99_wait_slots']:.0f}</td>"
            f"<td>{row['max_wait_slots']:.0f}</td>"
            f"<td>{100*row['observed_bler']:.2f}%</td>"
            f"<td>{row['p99_inference_us']:.1f}</td>"
            "</tr>"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 13 PPO vs RPPO</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1200px;margin:32px auto;padding:0 16px;line-height:1.55}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f5f7}</style>"
        "</head><body><h1>Round 13 — PPO vs RPPO checkpoint</h1>"
        "<p>Only policy architecture changes. RPPO uses a shared per-UE GRU with truncated BPTT; channel, CSI delay, CQI→MCS→BLER, T–J–S and PPO objective settings are held fixed.</p>"
        "<table><thead><tr><th>Case</th><th>Seed</th><th>Architecture</th><th>Goodput</th><th>SE</th><th>JFI</th><th>Starv.</th><th>P99</th><th>Max wait</th><th>BLER</th><th>P99 inference us</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return output_path
