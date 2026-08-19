from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from .reward_study import RewardStudyPlan, read_csv_rows, safe_float


SERVICE_STARVATION_LIMIT = 0.0
SERVICE_P99_LIMIT = 50.0
SERVICE_MAX_WAIT_LIMIT = 60.0
_TOL = 1e-12


def _profile(case) -> str:
    w = case.positive_weights
    t = float(w["throughput"])
    j = float(w["fairness"])
    s = float(w["service"])
    if abs(t - 1 / 3) < 1e-9 and abs(j - 1 / 3) < 1e-9 and abs(s - 1 / 3) < 1e-9:
        return "equal"
    if abs(t - 0.4) < 1e-9 and abs(j - 0.3) < 1e-9 and abs(s - 0.3) < 1e-9:
        return "throughput40"
    if abs(t - 0.3) < 1e-9 and abs(j - 0.4) < 1e-9 and abs(s - 0.3) < 1e-9:
        return "jain40"
    if abs(t - 0.3) < 1e-9 and abs(j - 0.3) < 1e-9 and abs(s - 0.4) < 1e-9:
        return "service40"
    return f"t{t:.3f}_j{j:.3f}_s{s:.3f}"


def _service_feasible(row: Mapping[str, Any]) -> bool:
    return (
        safe_float(row, "max_starvation_rate") <= SERVICE_STARVATION_LIMIT + _TOL
        and safe_float(row, "max_p99_wait_slots") <= SERVICE_P99_LIMIT + _TOL
        and safe_float(row, "max_wait_slots") <= SERVICE_MAX_WAIT_LIMIT + _TOL
    )


def _target_feasible(row: Mapping[str, Any]) -> bool:
    raw = str(row.get("constraint_feasible", "")).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    return (
        _service_feasible(row)
        and safe_float(row, "final_jain_fairness") >= 0.60 - _TOL
    )


def _full_collapse(row: Mapping[str, Any]) -> bool:
    return (
        safe_float(row, "max_starvation_rate") >= 0.5
        or safe_float(row, "max_p99_wait_slots") >= 4999.0
        or safe_float(row, "max_wait_slots") >= 4999.0
    )


def _find_best_tradeoff_row(case_dir: Path, validations: list[dict[str, str]]) -> tuple[dict[str, str], int]:
    manifest = read_csv_rows(case_dir / "checkpoint_manifest.csv")
    tradeoff_rows = [row for row in manifest if row.get("tag") == "best_tradeoff"]
    if not tradeoff_rows:
        raise ValueError(f"missing best_tradeoff checkpoint manifest row for {case_dir.name}")
    update = int(float(tradeoff_rows[-1]["update"]))
    matches = [row for row in validations if int(float(row.get("update", -1))) == update]
    if not matches:
        raise ValueError(
            f"best_tradeoff update {update} has no matching validation row for {case_dir.name}"
        )
    return matches[-1], update


def _metric_payload(prefix: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_update": int(float(row.get("update", 0))),
        f"{prefix}_global_env_steps": int(float(row.get("global_env_steps", 0))),
        f"{prefix}_goodput_bits_per_slot": safe_float(row, "mean_goodput_bits_per_slot"),
        f"{prefix}_spectral_efficiency_bps_hz": safe_float(row, "mean_spectral_efficiency_bps_hz"),
        f"{prefix}_jain_fairness": safe_float(row, "final_jain_fairness"),
        f"{prefix}_starvation_rate": safe_float(row, "max_starvation_rate"),
        f"{prefix}_p99_wait_slots": safe_float(row, "max_p99_wait_slots"),
        f"{prefix}_max_wait_slots": safe_float(row, "max_wait_slots"),
        f"{prefix}_observed_bler": safe_float(row, "mean_observed_bler"),
        f"{prefix}_harq_retx_fraction": safe_float(row, "mean_harq_retransmission_fraction"),
        f"{prefix}_target_reward": safe_float(row, "mean_final_target_reward"),
        f"{prefix}_service_feasible": int(_service_feasible(row)),
        f"{prefix}_target_feasible": int(_target_feasible(row)),
        f"{prefix}_full_collapse": int(_full_collapse(row)),
    }


def _summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_reward_checkpoint_stability_analysis(
    *, plan: RewardStudyPlan, round_dir: Path, output_path: Path
) -> Path:
    case_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    for case in plan.cases:
        case_dir = round_dir / case.case_id
        validations = read_csv_rows(case_dir / "validation.csv")
        if not validations:
            raise ValueError(f"missing validation.csv rows for {case.case_id}")

        common = dict(plan.common)
        common.update(case.common_overrides)
        seed = int(common.get("seed", 1701))
        profile = _profile(case)
        best, best_update = _find_best_tradeoff_row(case_dir, validations)
        latest = validations[-1]

        feasible_rows = [row for row in validations if _service_feasible(row)]
        target_feasible_rows = [row for row in validations if _target_feasible(row)]
        ever_service = bool(feasible_rows)
        latest_service = _service_feasible(latest)
        best_service = _service_feasible(best)
        drift = ever_service and not latest_service
        never_service = not ever_service

        first_feasible_step = (
            min(int(float(row.get("global_env_steps", 0))) for row in feasible_rows)
            if feasible_rows else -1
        )
        last_feasible_step = (
            max(int(float(row.get("global_env_steps", 0))) for row in feasible_rows)
            if feasible_rows else -1
        )

        row: dict[str, Any] = {
            "case_id": case.case_id,
            "label": case.label,
            "profile": profile,
            "seed": seed,
            "weight_throughput": float(case.positive_weights["throughput"]),
            "weight_jain": float(case.positive_weights["fairness"]),
            "weight_service": float(case.positive_weights["service"]),
            "validation_points": len(validations),
            "service_feasible_validation_points": len(feasible_rows),
            "service_feasible_validation_fraction": len(feasible_rows) / len(validations),
            "target_feasible_validation_points": len(target_feasible_rows),
            "ever_service_feasible": int(ever_service),
            "never_service_feasible": int(never_service),
            "learn_then_drift": int(drift),
            "latest_service_feasible": int(latest_service),
            "best_tradeoff_service_feasible": int(best_service),
            "first_service_feasible_global_env_steps": first_feasible_step,
            "last_service_feasible_global_env_steps": last_feasible_step,
            "best_tradeoff_manifest_update": best_update,
        }
        row.update(_metric_payload("best", best))
        row.update(_metric_payload("latest", latest))
        row.update({
            "latest_minus_best_goodput": row["latest_goodput_bits_per_slot"] - row["best_goodput_bits_per_slot"],
            "latest_minus_best_jain": row["latest_jain_fairness"] - row["best_jain_fairness"],
            "latest_minus_best_starvation": row["latest_starvation_rate"] - row["best_starvation_rate"],
            "latest_minus_best_p99": row["latest_p99_wait_slots"] - row["best_p99_wait_slots"],
            "latest_minus_best_max_wait": row["latest_max_wait_slots"] - row["best_max_wait_slots"],
        })
        case_rows.append(row)

        for validation in validations:
            trajectory_rows.append({
                "case_id": case.case_id,
                "profile": profile,
                "seed": seed,
                "update": int(float(validation.get("update", 0))),
                "global_env_steps": int(float(validation.get("global_env_steps", 0))),
                "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                "spectral_efficiency_bps_hz": safe_float(validation, "mean_spectral_efficiency_bps_hz"),
                "jain_fairness": safe_float(validation, "final_jain_fairness"),
                "starvation_rate": safe_float(validation, "max_starvation_rate"),
                "p99_wait_slots": safe_float(validation, "max_p99_wait_slots"),
                "max_wait_slots": safe_float(validation, "max_wait_slots"),
                "service_feasible": int(_service_feasible(validation)),
                "target_feasible": int(_target_feasible(validation)),
                "is_best_tradeoff": int(int(float(validation.get("update", 0))) == best_update),
                "is_latest": int(validation is latest),
            })

    metrics_output = Path(str(plan.analysis.get("metrics_output", output_path.with_suffix(".csv"))))
    trajectory_output = Path(str(plan.analysis.get("trajectory_output", output_path.with_name("trajectory.csv"))))
    summary_output = Path(str(plan.analysis.get("summary_output", output_path.with_name("profile_summary.csv"))))
    ranking_output = Path(str(plan.analysis.get("ranking_output", output_path.with_name("profile_ranking.csv"))))
    decision_output = Path(str(plan.analysis.get("decision_output", output_path.with_name("decision.json"))))
    markdown_output = Path(str(plan.analysis.get("markdown_output", output_path.with_suffix(".md"))))

    _write_csv(metrics_output, case_rows)
    _write_csv(trajectory_output, trajectory_rows)

    profile_order = ["equal", "throughput40", "jain40", "service40"]
    summary_rows: list[dict[str, Any]] = []
    for profile in profile_order:
        subset = [row for row in case_rows if row["profile"] == profile]
        if not subset:
            continue
        entry: dict[str, Any] = {
            "profile": profile,
            "seeds": len(subset),
            "weight_throughput": subset[0]["weight_throughput"],
            "weight_jain": subset[0]["weight_jain"],
            "weight_service": subset[0]["weight_service"],
            "ever_service_feasible_seeds": sum(int(row["ever_service_feasible"]) for row in subset),
            "best_tradeoff_service_feasible_seeds": sum(int(row["best_tradeoff_service_feasible"]) for row in subset),
            "latest_service_feasible_seeds": sum(int(row["latest_service_feasible"]) for row in subset),
            "never_service_feasible_seeds": sum(int(row["never_service_feasible"]) for row in subset),
            "learn_then_drift_seeds": sum(int(row["learn_then_drift"]) for row in subset),
            "latest_full_collapse_seeds": sum(int(row["latest_full_collapse"]) for row in subset),
            "mean_service_feasible_validation_fraction": mean(
                float(row["service_feasible_validation_fraction"]) for row in subset
            ),
        }
        for prefix in ("best", "latest"):
            for metric in (
                "goodput_bits_per_slot",
                "spectral_efficiency_bps_hz",
                "jain_fairness",
                "starvation_rate",
                "p99_wait_slots",
                "max_wait_slots",
                "observed_bler",
            ):
                avg, sd = _summary([float(row[f"{prefix}_{metric}"]) for row in subset])
                entry[f"mean_{prefix}_{metric}"] = avg
                entry[f"std_{prefix}_{metric}"] = sd
        summary_rows.append(entry)

    _write_csv(summary_output, summary_rows)

    # Robustness-first ranking. The ranking is descriptive rather than a significance test.
    ranking_rows = sorted(
        summary_rows,
        key=lambda row: (
            -int(row["latest_service_feasible_seeds"]),
            -int(row["ever_service_feasible_seeds"]),
            int(row["never_service_feasible_seeds"]),
            int(row["learn_then_drift_seeds"]),
            int(row["latest_full_collapse_seeds"]),
            float(row["mean_latest_starvation_rate"]),
            float(row["mean_latest_p99_wait_slots"]),
            -float(row["mean_latest_jain_fairness"]),
            -float(row["mean_latest_goodput_bits_per_slot"]),
        ),
    )
    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(ranking_rows, start=1):
        ranked.append({
            "rank": rank,
            "profile": row["profile"],
            "latest_service_feasible_seeds": row["latest_service_feasible_seeds"],
            "ever_service_feasible_seeds": row["ever_service_feasible_seeds"],
            "never_service_feasible_seeds": row["never_service_feasible_seeds"],
            "learn_then_drift_seeds": row["learn_then_drift_seeds"],
            "latest_full_collapse_seeds": row["latest_full_collapse_seeds"],
            "mean_latest_goodput_bits_per_slot": row["mean_latest_goodput_bits_per_slot"],
            "mean_latest_jain_fairness": row["mean_latest_jain_fairness"],
            "mean_latest_starvation_rate": row["mean_latest_starvation_rate"],
            "mean_latest_p99_wait_slots": row["mean_latest_p99_wait_slots"],
        })
    _write_csv(ranking_output, ranked)

    winner = ranked[0]
    decision = {
        "recommended_profile": winner["profile"],
        "ranking_rule": "robustness_first_latest_then_ever_feasibility_then_drift_then_kpis",
        "latest_service_feasible_seeds": int(winner["latest_service_feasible_seeds"]),
        "ever_service_feasible_seeds": int(winner["ever_service_feasible_seeds"]),
        "never_service_feasible_seeds": int(winner["never_service_feasible_seeds"]),
        "learn_then_drift_seeds": int(winner["learn_then_drift_seeds"]),
        "caution": "Ranking is descriptive across three paired seeds; do not claim statistical significance.",
        "next_step_if_no_profile_is_robust": "Keep the strongest controlled profile and test checkpoint/early-stop or constraint mechanisms before PPO-vs-RPPO.",
    }
    decision_output.parent.mkdir(parents=True, exist_ok=True)
    decision_output.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = [
        "# Round 15 — TJS Reward × Checkpoint Stability",
        "",
        "Only T/J/S positive reward weights change. PPO optimizer, 16-feature observation, feed-forward architecture, Dynamic CQI, delayed CSI and CQI→MCS→BLER/HARQ remain fixed.",
        "",
        "Service-feasible means max starvation = 0, max P99 wait ≤ 50 slots, and max wait ≤ 60 slots. This is intentionally separate from the stricter full target that also includes Jain ≥ 0.60.",
        "",
        "| Profile | Ever feasible | Latest feasible | Never feasible | Learn→drift | Latest collapse | Latest GP | Latest JFI | Latest starvation | Latest P99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['profile']} | {row['ever_service_feasible_seeds']}/{row['seeds']} | "
            f"{row['latest_service_feasible_seeds']}/{row['seeds']} | {row['never_service_feasible_seeds']}/{row['seeds']} | "
            f"{row['learn_then_drift_seeds']}/{row['seeds']} | {row['latest_full_collapse_seeds']}/{row['seeds']} | "
            f"{row['mean_latest_goodput_bits_per_slot']:.0f} | {row['mean_latest_jain_fairness']:.4f} | "
            f"{100*row['mean_latest_starvation_rate']:.2f}% | {row['mean_latest_p99_wait_slots']:.1f} |"
        )
    lines.extend(["", f"Recommended by robustness-first ranking: **{winner['profile']}**.", ""])
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(lines), encoding="utf-8")

    profile_table = []
    for row in summary_rows:
        profile_table.append(
            "<tr>"
            f"<td>{html.escape(str(row['profile']))}</td>"
            f"<td>{row['weight_throughput']:.3f}/{row['weight_jain']:.3f}/{row['weight_service']:.3f}</td>"
            f"<td>{row['ever_service_feasible_seeds']}/{row['seeds']}</td>"
            f"<td>{row['latest_service_feasible_seeds']}/{row['seeds']}</td>"
            f"<td>{row['never_service_feasible_seeds']}/{row['seeds']}</td>"
            f"<td>{row['learn_then_drift_seeds']}/{row['seeds']}</td>"
            f"<td>{row['mean_latest_goodput_bits_per_slot']:.0f}</td>"
            f"<td>{row['mean_latest_jain_fairness']:.4f}</td>"
            f"<td>{100*row['mean_latest_starvation_rate']:.2f}%</td>"
            f"<td>{row['mean_latest_p99_wait_slots']:.1f}</td>"
            "</tr>"
        )

    case_table = []
    for row in case_rows:
        failure = "never-feasible" if row["never_service_feasible"] else ("learn→drift" if row["learn_then_drift"] else "retained")
        case_table.append(
            "<tr>"
            f"<td>{html.escape(str(row['case_id']))}</td><td>{row['seed']}</td><td>{failure}</td>"
            f"<td>{row['best_goodput_bits_per_slot']:.0f}</td><td>{row['best_jain_fairness']:.4f}</td>"
            f"<td>{100*row['best_starvation_rate']:.2f}%</td><td>{row['best_p99_wait_slots']:.0f}</td>"
            f"<td>{row['latest_goodput_bits_per_slot']:.0f}</td><td>{row['latest_jain_fairness']:.4f}</td>"
            f"<td>{100*row['latest_starvation_rate']:.2f}%</td><td>{row['latest_p99_wait_slots']:.0f}</td>"
            "</tr>"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 15 TJS Reward × Checkpoint Stability</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1280px;margin:32px auto;padding:0 16px;line-height:1.55;color:#1f2937}"
        "table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{border:1px solid #ddd;padding:8px}th{background:#f3f5f7}"
        ".callout{border-left:5px solid #2563eb;background:#eff6ff;padding:14px 16px;margin:16px 0;border-radius:6px}</style>"
        "</head><body><h1>Round 15 — TJS Reward × Checkpoint Stability</h1>"
        "<p>Controlled reward re-check: only T/J/S weights change. All PPO, observation and radio settings are fixed.</p>"
        "<div class='callout'><b>Service-feasible definition:</b> max starvation = 0, max P99 wait ≤ 50, max wait ≤ 60. "
        "The report separately preserves the stricter full constraint target including Jain ≥ 0.60.</div>"
        "<h2>Profile robustness</h2><table><thead><tr><th>Profile</th><th>T/J/S</th><th>Ever feasible</th><th>Latest feasible</th>"
        "<th>Never feasible</th><th>Learn→drift</th><th>Latest GP</th><th>Latest JFI</th><th>Latest starvation</th><th>Latest P99</th></tr></thead><tbody>"
        + "".join(profile_table)
        + "</tbody></table>"
        f"<div class='callout'><b>Robustness-first recommendation:</b> {html.escape(str(winner['profile']))}. "
        "This is a descriptive three-seed screen, not a statistical significance claim.</div>"
        "<h2>Best-tradeoff vs latest by case</h2><table><thead><tr><th>Case</th><th>Seed</th><th>Failure mode</th>"
        "<th>Best GP</th><th>Best JFI</th><th>Best starvation</th><th>Best P99</th>"
        "<th>Latest GP</th><th>Latest JFI</th><th>Latest starvation</th><th>Latest P99</th></tr></thead><tbody>"
        + "".join(case_table)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    return output_path
