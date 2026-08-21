from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from scalemac_rl.reward_study import RewardStudyPlan


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _b(row: dict[str, str], key: str, default: bool = False) -> bool:
    value = str(row.get(key, str(default))).strip().lower()
    return value in {"1", "true", "yes"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_service_rollback_analysis(*, plan: RewardStudyPlan, round_dir: Path, output_path: Path) -> Path:
    metrics: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for case in plan.cases:
        validation = _read(round_dir / case.case_id / "validation_summary.csv")
        training = _read(round_dir / case.case_id / "training.csv")
        if not validation:
            continue
        latest = validation[-1]
        profile = case.case_id.rsplit("_seed", 1)[0]
        effective_feasible = [_b(row, "post_rollback_service_feasible", _b(row, "service_feasible")) for row in validation]
        rollback_count = int(max((_f(row, "rollback_count") for row in validation), default=0.0))
        tail = training[-max(1, len(training)//5):] if training else []
        metrics.append({
            "case_id": case.case_id,
            "profile": profile,
            "seed": int(case.common_overrides.get("seed", 0)),
            "ever_service_feasible": int(any(effective_feasible)),
            "latest_service_feasible": int(effective_feasible[-1]),
            "learn_then_drift_after_protection": int(any(effective_feasible) and not effective_feasible[-1]),
            "rollback_count": rollback_count,
            "latest_goodput_bits_per_slot": _f(latest, "post_rollback_goodput_bits_per_slot", _f(latest, "mean_goodput_bits_per_slot")),
            "latest_jain_fairness": _f(latest, "post_rollback_jain_fairness", _f(latest, "mean_jain_fairness")),
            "latest_starvation_rate": _f(latest, "post_rollback_starvation_rate", _f(latest, "worst_starvation_rate")),
            "latest_p99_wait_slots": _f(latest, "post_rollback_p99_wait_slots", _f(latest, "worst_p99_wait_slots")),
            "latest_max_wait_slots": _f(latest, "post_rollback_max_wait_slots", _f(latest, "worst_max_wait_slots")),
            "final_rollback_lr_multiplier": _f(latest, "rollback_lr_multiplier_after_validation", 1.0),
            "tail_mean_approx_kl": sum(_f(r, "approx_kl") for r in tail) / max(len(tail), 1),
        })
        for row in validation:
            trajectory.append({
                "case_id": case.case_id,
                "profile": profile,
                "seed": int(case.common_overrides.get("seed", 0)),
                "update": int(_f(row, "update")),
                "global_env_steps": int(_f(row, "global_env_steps")),
                "pre_rollback_service_feasible": int(_b(row, "service_feasible")),
                "rolled_back": int(_b(row, "rolled_back")),
                "post_rollback_service_feasible": int(_b(row, "post_rollback_service_feasible", _b(row, "service_feasible"))),
                "rollback_count": int(_f(row, "rollback_count")),
                "rollback_lr_multiplier": _f(row, "rollback_lr_multiplier_after_validation", 1.0),
                "starvation_rate": _f(row, "worst_starvation_rate"),
                "p99_wait_slots": _f(row, "worst_p99_wait_slots"),
                "jain_fairness": _f(row, "mean_jain_fairness"),
            })

    if len(metrics) != 9:
        raise ValueError(f"service rollback analysis requires 9 completed cases; found {len(metrics)}")

    summaries: list[dict[str, Any]] = []
    profiles = ["baseline", "rollback", "rollback_lr50"]
    for profile in profiles:
        rows = [r for r in metrics if r["profile"] == profile]
        summaries.append({
            "profile": profile,
            "latest_service_feasible_seeds": sum(int(r["latest_service_feasible"]) for r in rows),
            "ever_service_feasible_seeds": sum(int(r["ever_service_feasible"]) for r in rows),
            "learn_then_drift_seeds": sum(int(r["learn_then_drift_after_protection"]) for r in rows),
            "total_rollbacks": sum(int(r["rollback_count"]) for r in rows),
            "mean_latest_goodput_bits_per_slot": sum(float(r["latest_goodput_bits_per_slot"]) for r in rows)/3.0,
            "mean_latest_jain_fairness": sum(float(r["latest_jain_fairness"]) for r in rows)/3.0,
            "mean_latest_starvation_rate": sum(float(r["latest_starvation_rate"]) for r in rows)/3.0,
            "mean_final_rollback_lr_multiplier": sum(float(r["final_rollback_lr_multiplier"]) for r in rows)/3.0,
        })

    ranked = sorted(
        summaries,
        key=lambda r: (
            int(r["latest_service_feasible_seeds"]),
            int(r["ever_service_feasible_seeds"]),
            -int(r["learn_then_drift_seeds"]),
            float(r["mean_latest_jain_fairness"]),
        ),
        reverse=True,
    )
    decision = {
        "recommended_profile": ranked[0]["profile"],
        "latest_service_feasible_seeds": ranked[0]["latest_service_feasible_seeds"],
        "ever_service_feasible_seeds": ranked[0]["ever_service_feasible_seeds"],
        "total_rollbacks": ranked[0]["total_rollbacks"],
        "selection_rule": "robustness first: latest service-feasible seeds, ever feasible seeds, no post-protection drift, then Jain fairness",
    }

    a = plan.analysis
    _write_csv(Path(str(a["metrics_output"])), metrics)
    _write_csv(Path(str(a["summary_output"])), summaries)
    _write_csv(Path(str(a["trajectory_output"])), trajectory)
    dpath = Path(str(a["decision_output"])); dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    rows_html = "".join(
        "<tr>" + f"<td>{html.escape(str(r['profile']))}</td>" +
        f"<td>{r['latest_service_feasible_seeds']}/3</td><td>{r['ever_service_feasible_seeds']}/3</td>" +
        f"<td>{r['learn_then_drift_seeds']}/3</td><td>{r['total_rollbacks']}</td>" +
        f"<td>{float(r['mean_latest_goodput_bits_per_slot']):,.0f}</td>" +
        f"<td>{float(r['mean_latest_jain_fairness']):.4f}</td>" +
        f"<td>{100*float(r['mean_latest_starvation_rate']):.2f}%</td></tr>"
        for r in summaries
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round 17B</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1100px;margin:32px auto;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:7px}th{background:#eee}</style>"
        "</head><body><h1>Round 17B — Service-Aware Rollback</h1>"
        f"<p><b>Recommended:</b> {html.escape(str(decision['recommended_profile']))}</p>"
        "<table><thead><tr><th>Profile</th><th>Latest feasible</th><th>Ever feasible</th><th>Drift</th>"
        "<th>Rollbacks</th><th>Goodput</th><th>JFI</th><th>Starvation</th></tr></thead><tbody>" + rows_html +
        "</tbody></table></body></html>", encoding="utf-8"
    )
    md = Path(str(a.get("markdown_output", output_path.with_suffix(".md"))))
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        f"# Round 17B — Service-Aware Rollback\n\nRecommended profile: **{decision['recommended_profile']}**.\n",
        encoding="utf-8",
    )
    return output_path
