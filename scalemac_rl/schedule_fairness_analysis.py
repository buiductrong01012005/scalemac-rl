from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .reward_study import RewardStudyPlan


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return default if value in {None, ""} else float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _profile(case_id: str) -> str:
    marker = "_seed"
    return case_id.rsplit(marker, 1)[0] if marker in case_id else case_id


def _seed(case_id: str, run_config: dict[str, Any]) -> int:
    common = run_config.get("effective_common", run_config.get("common", {}))
    try:
        return int(common.get("seed", case_id.rsplit("_seed", 1)[-1]))
    except Exception:
        return -1


def _service_feasible(row: dict[str, Any]) -> bool:
    return (
        _f(row, "max_starvation_rate") <= 1e-12
        and _f(row, "max_p99_wait_slots", 1e9) <= 50.0
        and _f(row, "max_wait_slots", 1e9) <= 60.0
    )


def _full_feasible(row: dict[str, Any]) -> bool:
    return _service_feasible(row) and _f(row, "final_jain_fairness") >= 0.60


def build_schedule_fairness_analysis(
    *,
    plan_path: str | Path,
    round_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    plan = RewardStudyPlan.from_json(plan_path)
    round_root = Path(round_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in plan.cases:
        case_dir = round_root / case.case_id
        validation = _read_csv(case_dir / "validation.csv")
        training = _read_csv(case_dir / "training.csv")
        config_path = case_dir / "run_config.json"
        run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        if not validation:
            raise FileNotFoundError(f"missing/empty validation.csv for {case.case_id}")

        latest = validation[-1]
        feasible_rows = [row for row in validation if _service_feasible(row)]
        full_rows = [row for row in validation if _full_feasible(row)]
        tail_count = max(1, len(training) // 5) if training else 0
        tail = training[-tail_count:] if tail_count else []

        weights = case.positive_weights
        row = {
            "case_id": case.case_id,
            "profile": _profile(case.case_id),
            "seed": _seed(case.case_id, run_config),
            "weight_throughput": weights.get("throughput", 0.0),
            "weight_bandwidth_fairness": weights.get("fairness", 0.0),
            "weight_schedule_fairness": weights.get("schedule_fairness", 0.0),
            "weight_service": weights.get("service", 0.0),
            "latest_service_feasible": int(_service_feasible(latest)),
            "ever_service_feasible": int(bool(feasible_rows)),
            "learn_then_drift": int(bool(feasible_rows) and not _service_feasible(latest)),
            "latest_full_feasible": int(_full_feasible(latest)),
            "ever_full_feasible": int(bool(full_rows)),
            "latest_goodput_bits_per_slot": _f(latest, "mean_goodput_bits_per_slot"),
            "latest_jain_fairness": _f(latest, "final_jain_fairness"),
            "latest_schedule_fairness": _f(latest, "final_schedule_fairness"),
            "latest_mean_schedule_fairness_score": _f(latest, "mean_schedule_fairness_score"),
            "latest_short_schedule_fairness": _f(latest, "mean_short_term_schedule_fairness"),
            "latest_starvation_rate": _f(latest, "max_starvation_rate"),
            "latest_scheduling_starvation_rate": _f(latest, "max_scheduling_starvation_rate"),
            "latest_p99_wait_slots": _f(latest, "max_p99_wait_slots"),
            "latest_max_wait_slots": _f(latest, "max_wait_slots"),
            "latest_max_scheduling_wait_slots": _f(latest, "max_scheduling_wait_slots"),
            "latest_bler": _f(latest, "mean_observed_bler"),
            "latest_service_score": _f(latest, "mean_service_score"),
            "tail_train_jain": mean(_f(r, "mean_jain_fairness") for r in tail) if tail else 0.0,
            "tail_train_schedule_fairness": mean(_f(r, "mean_schedule_fairness_score") for r in tail) if tail else 0.0,
            "tail_train_starvation": mean(_f(r, "mean_starvation_rate") for r in tail) if tail else 0.0,
            "tail_train_scheduling_starvation": mean(_f(r, "mean_scheduling_starvation_rate") for r in tail) if tail else 0.0,
            "tail_train_approx_kl": mean(_f(r, "approx_kl") for r in tail) if tail else 0.0,
        }
        case_rows.append(row)
        grouped[row["profile"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for profile, rows in grouped.items():
        first = rows[0]
        summary_rows.append({
            "profile": profile,
            "weight_throughput": first["weight_throughput"],
            "weight_bandwidth_fairness": first["weight_bandwidth_fairness"],
            "weight_schedule_fairness": first["weight_schedule_fairness"],
            "weight_service": first["weight_service"],
            "latest_service_feasible_seeds": sum(r["latest_service_feasible"] for r in rows),
            "ever_service_feasible_seeds": sum(r["ever_service_feasible"] for r in rows),
            "learn_then_drift_seeds": sum(r["learn_then_drift"] for r in rows),
            "latest_full_feasible_seeds": sum(r["latest_full_feasible"] for r in rows),
            "ever_full_feasible_seeds": sum(r["ever_full_feasible"] for r in rows),
            "mean_latest_goodput_bits_per_slot": mean(r["latest_goodput_bits_per_slot"] for r in rows),
            "mean_latest_jain_fairness": mean(r["latest_jain_fairness"] for r in rows),
            "mean_latest_schedule_fairness": mean(r["latest_schedule_fairness"] for r in rows),
            "mean_latest_starvation_rate": mean(r["latest_starvation_rate"] for r in rows),
            "mean_latest_scheduling_starvation_rate": mean(r["latest_scheduling_starvation_rate"] for r in rows),
            "mean_latest_p99_wait_slots": mean(r["latest_p99_wait_slots"] for r in rows),
            "mean_latest_max_wait_slots": mean(r["latest_max_wait_slots"] for r in rows),
            "mean_latest_max_scheduling_wait_slots": mean(r["latest_max_scheduling_wait_slots"] for r in rows),
            "mean_latest_bler": mean(r["latest_bler"] for r in rows),
            "mean_tail_train_approx_kl": mean(r["tail_train_approx_kl"] for r in rows),
        })

    ranking = sorted(
        summary_rows,
        key=lambda r: (
            -int(r["latest_service_feasible_seeds"]),
            -int(r["ever_service_feasible_seeds"]),
            -float(r["mean_latest_schedule_fairness"]),
            -float(r["mean_latest_jain_fairness"]),
            -float(r["mean_latest_goodput_bits_per_slot"]),
        ),
    )
    ranking_rows = [{"rank": i + 1, **row} for i, row in enumerate(ranking)]

    baseline = next((r for r in summary_rows if r["profile"].startswith("baseline_")), None)
    best = ranking_rows[0] if ranking_rows else None
    decision = {
        "study_id": plan.study_id,
        "round_id": plan.round_id,
        "cases": len(case_rows),
        "profiles": len(summary_rows),
        "primary_metric": "multi-seed deterministic service feasibility",
        "secondary_metrics": [
            "schedule-frequency fairness",
            "throughput Jain fairness",
            "goodput",
            "scheduling starvation",
        ],
        "baseline_profile": baseline["profile"] if baseline else None,
        "baseline_latest_service_feasible_seeds": baseline["latest_service_feasible_seeds"] if baseline else None,
        "best_profile": best["profile"] if best else None,
        "best_latest_service_feasible_seeds": best["latest_service_feasible_seeds"] if best else None,
        "best_ever_service_feasible_seeds": best["ever_service_feasible_seeds"] if best else None,
        "note": (
            "Schedule-frequency fairness is computed from UE selection counts/rates, not delivered bits. "
            "It therefore isolates scheduling opportunity balance from CQI/BLER effects."
        ),
    }

    case_csv = out / "schedule_fairness_case_metrics.csv"
    summary_csv = out / "schedule_fairness_profile_summary.csv"
    ranking_csv = out / "schedule_fairness_ranking.csv"
    decision_json = out / "schedule_fairness_decision.json"
    html_path = out / "schedule_fairness_analysis.html"
    _write_csv(case_csv, case_rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(ranking_csv, ranking_rows)
    decision_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    table_rows = []
    for row in ranking_rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['rank']}</td>"
            f"<td>{html.escape(str(row['profile']))}</td>"
            f"<td>{row['weight_throughput']:.2f}/{row['weight_bandwidth_fairness']:.2f}/{row['weight_schedule_fairness']:.2f}/{row['weight_service']:.2f}</td>"
            f"<td>{row['latest_service_feasible_seeds']}/3</td>"
            f"<td>{row['ever_service_feasible_seeds']}/3</td>"
            f"<td>{row['mean_latest_schedule_fairness']:.4f}</td>"
            f"<td>{row['mean_latest_jain_fairness']:.4f}</td>"
            f"<td>{100*row['mean_latest_starvation_rate']:.2f}%</td>"
            f"<td>{row['mean_latest_p99_wait_slots']:.1f}</td>"
            f"<td>{row['mean_latest_goodput_bits_per_slot']:,.0f}</td>"
            "</tr>"
        )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Schedule fairness reward study</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1200px;margin:30px auto;padding:0 18px;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:7px}th{background:#f3f4f6}</style>"
        "</head><body>"
        f"<h1>{html.escape(plan.round_id)}</h1>"
        "<p><b>T/B/F/S</b> = Throughput / throughput-bandwidth Jain fairness / schedule-frequency fairness / Service.</p>"
        "<p>Schedule-frequency fairness measures how evenly UE scheduling opportunities are distributed. It is separate from successful-delivery throughput fairness, so a low-CQI UE can still receive fair scheduling opportunities even if it delivers fewer bits.</p>"
        "<table><thead><tr><th>Rank</th><th>Profile</th><th>T/B/F/S</th><th>Latest service feasible</th><th>Ever service feasible</th><th>Schedule fair</th><th>Jain</th><th>Starvation</th><th>P99</th><th>Goodput</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
        f"<pre>{html.escape(json.dumps(decision, indent=2))}</pre>"
        "</body></html>",
        encoding="utf-8",
    )

    return {
        "case_metrics": case_csv,
        "profile_summary": summary_csv,
        "ranking": ranking_csv,
        "decision": decision_json,
        "html": html_path,
    }
