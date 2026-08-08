from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


Row = dict[str, Any]


def parse_rule_reserves(value: str, *, max_selected_ues: int = 64) -> list[int]:
    """Parse a stable, duplicate-free list of rule-reserve sizes."""
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("rule reserves must be comma-separated integers") from exc
    if not values:
        raise ValueError("at least one rule reserve is required")
    if any(value < 0 or value > max_selected_ues for value in values):
        raise ValueError(f"rule reserves must be in [0, {max_selected_ues}]")
    return list(dict.fromkeys(values))


def add_rule_lift_deltas(
    rows: Iterable[Row],
    *,
    reference_method: str = "ppo_same_weights",
    group_key: str = "seed",
) -> list[Row]:
    """Add per-seed KPI deltas against the same actor with all rules disabled.

    Positive ``rule_lift_*`` values always mean an improvement. For wait and
    starvation KPIs, the delta is reference minus current so lower delay becomes
    a positive lift.
    """
    annotated = [dict(row) for row in rows]
    references: dict[Any, Row] = {}
    for row in annotated:
        if row.get("method") == reference_method:
            key = row.get(group_key)
            if key in references:
                raise ValueError(
                    f"duplicate {reference_method!r} row for {group_key}={key!r}"
                )
            references[key] = row

    groups = {row.get(group_key) for row in annotated}
    missing = groups - set(references)
    if missing:
        raise ValueError(
            f"missing {reference_method!r} reference for {group_key} values: {sorted(missing)}"
        )

    for row in annotated:
        reference = references[row.get(group_key)]
        row["rule_lift_reference_method"] = reference_method
        row["rule_lift_goodput"] = float(row["mean_goodput_bits_per_slot"]) - float(
            reference["mean_goodput_bits_per_slot"]
        )
        row["rule_lift_fairness"] = float(row["final_jain_fairness"]) - float(
            reference["final_jain_fairness"]
        )
        row["rule_lift_p99_reduction"] = float(reference["max_p99_wait_slots"]) - float(
            row["max_p99_wait_slots"]
        )
        row["rule_lift_max_wait_reduction"] = float(reference["max_wait_slots"]) - float(
            row["max_wait_slots"]
        )
        row["rule_lift_starvation_reduction"] = float(
            reference["mean_starvation_rate"]
        ) - float(row["mean_starvation_rate"])
        row["rule_lift_balanced_score"] = float(row.get("balanced_score", 0.0)) - float(
            reference.get("balanced_score", 0.0)
        )
        row["rule_lift_improved_kpi_count"] = sum(
            float(row[key]) > 1e-12
            for key in (
                "rule_lift_goodput",
                "rule_lift_fairness",
                "rule_lift_p99_reduction",
                "rule_lift_max_wait_reduction",
                "rule_lift_starvation_reduction",
            )
        )
    return annotated


def same_actor_curve(rows: Iterable[Row]) -> list[Row]:
    """Return only fixed-weight rule/PPO split points, ordered by rule reserve."""
    selected = [
        dict(row)
        for row in rows
        if bool(row.get("same_actor_weights"))
        and row.get("ablation_family") == "same_actor_rule_split"
    ]
    return sorted(
        selected,
        key=lambda row: (
            int(row.get("seed", 0)),
            int(row.get("target_rule_reserve_ues", 0)),
            str(row.get("method", "")),
        ),
    )


def summarize_rule_dependency(rows: Iterable[Row]) -> list[Row]:
    """Aggregate the same-actor curve without hiding seed-level variability."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in same_actor_curve(rows):
        groups[str(row["method"])].append(row)

    fields = (
        "target_rule_reserve_ues",
        "mean_rule_selected_count",
        "mean_ppo_selected_count",
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "mean_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "balanced_score",
        "worst_kpi_gap",
        "rule_lift_goodput",
        "rule_lift_fairness",
        "rule_lift_p99_reduction",
        "rule_lift_max_wait_reduction",
        "rule_lift_starvation_reduction",
        "rule_lift_balanced_score",
    )
    summaries: list[Row] = []
    for method, method_rows in groups.items():
        summary: Row = {"method": method, "runs": len(method_rows)}
        for field in fields:
            values = [float(row[field]) for row in method_rows]
            summary[f"{field}_mean"] = sum(values) / len(values)
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            float(row["target_rule_reserve_ues_mean"]), str(row["method"])
        ),
    )
