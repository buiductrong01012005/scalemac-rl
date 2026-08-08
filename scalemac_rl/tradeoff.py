from __future__ import annotations

from collections import defaultdict
from math import exp, prod
from typing import Any, Iterable


Row = dict[str, Any]

TRADEOFF_SCORING_VERSION = "feasible-ideal-v1"


def _safe_ratio(value: float, reference: float, *, higher_is_better: bool) -> float:
    value = max(float(value), 0.0)
    reference = max(float(reference), 0.0)
    if higher_is_better:
        if reference <= 1e-12:
            return 1.0
        return min(max(value / reference, 0.0), 1.0)
    if value <= 1e-12:
        return 1.0
    if reference <= 1e-12:
        return 0.0
    return min(max(reference / value, 0.0), 1.0)


def _dominates(left: Row, right: Row) -> bool:
    """Return True when left is no worse in every KPI and better in one."""
    directions = {
        "mean_goodput_bits_per_slot": 1,
        "final_jain_fairness": 1,
        "max_p99_wait_slots": -1,
        "max_wait_slots": -1,
        "mean_starvation_rate": -1,
    }
    no_worse = True
    strictly_better = False
    for field, direction in directions.items():
        a = float(left[field])
        b = float(right[field])
        if direction > 0:
            no_worse = no_worse and a >= b - 1e-12
            strictly_better = strictly_better or a > b + 1e-12
        else:
            no_worse = no_worse and a <= b + 1e-12
            strictly_better = strictly_better or a < b - 1e-12
    return no_worse and strictly_better


def annotate_tradeoff_scores(
    rows: Iterable[Row],
    *,
    max_starvation_rate: float = 0.0,
    group_key: str = "seed",
) -> list[Row]:
    """Annotate scheduler rows with comparable multi-KPI trade-off metrics.

    The ideal point is formed only from starvation-feasible methods in the same
    evaluation scenario. This prevents an unsafe Max-CQI policy from defining the
    deployable goodput target. All methods are still scored and remain visible.

    ``balanced_score`` is the geometric mean of proximity to the feasible ideal for
    goodput, Jain fairness, maximum P99 wait, maximum UE wait, and starvation.
    ``worst_kpi_gap`` is the largest gap to that ideal. Lower is better.
    """
    annotated = [dict(row) for row in rows]
    groups: dict[Any, list[Row]] = defaultdict(list)
    for row in annotated:
        groups[row.get(group_key, "all")].append(row)

    for group_rows in groups.values():
        eligible = [
            row
            for row in group_rows
            if float(row["mean_starvation_rate"]) <= max_starvation_rate + 1e-12
        ]
        reference_rows = eligible or group_rows
        ideal_goodput = max(float(row["mean_goodput_bits_per_slot"]) for row in reference_rows)
        ideal_fairness = max(float(row["final_jain_fairness"]) for row in reference_rows)
        ideal_p99 = min(float(row["max_p99_wait_slots"]) for row in reference_rows)
        ideal_max_wait = min(float(row["max_wait_slots"]) for row in reference_rows)

        for row in group_rows:
            starvation_rate = max(float(row["mean_starvation_rate"]), 0.0)
            starvation_feasible = starvation_rate <= max_starvation_rate + 1e-12
            starvation_scale = max(0.01, max_starvation_rate if max_starvation_rate > 0 else 0.01)
            starvation_proximity = (
                1.0
                if starvation_feasible
                else max(min(exp(-(starvation_rate - max_starvation_rate) / starvation_scale), 1.0), 0.0)
            )
            proximities = {
                "goodput_proximity": _safe_ratio(
                    float(row["mean_goodput_bits_per_slot"]),
                    ideal_goodput,
                    higher_is_better=True,
                ),
                "fairness_proximity": _safe_ratio(
                    float(row["final_jain_fairness"]),
                    ideal_fairness,
                    higher_is_better=True,
                ),
                "p99_wait_proximity": _safe_ratio(
                    float(row["max_p99_wait_slots"]),
                    ideal_p99,
                    higher_is_better=False,
                ),
                "max_wait_proximity": _safe_ratio(
                    float(row["max_wait_slots"]),
                    ideal_max_wait,
                    higher_is_better=False,
                ),
                "starvation_proximity": starvation_proximity,
            }
            row.update(proximities)
            row["tradeoff_scoring_version"] = TRADEOFF_SCORING_VERSION
            row["tradeoff_eligible"] = starvation_feasible
            row["balanced_score"] = prod(max(value, 1e-12) for value in proximities.values()) ** (
                1.0 / len(proximities)
            )
            row["worst_kpi_gap"] = max(1.0 - value for value in proximities.values())
            row["pareto_dominated"] = any(
                other is not row and _dominates(other, row) for other in group_rows
            )

        ordered = sorted(
            group_rows,
            key=lambda row: (
                not bool(row["tradeoff_eligible"]),
                float(row["worst_kpi_gap"]),
                -float(row["balanced_score"]),
                -float(row["mean_goodput_bits_per_slot"]),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row["tradeoff_rank"] = rank

    return annotated


def validation_tradeoff_metrics(
    *,
    mean_throughput_score: float,
    minimum_jain_fairness: float,
    worst_starvation_rate: float,
    worst_p99_wait_slots: float,
    worst_max_wait_slots: float,
    target_throughput_score: float,
    target_jain_fairness: float,
    target_starvation_rate: float,
    target_p99_wait_slots: float,
    target_max_wait_slots: float,
) -> dict[str, float | bool]:
    """Compute fixed-target checkpoint metrics from one validation summary."""
    throughput_attainment = min(
        max(float(mean_throughput_score) / max(float(target_throughput_score), 1e-12), 0.0),
        1.0,
    )
    fairness_attainment = min(
        max(float(minimum_jain_fairness) / max(float(target_jain_fairness), 1e-12), 0.0),
        1.0,
    )
    p99_attainment = min(
        max(float(target_p99_wait_slots) / max(float(worst_p99_wait_slots), 1e-12), 0.0),
        1.0,
    )
    max_wait_attainment = min(
        max(float(target_max_wait_slots) / max(float(worst_max_wait_slots), 1e-12), 0.0),
        1.0,
    )
    starvation_rate = max(float(worst_starvation_rate), 0.0)
    starvation_feasible = starvation_rate <= float(target_starvation_rate) + 1e-12
    starvation_scale = max(0.01, float(target_starvation_rate) if target_starvation_rate > 0 else 0.01)
    starvation_attainment = (
        1.0
        if starvation_feasible
        else max(
            min(
                exp(-(starvation_rate - float(target_starvation_rate)) / starvation_scale),
                1.0,
            ),
            0.0,
        )
    )
    values = (
        throughput_attainment,
        fairness_attainment,
        p99_attainment,
        max_wait_attainment,
        starvation_attainment,
    )
    return {
        "target_throughput_attainment": throughput_attainment,
        "target_fairness_attainment": fairness_attainment,
        "target_p99_attainment": p99_attainment,
        "target_max_wait_attainment": max_wait_attainment,
        "target_starvation_attainment": starvation_attainment,
        "target_starvation_feasible": starvation_feasible,
        "target_balanced_score": prod(max(value, 1e-12) for value in values) ** (1.0 / len(values)),
        "target_worst_kpi_gap": max(1.0 - value for value in values),
    }
