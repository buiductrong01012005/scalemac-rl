import pytest

from scalemac_rl.rule_attribution import (
    add_rule_lift_deltas,
    parse_rule_reserves,
    same_actor_curve,
    summarize_rule_dependency,
)


def _row(method: str, seed: int, reserve: int, *, goodput: float, fairness: float, p99: float, wait: float):
    return {
        "method": method,
        "seed": seed,
        "ablation_family": "same_actor_rule_split",
        "same_actor_weights": True,
        "target_rule_reserve_ues": reserve,
        "mean_rule_selected_count": float(reserve),
        "mean_ppo_selected_count": float(64 - reserve),
        "mean_goodput_bits_per_slot": goodput,
        "final_jain_fairness": fairness,
        "mean_starvation_rate": 0.0,
        "max_p99_wait_slots": p99,
        "max_wait_slots": wait,
        "balanced_score": 0.5,
        "worst_kpi_gap": 0.5,
    }


def test_parse_rule_reserves_is_stable_and_validated() -> None:
    assert parse_rule_reserves("8,16,8,32") == [8, 16, 32]
    with pytest.raises(ValueError):
        parse_rule_reserves("65")
    with pytest.raises(ValueError):
        parse_rule_reserves("")


def test_rule_lift_uses_positive_direction_for_wait_reduction() -> None:
    rows = [
        _row("ppo_same_weights", 1, 0, goodput=80, fairness=0.50, p99=60, wait=70),
        _row("hybrid_rule_16", 1, 16, goodput=90, fairness=0.60, p99=40, wait=45),
    ]
    annotated = add_rule_lift_deltas(rows)
    hybrid = next(row for row in annotated if row["method"] == "hybrid_rule_16")
    assert hybrid["rule_lift_goodput"] == 10
    assert hybrid["rule_lift_fairness"] == pytest.approx(0.10)
    assert hybrid["rule_lift_p99_reduction"] == 20
    assert hybrid["rule_lift_max_wait_reduction"] == 25
    assert hybrid["rule_lift_improved_kpi_count"] == 4


def test_same_actor_curve_and_dependency_summary_are_ordered() -> None:
    rows = [
        _row("hybrid_rule_16", 1, 16, goodput=90, fairness=0.60, p99=40, wait=45),
        _row("ppo_same_weights", 1, 0, goodput=80, fairness=0.50, p99=60, wait=70),
    ]
    annotated = add_rule_lift_deltas(rows)
    curve = same_actor_curve(annotated)
    assert [row["target_rule_reserve_ues"] for row in curve] == [0, 16]
    summary = summarize_rule_dependency(annotated)
    assert [row["method"] for row in summary] == ["ppo_same_weights", "hybrid_rule_16"]
