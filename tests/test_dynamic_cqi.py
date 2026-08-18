from pathlib import Path

from scalemac_rl.reward_study import RewardStudyPlan


def test_round10_dynamic_cqi_plan_is_controlled_three_case_screen() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/channel_study/round_10_dynamic_cqi.json")
    )
    assert [case.case_id for case in plan.cases] == [
        "static_cqi_baseline",
        "slow_dynamic_cqi",
        "fast_dynamic_cqi",
    ]
    for case in plan.cases:
        weights = case.positive_weights
        assert abs(weights["throughput"] - 1.0 / 3.0) < 1e-12
        assert abs(weights["fairness"] - 1.0 / 3.0) < 1e-12
        assert abs(weights["service"] - 1.0 / 3.0) < 1e-12
        assert sum(weights[name] for name in weights if name not in {"throughput", "fairness", "service"}) == 0.0
    assert plan.cases[0].common_overrides["cqi_mode"] == "static"
    assert plan.cases[1].common_overrides["cqi_mode"] == "correlated"
    assert plan.cases[2].common_overrides["cqi_mode"] == "correlated"
    assert plan.cases[1].common_overrides["cqi_temporal_correlation"] > plan.cases[2].common_overrides["cqi_temporal_correlation"]
    assert plan.cases[1].common_overrides["cqi_innovation_std"] < plan.cases[2].common_overrides["cqi_innovation_std"]
