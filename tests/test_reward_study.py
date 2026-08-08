from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.reward_study import RewardCase, RewardStudyPlan, pareto_front_indices


def test_reward_positive_scale_sets_actual_component_coefficient() -> None:
    cfg = ScaleMacConfig(
        num_ues=16,
        num_prbs=16,
        max_selected_ues=8,
        episode_slots=2,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        reward_positive_scale=0.5,
        reward_throughput_weight=0.5,
        reward_fairness_weight=0.5,
        reward_service_weight=0.0,
        reward_deficit_service_weight=0.0,
        reward_starvation_penalty_weight=0.0,
        reward_deadline_risk_penalty_weight=0.0,
        reward_max_wait_risk_penalty_weight=0.0,
        reward_population_wait_penalty_weight=0.0,
        reward_fairness_delta_weight=0.0,
        reward_pf_utility_delta_weight=0.0,
        seed=11,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=11)
    action = np.ones((cfg.num_ues, 2), dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    assert np.isclose(
        info["reward_throughput_component"], 0.25 * info["throughput_score"]
    )
    assert np.isclose(
        info["reward_fairness_component"], 0.25 * info["fairness_score"]
    )
    assert np.isclose(
        reward,
        info["reward_throughput_component"] + info["reward_fairness_component"],
    )


def test_reward_case_builds_equal_actual_coefficients() -> None:
    case = RewardCase.from_dict(
        {
            "id": "equal_three",
            "positive_scale": 2.0 / 3.0,
            "positive_weights": {"throughput": 0.5, "fairness": 0.5},
            "penalty_weights": {"starvation": 1.0 / 3.0},
        }
    )
    coefficients = case.actual_coefficients()
    assert np.isclose(coefficients["coef_throughput"], 1.0 / 3.0)
    assert np.isclose(coefficients["coef_fairness"], 1.0 / 3.0)
    assert np.isclose(coefficients["coef_starvation_penalty"], 1.0 / 3.0)
    args = case.cli_args()
    assert args[args.index("--reward-positive-scale") + 1] == str(2.0 / 3.0)
    assert "--reward-starvation-penalty-weight" in args


def test_reward_case_rejects_non_normalized_relative_positive_weights() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        RewardCase.from_dict(
            {
                "id": "bad",
                "positive_weights": {"throughput": 0.4, "fairness": 0.4},
            }
        )


def test_round_plans_are_valid_and_unique() -> None:
    for path in (
        Path("configs/reward_study/round_01_component_screen.json"),
        Path("configs/reward_study/round_02_throughput_jain_sweep.json"),
    ):
        plan = RewardStudyPlan.from_json(path)
        assert plan.cases
        assert len({case.case_id for case in plan.cases}) == len(plan.cases)


def test_pareto_front_mixed_objectives() -> None:
    rows = [
        {"goodput": 10, "fairness": 0.5, "delay": 20},
        {"goodput": 9, "fairness": 0.6, "delay": 18},
        {"goodput": 8, "fairness": 0.4, "delay": 25},
    ]
    front = pareto_front_indices(
        rows,
        maximize=("goodput", "fairness"),
        minimize=("delay",),
    )
    assert front == {0, 1}


def test_baseline_analysis_is_stored_in_docs() -> None:
    path = Path(
        "docs/analysis/reward_study/baselines/v083_full_control_baseline.html"
    )
    assert path.is_file()
    assert "ScaleMAC-RL v0.8.3" in path.read_text(encoding="utf-8")


def test_round_02_is_a_four_case_throughput_jain_sweep() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/reward_study/round_02_throughput_jain_sweep.json")
    )
    coefficients = [
        (case.actual_coefficients()["coef_throughput"], case.actual_coefficients()["coef_fairness"])
        for case in plan.cases
    ]
    assert coefficients == [
        (0.75, 0.25),
        (0.5, 0.5),
        (0.375, 0.625),
        (0.25, 0.75),
    ]
    for case in plan.cases:
        actual = case.actual_coefficients()
        assert actual["coef_service"] == 0.0
        assert actual["coef_starvation_penalty"] == 0.0
        assert actual["coef_deadline_risk_penalty"] == 0.0


def test_round_01_analysis_explains_p99_wait_in_plain_language() -> None:
    path = Path(
        "docs/analysis/reward_study/round_01/round_01_component_screen_analysis.html"
    )
    content = path.read_text(encoding="utf-8")
    assert "99% UE" in content
    assert "Worst P99 wait" in content
    assert "truyền thành công gần nhất" in content
