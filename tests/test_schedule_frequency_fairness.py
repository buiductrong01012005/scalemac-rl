from __future__ import annotations

import numpy as np

from scalemac_rl.config import ScaleMacConfig
from scalemac_rl.env import ScaleMacDownlinkEnv
from scalemac_rl.reward_study import RewardCase


def _config() -> ScaleMacConfig:
    return ScaleMacConfig(
        num_ues=8,
        num_prbs=8,
        max_selected_ues=2,
        episode_slots=16,
        scheduler_mode="ppo_only",
        harq_enabled=False,
        freeze_static_profiles=True,
        static_profile_seed=11,
        reward_throughput_weight=0.0,
        reward_fairness_weight=0.0,
        reward_schedule_fairness_weight=1.0,
        reward_service_weight=0.0,
        reward_deficit_service_weight=0.0,
        reward_pf_utility_weight=0.0,
        reward_low_throughput_weight=0.0,
        reward_urgency_service_weight=0.0,
        reward_fairness_delta_weight=0.0,
        reward_pf_utility_delta_weight=0.0,
        reward_starvation_penalty_weight=0.0,
        reward_deadline_risk_penalty_weight=0.0,
        reward_max_wait_risk_penalty_weight=0.0,
        reward_population_wait_penalty_weight=0.0,
    )


def _action_for(selected: tuple[int, int], n: int = 8) -> np.ndarray:
    action = np.full((n, 2), 0.05, dtype=np.float32)
    action[:, 1] = 0.5
    action[list(selected), 0] = 0.95
    return action


def test_schedule_fairness_rewards_rotation_over_repeated_pair() -> None:
    repeated = ScaleMacDownlinkEnv(_config())
    repeated.reset(seed=11)
    info_repeat = None
    for _ in range(4):
        _, _, _, _, info_repeat = repeated.step(_action_for((0, 1)))

    rotating = ScaleMacDownlinkEnv(_config())
    rotating.reset(seed=11)
    info_rotate = None
    for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
        _, _, _, _, info_rotate = rotating.step(_action_for(pair))

    assert info_repeat is not None and info_rotate is not None
    assert info_repeat["cumulative_schedule_fairness"] == 0.25
    assert info_rotate["cumulative_schedule_fairness"] == 1.0
    assert info_rotate["schedule_fairness_score"] > info_repeat["schedule_fairness_score"]
    assert info_rotate["reward_schedule_fairness_component"] == info_rotate["schedule_fairness_score"]


def test_reward_case_accepts_four_way_schedule_fairness_mix() -> None:
    case = RewardCase.from_dict(
        {
            "id": "equal4",
            "positive_weights": {
                "throughput": 0.25,
                "fairness": 0.25,
                "schedule_fairness": 0.25,
                "service": 0.25,
            },
        }
    )
    assert case.positive_weights["schedule_fairness"] == 0.25
    args = case.cli_args()
    idx = args.index("--reward-schedule-fairness-weight")
    assert args[idx + 1] == "0.25"


def test_schedule_fairness_weight_participates_in_positive_sum_validation() -> None:
    cfg = _config()
    cfg.validate()
    cfg.reward_schedule_fairness_weight = 0.9
    cfg.reward_throughput_weight = 0.2
    try:
        cfg.validate()
    except ValueError as exc:
        assert "positive reward weights must sum to 1" in str(exc)
    else:
        raise AssertionError("invalid positive reward sum should fail")
