from __future__ import annotations

import sys

import numpy as np

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.env import (
    CQI_RANK,
    LAST_PRB_SHARE,
    OBSERVATION_FEATURES,
    THROUGHPUT_DEFICIT_RANK,
    THROUGHPUT_TO_MEAN,
    WAIT_RANK,
    WAIT_TO_DEADLINE,
)
from scalemac_rl.scripts import train_full_control_ppo_v2


def test_v2_observation_contains_relative_and_history_features() -> None:
    cfg = ScaleMacConfig(
        num_ues=16,
        num_prbs=16,
        max_selected_ues=8,
        episode_slots=2,
        seed=3,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=3)
    assert observation.shape == (16, OBSERVATION_FEATURES)
    for column in (
        CQI_RANK,
        THROUGHPUT_DEFICIT_RANK,
        WAIT_RANK,
        THROUGHPUT_TO_MEAN,
        WAIT_TO_DEADLINE,
        LAST_PRB_SHARE,
    ):
        assert np.all(np.isfinite(observation[:, column]))
        assert np.all(observation[:, column] >= 0.0)


def test_v2_reward_components_reconstruct_total() -> None:
    cfg = ScaleMacConfig(
        num_ues=32,
        num_prbs=32,
        max_selected_ues=8,
        episode_slots=2,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        reward_throughput_weight=0.40,
        reward_fairness_weight=0.15,
        reward_service_weight=0.10,
        reward_deficit_service_weight=0.05,
        reward_pf_utility_weight=0.15,
        reward_low_throughput_weight=0.10,
        reward_urgency_service_weight=0.05,
        reward_population_wait_penalty_weight=0.08,
        seed=5,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=5)
    action = np.ones((cfg.num_ues, 2), dtype=np.float32)
    _, reward, _, _, info = env.step(action)
    positive = sum(
        info[key]
        for key in (
            "reward_throughput_component",
            "reward_fairness_component",
            "reward_service_component",
            "reward_deficit_service_component",
            "reward_pf_utility_component",
            "reward_low_throughput_component",
            "reward_urgency_service_component",
        )
    )
    core = positive - info["reward_starvation_penalty"]
    shaped = (
        core
        + info["reward_fairness_progress_component"]
        + info["reward_pf_utility_progress_component"]
    )
    reconstructed = (
        shaped
        - info["reward_deadline_risk_penalty"]
        - info["reward_max_wait_risk_penalty"]
        - info["reward_population_wait_penalty"]
    )
    assert np.isclose(reward, reconstructed)


def test_full_control_entrypoint_disables_all_external_selection(monkeypatch) -> None:
    captured: list[str] = []

    def fake_main() -> None:
        captured.extend(sys.argv[1:])

    monkeypatch.setattr(train_full_control_ppo_v2, "train_ppo_main", fake_main)
    monkeypatch.setattr(sys, "argv", ["train_full_control_ppo_v2", "--profile", "balanced"])
    train_full_control_ppo_v2.main()

    assert captured[captured.index("--candidate-mode") + 1] == "all"
    assert captured[captured.index("--scheduler-mode") + 1] == "ppo_only"
    assert captured[captured.index("--safety-reserve-ues") + 1] == "0"
    assert "--no-force-harq-retransmissions" in captured
    assert captured[captured.index("--max-candidates") + 1] == "1200"


def test_full_control_profiles_have_normalized_positive_reward_weights() -> None:
    for profile in train_full_control_ppo_v2.PROFILES.values():
        assert np.isclose(sum(profile.reward_weights), 1.0)
