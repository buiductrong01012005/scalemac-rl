import numpy as np

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.env import deadline_risk_score
from scalemac_rl.schedulers import MaxCqiScheduler, RoundRobinScheduler


def test_reward_components_are_normalized_and_sum_correctly() -> None:
    cfg = ScaleMacConfig(
        num_ues=64,
        num_prbs=32,
        max_selected_ues=16,
        episode_slots=2,
        starvation_threshold_slots=10,
        seed=5,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=5)
    action = RoundRobinScheduler(cfg.max_selected_ues).act(observation)
    _, reward, _, _, info = env.step(action)

    for key in (
        "throughput_score",
        "fairness_score",
        "service_score",
        "starvation_violation",
    ):
        assert 0.0 <= info[key] <= 1.0
    assert info["deadline_risk"] >= 0.0
    assert info["reference_deadline_risk"] >= 0.0

    core = (
        info["reward_throughput_component"]
        + info["reward_fairness_component"]
        + info["reward_service_component"]
        - info["reward_starvation_penalty"]
    )
    reconstructed = core - info["reward_deadline_risk_penalty"]
    assert np.isclose(reward, reconstructed)
    assert np.isclose(reward, info["reward_total"])
    assert np.isclose(core, info["reward_core_total"])
    assert np.isclose(
        info["reward_final_target_total"],
        core - info["reward_reference_deadline_risk_penalty"],
    )


def test_starvation_penalty_activates_for_max_cqi() -> None:
    cfg = ScaleMacConfig(
        num_ues=100,
        num_prbs=32,
        max_selected_ues=8,
        episode_slots=20,
        starvation_threshold_slots=5,
        seed=9,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=9)
    scheduler = MaxCqiScheduler()
    final_info = {}
    for _ in range(cfg.episode_slots):
        observation, _, _, _, final_info = env.step(scheduler.act(observation))

    assert final_info["starvation_rate"] > 0.0
    assert final_info["reward_starvation_penalty"] > 0.0


def test_deadline_risk_is_non_saturating_beyond_target() -> None:
    at_target = deadline_risk_score(
        np.asarray([50.0]), target_slots=50.0, start_ratio=0.60
    )
    moderately_late = deadline_risk_score(
        np.asarray([60.0]), target_slots=50.0, start_ratio=0.60
    )
    severely_late = deadline_risk_score(
        np.asarray([100.0]), target_slots=50.0, start_ratio=0.60
    )

    assert np.isclose(at_target, 1.0)
    assert moderately_late > at_target
    assert severely_late > moderately_late


def test_reference_reward_uses_fixed_target() -> None:
    cfg = ScaleMacConfig(
        num_ues=64,
        num_prbs=32,
        max_selected_ues=16,
        episode_slots=4,
        deadline_target_slots=80.0,
        reference_deadline_target_slots=50.0,
        seed=13,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=13)
    scheduler = RoundRobinScheduler(cfg.max_selected_ues)
    final_info = {}
    for _ in range(cfg.episode_slots):
        observation, _, _, _, final_info = env.step(scheduler.act(observation))

    assert "reward_core_total" in final_info
    assert "reward_final_target_total" in final_info
    assert "reference_deadline_risk" in final_info
    assert final_info["reward_final_target_total"] <= final_info["reward_core_total"]
