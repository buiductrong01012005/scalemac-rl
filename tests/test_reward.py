import numpy as np

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
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

    reconstructed = (
        info["reward_throughput_component"]
        + info["reward_fairness_component"]
        + info["reward_service_component"]
        - info["reward_starvation_penalty"]
    )
    assert np.isclose(reward, reconstructed)
    assert np.isclose(reward, info["reward_total"])


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
