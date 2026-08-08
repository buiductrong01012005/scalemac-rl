import numpy as np

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.schedulers import RoundRobinScheduler


def test_environment_step_contract() -> None:
    cfg = ScaleMacConfig(
        num_ues=128,
        num_prbs=64,
        max_selected_ues=16,
        episode_slots=3,
        seed=11,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, info = env.reset(seed=11)
    scheduler = RoundRobinScheduler(cfg.max_selected_ues)

    assert observation.shape == (128, 8)
    assert info["num_active_ues"] == 128

    action = scheduler.act(observation)
    next_observation, reward, terminated, truncated, step_info = env.step(action)

    assert next_observation.shape == observation.shape
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert step_info["selected_ues"].size == 16
    assert int(step_info["prbs_per_selected_ue"].sum()) == 64
    assert step_info["prb_utilization"] == 1.0


def test_static_cqi_within_episode() -> None:
    cfg = ScaleMacConfig(num_ues=64, num_prbs=32, max_selected_ues=16, episode_slots=2)
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=3)
    cqi_before = observation[:, 0].copy()
    action = np.ones((64, 2), dtype=np.float32)
    observation, *_ = env.step(action)
    assert np.array_equal(cqi_before, observation[:, 0])


def test_frozen_static_profile_is_reused_across_resets() -> None:
    cfg = ScaleMacConfig(
        num_ues=64,
        num_prbs=32,
        max_selected_ues=16,
        freeze_static_profiles=True,
        static_profile_seed=1234,
        seed=7,
    )
    env = ScaleMacDownlinkEnv(cfg)
    env.reset(seed=10)
    first_cqi = env.cqi.copy()
    first_demand = env.demand_factor.copy()
    env.reset(seed=999)
    assert np.array_equal(env.cqi, first_cqi)
    assert np.array_equal(env.demand_factor, first_demand)


def test_delivery_starvation_requires_successful_delivery() -> None:
    """Scheduling a UE is not service when its HARQ transmission fails."""
    import numpy as np

    from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
    from scalemac_rl.schedulers import RoundRobinScheduler

    class AlwaysFailRng:
        def random(self, size: int) -> np.ndarray:
            return np.zeros(size, dtype=np.float64)

    cfg = ScaleMacConfig(
        num_ues=8,
        num_prbs=8,
        max_selected_ues=8,
        episode_slots=2,
        starvation_threshold_slots=2,
        target_bler=0.1,
        seed=21,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=21)
    env.rng = AlwaysFailRng()  # type: ignore[assignment]
    scheduler = RoundRobinScheduler(cfg.max_selected_ues)

    final_info = {}
    for _ in range(2):
        observation, _, _, _, final_info = env.step(scheduler.act(observation))

    assert final_info["scheduling_starvation_rate"] == 0.0
    assert final_info["delivery_starvation_rate"] == 1.0
    assert final_info["starvation_rate"] == 1.0
    assert final_info["scheduling_max_wait_slots"] == 0.0
    assert final_info["max_wait_slots"] == 2.0
