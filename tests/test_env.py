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

    assert observation.shape == (128, 16)
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


def test_correlated_cqi_changes_over_time_but_stays_bounded_and_rate_limited() -> None:
    cfg = ScaleMacConfig(
        num_ues=128,
        num_prbs=64,
        max_selected_ues=16,
        episode_slots=20,
        cqi_mode="correlated",
        cqi_temporal_correlation=0.8,
        cqi_innovation_std=1.2,
        cqi_update_interval_slots=1,
        cqi_max_delta_per_update=2,
        seed=17,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=17)
    action = np.ones((cfg.num_ues, 2), dtype=np.float32)
    previous = env.cqi.copy()
    changed_any = False

    for _ in range(10):
        observation, _, _, _, info = env.step(action)
        current = env.cqi.copy()
        delta = np.abs(current.astype(np.int32) - previous.astype(np.int32))
        assert int(delta.max(initial=0)) <= 2
        assert int(current.min()) >= 1
        assert int(current.max()) <= 15
        changed_any = changed_any or bool(np.any(delta > 0))
        assert 0.0 <= float(info["cqi_changed_fraction"]) <= 1.0
        previous = current

    assert changed_any
    assert np.allclose(observation[:, 0], env.cqi / 15.0)


def test_dynamic_cqi_is_reproducible_for_same_seed() -> None:
    cfg = ScaleMacConfig(
        num_ues=64,
        num_prbs=32,
        max_selected_ues=16,
        episode_slots=8,
        cqi_mode="correlated",
        cqi_temporal_correlation=0.9,
        cqi_innovation_std=0.7,
        cqi_max_delta_per_update=2,
        seed=23,
    )
    env_a = ScaleMacDownlinkEnv(cfg)
    env_b = ScaleMacDownlinkEnv(cfg)
    obs_a, _ = env_a.reset(seed=23)
    obs_b, _ = env_b.reset(seed=23)
    action = np.ones((cfg.num_ues, 2), dtype=np.float32)

    for _ in range(6):
        obs_a, reward_a, *_ = env_a.step(action)
        obs_b, reward_b, *_ = env_b.step(action)
        assert np.array_equal(env_a.cqi, env_b.cqi)
        assert np.array_equal(obs_a, obs_b)
        assert reward_a == reward_b


def test_static_cqi_mode_does_not_consume_main_harq_rng_stream() -> None:
    static_cfg = ScaleMacConfig(
        num_ues=32,
        num_prbs=32,
        max_selected_ues=16,
        episode_slots=3,
        cqi_mode="static",
        seed=31,
    )
    dynamic_cfg = ScaleMacConfig(
        num_ues=32,
        num_prbs=32,
        max_selected_ues=16,
        episode_slots=3,
        cqi_mode="correlated",
        cqi_temporal_correlation=0.9,
        cqi_innovation_std=0.8,
        cqi_max_delta_per_update=2,
        seed=31,
    )
    env_static = ScaleMacDownlinkEnv(static_cfg)
    env_dynamic = ScaleMacDownlinkEnv(dynamic_cfg)
    env_static.reset(seed=31)
    env_dynamic.reset(seed=31)
    action = np.ones((32, 2), dtype=np.float32)

    # First transmission happens before the first CQI transition. With a separate
    # channel RNG stream, HARQ outcomes therefore remain identical.
    _, _, _, _, info_static = env_static.step(action)
    _, _, _, _, info_dynamic = env_dynamic.step(action)
    assert info_static["failed_transmissions"] == info_dynamic["failed_transmissions"]
    assert np.array_equal(env_static.last_success, env_dynamic.last_success)
