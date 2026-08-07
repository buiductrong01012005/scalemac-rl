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
