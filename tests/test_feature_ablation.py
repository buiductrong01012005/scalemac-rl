from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scalemac_rl.config import ScaleMacConfig
from scalemac_rl.env import ScaleMacDownlinkEnv, OBSERVATION_FEATURES
from scalemac_rl.reward_study import RewardStudyPlan


def _cfg(*, age: bool = False, trend: bool = False) -> ScaleMacConfig:
    return ScaleMacConfig(
        num_ues=32,
        num_prbs=64,
        max_selected_ues=16,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        cqi_mode="correlated",
        csi_report_mode="periodic",
        csi_report_period_slots=4,
        csi_report_delay_slots=2,
        link_adaptation_mode="cqi_mcs_bler",
        observation_include_csi_age=age,
        observation_include_reported_cqi_trend=trend,
    )


def test_feature_ablation_preserves_legacy_16_feature_default() -> None:
    env = ScaleMacDownlinkEnv(_cfg())
    obs, _ = env.reset(seed=1701)
    assert obs.shape == (32, OBSERVATION_FEATURES)


def test_csi_age_and_trend_extend_observation_only_when_enabled() -> None:
    assert ScaleMacDownlinkEnv(_cfg(age=True)).observation_shape == (32, 17)
    assert ScaleMacDownlinkEnv(_cfg(trend=True)).observation_shape == (32, 17)
    assert ScaleMacDownlinkEnv(_cfg(age=True, trend=True)).observation_shape == (32, 18)


def test_csi_age_feature_increases_between_delivered_reports() -> None:
    env = ScaleMacDownlinkEnv(_cfg(age=True))
    obs, _ = env.reset(seed=1701)
    assert np.allclose(obs[:, -1], 0.0)
    action = np.zeros((32, 2), dtype=np.float32)
    obs, *_ = env.step(action)
    assert float(obs[0, -1]) > 0.0


def test_reported_cqi_trend_is_zero_at_reset() -> None:
    env = ScaleMacDownlinkEnv(_cfg(trend=True))
    obs, _ = env.reset(seed=1701)
    assert np.allclose(obs[:, -1], 0.0)


def test_round14b_plan_contains_four_profiles_and_three_paired_seeds() -> None:
    path = Path("configs/optimization/round_14b_feature_ablation.json")
    plan = RewardStudyPlan.from_json(path)
    assert len(plan.cases) == 12
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = {}
    for case in payload["cases"]:
        ov = case["common_overrides"]
        key = (ov["observation_include_csi_age"], ov["observation_include_reported_cqi_trend"])
        profiles.setdefault(key, set()).add(ov["seed"])
    assert set(profiles) == {(False, False), (True, False), (False, True), (True, True)}
    assert all(seeds == {1701, 2701, 3701} for seeds in profiles.values())
