from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.env import OBSERVATION_FEATURES, observation_feature_count
from scalemac_rl.models import SharedSetActorCritic, build_baseline_compatible_expanded_model


ROOT = Path(__file__).resolve().parents[1]


def test_schedule_state_observation_features_are_deployable_and_ordered() -> None:
    cfg = ScaleMacConfig(
        num_ues=4,
        num_prbs=8,
        max_selected_ues=2,
        episode_slots=16,
        scheduler_mode='ppo_only',
        force_harq_retransmissions=False,
        safety_reserve_ues=0,
        observation_include_time_since_schedule=True,
        observation_include_schedule_rate_deficit=True,
        observation_include_schedule_rate_rank=True,
    )
    env = ScaleMacDownlinkEnv(cfg)
    env.time_since_schedule[:] = np.asarray([0, 32, 64, 128], dtype=np.int32)
    env.ewma_schedule_rate[:] = np.asarray([0.0, 0.25, 0.50, 1.0], dtype=np.float64)
    obs = env._observation()

    assert observation_feature_count(cfg) == OBSERVATION_FEATURES + 3
    assert obs.shape == (4, 19)
    np.testing.assert_allclose(obs[:, 16], [0.0, 0.5, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(obs[:, 17], [1.0, 0.5, 0.0, 0.0], atol=1e-6)
    # Low recent scheduling rate must map to a larger under-scheduled rank.
    assert obs[0, 18] > obs[1, 18] > obs[2, 18] > obs[3, 18]


def test_expanded_schedule_state_model_matches_16_feature_actor_at_step_zero() -> None:
    torch.manual_seed(2468)
    baseline = SharedSetActorCritic(input_dim=16, hidden_dim=64, initial_concentration=20.0)
    baseline_rng = torch.get_rng_state().clone()

    torch.manual_seed(2468)
    expanded = build_baseline_compatible_expanded_model(
        input_dim=19, hidden_dim=64, initial_concentration=20.0
    )
    expanded_rng = torch.get_rng_state().clone()

    obs16 = torch.linspace(0.0, 1.0, steps=5 * 16).reshape(5, 16)
    extras = torch.tensor(
        [[0.1, 0.9, 0.8], [0.2, 0.7, 0.6], [0.3, 0.5, 0.4], [0.4, 0.3, 0.2], [0.5, 0.1, 0.0]],
        dtype=torch.float32,
    )
    obs19 = torch.cat([obs16, extras], dim=-1)

    with torch.no_grad():
        a16, v16 = baseline.action_mean_and_value(obs16)
        a19, v19 = expanded.action_mean_and_value(obs19)
    torch.testing.assert_close(a16, a19, rtol=0.0, atol=0.0)
    torch.testing.assert_close(v16, v19, rtol=0.0, atol=0.0)
    assert torch.equal(baseline_rng, expanded_rng)


def test_round21_plans_are_pure_ppo_and_have_nine_cases_each() -> None:
    for name in [
        'round_21a_equal4_retention_alignment.json',
        'round_21b_schedule_state_observability.json',
    ]:
        payload = json.loads((ROOT / 'configs' / 'optimization' / name).read_text())
        assert len(payload['cases']) == 9
        for case in payload['cases']:
            assert case['positive_weights'] == {
                'throughput': 0.25,
                'fairness': 0.25,
                'schedule_fairness': 0.25,
                'service': 0.25,
            }
            assert 'init_checkpoint' not in case.get('common_overrides', {})


def test_round21b_full_schedule_state_is_baseline_compatible() -> None:
    payload = json.loads(
        (ROOT / 'configs' / 'optimization' / 'round_21b_schedule_state_observability.json').read_text()
    )
    full = [c for c in payload['cases'] if c['id'].startswith('schedstate19_')]
    assert len(full) == 3
    for case in full:
        common = case['common_overrides']
        assert common['observation_include_time_since_schedule'] is True
        assert common['observation_include_schedule_rate_deficit'] is True
        assert common['observation_include_schedule_rate_rank'] is True
        assert common['baseline_compatible_feature_init'] is True
