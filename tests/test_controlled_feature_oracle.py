from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scalemac_rl.config import ScaleMacConfig
from scalemac_rl.env import ScaleMacDownlinkEnv
from scalemac_rl.models import SharedSetActorCritic, build_baseline_compatible_expanded_model
from scalemac_rl.oracle_sanity import evaluate_policy, service_aware_oracle_action
from scalemac_rl.reward_study import RewardStudyPlan


def test_expanded_feature_init_matches_baseline_policy_and_rng() -> None:
    obs16 = torch.randn(32, 16, generator=torch.Generator().manual_seed(99))
    obs17 = torch.cat([obs16, torch.randn(32, 1, generator=torch.Generator().manual_seed(100))], dim=1)

    torch.manual_seed(1701)
    baseline = SharedSetActorCritic(input_dim=16, hidden_dim=64)
    baseline_rng = torch.get_rng_state().clone()
    baseline_action = baseline.deterministic_action(obs16).action.detach().clone()

    torch.manual_seed(1701)
    expanded = build_baseline_compatible_expanded_model(input_dim=17, hidden_dim=64)
    expanded_rng = torch.get_rng_state().clone()
    expanded_action = expanded.deterministic_action(obs17).action.detach().clone()

    assert torch.equal(baseline_rng, expanded_rng)
    assert torch.equal(baseline_action, expanded_action)
    assert torch.count_nonzero(expanded.encoder[0].weight[:, 16:]) == 0


def test_round14c_plan_has_three_profiles_three_seeds_and_controlled_init() -> None:
    path = Path("configs/optimization/round_14c_controlled_features_oracle.json")
    plan = RewardStudyPlan.from_json(path)
    assert len(plan.cases) == 9
    payload = json.loads(path.read_text(encoding="utf-8"))
    seen = {}
    for case in payload["cases"]:
        ov = case["common_overrides"]
        profile = (ov["observation_include_csi_age"], ov["observation_include_reported_cqi_trend"])
        seen.setdefault(profile, set()).add(ov["seed"])
        if profile != (False, False):
            assert ov["baseline_compatible_feature_init"] is True
    assert set(seen) == {(False, False), (True, False), (False, True)}
    assert all(v == {1701, 2701, 3701} for v in seen.values())


def test_service_oracle_can_keep_small_environment_service_feasible() -> None:
    cfg = ScaleMacConfig(
        num_ues=32, num_prbs=64, max_selected_ues=16, episode_slots=128,
        scheduler_mode="ppo_only", force_harq_retransmissions=False,
        freeze_static_profiles=True, static_profile_seed=1701,
        cqi_mode="correlated", csi_report_mode="periodic",
        csi_report_period_slots=4, csi_report_delay_slots=2,
        link_adaptation_mode="cqi_mcs_bler", starvation_threshold_slots=64,
        reward_throughput_weight=1/3, reward_fairness_weight=1/3,
        reward_service_weight=1/3, reward_deficit_service_weight=0.0,
        reward_pf_utility_weight=0.0, reward_low_throughput_weight=0.0,
        reward_urgency_service_weight=0.0,
    )
    row = evaluate_policy(
        name="oracle", config=cfg, seed=1701,
        action_fn=lambda env, obs: service_aware_oracle_action(env),
    )
    assert row["zero_starvation"] == 1
    assert row["max_p99_wait_slots"] < 64


def test_oracle_sanity_script_and_combined_report_modules_import() -> None:
    from scalemac_rl.controlled_feature_oracle_analysis import build_controlled_feature_oracle_report
    from scalemac_rl.scripts import run_oracle_sanity
    assert callable(build_controlled_feature_oracle_report)
    assert callable(run_oracle_sanity.main)
