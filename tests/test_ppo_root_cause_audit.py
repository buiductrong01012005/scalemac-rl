from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.ppo_root_cause_analysis import build_ppo_root_cause_analysis
from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command
from scalemac_rl.scripts.train_ppo import PpoHyperparameters, _ppo_update


PLAN = Path("configs/optimization/round_16a_ppo_root_cause_audit.json")


def test_round16a_is_two_by_two_sampling_value_clip_factorial(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    assert plan.analysis["design"] == "ppo_root_cause_audit"
    assert len(plan.cases) == 12
    profiles: dict[str, set[int]] = {}
    for case in plan.cases:
        effective = dict(plan.common)
        effective.update(case.common_overrides)
        profile = str(effective["audit_profile"])
        profiles.setdefault(profile, set()).add(int(effective["seed"]))
        assert case.positive_weights["throughput"] == 0.3
        assert case.positive_weights["fairness"] == 0.3
        assert case.positive_weights["service"] == 0.4
        assert all(case.positive_weights[name] == 0.0 for name in ("deficit_service", "pf_utility", "low_throughput", "urgency_service"))
        assert effective["observation_include_csi_age"] is False
        assert effective["observation_include_reported_cqi_trend"] is False
        assert effective["policy_architecture"] == "feedforward"
        command = _common_command(
            common=effective,
            run_dir=tmp_path / case.case_id,
            steps_override=1024,
            validation_slots_override=64,
            progress=False,
            device="cpu",
        )
        assert "--audit-ppo-diagnostics" in command
        assert float(command[command.index("--value-clip-coef") + 1]) == float(effective["value_clip_coef"])
        assert int(command[command.index("--rollout-steps") + 1]) == int(effective["rollout_steps"])
        assert int(command[command.index("--minibatch-size") + 1]) == int(effective["minibatch_size"])
    assert set(profiles) == {
        "baseline", "large_sampling", "critic_stabilized", "large_sampling_critic_stabilized"
    }
    assert all(seeds == {1701, 2701, 3701} for seeds in profiles.values())


def test_ppo_update_exports_root_cause_diagnostics_and_value_clipping() -> None:
    torch.manual_seed(7)
    model = SharedSetActorCritic(input_dim=4, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    observations = torch.randn(16, 5, 4)
    masks = torch.ones(16, 5, dtype=torch.bool)
    with torch.no_grad():
        initial = model.get_action_and_value(observations, masks)
    actions = initial.action.detach()
    old_log_probs = initial.log_prob.detach()
    old_values = initial.value.detach()
    returns = old_values + torch.linspace(-2.0, 2.0, 16)
    advantages = torch.linspace(-1.0, 1.0, 16)
    hyper = PpoHyperparameters(
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.1,
        value_coef=0.5,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        update_epochs=2,
        minibatch_size=4,
        target_kl=10.0,
        value_clip_coef=0.2,
        audit_gradients=True,
    )
    metrics = _ppo_update(
        model=model,
        optimizer=optimizer,
        observations=observations,
        actions=actions,
        candidate_masks=masks,
        old_log_probs=old_log_probs,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        hyper=hyper,
    )
    required = {
        "value_clip_fraction", "max_approx_kl", "grad_clip_fraction",
        "actor_grad_norm_probe", "critic_grad_norm_probe",
        "critic_to_actor_grad_ratio_probe", "ppo_minibatches_processed",
        "ppo_epochs_started", "ppo_early_stop", "ppo_sample_reuse",
    }
    assert required.issubset(metrics)
    assert metrics["ppo_minibatches_processed"] > 0
    assert metrics["actor_grad_norm_probe"] >= 0
    assert metrics["critic_grad_norm_probe"] >= 0


def test_root_cause_analysis_exports_profile_factor_and_decision_tables(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    round_dir = tmp_path / plan.round_id
    training_fields = [
        "approx_kl", "max_approx_kl", "clip_fraction", "grad_clip_fraction",
        "max_grad_norm_preclip", "ppo_early_stop", "ppo_minibatches_processed",
        "ppo_sample_reuse", "value_loss", "value_explained_variance_preupdate",
        "value_rmse_preupdate", "actor_grad_norm_probe", "critic_grad_norm_probe",
        "critic_to_actor_grad_ratio_probe", "value_clip_fraction", "steps_per_second",
        "max_ratio", "max_abs_log_ratio",
    ]
    validation_fields = [
        "update", "global_env_steps", "mean_goodput_bits_per_slot",
        "mean_spectral_efficiency_bps_hz", "final_jain_fairness",
        "max_starvation_rate", "max_p99_wait_slots", "max_wait_slots",
        "mean_observed_bler", "mean_harq_retransmission_fraction",
    ]
    for idx, case in enumerate(plan.cases):
        case_dir = round_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "training.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=training_fields)
            writer.writeheader()
            for _ in range(10):
                writer.writerow({
                    "approx_kl": 0.015, "max_approx_kl": 0.025, "clip_fraction": 0.3,
                    "grad_clip_fraction": 0.2, "max_grad_norm_preclip": 0.7,
                    "ppo_early_stop": 1, "ppo_minibatches_processed": 10,
                    "ppo_sample_reuse": 0.4, "value_loss": 12.0,
                    "value_explained_variance_preupdate": 0.2, "value_rmse_preupdate": 4.0,
                    "actor_grad_norm_probe": 0.4, "critic_grad_norm_probe": 2.0,
                    "critic_to_actor_grad_ratio_probe": 5.0, "value_clip_fraction": 0.1,
                    "steps_per_second": 150.0, "max_ratio": 1.4, "max_abs_log_ratio": 0.4,
                })
        seed = int(case.common_overrides["seed"])
        feasible = seed != 2701
        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=validation_fields)
            writer.writeheader()
            for update in (1, 2):
                writer.writerow({
                    "update": update, "global_env_steps": update * 16384,
                    "mean_goodput_bits_per_slot": 90000 + idx * 10,
                    "mean_spectral_efficiency_bps_hz": 1.9,
                    "final_jain_fairness": 0.45,
                    "max_starvation_rate": 0.0 if feasible else 0.8,
                    "max_p99_wait_slots": 45 if feasible else 5000,
                    "max_wait_slots": 55 if feasible else 5000,
                    "mean_observed_bler": 0.16,
                    "mean_harq_retransmission_fraction": 0.14,
                })
    plan.analysis.update({
        "output": str(tmp_path / "analysis.html"),
        "markdown_output": str(tmp_path / "analysis.md"),
        "metrics_output": str(tmp_path / "metrics.csv"),
        "summary_output": str(tmp_path / "summary.csv"),
        "ranking_output": str(tmp_path / "ranking.csv"),
        "factor_output": str(tmp_path / "factors.csv"),
        "trajectory_output": str(tmp_path / "trajectory.csv"),
        "decision_output": str(tmp_path / "decision.json"),
    })
    result = build_ppo_root_cause_analysis(
        plan=plan, round_dir=round_dir, output_path=tmp_path / "analysis.html"
    )
    assert result.is_file()
    for name in ("analysis.md", "metrics.csv", "summary.csv", "ranking.csv", "factors.csv", "trajectory.csv", "decision.json"):
        assert (tmp_path / name).is_file()
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["recommended_profile"] in {
        "baseline", "large_sampling", "critic_stabilized", "large_sampling_critic_stabilized"
    }
