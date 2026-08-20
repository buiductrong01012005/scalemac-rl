from __future__ import annotations

import csv
import json
from pathlib import Path

from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.sampling_budget_analysis import build_sampling_budget_analysis
from scalemac_rl.scripts.run_reward_study import _common_command


PLAN = Path("configs/optimization/round_16b_sampling_budget_control.json")


def test_round16b_sampling_budget_plan_is_controlled(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    assert plan.analysis["design"] == "ppo_sampling_budget_control"
    assert len(plan.cases) == 12

    profiles: dict[str, dict[str, object]] = {}
    for case in plan.cases:
        effective = dict(plan.common)
        effective.update(case.common_overrides)
        profile = str(effective["sampling_profile"])
        profiles.setdefault(profile, {
            "seeds": set(),
            "rollout": int(effective["rollout_steps"]),
            "minibatch": int(effective["minibatch_size"]),
            "steps": int(effective["environment_steps"]),
        })
        profiles[profile]["seeds"].add(int(effective["seed"]))  # type: ignore[index]

        assert case.positive_weights["throughput"] == 0.3
        assert case.positive_weights["fairness"] == 0.3
        assert case.positive_weights["service"] == 0.4
        assert effective["value_coef"] == 0.5
        assert effective["value_clip_coef"] == 0.0
        assert effective["target_kl"] == 0.02
        assert effective["clip_coef"] == 0.1
        assert effective["policy_architecture"] == "feedforward"
        assert effective["observation_include_csi_age"] is False
        assert effective["observation_include_reported_cqi_trend"] is False

        command = _common_command(
            common=effective,
            run_dir=tmp_path / case.case_id,
            steps_override=None,
            validation_slots_override=64,
            progress=False,
            device="cpu",
        )
        assert "--audit-ppo-diagnostics" in command
        assert int(command[command.index("--rollout-steps") + 1]) == int(effective["rollout_steps"])
        assert int(command[command.index("--minibatch-size") + 1]) == int(effective["minibatch_size"])
        assert int(command[command.index("--steps-per-stage") + 1]) == int(effective["environment_steps"])

    assert set(profiles) == {"r256_e100k", "r1024_e100k", "r512_e200k", "r1024_e400k"}
    assert profiles["r256_e100k"]["rollout"] == 256
    assert profiles["r256_e100k"]["minibatch"] == 8
    assert profiles["r256_e100k"]["steps"] == 98304
    assert profiles["r1024_e100k"]["steps"] == 98304
    assert profiles["r512_e200k"]["steps"] == 196608
    assert profiles["r1024_e400k"]["steps"] == 393216
    assert all(v["seeds"] == {1701, 2701, 3701} for v in profiles.values())

    # Same-update-count control: exactly 384 requested outer updates.
    assert profiles["r256_e100k"]["steps"] // profiles["r256_e100k"]["rollout"] == 384
    assert profiles["r512_e200k"]["steps"] // profiles["r512_e200k"]["rollout"] == 384
    assert profiles["r1024_e400k"]["steps"] // profiles["r1024_e400k"]["rollout"] == 384
    assert profiles["r1024_e100k"]["steps"] // profiles["r1024_e100k"]["rollout"] == 96


def test_round16b_validation_cadence_matches_env_steps() -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    for case in plan.cases:
        effective = dict(plan.common)
        effective.update(case.common_overrides)
        assert int(effective["rollout_steps"]) * int(effective["validate_every"]) == 16384
        assert int(effective["rollout_steps"]) * int(effective["checkpoint_every"]) == 32768


def test_sampling_budget_analysis_exports_controls(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    round_dir = tmp_path / plan.round_id
    training_fields = [
        "approx_kl", "max_approx_kl", "clip_fraction", "grad_clip_fraction",
        "ppo_early_stop", "ppo_minibatches_processed", "ppo_sample_reuse",
        "value_loss", "value_explained_variance_preupdate", "value_rmse_preupdate",
        "critic_to_actor_grad_ratio_probe", "steps_per_second",
    ]
    validation_fields = [
        "update", "global_env_steps", "mean_goodput_bits_per_slot",
        "mean_spectral_efficiency_bps_hz", "final_jain_fairness",
        "max_starvation_rate", "max_p99_wait_slots", "max_wait_slots",
        "mean_observed_bler", "mean_harq_retransmission_fraction",
    ]

    for idx, case in enumerate(plan.cases):
        common = dict(plan.common)
        common.update(case.common_overrides)
        case_dir = round_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "training.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=training_fields)
            writer.writeheader()
            for _ in range(4):
                writer.writerow({
                    "approx_kl": 0.01, "max_approx_kl": 0.03, "clip_fraction": 0.25,
                    "grad_clip_fraction": 1.0, "ppo_early_stop": 1.0,
                    "ppo_minibatches_processed": 8.0, "ppo_sample_reuse": 0.25,
                    "value_loss": 10.0, "value_explained_variance_preupdate": -0.1,
                    "value_rmse_preupdate": 3.0, "critic_to_actor_grad_ratio_probe": 8.0,
                    "steps_per_second": 100.0,
                })
        seed = int(common["seed"])
        feasible = seed != 2701
        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=validation_fields)
            writer.writeheader()
            for update in (1, 2):
                writer.writerow({
                    "update": update,
                    "global_env_steps": update * int(common["rollout_steps"]),
                    "mean_goodput_bits_per_slot": 90000 + idx,
                    "mean_spectral_efficiency_bps_hz": 1.9,
                    "final_jain_fairness": 0.45,
                    "max_starvation_rate": 0.0 if feasible else 0.8,
                    "max_p99_wait_slots": 45 if feasible else 5000,
                    "max_wait_slots": 55 if feasible else 5000,
                    "mean_observed_bler": 0.15,
                    "mean_harq_retransmission_fraction": 0.13,
                })

    plan.analysis.update({
        "output": str(tmp_path / "analysis.html"),
        "markdown_output": str(tmp_path / "analysis.md"),
        "metrics_output": str(tmp_path / "metrics.csv"),
        "summary_output": str(tmp_path / "summary.csv"),
        "ranking_output": str(tmp_path / "ranking.csv"),
        "paired_output": str(tmp_path / "paired.csv"),
        "trajectory_output": str(tmp_path / "trajectory.csv"),
        "decision_output": str(tmp_path / "decision.json"),
    })
    result = build_sampling_budget_analysis(plan=plan, round_dir=round_dir, output_path=tmp_path / "analysis.html")
    assert result.is_file()
    for name in ("analysis.md", "metrics.csv", "summary.csv", "ranking.csv", "paired.csv", "trajectory.csv", "decision.json"):
        assert (tmp_path / name).is_file()
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["best_same_environment_budget_profile"] in {"r256_e100k", "r1024_e100k"}
    assert decision["best_same_outer_update_count_profile"] in {"r256_e100k", "r512_e200k", "r1024_e400k"}

    paired = list(csv.DictReader((tmp_path / "paired.csv").open(encoding="utf-8")))
    comparisons = {row["comparison"] for row in paired}
    assert {
        "same_env_budget_large_minus_baseline",
        "same_update_count_r512_minus_baseline",
        "same_update_count_r1024_minus_baseline",
        "r1024_long_budget_minus_short_budget",
    }.issubset(comparisons)
