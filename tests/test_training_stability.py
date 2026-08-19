from __future__ import annotations

import csv
from pathlib import Path

from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command
from scalemac_rl.training_stability_analysis import build_training_stability_analysis


PLAN = Path("configs/optimization/round_14a_ppo_training_stability.json")


def test_round14a_is_self_contained_two_by_two_factorial_on_three_seeds(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    assert plan.analysis["design"] == "ppo_training_stability_screen"
    assert len(plan.cases) == 12
    expected_seeds = {1701, 2701, 3701}
    recipes: dict[tuple[float, int], set[int]] = {}
    for case in plan.cases:
        effective = dict(plan.common)
        effective.update(case.common_overrides)
        assert effective["policy_architecture"] == "feedforward"
        assert case.positive_weights["throughput"] == case.positive_weights["fairness"]
        assert case.positive_weights["fairness"] == case.positive_weights["service"]
        key = (float(effective["learning_rate_start"]), int(effective["update_epochs"]))
        recipes.setdefault(key, set()).add(int(effective["seed"]))
        command = _common_command(
            common=effective,
            run_dir=tmp_path / case.case_id,
            steps_override=256,
            validation_slots_override=64,
            progress=False,
            device="cpu",
        )
        assert command[command.index("--policy-architecture") + 1] == "feedforward"
        assert command[command.index("--link-adaptation-mode") + 1] == "cqi_mcs_bler"
        assert command[command.index("--csi-report-delay-slots") + 1] == "2"
        assert float(command[command.index("--lr") + 1]) == key[0]
        assert int(command[command.index("--update-epochs") + 1]) == key[1]
    assert set(recipes) == {(1e-4, 4), (5e-5, 4), (1e-4, 2), (5e-5, 2)}
    assert all(seeds == expected_seeds for seeds in recipes.values())


def test_training_stability_analysis_exports_recipe_and_factor_tables(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    round_dir = tmp_path / plan.round_id
    validation_fields = [
        "mean_goodput_bits_per_slot", "mean_spectral_efficiency_bps_hz", "final_jain_fairness",
        "max_starvation_rate", "max_p99_wait_slots", "max_wait_slots", "mean_observed_bler",
        "mean_harq_retransmission_fraction",
    ]
    training_fields = ["approx_kl", "clip_fraction", "mean_jain_fairness", "mean_starvation_rate"]
    for idx, case in enumerate(plan.cases):
        case_dir = round_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_fields)
            writer.writeheader()
            writer.writerow({
                "mean_goodput_bits_per_slot": 90000 + idx * 100,
                "mean_spectral_efficiency_bps_hz": 1.9 + idx * 0.01,
                "final_jain_fairness": 0.30 + idx * 0.005,
                "max_starvation_rate": 0.0,
                "max_p99_wait_slots": 50 + idx,
                "max_wait_slots": 55 + idx,
                "mean_observed_bler": 0.16,
                "mean_harq_retransmission_fraction": 0.14,
            })
        with (case_dir / "training.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=training_fields)
            writer.writeheader()
            for j in range(40):
                writer.writerow({"approx_kl": 0.01 + idx * 1e-4, "clip_fraction": 0.3, "mean_jain_fairness": 0.55, "mean_starvation_rate": 0.01})
    plan.analysis.update({
        "output": str(tmp_path / "analysis.html"),
        "markdown_output": str(tmp_path / "summary.md"),
        "metrics_output": str(tmp_path / "metrics.csv"),
        "summary_output": str(tmp_path / "recipes.csv"),
        "factor_effects_output": str(tmp_path / "effects.csv"),
    })
    result = build_training_stability_analysis(plan=plan, round_dir=round_dir, output_path=tmp_path / "analysis.html")
    assert result.is_file()
    for name in ("summary.md", "metrics.csv", "recipes.csv", "effects.csv"):
        assert (tmp_path / name).is_file()
    with (tmp_path / "recipes.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["recipe"] for row in rows} == {"baseline", "low_lr", "epochs2", "low_lr_epochs2"}
    assert all(int(row["seeds"]) == 3 for row in rows)
