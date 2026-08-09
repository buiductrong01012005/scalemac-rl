from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scalemac_rl.multiseed_analysis import build_multiseed_confirmation_analysis
from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command


PLAN_PATH = Path("configs/reward_study/round_08_multiseed_confirmation.json")


def test_round08_plan_has_three_profiles_and_three_common_seeds() -> None:
    plan = RewardStudyPlan.from_json(PLAN_PATH)
    assert plan.round_id == "round_08_multiseed_confirmation"
    assert len(plan.cases) == 9
    assert plan.analysis["design"] == "multiseed_confirmation"

    expected_seeds = {1701, 2701, 3701}
    by_profile: dict[str, list] = {}
    for case in plan.cases:
        profile = plan.analysis["case_profile"][case.case_id]
        by_profile.setdefault(profile, []).append(case)
        seed = int(case.common_overrides["seed"])
        assert int(case.common_overrides["profile_seed"]) == seed
        assert case.common_overrides["validation_seeds"] == [seed]
    assert set(by_profile) == {"tjs_equal", "urgency_hold_throughput", "deficit_group_ts"}
    assert all({int(case.common_overrides["seed"]) for case in cases} == expected_seeds for cases in by_profile.values())

    baseline = by_profile["tjs_equal"][0].actual_coefficients()
    assert np.isclose(baseline["coef_throughput"], 1 / 3)
    assert np.isclose(baseline["coef_fairness"], 1 / 3)
    assert np.isclose(baseline["coef_service"], 1 / 3)

    urgency = by_profile["urgency_hold_throughput"][0].actual_coefficients()
    assert urgency["coef_throughput"] == 0.25
    assert urgency["coef_fairness"] == 0.20
    assert urgency["coef_service"] == 0.20
    assert urgency["coef_urgency_service"] == 0.35

    deficit = by_profile["deficit_group_ts"][0].actual_coefficients()
    assert deficit["coef_throughput"] == 0.30
    assert deficit["coef_fairness"] == 0.10
    assert deficit["coef_service"] == 0.30
    assert deficit["coef_deficit_service"] == 0.30


def test_case_common_overrides_produce_matching_train_profile_and_validation_seeds(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN_PATH)
    for case in plan.cases:
        effective = dict(plan.common)
        effective.update(case.common_overrides)
        command = _common_command(
            common=effective,
            run_dir=tmp_path / case.case_id,
            steps_override=None,
            validation_slots_override=None,
            progress=False,
            device="cpu",
        )
        seed = str(case.common_overrides["seed"])
        assert command[command.index("--seed") + 1] == seed
        assert command[command.index("--fixed-profile-seed") + 1] == seed
        assert command[command.index("--validation-seeds") + 1] == seed
        assert command[command.index("--device") + 1] == "cpu"


def _write_validation(path: Path, *, seed: int, goodput: float, jain: float, p99: float, wait: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "global_env_steps",
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "mean_service_score",
        "mean_urgency_service_score",
        "mean_deficit_service_score",
        "mean_final_target_reward",
        "device",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "seed": seed,
                "global_env_steps": 50048,
                "mean_goodput_bits_per_slot": goodput * 0.98,
                "final_jain_fairness": jain * 0.95,
                "max_starvation_rate": 0,
                "max_p99_wait_slots": p99 + 2,
                "max_wait_slots": wait + 2,
                "mean_service_score": 0.9,
                "mean_urgency_service_score": 0.2,
                "mean_deficit_service_score": 0.2,
                "mean_final_target_reward": 0.5,
                "device": "cpu",
            }
        )
        writer.writerow(
            {
                "seed": seed,
                "global_env_steps": 100096,
                "mean_goodput_bits_per_slot": goodput,
                "final_jain_fairness": jain,
                "max_starvation_rate": 0,
                "max_p99_wait_slots": p99,
                "max_wait_slots": wait,
                "mean_service_score": 0.92,
                "mean_urgency_service_score": 0.25,
                "mean_deficit_service_score": 0.25,
                "mean_final_target_reward": 0.55,
                "device": "cpu",
            }
        )


def test_multiseed_analysis_exports_seed_summary_paired_and_html(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN_PATH)
    round_dir = tmp_path / plan.round_id
    for case in plan.cases:
        profile = plan.analysis["case_profile"][case.case_id]
        seed = int(case.common_overrides["seed"])
        if profile == "tjs_equal":
            metrics = (100000 + seed % 10, 0.27, 47, 48)
        elif profile == "urgency_hold_throughput":
            metrics = (99500 + seed % 10, 0.31, 45, 47)
        else:
            metrics = (95000 + seed % 10, 0.29, 42, 44)
        _write_validation(
            round_dir / case.case_id / "validation.csv",
            seed=seed,
            goodput=metrics[0],
            jain=metrics[1],
            p99=metrics[2],
            wait=metrics[3],
        )

    output = tmp_path / "report.html"
    # Redirect outputs into tmp_path without changing the source plan.
    plan.analysis.update(
        {
            "output": str(output),
            "markdown_output": str(tmp_path / "summary.md"),
            "seed_metrics_output": str(tmp_path / "seed.csv"),
            "profile_summary_output": str(tmp_path / "profile.csv"),
            "trajectory_output": str(tmp_path / "trajectory.csv"),
            "stability_output": str(tmp_path / "stability.csv"),
            "comparison_output": str(tmp_path / "paired.csv"),
        }
    )
    result = build_multiseed_confirmation_analysis(plan=plan, round_dir=round_dir, output_path=output)
    assert result == output
    assert output.is_file()
    content = output.read_text(encoding="utf-8")
    assert "mean ± std" in content
    assert "Urgency hold-Throughput" in content
    assert "Deficit giữ nhóm T+S" in content
    assert "3/3" in content
    for name in ["seed.csv", "profile.csv", "trajectory.csv", "stability.csv", "paired.csv", "summary.md"]:
        assert (tmp_path / name).is_file()

    with (tmp_path / "profile.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(int(row["completed_seeds"]) == 3 for row in rows)
    assert all(int(row["stable_seeds"]) == 3 for row in rows)

    with (tmp_path / "paired.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 6
