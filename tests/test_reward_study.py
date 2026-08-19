from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.reward_study import (
    POSITIVE_COMPONENTS,
    RewardCase,
    RewardStudyPlan,
    pareto_front_indices,
)


def test_reward_positive_scale_sets_actual_component_coefficient() -> None:
    cfg = ScaleMacConfig(
        num_ues=16,
        num_prbs=16,
        max_selected_ues=8,
        episode_slots=2,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        reward_positive_scale=0.5,
        reward_throughput_weight=0.5,
        reward_fairness_weight=0.5,
        reward_service_weight=0.0,
        reward_deficit_service_weight=0.0,
        reward_starvation_penalty_weight=0.0,
        reward_deadline_risk_penalty_weight=0.0,
        reward_max_wait_risk_penalty_weight=0.0,
        reward_population_wait_penalty_weight=0.0,
        reward_fairness_delta_weight=0.0,
        reward_pf_utility_delta_weight=0.0,
        seed=11,
    )
    env = ScaleMacDownlinkEnv(cfg)
    observation, _ = env.reset(seed=11)
    action = np.ones((cfg.num_ues, 2), dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    assert np.isclose(
        info["reward_throughput_component"], 0.25 * info["throughput_score"]
    )
    assert np.isclose(
        info["reward_fairness_component"], 0.25 * info["fairness_score"]
    )
    assert np.isclose(
        reward,
        info["reward_throughput_component"] + info["reward_fairness_component"],
    )


def test_reward_case_builds_equal_actual_coefficients() -> None:
    case = RewardCase.from_dict(
        {
            "id": "equal_three",
            "positive_scale": 2.0 / 3.0,
            "positive_weights": {"throughput": 0.5, "fairness": 0.5},
            "penalty_weights": {"starvation": 1.0 / 3.0},
        }
    )
    coefficients = case.actual_coefficients()
    assert np.isclose(coefficients["coef_throughput"], 1.0 / 3.0)
    assert np.isclose(coefficients["coef_fairness"], 1.0 / 3.0)
    assert np.isclose(coefficients["coef_starvation_penalty"], 1.0 / 3.0)
    args = case.cli_args()
    assert args[args.index("--reward-positive-scale") + 1] == str(2.0 / 3.0)
    assert "--reward-starvation-penalty-weight" in args


def test_reward_case_rejects_non_normalized_relative_positive_weights() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        RewardCase.from_dict(
            {
                "id": "bad",
                "positive_weights": {"throughput": 0.4, "fairness": 0.4},
            }
        )


def test_round_plans_are_valid_and_unique() -> None:
    for path in (
        Path("configs/reward_study/round_01_component_screen.json"),
        Path("configs/reward_study/round_02_throughput_jain_sweep.json"),
        Path("configs/reward_study/round_04_add_service_equal.json"),
        Path("configs/reward_study/round_05_three_component_directional.json"),
        Path("configs/reward_study/round_06_three_component_coordinate.json"),
        Path("configs/reward_study/round_07_fourth_component_screen.json"),
    ):
        plan = RewardStudyPlan.from_json(path)
        assert plan.cases
        assert len({case.case_id for case in plan.cases}) == len(plan.cases)


def test_pareto_front_mixed_objectives() -> None:
    rows = [
        {"goodput": 10, "fairness": 0.5, "delay": 20},
        {"goodput": 9, "fairness": 0.6, "delay": 18},
        {"goodput": 8, "fairness": 0.4, "delay": 25},
    ]
    front = pareto_front_indices(
        rows,
        maximize=("goodput", "fairness"),
        minimize=("delay",),
    )
    assert front == {0, 1}


def test_round_02_is_a_four_case_throughput_jain_sweep() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/reward_study/round_02_throughput_jain_sweep.json")
    )
    coefficients = [
        (case.actual_coefficients()["coef_throughput"], case.actual_coefficients()["coef_fairness"])
        for case in plan.cases
    ]
    assert coefficients == [
        (0.75, 0.25),
        (0.5, 0.5),
        (0.375, 0.625),
        (0.25, 0.75),
    ]
    for case in plan.cases:
        actual = case.actual_coefficients()
        assert actual["coef_service"] == 0.0
        assert actual["coef_starvation_penalty"] == 0.0
        assert actual["coef_deadline_risk_penalty"] == 0.0


def test_round_04_adds_service_with_equal_coefficients() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/reward_study/round_04_add_service_equal.json")
    )
    assert len(plan.cases) == 1
    case = plan.cases[0]
    actual = case.actual_coefficients()
    assert np.isclose(actual["coef_throughput"], 1.0 / 3.0)
    assert np.isclose(actual["coef_fairness"], 1.0 / 3.0)
    assert np.isclose(actual["coef_service"], 1.0 / 3.0)
    assert actual["coef_deficit_service"] == 0.0
    assert actual["coef_pf_utility"] == 0.0
    assert actual["coef_starvation_penalty"] == 0.0
    assert plan.analysis["focus_component"] == "service"


def test_incremental_reward_analysis_contains_formula_and_plain_kpi_explanations(
    tmp_path: Path,
) -> None:
    from scalemac_rl.reward_analysis import build_incremental_reward_analysis

    plan_path = tmp_path / "plan.json"
    reference_dir = tmp_path / "reference"
    round_dir = tmp_path / "round"
    case_dir = round_dir / "equal_three"
    reference_dir.mkdir()
    case_dir.mkdir(parents=True)

    payload = {
        "study_id": "test",
        "round_id": "round_test",
        "description": "test incremental reward analysis",
        "common": {},
        "analysis": {
            "focus_component": "service",
            "reference_run": str(reference_dir),
            "reference_label": "reference",
            "output": str(tmp_path / "analysis.html"),
        },
        "cases": [
            {
                "id": "equal_three",
                "label": "equal three",
                "hypothesis": "test what service changes",
                "positive_weights": {
                    "throughput": 1.0 / 3.0,
                    "fairness": 1.0 / 3.0,
                    "service": 1.0 / 3.0,
                },
            }
        ],
    }
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    fieldnames = [
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
    ]
    with (reference_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "mean_goodput_bits_per_slot": 100000,
                "final_jain_fairness": 0.30,
                "max_starvation_rate": 0.0,
                "max_p99_wait_slots": 49,
                "max_wait_slots": 50,
            }
        )
    with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "mean_goodput_bits_per_slot": 98000,
                "final_jain_fairness": 0.32,
                "max_starvation_rate": 0.0,
                "max_p99_wait_slots": 45,
                "max_wait_slots": 47,
            }
        )

    plan = RewardStudyPlan.from_json(plan_path)
    output = build_incremental_reward_analysis(
        plan=plan,
        round_dir=round_dir,
        output_path=tmp_path / "analysis.html",
    )
    content = output.read_text(encoding="utf-8")
    assert "0.3333 × Throughput" in content
    assert "0.3333 × Jain fairness" in content
    assert "0.3333 × Service" in content
    assert "99% UE" in content
    assert "64 slot" in content
    assert "Goodput giảm" in content


def test_round_05_increases_one_component_and_reduces_the_other_two_equally() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/reward_study/round_05_three_component_directional.json")
    )
    assert [case.case_id for case in plan.cases] == [
        "throughput_heavy",
        "jain_heavy",
        "service_heavy",
    ]
    expected = [
        (0.50, 0.25, 0.25),
        (0.25, 0.50, 0.25),
        (0.25, 0.25, 0.50),
    ]
    for case, (throughput, fairness, service) in zip(plan.cases, expected, strict=True):
        actual = case.actual_coefficients()
        assert np.isclose(actual["coef_throughput"], throughput)
        assert np.isclose(actual["coef_fairness"], fairness)
        assert np.isclose(actual["coef_service"], service)
        assert np.isclose(throughput + fairness + service, 1.0)
        assert actual["coef_deficit_service"] == 0.0
        assert actual["coef_pf_utility"] == 0.0
        assert actual["coef_starvation_penalty"] == 0.0
    assert plan.analysis["design"] == "directional_three_component"


def test_directional_three_component_analysis_explains_all_three_cases(tmp_path: Path) -> None:
    from scalemac_rl.reward_analysis import build_incremental_reward_analysis

    reference_dir = tmp_path / "reference"
    round_dir = tmp_path / "round"
    reference_dir.mkdir()
    fieldnames = [
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "mean_reward_throughput_component",
        "mean_reward_fairness_component",
        "mean_reward_service_component",
    ]
    reference_row = {
        "mean_goodput_bits_per_slot": 97000,
        "final_jain_fairness": 0.27,
        "max_starvation_rate": 0.0,
        "max_p99_wait_slots": 47,
        "max_wait_slots": 48,
        "mean_reward_throughput_component": 0.16,
        "mean_reward_fairness_component": 0.08,
        "mean_reward_service_component": 0.30,
    }
    with (reference_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(reference_row)

    cases = [
        ("throughput_heavy", {"throughput": 0.5, "fairness": 0.25, "service": 0.25}),
        ("jain_heavy", {"throughput": 0.25, "fairness": 0.5, "service": 0.25}),
        ("service_heavy", {"throughput": 0.25, "fairness": 0.25, "service": 0.5}),
    ]
    payload_cases = []
    for index, (case_id, weights) in enumerate(cases):
        case_dir = round_dir / case_id
        case_dir.mkdir(parents=True)
        row = dict(reference_row)
        row["mean_goodput_bits_per_slot"] = 97000 + index * 1000
        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        payload_cases.append(
            {
                "id": case_id,
                "label": case_id,
                "hypothesis": f"understand {case_id}",
                "positive_weights": weights,
            }
        )

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "study_id": "test",
                "round_id": "round_directional",
                "description": "directional test",
                "common": {},
                "analysis": {
                    "design": "directional_three_component",
                    "reference_run": str(reference_dir),
                    "reference_label": "equal baseline",
                    "output": str(tmp_path / "analysis.html"),
                },
                "cases": payload_cases,
            }
        ),
        encoding="utf-8",
    )
    output = build_incremental_reward_analysis(
        plan=RewardStudyPlan.from_json(plan_path),
        round_dir=round_dir,
        output_path=tmp_path / "analysis.html",
    )
    content = output.read_text(encoding="utf-8")
    assert "Bảng so sánh tổng quát" in content
    assert "0.5 × Throughput" in content
    assert "0.5 × Jain fairness" in content
    assert "0.5 × Service" in content
    assert "giảm đều hai thành phần còn lại" in content
    assert "99% UE" in content


def test_round_06_includes_the_hold_service_pair() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/reward_study/round_06_three_component_coordinate.json")
    )
    by_id = {case.case_id: case for case in plan.cases}
    expected = {
        "hold_service_raise_throughput": (0.40, 0.2666666666666667, 1.0 / 3.0),
        "hold_service_raise_jain": (0.2666666666666667, 0.40, 1.0 / 3.0),
    }
    for case_id, (throughput, fairness, service) in expected.items():
        actual = by_id[case_id].actual_coefficients()
        assert np.isclose(actual["coef_throughput"], throughput)
        assert np.isclose(actual["coef_fairness"], fairness)
        assert np.isclose(actual["coef_service"], service)
        assert np.isclose(throughput + fairness + service, 1.0)



def test_round_06_covers_all_six_local_coordinate_directions() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/reward_study/round_06_three_component_coordinate.json")
    )
    assert plan.analysis["design"] == "three_component_coordinate_perturbation"
    assert len(plan.cases) == 6
    expected = {
        (0.4, 0.2666666666666667, 0.3333333333333333),
        (0.2666666666666667, 0.4, 0.3333333333333333),
        (0.4, 0.3333333333333333, 0.2666666666666667),
        (0.2666666666666667, 0.3333333333333333, 0.4),
        (0.3333333333333333, 0.4, 0.2666666666666667),
        (0.3333333333333333, 0.2666666666666667, 0.4),
    }
    observed = {
        (
            case.actual_coefficients()["coef_throughput"],
            case.actual_coefficients()["coef_fairness"],
            case.actual_coefficients()["coef_service"],
        )
        for case in plan.cases
    }
    assert observed == expected
    for case in plan.cases:
        actual = case.actual_coefficients()
        assert actual["coef_deficit_service"] == 0.0
        assert actual["coef_pf_utility"] == 0.0
        assert actual["coef_starvation_penalty"] == 0.0


def test_round_07_integrates_eight_regimes_for_each_remaining_component() -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/reward_study/round_07_fourth_component_screen.json")
    )
    assert plan.analysis["design"] == "fourth_component_comprehensive_screen"
    assert len(plan.cases) == 32
    component_case_map = plan.analysis["component_case_map"]
    expected_components = {
        "deficit_service",
        "pf_utility",
        "low_throughput",
        "urgency_service",
    }
    expected_regimes = {
        "equal_quarter",
        "new_component_heavy",
        "hold_throughput",
        "hold_fairness",
        "hold_service",
        "group_throughput_fairness",
        "group_throughput_service",
        "group_fairness_service",
    }
    assert set(component_case_map) == expected_components
    assert set(plan.analysis["regime_order"]) == expected_regimes
    by_id = {case.case_id: case for case in plan.cases}
    expected_weights = {
        "equal_quarter": (0.25, 0.25, 0.25, 0.25),
        "new_component_heavy": (0.20, 0.20, 0.20, 0.40),
        "hold_throughput": (0.25, 0.20, 0.20, 0.35),
        "hold_fairness": (0.20, 0.25, 0.20, 0.35),
        "hold_service": (0.20, 0.20, 0.25, 0.35),
        "group_throughput_fairness": (0.30, 0.30, 0.10, 0.30),
        "group_throughput_service": (0.30, 0.10, 0.30, 0.30),
        "group_fairness_service": (0.10, 0.30, 0.30, 0.30),
    }
    for component, mapping in component_case_map.items():
        assert set(mapping) == expected_regimes
        for regime, expected in expected_weights.items():
            coefficients = by_id[mapping[regime]].actual_coefficients()
            observed = (
                coefficients["coef_throughput"],
                coefficients["coef_fairness"],
                coefficients["coef_service"],
                coefficients[f"coef_{component}"],
            )
            assert np.allclose(observed, expected)
            assert np.isclose(
                sum(coefficients[f"coef_{name}"] for name in POSITIVE_COMPONENTS),
                1.0,
            )
            assert coefficients["coef_starvation_penalty"] == 0.0
            assert coefficients["coef_deadline_risk_penalty"] == 0.0


def test_comprehensive_fourth_component_analysis_exports_matrix_and_summary(
    tmp_path: Path,
) -> None:
    from scalemac_rl.reward_analysis import build_incremental_reward_analysis

    round_dir = tmp_path / "round"
    case_dir = round_dir / "pf_equal"
    case_dir.mkdir(parents=True)
    fields = [
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
    ]
    with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "mean_goodput_bits_per_slot": 98000,
            "final_jain_fairness": 0.28,
            "max_starvation_rate": 0,
            "max_p99_wait_slots": 48,
            "max_wait_slots": 49,
        })

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "study_id": "test",
        "round_id": "round_comprehensive",
        "description": "comprehensive",
        "common": {},
        "analysis": {
            "design": "fourth_component_comprehensive_screen",
            "case_focus": {"pf_equal": "pf_utility"},
            "case_regime": {"pf_equal": "equal_quarter"},
            "component_case_map": {"pf_utility": {"equal_quarter": "pf_equal"}},
            "regime_order": ["equal_quarter"],
            "regime_labels": {"equal_quarter": "Equal-quarter"},
            "regime_family": {"equal_quarter": "baseline_addition"},
            "final_metrics_output": str(tmp_path / "final.csv"),
            "trajectory_output": str(tmp_path / "trajectory.csv"),
            "comparison_output": str(tmp_path / "comparison.csv"),
            "stability_output": str(tmp_path / "stability.csv"),
            "regime_summary_output": str(tmp_path / "summary.csv"),
            "reference_metrics": {
                "mean_goodput_bits_per_slot": 97000,
                "final_jain_fairness": 0.27,
                "max_starvation_rate": 0,
                "max_p99_wait_slots": 47,
                "max_wait_slots": 48,
            },
        },
        "cases": [{
            "id": "pf_equal",
            "label": "PF equal",
            "hypothesis": "test",
            "positive_weights": {
                "throughput": 0.25,
                "fairness": 0.25,
                "service": 0.25,
                "pf_utility": 0.25,
            },
        }],
    }), encoding="utf-8")
    output = build_incremental_reward_analysis(
        plan=RewardStudyPlan.from_json(plan_path),
        round_dir=round_dir,
        output_path=tmp_path / "analysis.html",
    )
    content = output.read_text(encoding="utf-8")
    assert "So sánh bốn reward tại regime" in content
    assert "<details" in content
    assert (tmp_path / "stability.csv").is_file()
    assert (tmp_path / "summary.csv").is_file()


def test_fourth_component_analysis_uses_case_focus_and_equal_quarter_formula(
    tmp_path: Path,
) -> None:
    from scalemac_rl.reward_analysis import build_incremental_reward_analysis

    round_dir = tmp_path / "round"
    case_dir = round_dir / "add_pf"
    case_dir.mkdir(parents=True)
    fields = [
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "mean_reward_throughput_component",
        "mean_reward_fairness_component",
        "mean_reward_service_component",
        "mean_reward_pf_utility_component",
    ]
    with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "mean_goodput_bits_per_slot": 98000,
                "final_jain_fairness": 0.28,
                "max_starvation_rate": 0,
                "max_p99_wait_slots": 48,
                "max_wait_slots": 49,
                "mean_reward_throughput_component": 0.12,
                "mean_reward_fairness_component": 0.06,
                "mean_reward_service_component": 0.23,
                "mean_reward_pf_utility_component": 0.03,
            }
        )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "study_id": "test",
                "round_id": "round_fourth",
                "description": "fourth component test",
                "common": {},
                "analysis": {
                    "design": "fourth_component_integrated_screen",
                    "focus_component": "service",
                    "case_focus": {"add_pf": "pf_utility"},
                    "case_regime": {"add_pf": "equal_quarter"},
                    "component_case_map": {
                        "pf_utility": {
                            "equal_quarter": "add_pf",
                            "new_component_heavy": "missing_heavy",
                            "anchor_preserving": "missing_anchor"
                        }
                    },
                    "regime_labels": {"equal_quarter": "Equal-quarter"},
                    "final_metrics_output": str(tmp_path / "final.csv"),
                    "trajectory_output": str(tmp_path / "trajectory.csv"),
                    "comparison_output": str(tmp_path / "comparison.csv"),
                    "reference_label": "equal-three",
                    "reference_metrics": {
                        "mean_goodput_bits_per_slot": 97000,
                        "final_jain_fairness": 0.27,
                        "max_starvation_rate": 0,
                        "max_p99_wait_slots": 47,
                        "max_wait_slots": 48,
                    },
                },
                "cases": [
                    {
                        "id": "add_pf",
                        "label": "add PF",
                        "hypothesis": "test PF",
                        "positive_weights": {
                            "throughput": 0.25,
                            "fairness": 0.25,
                            "service": 0.25,
                            "pf_utility": 0.25,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = build_incremental_reward_analysis(
        plan=RewardStudyPlan.from_json(plan_path),
        round_dir=round_dir,
        output_path=tmp_path / "analysis.html",
    )
    content = output.read_text(encoding="utf-8")
    assert "0.25 × Throughput" in content
    assert "0.25 × PF utility" in content
    assert "ba regime" in content
    assert "PF utility" in content
    assert "Equal-quarter" in content
    assert (tmp_path / "final.csv").is_file()
    assert (tmp_path / "trajectory.csv").is_file()
    assert (tmp_path / "comparison.csv").is_file()
