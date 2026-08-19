from __future__ import annotations

import csv
import json
from pathlib import Path

from scalemac_rl.reward_checkpoint_analysis import build_reward_checkpoint_stability_analysis
from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command


PLAN = Path("configs/optimization/round_15_tjs_reward_checkpoint_stability.json")


def test_round15_is_four_tjs_profiles_on_three_paired_seeds(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    assert plan.analysis["design"] == "reward_checkpoint_stability_screen"
    assert len(plan.cases) == 12
    assert plan.common["policy_architecture"] == "feedforward"
    assert plan.common["observation_include_csi_age"] is False
    assert plan.common["observation_include_reported_cqi_trend"] is False

    profiles: dict[tuple[float, float, float], set[int]] = {}
    for case in plan.cases:
        weights = (
            round(case.positive_weights["throughput"], 6),
            round(case.positive_weights["fairness"], 6),
            round(case.positive_weights["service"], 6),
        )
        seed = int(case.common_overrides["seed"])
        profiles.setdefault(weights, set()).add(seed)

        effective = dict(plan.common)
        effective.update(case.common_overrides)
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
        assert "--no-observation-include-csi-age" in command
        assert "--no-observation-include-reported-cqi-trend" in command

    assert set(profiles) == {
        (0.333333, 0.333333, 0.333333),
        (0.4, 0.3, 0.3),
        (0.3, 0.4, 0.3),
        (0.3, 0.3, 0.4),
    }
    assert all(seeds == {1701, 2701, 3701} for seeds in profiles.values())


def test_round15_analysis_distinguishes_never_feasible_and_drift(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    round_dir = tmp_path / plan.round_id

    fields = [
        "update",
        "global_env_steps",
        "mean_goodput_bits_per_slot",
        "mean_spectral_efficiency_bps_hz",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "mean_observed_bler",
        "mean_harq_retransmission_fraction",
        "mean_final_target_reward",
        "constraint_feasible",
    ]

    for idx, case in enumerate(plan.cases):
        case_dir = round_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        seed = int(case.common_overrides["seed"])

        if seed == 1701:
            vals = [
                [64, 16384, 90000, 1.9, 0.50, 0.0, 45, 55, 0.16, 0.14, 0.55, False],
                [128, 32768, 92000, 2.0, 0.52, 0.0, 44, 54, 0.16, 0.14, 0.57, False],
            ]
            best_update = 128
        elif seed == 2701:
            vals = [
                [64, 16384, 125000, 2.5, 0.15, 0.8, 5000, 5000, 0.16, 0.14, 0.50, False],
                [128, 32768, 130000, 2.6, 0.12, 0.9, 5000, 5000, 0.16, 0.14, 0.49, False],
            ]
            best_update = 64
        else:
            vals = [
                [64, 16384, 88000, 1.8, 0.53, 0.0, 40, 50, 0.16, 0.14, 0.58, False],
                [128, 32768, 103000, 2.1, 0.45, 0.1, 5000, 5000, 0.16, 0.14, 0.56, False],
            ]
            best_update = 64

        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(vals)

        with (case_dir / "checkpoint_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tag", "update"])
            writer.writeheader()
            writer.writerow({"tag": "best_tradeoff", "update": best_update})

    plan.analysis.update({
        "output": str(tmp_path / "analysis.html"),
        "markdown_output": str(tmp_path / "summary.md"),
        "metrics_output": str(tmp_path / "metrics.csv"),
        "summary_output": str(tmp_path / "profiles.csv"),
        "trajectory_output": str(tmp_path / "trajectory.csv"),
        "ranking_output": str(tmp_path / "ranking.csv"),
        "decision_output": str(tmp_path / "decision.json"),
    })
    result = build_reward_checkpoint_stability_analysis(
        plan=plan, round_dir=round_dir, output_path=tmp_path / "analysis.html"
    )
    assert result.is_file()
    for name in ("summary.md", "metrics.csv", "profiles.csv", "trajectory.csv", "ranking.csv", "decision.json"):
        assert (tmp_path / name).is_file()

    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    by_seed = {1701: [], 2701: [], 3701: []}
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    assert all(int(row["latest_service_feasible"]) == 1 for row in by_seed[1701])
    assert all(int(row["never_service_feasible"]) == 1 for row in by_seed[2701])
    assert all(int(row["learn_then_drift"]) == 1 for row in by_seed[3701])

    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["recommended_profile"] in {"equal", "throughput40", "jain40", "service40"}
