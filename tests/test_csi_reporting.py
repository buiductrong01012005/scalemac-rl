from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scalemac_rl.config import ScaleMacConfig
from scalemac_rl.env import CQI, CQI_RANK, ScaleMacDownlinkEnv
from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command


def _config(**overrides) -> ScaleMacConfig:
    values = dict(
        num_ues=24,
        num_prbs=24,
        max_selected_ues=8,
        episode_slots=20,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        cqi_mode="correlated",
        cqi_temporal_correlation=0.80,
        cqi_innovation_std=1.20,
        cqi_update_interval_slots=1,
        cqi_max_delta_per_update=3,
        reward_throughput_weight=1 / 3,
        reward_fairness_weight=1 / 3,
        reward_service_weight=1 / 3,
        reward_deficit_service_weight=0.0,
        reward_fairness_delta_weight=0.0,
        reward_pf_utility_delta_weight=0.0,
        reward_starvation_penalty_weight=0.0,
        reward_deadline_risk_penalty_weight=0.0,
        reward_max_wait_risk_penalty_weight=0.0,
        seed=41,
    )
    values.update(overrides)
    return ScaleMacConfig(**values)


def _action(env: ScaleMacDownlinkEnv) -> np.ndarray:
    action = np.zeros(env.action_shape, dtype=np.float32)
    action[:, 0] = np.linspace(0.0, 1.0, env.config.num_ues, dtype=np.float32)
    action[:, 1] = 0.5
    return action


def test_perfect_csi_preserves_current_true_cqi_observation() -> None:
    env = ScaleMacDownlinkEnv(_config(csi_report_mode="perfect"))
    obs, _ = env.reset(seed=41)
    for _ in range(6):
        obs, _, _, _, info = env.step(_action(env))
        np.testing.assert_array_equal(env.reported_cqi, env.cqi)
        np.testing.assert_allclose(obs[:, CQI], env.cqi / 15.0)
        assert info["mean_csi_abs_error"] == 0.0
        assert info["csi_report_age_slots"] == 0.0


def test_periodic_delayed_csi_is_stale_until_report_arrives() -> None:
    env = ScaleMacDownlinkEnv(
        _config(
            csi_report_mode="periodic",
            csi_report_period_slots=4,
            csi_report_delay_slots=2,
            csi_report_error_std=0.0,
        )
    )
    initial_obs, _ = env.reset(seed=41)
    initial_report = env.reported_cqi.copy()
    assert np.array_equal(initial_report, env.cqi)

    for slot in range(1, 6):
        obs, _, _, _, info = env.step(_action(env))
        np.testing.assert_array_equal(env.reported_cqi, initial_report)
        np.testing.assert_allclose(obs[:, CQI], env.reported_cqi / 15.0)
        assert info["csi_report_delivered"] == 0.0
        assert info["csi_report_age_slots"] == float(slot)

    obs, _, _, _, info = env.step(_action(env))
    assert info["csi_report_delivered"] == 1.0
    assert info["csi_report_age_slots"] == 2.0
    np.testing.assert_allclose(obs[:, CQI], env.reported_cqi / 15.0)
    # CQI rank must also be based on scheduler-visible reported CSI.
    np.testing.assert_allclose(
        obs[:, CQI_RANK], env._percentile_rank(env.reported_cqi.astype(np.float64))
    )


def test_csi_reporting_rng_does_not_change_true_channel_path() -> None:
    perfect = ScaleMacDownlinkEnv(_config(csi_report_mode="perfect"))
    noisy = ScaleMacDownlinkEnv(
        _config(
            csi_report_mode="periodic",
            csi_report_period_slots=4,
            csi_report_delay_slots=2,
            csi_report_error_std=1.0,
        )
    )
    perfect.reset(seed=41)
    noisy.reset(seed=41)
    for _ in range(10):
        perfect.step(_action(perfect))
        noisy.step(_action(noisy))
        np.testing.assert_array_equal(perfect.cqi, noisy.cqi)


def test_csi_config_validation() -> None:
    with pytest.raises(ValueError, match="csi_report_period_slots"):
        _config(csi_report_period_slots=0).validate()
    with pytest.raises(ValueError, match="csi_report_delay_slots"):
        _config(csi_report_delay_slots=-1).validate()
    with pytest.raises(ValueError, match="csi_report_error_std"):
        _config(csi_report_error_std=-0.1).validate()


def test_round11_csi_plan_is_controlled_four_case_screen(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(Path("configs/channel_study/round_11_csi_reporting.json"))
    assert plan.analysis["design"] == "csi_reporting_screen"
    assert [case.case_id for case in plan.cases] == [
        "perfect_csi_baseline",
        "periodic_csi_only",
        "periodic_delayed_csi",
        "periodic_delayed_noisy_csi",
    ]
    for case in plan.cases:
        assert case.positive_weights["throughput"] == pytest.approx(1 / 3)
        assert case.positive_weights["fairness"] == pytest.approx(1 / 3)
        assert case.positive_weights["service"] == pytest.approx(1 / 3)
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
        assert command[command.index("--cqi-mode") + 1] == "correlated"
        assert command[command.index("--csi-report-mode") + 1] == effective["csi_report_mode"]
        assert int(command[command.index("--csi-report-period-slots") + 1]) == effective["csi_report_period_slots"]
        assert int(command[command.index("--csi-report-delay-slots") + 1]) == effective["csi_report_delay_slots"]


def test_csi_analysis_exports_html_markdown_and_metrics(tmp_path: Path) -> None:
    import csv
    from scalemac_rl.csi_analysis import build_csi_reporting_analysis

    plan = RewardStudyPlan.from_json(Path("configs/channel_study/round_11_csi_reporting.json"))
    round_dir = tmp_path / plan.round_id
    for index, case in enumerate(plan.cases):
        case_dir = round_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        common = dict(plan.common)
        common.update(case.common_overrides)
        fields = [
            "mean_goodput_bits_per_slot", "final_jain_fairness", "max_starvation_rate",
            "max_p99_wait_slots", "max_wait_slots", "mean_cqi", "mean_reported_cqi",
            "mean_csi_abs_error", "max_p95_csi_abs_error", "mean_csi_stale_fraction",
            "mean_csi_report_age_slots",
        ]
        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "mean_goodput_bits_per_slot": 100000 - index * 1000,
                    "final_jain_fairness": 0.30 - index * 0.02,
                    "max_starvation_rate": index * 0.01,
                    "max_p99_wait_slots": 45 + index,
                    "max_wait_slots": 48 + index,
                    "mean_cqi": 8.0,
                    "mean_reported_cqi": 8.0,
                    "mean_csi_abs_error": index * 0.2,
                    "max_p95_csi_abs_error": index,
                    "mean_csi_stale_fraction": index * 0.1,
                    "mean_csi_report_age_slots": index,
                }
            )
    output = tmp_path / "csi.html"
    plan.analysis.update(
        {
            "output": str(output),
            "markdown_output": str(tmp_path / "csi.md"),
            "metrics_output": str(tmp_path / "csi.csv"),
        }
    )
    result = build_csi_reporting_analysis(plan=plan, round_dir=round_dir, output_path=output)
    assert result == output
    assert output.is_file()
    assert (tmp_path / "csi.md").is_file()
    assert (tmp_path / "csi.csv").is_file()
    assert "Periodic + delayed + noisy CSI" in output.read_text(encoding="utf-8")
