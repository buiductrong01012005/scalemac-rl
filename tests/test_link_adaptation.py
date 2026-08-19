from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scalemac_rl.config import ScaleMacConfig
from scalemac_rl.env import ScaleMacDownlinkEnv
from scalemac_rl.link_adaptation import (
    CQI_TABLE1_EFFICIENCY,
    MCS_TABLE1_EFFICIENCY,
    bler_probability_from_cqi_mismatch,
    required_cqi_for_mcs,
    select_mcs_from_reported_cqi,
)
from scalemac_rl.link_adaptation_analysis import build_link_adaptation_analysis
from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command


def _cfg(**overrides) -> ScaleMacConfig:
    values = dict(
        num_ues=16,
        num_prbs=16,
        max_selected_ues=8,
        episode_slots=20,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        cqi_mode="static",
        csi_report_mode="perfect",
        reward_throughput_weight=1 / 3,
        reward_fairness_weight=1 / 3,
        reward_service_weight=1 / 3,
        reward_deficit_service_weight=0.0,
        reward_fairness_delta_weight=0.0,
        reward_pf_utility_delta_weight=0.0,
        reward_starvation_penalty_weight=0.0,
        reward_deadline_risk_penalty_weight=0.0,
        reward_max_wait_risk_penalty_weight=0.0,
        seed=17,
    )
    values.update(overrides)
    return ScaleMacConfig(**values)


def _action(env: ScaleMacDownlinkEnv) -> np.ndarray:
    action = np.zeros(env.action_shape, dtype=np.float32)
    action[:, 0] = np.linspace(0.0, 1.0, env.config.num_ues, dtype=np.float32)
    action[:, 1] = 0.5
    return action


def test_nr_table1_cqi_and_mcs_reference_points() -> None:
    assert CQI_TABLE1_EFFICIENCY[0] == pytest.approx(0.1523)
    assert CQI_TABLE1_EFFICIENCY[-1] == pytest.approx(5.5547)
    assert MCS_TABLE1_EFFICIENCY[0] == pytest.approx(0.2344)
    assert MCS_TABLE1_EFFICIENCY[28] == pytest.approx(5.5547)


def test_cqi_to_mcs_mapping_never_exceeds_reported_support_except_cqi1_floor() -> None:
    cqi = np.arange(1, 16, dtype=np.int16)
    mcs = select_mcs_from_reported_cqi(cqi)
    eff = MCS_TABLE1_EFFICIENCY[mcs]
    assert mcs.shape == cqi.shape
    assert np.all(eff[1:] <= CQI_TABLE1_EFFICIENCY[1:] + 1e-12)
    assert mcs[0] == 0  # Table-1 has no MCS below CQI-1 efficiency.
    assert mcs[-1] == 28


def test_bler_increases_smoothly_when_selected_mcs_exceeds_true_cqi_support() -> None:
    mcs = select_mcs_from_reported_cqi(np.asarray([10], dtype=np.int16))[0]
    required = int(required_cqi_for_mcs(np.asarray([mcs]))[0])
    true = np.asarray([required + 1, required, required - 1, required - 2])
    bler = bler_probability_from_cqi_mismatch(
        true_cqi=true,
        mcs_index=np.full(true.shape, mcs, dtype=np.int16),
        target_bler=0.10,
        mismatch_slope=1.5,
    )
    assert bler[0] < bler[1] < bler[2] < bler[3]
    assert bler[1] == pytest.approx(0.10)
    assert bler[2] > 0.25
    assert bler[3] > 0.60


def test_env_uses_reported_cqi_for_mcs_and_true_cqi_for_bler() -> None:
    env = ScaleMacDownlinkEnv(_cfg(link_adaptation_mode="cqi_mcs_bler"))
    env.reset(seed=17)
    env.cqi[:] = 5
    env.reported_cqi[:] = 10
    _, _, _, _, stale = env.step(_action(env))

    env2 = ScaleMacDownlinkEnv(_cfg(link_adaptation_mode="cqi_mcs_bler"))
    env2.reset(seed=17)
    env2.cqi[:] = 10
    env2.reported_cqi[:] = 10
    _, _, _, _, matched = env2.step(_action(env2))

    assert stale["mean_mcs_index"] == pytest.approx(matched["mean_mcs_index"])
    assert stale["mean_predicted_bler"] > matched["mean_predicted_bler"]
    assert stale["spectral_efficiency_bps_hz"] >= 0.0
    assert 0.0 <= stale["observed_bler"] <= 1.0


def test_legacy_mode_preserves_fixed_bler_diagnostics() -> None:
    env = ScaleMacDownlinkEnv(_cfg(link_adaptation_mode="legacy_fixed_bler"))
    env.reset(seed=17)
    _, _, _, _, info = env.step(_action(env))
    assert info["mean_mcs_index"] == -1.0
    assert info["mean_predicted_bler"] == pytest.approx(env.config.target_bler)


def test_link_adaptation_config_validation() -> None:
    with pytest.raises(ValueError, match="link_adaptation_mode"):
        _cfg(link_adaptation_mode="bad").validate()
    with pytest.raises(ValueError, match="link_adaptation_cqi_backoff"):
        _cfg(link_adaptation_cqi_backoff=-1).validate()
    with pytest.raises(ValueError, match="bler_mismatch_slope"):
        _cfg(bler_mismatch_slope=0.0).validate()


def test_round12_plan_has_controlled_link_adaptation_screen(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/channel_study/round_12_link_adaptation.json")
    )
    assert plan.analysis["design"] == "link_adaptation_screen"
    assert [case.case_id for case in plan.cases] == [
        "legacy_delayed_csi_reference",
        "mcs_bler_perfect_csi",
        "mcs_bler_delayed_csi",
        "mcs_bler_delayed_noisy_csi",
    ]
    for case in plan.cases:
        assert case.positive_weights["throughput"] == pytest.approx(1 / 3)
        assert case.positive_weights["fairness"] == pytest.approx(1 / 3)
        assert case.positive_weights["service"] == pytest.approx(1 / 3)
        common = dict(plan.common)
        common.update(case.common_overrides)
        command = _common_command(
            common=common,
            run_dir=tmp_path / case.case_id,
            steps_override=256,
            validation_slots_override=64,
            progress=False,
            device="cpu",
        )
        assert command[command.index("--link-adaptation-mode") + 1] == common[
            "link_adaptation_mode"
        ]
        assert float(command[command.index("--bler-mismatch-slope") + 1]) == pytest.approx(
            1.5
        )


def test_link_adaptation_analysis_exports_metrics(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(
        Path("configs/channel_study/round_12_link_adaptation.json")
    )
    round_dir = tmp_path / plan.round_id
    fields = [
        "mean_goodput_bits_per_slot",
        "mean_spectral_efficiency_bps_hz",
        "mean_attempted_spectral_efficiency_bps_hz",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "mean_mcs_index",
        "mean_modulation_order",
        "mean_predicted_bler",
        "mean_observed_bler",
        "mean_harq_retransmission_fraction",
        "mean_csi_abs_error",
        "mean_csi_report_age_slots",
    ]
    for idx, case in enumerate(plan.cases):
        case_dir = round_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "mean_goodput_bits_per_slot": 100000 - idx * 1000,
                    "mean_spectral_efficiency_bps_hz": 2.0 - idx * 0.1,
                    "mean_attempted_spectral_efficiency_bps_hz": 2.2,
                    "final_jain_fairness": 0.30 + idx * 0.01,
                    "max_starvation_rate": 0.0,
                    "max_p99_wait_slots": 45 + idx,
                    "max_wait_slots": 48 + idx,
                    "mean_mcs_index": 12 + idx,
                    "mean_modulation_order": 4.0,
                    "mean_predicted_bler": 0.10 + idx * 0.01,
                    "mean_observed_bler": 0.11 + idx * 0.01,
                    "mean_harq_retransmission_fraction": 0.08 + idx * 0.01,
                    "mean_csi_abs_error": idx * 0.2,
                    "mean_csi_report_age_slots": idx,
                }
            )
    plan.analysis.update(
        {
            "output": str(tmp_path / "la.html"),
            "markdown_output": str(tmp_path / "la.md"),
            "metrics_output": str(tmp_path / "la.csv"),
        }
    )
    out = build_link_adaptation_analysis(
        plan=plan, round_dir=round_dir, output_path=tmp_path / "la.html"
    )
    assert out.is_file()
    assert (tmp_path / "la.md").is_file()
    assert (tmp_path / "la.csv").is_file()
    assert "MCS/BLER + periodic delayed CSI" in out.read_text(encoding="utf-8")


def test_legacy_mode_preserves_pre_v013_float32_attempted_bits() -> None:
    """Legacy PHY must retain the exact pre-v0.13 float32 CQI arithmetic path.

    PPO is sensitive enough that a float32→float64 change in the legacy branch
    can perturb the first reward and eventually send training to another basin.
    """
    env = ScaleMacDownlinkEnv(
        _cfg(
            link_adaptation_mode="legacy_fixed_bler",
            harq_enabled=False,
        )
    )
    env.reset(seed=17)
    _, _, _, _, info = env.step(_action(env))

    selected = np.asarray(info["selected_ues"], dtype=np.int64)
    grants = np.asarray(info["prbs_per_selected_ue"], dtype=np.float64)
    legacy_efficiency = CQI_TABLE1_EFFICIENCY.astype(np.float32)[env.cqi[selected] - 1]
    legacy_bits_per_prb = 12.0 * 14.0 * legacy_efficiency * 0.86
    expected_attempted = float(np.sum(grants * legacy_bits_per_prb))

    assert info["cell_attempted_bits"] == expected_attempted
