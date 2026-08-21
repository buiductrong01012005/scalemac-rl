from __future__ import annotations

from scalemac_rl.environment_stress_audit import (
    DEFAULT_STRESS_SEEDS,
    KEY_SEEDS,
    _percentile_rank,
    summarize_stress_rows,
    stress_config,
)


def test_round17c_default_seed_set_is_unique_and_contains_key_seeds() -> None:
    assert 10 <= len(DEFAULT_STRESS_SEEDS) <= 20
    assert len(DEFAULT_STRESS_SEEDS) == len(set(DEFAULT_STRESS_SEEDS))
    assert set(KEY_SEEDS).issubset(DEFAULT_STRESS_SEEDS)


def test_round17c_environment_matches_current_realism_anchor() -> None:
    cfg = stress_config(2701, 5000)
    assert cfg.num_ues == 1200
    assert cfg.num_prbs == 273
    assert cfg.cqi_mode == "correlated"
    assert cfg.csi_report_mode == "periodic"
    assert cfg.csi_report_period_slots == 4
    assert cfg.csi_report_delay_slots == 2
    assert cfg.link_adaptation_mode == "cqi_mcs_bler"
    assert cfg.reward_throughput_weight == 0.30
    assert cfg.reward_fairness_weight == 0.30
    assert cfg.reward_service_weight == 0.40
    assert cfg.static_profile_seed == 2701
    assert cfg.seed == 2701


def test_hardness_percentile_direction_is_explicit() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile_rank(values, 4.0, higher_is_harder=True) == 100.0
    assert _percentile_rank(values, 1.0, higher_is_harder=False) == 100.0


def test_environment_stress_script_imports() -> None:
    from scalemac_rl.scripts import run_environment_stress_audit
    assert callable(run_environment_stress_audit.main)


def test_summary_can_be_built_from_partial_rows() -> None:
    rows = [
        {"policy":"oracle","zero_starvation":1,"service_feasible_under_64":1,"mean_goodput_bits_per_slot":10.0,"mean_spectral_efficiency_bps_hz":1.0,"final_jain_fairness":0.5,"max_starvation_rate":0.0,"max_p99_wait_slots":20.0,"max_wait_slots":30.0,"mean_observed_bler":0.1,"mean_harq_retransmission_fraction":0.1},
        {"policy":"pf","zero_starvation":0,"service_feasible_under_64":0,"mean_goodput_bits_per_slot":11.0,"mean_spectral_efficiency_bps_hz":1.1,"final_jain_fairness":0.8,"max_starvation_rate":0.01,"max_p99_wait_slots":80.0,"max_wait_slots":90.0,"mean_observed_bler":0.1,"mean_harq_retransmission_fraction":0.1},
    ]
    summary = summarize_stress_rows(rows)
    assert {row["policy"] for row in summary} == {"oracle", "pf"}
