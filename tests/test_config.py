import pytest

from scalemac_rl import ScaleMacConfig


def test_default_config_is_valid() -> None:
    ScaleMacConfig().validate()


def test_invalid_prb_budget_is_rejected() -> None:
    cfg = ScaleMacConfig(num_prbs=16, max_selected_ues=64)
    with pytest.raises(ValueError):
        cfg.validate()


def test_safety_reserve_must_leave_valid_top_k_range() -> None:
    cfg = ScaleMacConfig(safety_reserve_ues=65)
    with pytest.raises(ValueError):
        cfg.validate()


def test_deadline_shaping_configuration_is_validated() -> None:
    cfg = ScaleMacConfig(deadline_risk_start_ratio=1.0)
    with pytest.raises(ValueError):
        cfg.validate()
