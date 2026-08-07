import pytest

from scalemac_rl import ScaleMacConfig


def test_default_config_is_valid() -> None:
    ScaleMacConfig().validate()


def test_invalid_prb_budget_is_rejected() -> None:
    cfg = ScaleMacConfig(num_prbs=16, max_selected_ues=64)
    with pytest.raises(ValueError):
        cfg.validate()
