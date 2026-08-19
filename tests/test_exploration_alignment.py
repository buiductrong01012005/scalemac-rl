from __future__ import annotations

import json
from pathlib import Path

import torch

from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.scripts.train_ppo import _current_beta_concentration, _set_beta_concentration


def test_beta_concentration_can_be_set_explicitly() -> None:
    model = SharedSetActorCritic(input_dim=16, hidden_dim=16, initial_concentration=20.0)
    _set_beta_concentration(model, 80.0)
    priority, demand = _current_beta_concentration(model)
    assert abs(priority - 80.0) < 1e-4
    assert abs(demand - 80.0) < 1e-4


def test_archived_exploration_alignment_changes_only_beta_exploration() -> None:
    payload = json.loads(
        Path("configs/reward_study/archive/exploration_alignment_25_75.json").read_text(
            encoding="utf-8"
        )
    )
    cases = payload["cases"]
    assert [case["beta_concentration_end"] for case in cases] == [20.0, 80.0, 200.0]
    assert all(case["beta_concentration_start"] == 20.0 for case in cases)
    assert payload["common"]["environment_steps"] == 100096


