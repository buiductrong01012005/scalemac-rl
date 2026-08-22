from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.scripts.train_ppo import PpoHyperparameters, _ppo_update

ROOT = Path(__file__).resolve().parents[1]


def _hyper(mode: str, *, alpha: float = 0.0, target: float = 0.0) -> PpoHyperparameters:
    return PpoHyperparameters(
        gamma=.99, gae_lambda=.95, clip_coef=.4, value_coef=.5,
        entropy_coef=0.0, max_grad_norm=.5, update_epochs=1, minibatch_size=8,
        target_kl=0.0, value_clip_coef=0.0, audit_gradients=True,
        ratio_mode=mode, disc_is_alpha=alpha, disc_is_target=target,
        disc_normalize_dimensions=False,
    )


def _batch(model: SharedSetActorCritic, n_ue: int = 6):
    torch.manual_seed(1902)
    obs = torch.randn(8, n_ue, 16)
    mask = torch.ones(8, n_ue, dtype=torch.bool)
    with torch.no_grad():
        out = model.get_action_and_value(obs, mask)
        old = model.action_log_prob_per_dimension(obs, mask, out.action)
    return obs, mask, out.action.detach(), old, out.value.detach()


def test_ue_group_sum_restores_actor_scale_by_number_of_ues():
    torch.manual_seed(1901)
    base = SharedSetActorCritic(input_dim=16, hidden_dim=8, initial_concentration=20)
    mean_model = copy.deepcopy(base)
    sum_model = copy.deepcopy(base)
    obs, mask, actions, old, values = _batch(base, n_ue=6)
    kwargs = dict(
        observations=obs, actions=actions, candidate_masks=mask,
        old_log_probs=old, old_values=values, returns=values + .1,
        advantages=torch.ones(8),
    )
    mean_result = _ppo_update(
        model=mean_model, optimizer=torch.optim.Adam(mean_model.parameters(), lr=0.0),
        hyper=_hyper("ue_group"), **kwargs,
    )
    sum_result = _ppo_update(
        model=sum_model, optimizer=torch.optim.Adam(sum_model.parameters(), lr=0.0),
        hyper=_hyper("ue_group_sum"), **kwargs,
    )
    ratio = sum_result["actor_grad_norm_probe"] / mean_result["actor_grad_norm_probe"]
    assert abs(ratio - 6.0) < 1e-3


def test_ue_group_sum_jis_controls_mean_but_penalizes_sum():
    torch.manual_seed(1903)
    model = SharedSetActorCritic(input_dim=16, hidden_dim=8, initial_concentration=20)
    obs = torch.randn(8, 6, 16)
    mask = torch.ones(8, 6, dtype=torch.bool)
    with torch.no_grad():
        out = model.get_action_and_value(obs, mask)
        current = model.action_log_prob_per_dimension(obs, mask, out.action)
        old = current - 0.01
    result = _ppo_update(
        model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.0),
        observations=obs, actions=out.action.detach(), candidate_masks=mask,
        old_log_probs=old, old_values=out.value.detach(), returns=out.value.detach(),
        advantages=torch.ones(8), hyper=_hyper("ue_group_sum", alpha=1.0, target=.001),
    )
    # Each 2-D UE group: 0.5*(0.01+0.01)^2 = 0.0002.
    assert abs(result["disc_is_loss"] - 0.0002) < 1e-6
    # Actor penalty uses six-UE sum, while controller still sees mean 0.0002.
    assert abs(result["disc_is_penalty"] - 0.0012) < 1e-6
    assert result["disc_is_alpha_after"] == 0.5


def test_round19b_plan_has_three_profiles_three_seeds():
    plan=json.loads((ROOT/"configs/optimization/round_19b_ue_group_sum_scale_correction.json").read_text())
    assert len(plan["cases"]) == 9
    ids=[c["id"] for c in plan["cases"]]
    expected = {
        "joint": {f"joint_seed{s}" for s in (1701,2701,3701)},
        "uegroup_sum": {f"uegroup_sum_seed{s}" for s in (1701,2701,3701)},
        "uegroup_sum_jis": {f"uegroup_sum_jis_seed{s}" for s in (1701,2701,3701)},
    }
    assert set(ids) == set().union(*expected.values())
