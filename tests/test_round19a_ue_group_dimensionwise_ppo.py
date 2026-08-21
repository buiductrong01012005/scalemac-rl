from __future__ import annotations

import json
from pathlib import Path

import torch

from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.scripts.train_ppo import PpoHyperparameters, _ppo_update

ROOT = Path(__file__).resolve().parents[1]


def _hyper(*, alpha: float, target: float) -> PpoHyperparameters:
    return PpoHyperparameters(
        gamma=.99,
        gae_lambda=.95,
        clip_coef=.4,
        value_coef=.5,
        entropy_coef=0.0,
        max_grad_norm=.5,
        update_epochs=1,
        minibatch_size=8,
        target_kl=0.0,
        value_clip_coef=0.0,
        audit_gradients=True,
        ratio_mode="ue_group",
        disc_is_alpha=alpha,
        disc_is_target=target,
        disc_normalize_dimensions=False,
    )


def test_ue_group_update_accepts_factorized_old_logprobs():
    torch.manual_seed(19)
    model = SharedSetActorCritic(input_dim=16, hidden_dim=8, initial_concentration=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    obs = torch.randn(8, 6, 16)
    mask = torch.ones(8, 6, dtype=torch.bool)
    with torch.no_grad():
        out = model.get_action_and_value(obs, mask)
        old = model.action_log_prob_per_dimension(obs, mask, out.action)
    result = _ppo_update(
        model=model,
        optimizer=optimizer,
        observations=obs,
        actions=out.action.detach(),
        candidate_masks=mask,
        old_log_probs=old,
        old_values=out.value.detach(),
        returns=out.value.detach() + 0.1,
        advantages=torch.ones(8),
        hyper=_hyper(alpha=0.0, target=0.0),
    )
    assert result["ppo_minibatches_processed"] == 1
    assert torch.isfinite(torch.tensor(result["policy_loss"]))
    assert result["actor_grad_norm_probe"] > 0.0


def test_ue_group_jis_is_per_ue_not_full_joint():
    torch.manual_seed(20)
    model = SharedSetActorCritic(input_dim=16, hidden_dim=8, initial_concentration=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    obs = torch.randn(8, 6, 16)
    mask = torch.ones(8, 6, dtype=torch.bool)
    with torch.no_grad():
        out = model.get_action_and_value(obs, mask)
        current = model.action_log_prob_per_dimension(obs, mask, out.action)
        # Make every UE's two-dimensional group log-ratio exactly +0.02.
        old = current - 0.01
    hyper = _hyper(alpha=1.0, target=0.001)
    result = _ppo_update(
        model=model,
        optimizer=optimizer,
        observations=obs,
        actions=out.action.detach(),
        candidate_masks=mask,
        old_log_probs=old,
        old_values=out.value.detach(),
        returns=out.value.detach(),
        advantages=torch.ones(8),
        hyper=hyper,
    )
    # Per-UE J_IS = 0.5 * (0.01 + 0.01)^2 = 0.0002.
    assert abs(result["disc_is_loss"] - 0.0002) < 1e-6
    # This is below target/1.5, so the adaptive controller relaxes alpha.
    assert result["disc_is_alpha_after"] == 0.5


def test_round19a_plan_has_three_profiles_three_seeds():
    plan = json.loads(
        (ROOT / "configs/optimization/round_19a_ue_group_dimensionwise_ppo.json").read_text()
    )
    assert len(plan["cases"]) == 9
    ids = [case["id"] for case in plan["cases"]]
    for prefix in ("joint", "uegroup_clip", "uegroup_jis"):
        assert sum(case_id.startswith(prefix + "_") for case_id in ids) == 3
