import copy

import torch

from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.scripts.train_ppo import PpoHyperparameters, _ppo_update


def _batch(seed=1):
    torch.manual_seed(seed)
    model = SharedSetActorCritic(input_dim=16, hidden_dim=16)
    obs = torch.randn(8, 12, 16)
    masks = torch.ones(8, 12, dtype=torch.bool)
    with torch.no_grad():
        out = model.get_action_and_value(obs, masks)
        per_ue = model.action_log_prob_per_ue(obs, masks, out.action)
    return model, obs, masks, out.action.detach(), out.log_prob.detach(), per_ue.detach(), out.value.detach()


def test_per_ue_log_probability_groups_two_action_heads():
    model, obs, masks, actions, joint, per_ue, _ = _batch()
    assert per_ue.shape == (8, 12)
    assert torch.allclose(per_ue.sum(dim=-1), joint, atol=1e-5, rtol=1e-5)


def test_per_ue_ppo_update_runs_with_vector_old_log_probs():
    model, obs, masks, actions, _, per_ue, old_values = _batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    returns = old_values + torch.linspace(-0.2, 0.2, 8)
    advantages = torch.linspace(-1.0, 1.0, 8)
    metrics = _ppo_update(
        model=model, optimizer=optimizer, observations=obs, actions=actions,
        candidate_masks=masks, old_log_probs=per_ue, old_values=old_values,
        returns=returns, advantages=advantages,
        hyper=PpoHyperparameters(0.99,0.95,0.1,0.5,0.0,0.5,1,4,0.02,0.0,True,"per_ue",False,0.02),
    )
    assert metrics["ppo_minibatches_processed"] > 0
    assert metrics["max_ratio"] >= metrics["min_ratio"]


def test_strict_kl_guard_rolls_back_rejected_step():
    model, obs, masks, actions, joint, _, old_values = _batch(seed=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    returns = old_values + 10.0
    advantages = torch.ones(8)
    metrics = _ppo_update(
        model=model, optimizer=optimizer, observations=obs, actions=actions,
        candidate_masks=masks, old_log_probs=joint, old_values=old_values,
        returns=returns, advantages=advantages,
        hyper=PpoHyperparameters(0.99,0.95,0.1,0.5,0.0,100.0,1,8,0.02,0.0,False,"joint",True,1e-8),
    )
    assert metrics["strict_kl_rejections"] == 1.0
    assert metrics["ppo_minibatches_processed"] == 0.0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
