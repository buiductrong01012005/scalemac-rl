from __future__ import annotations
import json
from pathlib import Path
import torch
from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.scripts.train_ppo import PpoHyperparameters, _ppo_update

ROOT=Path(__file__).resolve().parents[1]

def test_dimension_log_prob_factorizes_to_joint():
    torch.manual_seed(2)
    model=SharedSetActorCritic(input_dim=16,hidden_dim=8,initial_concentration=20)
    obs=torch.randn(3,5,16)
    mask=torch.ones(3,5,dtype=torch.bool)
    out=model.get_action_and_value(obs,mask)
    per_dim=model.action_log_prob_per_dimension(obs,mask,out.action)
    per_ue=model.action_log_prob_per_ue(obs,mask,out.action)
    assert per_dim.shape==(3,5,2)
    assert torch.allclose(per_dim.sum(-1),per_ue,atol=1e-5)
    assert torch.allclose(per_dim.sum((-2,-1)),out.log_prob,atol=1e-5)

def test_dimension_ppo_update_runs_and_adapts_is_coefficient():
    torch.manual_seed(3)
    model=SharedSetActorCritic(input_dim=16,hidden_dim=8,initial_concentration=20)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    obs=torch.randn(8,6,16); mask=torch.ones(8,6,dtype=torch.bool)
    with torch.no_grad():
        out=model.get_action_and_value(obs,mask)
        old=model.action_log_prob_per_dimension(obs,mask,out.action)
    hyper=PpoHyperparameters(gamma=.99,gae_lambda=.95,clip_coef=.4,value_coef=.5,entropy_coef=0,max_grad_norm=.5,update_epochs=1,minibatch_size=8,target_kl=0,value_clip_coef=0,audit_gradients=False,ratio_mode='dimension',disc_is_alpha=1,disc_is_target=.001,disc_normalize_dimensions=True)
    result=_ppo_update(model=model,optimizer=opt,observations=obs,actions=out.action.detach(),candidate_masks=mask,old_log_probs=old,old_values=out.value.detach(),returns=out.value.detach()+0.1,advantages=torch.ones(8),hyper=hyper)
    assert 'disc_is_loss' in result
    assert result['ppo_minibatches_processed']==1
    assert torch.isfinite(torch.tensor(result['policy_loss']))

def test_round18b_plan_has_three_profiles_three_seeds():
    p=json.loads((ROOT/'configs/optimization/round_18b_dimensionwise_ppo.json').read_text())
    assert len(p['cases'])==9
    ids=[c['id'] for c in p['cases']]
    for prefix in ('joint','disc_exact','disc_norm'):
        assert sum(i.startswith(prefix+'_') for i in ids)==3
