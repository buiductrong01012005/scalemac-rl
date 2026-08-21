import torch

from scalemac_rl.models import SharedSetActorCritic, SplitEncoderActorCritic


def test_split_encoder_matches_shared_step_zero_and_rng():
    obs=torch.randn(3,12,16); mask=torch.ones(3,12,dtype=torch.bool)
    torch.manual_seed(1234); shared=SharedSetActorCritic(input_dim=16,hidden_dim=16); rng_shared=torch.get_rng_state().clone()
    torch.manual_seed(1234); split=SplitEncoderActorCritic(input_dim=16,hidden_dim=16); rng_split=torch.get_rng_state().clone()
    assert torch.equal(rng_shared,rng_split)
    with torch.no_grad():
        a=shared.get_action_and_value(obs,mask,deterministic=True)
        b=split.get_action_and_value(obs,mask,deterministic=True)
    assert torch.equal(a.action,b.action)
    assert torch.equal(a.value,b.value)


def test_split_critic_loss_does_not_touch_actor_encoder():
    torch.manual_seed(7); model=SplitEncoderActorCritic(input_dim=16,hidden_dim=16)
    obs=torch.randn(4,10,16); mask=torch.ones(4,10,dtype=torch.bool)
    out=model.get_action_and_value(obs,mask,deterministic=True)
    loss=(out.value**2).mean(); model.zero_grad(set_to_none=True); loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.encoder.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.critic_encoder.parameters())
