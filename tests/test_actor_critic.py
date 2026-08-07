import torch

from scalemac_rl.models import SharedSetActorCritic, SharedSetPolicy


def test_actor_critic_supports_variable_ue_count() -> None:
    model = SharedSetActorCritic(input_dim=8, hidden_dim=16)
    observation = torch.zeros(2, 32, 8)
    mask = torch.ones(2, 32, dtype=torch.bool)
    output = model.get_action_and_value(observation, mask)
    assert output.action.shape == (2, 32, 2)
    assert output.log_prob.shape == (2,)
    assert output.value.shape == (2,)
    assert torch.isfinite(output.log_prob).all()


def test_actor_critic_loads_imitation_actor() -> None:
    imitation = SharedSetPolicy(input_dim=8, hidden_dim=16)
    model = SharedSetActorCritic(input_dim=8, hidden_dim=16)
    missing, unexpected = model.load_imitation_state_dict(imitation.state_dict())
    assert any(key.startswith("critic") for key in missing)
    assert unexpected == []


def test_deterministic_action_skips_sampling_and_supports_compact_set() -> None:
    model = SharedSetActorCritic(input_dim=8, hidden_dim=16)
    observation = torch.zeros(64, 8)
    output = model.deterministic_action(observation)
    mean_action, _ = model.action_mean_and_value(observation)
    assert output.action.shape == (64, 2)
    assert torch.allclose(output.action, mean_action)
