import torch

from scalemac_rl.models import SharedSetPolicy


def test_policy_supports_variable_ue_count() -> None:
    model = SharedSetPolicy(input_dim=8, hidden_dim=16)
    output_32 = model(torch.zeros(32, 8))
    output_120 = model(torch.zeros(120, 8))
    assert output_32.shape == (32, 2)
    assert output_120.shape == (120, 2)
    assert torch.all((output_32 >= 0.0) & (output_32 <= 1.0))
