from __future__ import annotations

import torch
from torch import nn


class SharedSetPolicy(nn.Module):
    """Permutation-equivariant per-UE policy with global mean context."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Accept [N,F] or [B,N,F], return matching [N,2] or [B,N,2]."""
        squeeze = observation.ndim == 2
        if squeeze:
            observation = observation.unsqueeze(0)
        if observation.ndim != 3:
            raise ValueError("observation must have shape [N,F] or [B,N,F]")

        local = self.encoder(observation)
        context = local.mean(dim=1, keepdim=True).expand_as(local)
        output = self.actor(torch.cat([local, context], dim=-1))
        return output.squeeze(0) if squeeze else output

    def load_compatible_state_dict(
        self, state_dict: dict[str, torch.Tensor], *, strict: bool = True
    ) -> tuple[list[str], list[str]]:
        adapted = dict(state_dict)
        key = "encoder.0.weight"
        if key in adapted and adapted[key].shape != self.encoder[0].weight.shape:
            old = adapted[key]
            target = torch.zeros_like(self.encoder[0].weight)
            rows = min(old.shape[0], target.shape[0])
            columns = min(old.shape[1], target.shape[1])
            target[:rows, :columns] = old[:rows, :columns]
            adapted[key] = target
        result = self.load_state_dict(adapted, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
