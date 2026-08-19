from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Beta


class RecurrentPolicyOutput(NamedTuple):
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    mean_action: torch.Tensor
    hidden_state: torch.Tensor


class RecurrentDeterministicPolicyOutput(NamedTuple):
    action: torch.Tensor
    mean_action: torch.Tensor
    hidden_state: torch.Tensor


class RecurrentSharedSetActorCritic(nn.Module):
    """Permutation-equivariant recurrent actor-critic with per-UE shared GRU memory.

    Every UE is encoded with shared weights and carries one hidden state of size
    ``hidden_dim``. The same GRUCell is applied independently to every UE, so the
    model remains permutation equivariant as long as UE ordering is preserved over
    time. Round 13 therefore uses ``candidate_mode=all`` so hidden state index i
    always belongs to UE i.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        initial_concentration: float = 20.0,
    ) -> None:
        super().__init__()
        if initial_concentration <= 2.0:
            raise ValueError("initial_concentration must be > 2")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        raw = math.log(math.exp(initial_concentration - 2.0) - 1.0)
        self.raw_concentration = nn.Parameter(
            torch.full((2,), raw, dtype=torch.float32)
        )

    def initial_state(
        self,
        batch_size: int,
        num_ues: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            int(batch_size),
            int(num_ues),
            self.hidden_dim,
            device=device if device is not None else parameter.device,
            dtype=dtype if dtype is not None else parameter.dtype,
        )

    def _normalize_inputs(
        self,
        observation: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        squeeze = observation.ndim == 2
        if squeeze:
            observation = observation.unsqueeze(0)
        if observation.ndim != 3:
            raise ValueError("observation must have shape [N,F] or [B,N,F]")
        if hidden_state.ndim == 2:
            hidden_state = hidden_state.unsqueeze(0)
        if hidden_state.ndim != 3:
            raise ValueError("hidden_state must have shape [N,H] or [B,N,H]")
        if hidden_state.shape[:2] != observation.shape[:2]:
            raise ValueError("hidden_state batch/UE dimensions must match observation")
        if hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError("hidden_state has wrong recurrent hidden dimension")
        return observation, hidden_state, squeeze

    def _features_and_state(
        self,
        observation: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        observation, hidden_state, squeeze = self._normalize_inputs(
            observation, hidden_state
        )
        encoded = self.encoder(observation)
        batch, num_ues, _ = encoded.shape
        next_hidden = self.gru(
            encoded.reshape(batch * num_ues, self.hidden_dim),
            hidden_state.reshape(batch * num_ues, self.hidden_dim),
        ).reshape(batch, num_ues, self.hidden_dim)
        mean_context = next_hidden.mean(dim=1, keepdim=True).expand_as(next_hidden)
        actor_features = torch.cat([next_hidden, mean_context], dim=-1)
        pooled = torch.cat(
            [next_hidden.mean(dim=1), next_hidden.amax(dim=1)], dim=-1
        )
        return actor_features, pooled, next_hidden, squeeze

    def action_mean_value_state(
        self,
        observation: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features, pooled, next_hidden, squeeze = self._features_and_state(
            observation, hidden_state
        )
        mean_action = self.actor(actor_features).clamp(1e-4, 1.0 - 1e-4)
        value = self.critic(pooled).squeeze(-1)
        if squeeze:
            return mean_action.squeeze(0), value.squeeze(0), next_hidden.squeeze(0)
        return mean_action, value, next_hidden

    def deterministic_action(
        self,
        observation: torch.Tensor,
        hidden_state: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> RecurrentDeterministicPolicyOutput:
        mean_action, _, next_hidden = self.action_mean_value_state(
            observation, hidden_state
        )
        squeeze = observation.ndim == 2
        if candidate_mask is None:
            masked_action = mean_action
        else:
            mask = candidate_mask
            if squeeze and mask.ndim == 1:
                masked_action = mean_action * mask.unsqueeze(-1).to(mean_action.dtype)
            else:
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                masked_action = mean_action * mask.unsqueeze(-1).to(mean_action.dtype)
        return RecurrentDeterministicPolicyOutput(
            action=masked_action,
            mean_action=mean_action,
            hidden_state=next_hidden,
        )

    def _distribution(self, mean_action: torch.Tensor) -> Beta:
        concentration = torch.nn.functional.softplus(self.raw_concentration) + 2.0
        alpha = 1.0 + mean_action * (concentration - 2.0)
        beta = 1.0 + (1.0 - mean_action) * (concentration - 2.0)
        return Beta(alpha, beta)

    @staticmethod
    def _masked_reduce(
        values: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        average: bool,
    ) -> torch.Tensor:
        if candidate_mask.ndim == 1:
            candidate_mask = candidate_mask.unsqueeze(0)
        expanded = candidate_mask.unsqueeze(-1).expand_as(values).to(values.dtype)
        summed = (values * expanded).sum(dim=(-2, -1))
        if average:
            denominator = expanded.sum(dim=(-2, -1)).clamp_min(1.0)
            return summed / denominator
        return summed

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        candidate_mask: torch.Tensor,
        hidden_state: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> RecurrentPolicyOutput:
        squeeze = observation.ndim == 2
        mean_action, value, next_hidden = self.action_mean_value_state(
            observation, hidden_state
        )
        if squeeze:
            mean_action_b = mean_action.unsqueeze(0)
            value_b = value.unsqueeze(0)
        else:
            mean_action_b = mean_action
            value_b = value
        candidate_mask_b = (
            candidate_mask.unsqueeze(0)
            if candidate_mask.ndim == 1
            else candidate_mask
        )
        distribution = self._distribution(mean_action_b)
        if action is None:
            sampled = mean_action_b if deterministic else distribution.rsample()
        else:
            sampled = action.unsqueeze(0) if action.ndim == 2 else action
            sampled = sampled.clamp(1e-5, 1.0 - 1e-5)
        log_prob = self._masked_reduce(
            distribution.log_prob(sampled), candidate_mask_b, average=False
        )
        entropy = self._masked_reduce(
            distribution.entropy(), candidate_mask_b, average=True
        )
        masked_action = sampled * candidate_mask_b.unsqueeze(-1).to(sampled.dtype)
        if squeeze:
            return RecurrentPolicyOutput(
                action=masked_action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                entropy=entropy.squeeze(0),
                value=value_b.squeeze(0),
                mean_action=mean_action_b.squeeze(0),
                hidden_state=next_hidden,
            )
        return RecurrentPolicyOutput(
            masked_action,
            log_prob,
            entropy,
            value_b,
            mean_action_b,
            next_hidden,
        )

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        candidate_masks: torch.Tensor,
        actions: torch.Tensor,
        initial_hidden_state: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate contiguous [B,T,N,*] sequences using truncated BPTT."""
        if observations.ndim != 4:
            raise ValueError("observations must have shape [B,T,N,F]")
        if actions.ndim != 4 or candidate_masks.ndim != 3 or dones.ndim != 2:
            raise ValueError("sequence tensors have incompatible shapes")
        hidden = initial_hidden_state
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        for step in range(observations.shape[1]):
            output = self.get_action_and_value(
                observations[:, step],
                candidate_masks[:, step],
                hidden,
                action=actions[:, step],
            )
            log_probs.append(output.log_prob)
            entropies.append(output.entropy)
            values.append(output.value)
            means.append(output.mean_action)
            hidden = output.hidden_state
            # A terminal transition resets memory before the next observation.
            reset = (1.0 - dones[:, step]).view(-1, 1, 1).to(hidden.dtype)
            hidden = hidden * reset
        return (
            torch.stack(log_probs, dim=1),
            torch.stack(entropies, dim=1),
            torch.stack(values, dim=1),
            torch.stack(means, dim=1),
        )

    def load_compatible_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        result = self.load_state_dict(state_dict, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)

    def load_imitation_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> tuple[list[str], list[str]]:
        raise RuntimeError(
            "recurrent policy must start from a recurrent checkpoint or random initialization"
        )
