from __future__ import annotations

import copy
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Beta


class PolicyOutput(NamedTuple):
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    mean_action: torch.Tensor


class DeterministicPolicyOutput(NamedTuple):
    action: torch.Tensor
    mean_action: torch.Tensor


class SharedSetActorCritic(nn.Module):
    """Permutation-equivariant stochastic actor with a pooled set critic.

    The actor preserves the v0.2 imitation architecture, allowing its encoder and
    actor weights to initialize PPO. A shared Beta concentration adds bounded
    exploration over priority and PRB-demand scores in [0, 1].
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, initial_concentration: float = 20.0):
        super().__init__()
        if initial_concentration <= 2.0:
            raise ValueError("initial_concentration must be > 2")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
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
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # softplus(raw) + 2 = initial concentration
        raw = math.log(math.exp(initial_concentration - 2.0) - 1.0)
        self.raw_concentration = nn.Parameter(torch.full((2,), raw, dtype=torch.float32))

    def encode_features(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = observation.ndim == 2
        if squeeze:
            observation = observation.unsqueeze(0)
        if observation.ndim != 3:
            raise ValueError("observation must have shape [N,F] or [B,N,F]")
        local = self.encoder(observation)
        mean_context = local.mean(dim=1, keepdim=True).expand_as(local)
        actor_features = torch.cat([local, mean_context], dim=-1)
        pooled = torch.cat([local.mean(dim=1), local.amax(dim=1)], dim=-1)
        return actor_features, pooled

    def action_mean_and_value(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = observation.ndim == 2
        actor_features, pooled = self.encode_features(observation)
        mean_action = self.actor(actor_features).clamp(1e-4, 1.0 - 1e-4)
        value = self.critic(pooled).squeeze(-1)
        if squeeze:
            return mean_action.squeeze(0), value.squeeze(0)
        return mean_action, value


    def deterministic_action(
        self,
        observation: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> DeterministicPolicyOutput:
        """Fast deployed-policy path without Beta construction or critic execution."""
        squeeze = observation.ndim == 2
        actor_features, _ = self.encode_features(observation)
        mean_action = self.actor(actor_features).clamp(1e-4, 1.0 - 1e-4)
        if candidate_mask is None:
            masked_action = mean_action
        else:
            mask = candidate_mask
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            masked_action = mean_action * mask.unsqueeze(-1).to(mean_action.dtype)
        if squeeze:
            return DeterministicPolicyOutput(
                action=masked_action.squeeze(0),
                mean_action=mean_action.squeeze(0),
            )
        return DeterministicPolicyOutput(action=masked_action, mean_action=mean_action)

    def _distribution(self, mean_action: torch.Tensor) -> Beta:
        concentration = torch.nn.functional.softplus(self.raw_concentration) + 2.0
        # Keep alpha and beta > 1 for smooth unimodal exploration around the mean.
        alpha = 1.0 + mean_action * (concentration - 2.0)
        beta = 1.0 + (1.0 - mean_action) * (concentration - 2.0)
        return Beta(alpha, beta)

    @staticmethod
    def _masked_reduce(values: torch.Tensor, candidate_mask: torch.Tensor, *, average: bool) -> torch.Tensor:
        if candidate_mask.ndim == 1:
            candidate_mask = candidate_mask.unsqueeze(0)
        expanded = candidate_mask.unsqueeze(-1).expand_as(values).to(values.dtype)
        summed = (values * expanded).sum(dim=(-2, -1))
        if average:
            denominator = expanded.sum(dim=(-2, -1)).clamp_min(1.0)
            return summed / denominator
        return summed

    def action_log_prob_per_dimension(
        self,
        observation: torch.Tensor,
        candidate_mask: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked log-probability for each UE action dimension.

        Shape is [N,2] for a single observation and [B,N,2] for a batch.
        The factorized Beta policy makes this the natural importance-sampling
        unit for dimension-wise clipping. Invalid candidates are zeroed.
        """
        squeeze = observation.ndim == 2
        mean_action, _ = self.action_mean_and_value(observation)
        mean_action_b = mean_action.unsqueeze(0) if squeeze else mean_action
        candidate_mask_b = candidate_mask.unsqueeze(0) if candidate_mask.ndim == 1 else candidate_mask
        sampled = action.unsqueeze(0) if action.ndim == 2 else action
        sampled = sampled.clamp(1e-5, 1.0 - 1e-5)
        distribution = self._distribution(mean_action_b)
        per_dimension = distribution.log_prob(sampled)
        per_dimension = per_dimension * candidate_mask_b.unsqueeze(-1).to(per_dimension.dtype)
        return per_dimension.squeeze(0) if squeeze else per_dimension

    def action_log_prob_per_ue(
        self,
        observation: torch.Tensor,
        candidate_mask: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked log-probability grouped per UE over priority+demand.

        Shape is [N] for a single observation and [B,N] for a batch.  This is
        intentionally separate from the legacy joint log-probability so PPO
        ratio geometry can be ablated without changing action sampling.
        """
        squeeze = observation.ndim == 2
        mean_action, _ = self.action_mean_and_value(observation)
        if squeeze:
            mean_action_b = mean_action.unsqueeze(0)
        else:
            mean_action_b = mean_action
        if candidate_mask.ndim == 1:
            candidate_mask_b = candidate_mask.unsqueeze(0)
        else:
            candidate_mask_b = candidate_mask
        sampled = action.unsqueeze(0) if action.ndim == 2 else action
        sampled = sampled.clamp(1e-5, 1.0 - 1e-5)
        distribution = self._distribution(mean_action_b)
        per_ue = distribution.log_prob(sampled).sum(dim=-1)
        per_ue = per_ue * candidate_mask_b.to(per_ue.dtype)
        return per_ue.squeeze(0) if squeeze else per_ue

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        candidate_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        squeeze = observation.ndim == 2
        mean_action, value = self.action_mean_and_value(observation)
        if squeeze:
            mean_action_b = mean_action.unsqueeze(0)
            value_b = value.unsqueeze(0)
        else:
            mean_action_b = mean_action
            value_b = value
        if candidate_mask.ndim == 1:
            candidate_mask_b = candidate_mask.unsqueeze(0)
        else:
            candidate_mask_b = candidate_mask

        distribution = self._distribution(mean_action_b)
        if action is None:
            sampled = mean_action_b if deterministic else distribution.rsample()
        else:
            sampled = action.unsqueeze(0) if action.ndim == 2 else action
            sampled = sampled.clamp(1e-5, 1.0 - 1e-5)

        log_prob = self._masked_reduce(distribution.log_prob(sampled), candidate_mask_b, average=False)
        entropy = self._masked_reduce(distribution.entropy(), candidate_mask_b, average=True)
        masked_action = sampled * candidate_mask_b.unsqueeze(-1).to(sampled.dtype)

        if squeeze:
            return PolicyOutput(
                action=masked_action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                entropy=entropy.squeeze(0),
                value=value_b.squeeze(0),
                mean_action=mean_action_b.squeeze(0),
            )
        return PolicyOutput(masked_action, log_prob, entropy, value_b, mean_action_b)

    def _expand_input_features(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Expand legacy 8-feature encoder weights into the current input size."""
        adapted = dict(state_dict)
        key = "encoder.0.weight"
        if key in adapted and adapted[key].shape != self.encoder[0].weight.shape:
            old = adapted[key]
            target = torch.zeros_like(self.encoder[0].weight)
            rows = min(old.shape[0], target.shape[0])
            columns = min(old.shape[1], target.shape[1])
            target[:rows, :columns] = old[:rows, :columns]
            adapted[key] = target
        return adapted

    def load_compatible_state_dict(
        self, state_dict: dict[str, torch.Tensor], *, strict: bool = True
    ) -> tuple[list[str], list[str]]:
        result = self.load_state_dict(self._expand_input_features(state_dict), strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)

    def load_imitation_state_dict(self, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        result = self.load_state_dict(self._expand_input_features(state_dict), strict=False)
        expected_missing = [key for key in result.missing_keys if key.startswith("critic") or key == "raw_concentration"]
        unexpected_missing = [key for key in result.missing_keys if key not in expected_missing]
        if unexpected_missing:
            raise RuntimeError(f"imitation checkpoint is missing actor keys: {unexpected_missing}")
        return list(result.missing_keys), list(result.unexpected_keys)



class SplitEncoderActorCritic(SharedSetActorCritic):
    """Feed-forward actor/critic with independent encoders after identical init.

    The actor encoder and critic encoder start as exact copies of the legacy
    shared encoder.  ``copy.deepcopy`` consumes no RNG, so when constructed from
    the same torch seed, step-0 actor outputs, critic values, and subsequent PPO
    sampling RNG match ``SharedSetActorCritic`` exactly.  Training can then
    separate the actor and critic representations without an initialization
    confound.
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, initial_concentration: float = 20.0):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            initial_concentration=initial_concentration,
        )
        self.critic_encoder = copy.deepcopy(self.encoder)

    def encode_features(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = observation.ndim == 2
        if squeeze:
            observation = observation.unsqueeze(0)
        if observation.ndim != 3:
            raise ValueError("observation must have shape [N,F] or [B,N,F]")
        actor_local = self.encoder(observation)
        critic_local = self.critic_encoder(observation)
        mean_context = actor_local.mean(dim=1, keepdim=True).expand_as(actor_local)
        actor_features = torch.cat([actor_local, mean_context], dim=-1)
        pooled = torch.cat([critic_local.mean(dim=1), critic_local.amax(dim=1)], dim=-1)
        return actor_features, pooled

def build_baseline_compatible_expanded_model(
    *, input_dim: int, hidden_dim: int = 64, initial_concentration: float = 20.0
) -> SharedSetActorCritic:
    """Build an expanded feed-forward model paired to the 16-feature baseline init.

    The first 16 input columns and every downstream parameter are exactly the
    baseline model for the current torch RNG state. Added input columns start at
    zero weight. The torch CPU RNG is restored to the state immediately after
    constructing the baseline, so subsequent stochastic PPO sampling is paired.
    """
    if input_dim <= 16:
        return SharedSetActorCritic(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            initial_concentration=initial_concentration,
        )
    baseline = SharedSetActorCritic(
        input_dim=16,
        hidden_dim=hidden_dim,
        initial_concentration=initial_concentration,
    )
    state = {k: v.detach().clone() for k, v in baseline.state_dict().items()}
    rng_after_baseline = torch.get_rng_state().clone()
    expanded = SharedSetActorCritic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        initial_concentration=initial_concentration,
    )
    expanded.load_compatible_state_dict(state, strict=True)
    torch.set_rng_state(rng_after_baseline)
    return expanded
