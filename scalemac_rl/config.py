from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ScaleMacConfig:
    """Configuration for the abstract single-cell downlink MVP."""

    num_ues: int = 1200
    num_prbs: int = 273
    max_selected_ues: int = 64
    episode_slots: int = 1000

    # Hybrid safety/learning split. The projector reserves up to this many
    # grants for HARQ and long-waiting UEs; PPO ranks the remaining grants.
    safety_reserve_ues: int = 0
    safety_wait_threshold_ratio: float = 0.80

    # Optional upper-bound experiment mode: keep one static CQI/demand profile
    # across episode resets while HARQ randomness continues to evolve.
    freeze_static_profiles: bool = False
    static_profile_seed: int | None = None

    # Dense tail-delay shaping starts before the hard P99 constraint is crossed.
    deadline_target_slots: float = 50.0
    # Fixed reference target used for comparable logging/checkpoint ranking even
    # while the training curriculum temporarily uses a looser active target.
    reference_deadline_target_slots: float = 50.0
    deadline_risk_start_ratio: float = 0.60
    reward_deadline_risk_penalty_weight: float = 0.15

    packet_size_bytes: int = 1500
    full_buffer_base_bytes: int = 1_000_000

    # A UE is considered starved when it has had no successful delivery for
    # this many consecutive slots. The scheduling-only wait is logged separately.
    starvation_threshold_slots: int = 64
    ewma_alpha: float = 0.02

    harq_enabled: bool = True
    target_bler: float = 0.10
    max_harq_retransmissions: int = 3

    # All positive reward scores are normalized to [0, 1]. Throughput remains
    # the main objective, while fairness and service regularize the solution.
    reward_throughput_weight: float = 0.50
    reward_fairness_weight: float = 0.35
    reward_service_weight: float = 0.15

    # Starvation is a constraint-like penalty applied outside the convex
    # combination above so a high-throughput policy cannot ignore most UEs.
    reward_starvation_penalty_weight: float = 0.50
    starvation_tolerance: float = 0.00

    # Extra dense pressure on the single worst-served UE. This complements
    # the top-1% P99-style risk and helps reduce the maximum service gap.
    max_wait_target_slots: float = 60.0
    reward_max_wait_risk_penalty_weight: float = 0.10

    seed: int = 7

    # Static heterogeneous UE profiles used by the MVP.
    low_cqi_fraction: float = 0.30
    medium_cqi_fraction: float = 0.40
    high_cqi_fraction: float = 0.30

    low_demand_fraction: float = 0.40
    medium_demand_fraction: float = 0.40
    high_demand_fraction: float = 0.20

    def validate(self) -> None:
        if self.num_ues <= 0:
            raise ValueError("num_ues must be positive")
        if not 1 <= self.max_selected_ues <= self.num_ues:
            raise ValueError("max_selected_ues must be in [1, num_ues]")
        if self.num_prbs < self.max_selected_ues:
            raise ValueError(
                "num_prbs must be >= max_selected_ues so every selected UE can receive at least 1 PRB"
            )
        if self.episode_slots <= 0:
            raise ValueError("episode_slots must be positive")
        if not 0 <= self.safety_reserve_ues <= self.max_selected_ues:
            raise ValueError("safety_reserve_ues must be in [0, max_selected_ues]")
        if self.safety_wait_threshold_ratio < 0.0:
            raise ValueError("safety_wait_threshold_ratio must be non-negative")
        if self.static_profile_seed is not None and self.static_profile_seed < 0:
            raise ValueError("static_profile_seed must be non-negative when provided")
        if self.deadline_target_slots <= 0.0:
            raise ValueError("deadline_target_slots must be positive")
        if self.reference_deadline_target_slots <= 0.0:
            raise ValueError("reference_deadline_target_slots must be positive")
        if not 0.0 <= self.deadline_risk_start_ratio < 1.0:
            raise ValueError("deadline_risk_start_ratio must be in [0, 1)")
        if self.reward_deadline_risk_penalty_weight < 0.0:
            raise ValueError("reward_deadline_risk_penalty_weight must be non-negative")
        if self.starvation_threshold_slots <= 0:
            raise ValueError("starvation_threshold_slots must be positive")
        if self.max_wait_target_slots <= 0.0:
            raise ValueError("max_wait_target_slots must be positive")
        if self.reward_max_wait_risk_penalty_weight < 0.0:
            raise ValueError("reward_max_wait_risk_penalty_weight must be non-negative")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not 0.0 <= self.target_bler < 1.0:
            raise ValueError("target_bler must be in [0, 1)")
        if self.max_harq_retransmissions < 0:
            raise ValueError("max_harq_retransmissions must be non-negative")

        cqi_sum = self.low_cqi_fraction + self.medium_cqi_fraction + self.high_cqi_fraction
        demand_sum = (
            self.low_demand_fraction
            + self.medium_demand_fraction
            + self.high_demand_fraction
        )
        if abs(cqi_sum - 1.0) > 1e-6:
            raise ValueError("CQI fractions must sum to 1")
        if abs(demand_sum - 1.0) > 1e-6:
            raise ValueError("demand fractions must sum to 1")

        reward_sum = (
            self.reward_throughput_weight
            + self.reward_fairness_weight
            + self.reward_service_weight
        )
        if abs(reward_sum - 1.0) > 1e-6:
            raise ValueError("positive reward weights must sum to 1")
        if self.reward_starvation_penalty_weight < 0.0:
            raise ValueError("reward_starvation_penalty_weight must be non-negative")
        if not 0.0 <= self.starvation_tolerance < 1.0:
            raise ValueError("starvation_tolerance must be in [0, 1)")

    @classmethod
    def from_json(cls, path: str | Path) -> "ScaleMacConfig":
        payload: dict[str, Any]
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = cls(**payload)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
