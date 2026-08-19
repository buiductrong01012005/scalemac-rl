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

    # Scheduler attribution mode. ``hybrid`` reserves rule-selected grants,
    # ``ppo_only`` lets PPO choose all grants, and ``rule_only`` ignores PPO
    # priority for UE selection. The action projector remains validity-only in
    # PPO-only mode.
    scheduler_mode: str = "hybrid"
    force_harq_retransmissions: bool = True

    # Hybrid safety/learning split. The projector reserves up to this many
    # grants for HARQ and long-waiting UEs; PPO ranks the remaining grants.
    safety_reserve_ues: int = 0
    safety_wait_threshold_ratio: float = 0.80

    # Optional controlled-profile mode: keep the same heterogeneous CQI anchor,
    # demand profile, and speed metadata across episode resets. Dynamic-CQI modes
    # may still evolve around the frozen anchor after reset.
    freeze_static_profiles: bool = False
    static_profile_seed: int | None = None

    # CQI dynamics. ``static`` preserves the pre-v0.11 behavior exactly.
    # ``correlated`` evolves a latent CQI state around each UE's heterogeneous
    # anchor using a mean-reverting AR(1)-style update, then quantizes to 1..15.
    cqi_mode: str = "static"
    cqi_temporal_correlation: float = 0.97
    cqi_innovation_std: float = 0.35
    cqi_update_interval_slots: int = 1
    cqi_max_delta_per_update: int = 1

    # CSI reporting abstraction. ``perfect`` exposes the current true CQI to the
    # scheduler and exactly preserves the v0.11 observation path. ``periodic``
    # samples the true CQI on a configurable period, optionally delays delivery
    # and adds measurement error in CQI-index units. The PHY still uses true CQI.
    csi_report_mode: str = "perfect"
    csi_report_period_slots: int = 1
    csi_report_delay_slots: int = 0
    csi_report_error_std: float = 0.0

    # Optional feed-forward observation enrichments. Defaults preserve the
    # pre-v0.16 16-feature observation exactly. CSI age exposes how old the
    # current report is; reported-CQI trend exposes the signed change between
    # the two most recently delivered CSI reports.
    observation_include_csi_age: bool = False
    observation_include_reported_cqi_trend: bool = False

    # Link-adaptation abstraction. ``legacy_fixed_bler`` preserves the pre-v0.13
    # PHY path: true-CQI efficiency with a fixed BLER. ``cqi_mcs_bler`` maps the
    # scheduler-visible reported CQI to 3GPP-inspired PDSCH MCS Table 1 and makes
    # BLER depend on the mismatch between selected MCS and true CQI.
    link_adaptation_mode: str = "legacy_fixed_bler"
    link_adaptation_cqi_backoff: int = 0
    bler_mismatch_slope: float = 1.5

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
    # the main objective, while fairness, service, and per-UE deficit credit
    # regularize the solution.
    # Global scale for the normalized positive reward mixture. This allows
    # reward-study cases to give positive and penalty objective families equal
    # absolute coefficients while preserving relative positive weights that sum
    # to one. The default keeps legacy behaviour unchanged.
    reward_positive_scale: float = 1.0
    reward_throughput_weight: float = 0.45
    reward_fairness_weight: float = 0.35
    reward_service_weight: float = 0.15
    reward_deficit_service_weight: float = 0.05
    # Optional v2 reward terms. Defaults are zero for legacy compatibility.
    reward_pf_utility_weight: float = 0.0
    reward_low_throughput_weight: float = 0.0
    reward_urgency_service_weight: float = 0.0

    # Signed dense shaping terms. They reward immediate improvements in Jain
    # fairness and proportional-fair utility without replacing KPI reporting.
    reward_fairness_delta_weight: float = 0.03
    reward_pf_utility_delta_weight: float = 0.02
    fairness_delta_scale: float = 20.0
    pf_utility_delta_scale: float = 8.0

    # Starvation is a constraint-like penalty applied outside the convex
    # combination above so a high-throughput policy cannot ignore most UEs.
    reward_starvation_penalty_weight: float = 0.50
    starvation_tolerance: float = 0.00

    # Extra dense pressure on the single worst-served UE. This complements
    # the top-1% P99-style risk and helps reduce the maximum service gap.
    max_wait_target_slots: float = 60.0
    reward_max_wait_risk_penalty_weight: float = 0.10
    # Dense population-wide pressure before P99/max-wait constraints are crossed.
    reward_population_wait_penalty_weight: float = 0.0
    low_throughput_percentile: float = 10.0

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
        if self.scheduler_mode not in {"hybrid", "ppo_only", "rule_only"}:
            raise ValueError("scheduler_mode must be hybrid, ppo_only, or rule_only")
        if not 0 <= self.safety_reserve_ues <= self.max_selected_ues:
            raise ValueError("safety_reserve_ues must be in [0, max_selected_ues]")
        if self.safety_wait_threshold_ratio < 0.0:
            raise ValueError("safety_wait_threshold_ratio must be non-negative")
        if self.static_profile_seed is not None and self.static_profile_seed < 0:
            raise ValueError("static_profile_seed must be non-negative when provided")
        if self.cqi_mode not in {"static", "correlated"}:
            raise ValueError("cqi_mode must be static or correlated")
        if not 0.0 <= self.cqi_temporal_correlation < 1.0:
            raise ValueError("cqi_temporal_correlation must be in [0, 1)")
        if self.cqi_innovation_std < 0.0:
            raise ValueError("cqi_innovation_std must be non-negative")
        if self.cqi_update_interval_slots <= 0:
            raise ValueError("cqi_update_interval_slots must be positive")
        if self.cqi_max_delta_per_update <= 0:
            raise ValueError("cqi_max_delta_per_update must be positive")
        if self.csi_report_mode not in {"perfect", "periodic"}:
            raise ValueError("csi_report_mode must be perfect or periodic")
        if self.csi_report_period_slots <= 0:
            raise ValueError("csi_report_period_slots must be positive")
        if self.csi_report_delay_slots < 0:
            raise ValueError("csi_report_delay_slots must be non-negative")
        if self.csi_report_error_std < 0.0:
            raise ValueError("csi_report_error_std must be non-negative")
        if self.link_adaptation_mode not in {"legacy_fixed_bler", "cqi_mcs_bler"}:
            raise ValueError("link_adaptation_mode must be legacy_fixed_bler or cqi_mcs_bler")
        if self.link_adaptation_cqi_backoff < 0:
            raise ValueError("link_adaptation_cqi_backoff must be non-negative")
        if self.bler_mismatch_slope <= 0.0:
            raise ValueError("bler_mismatch_slope must be positive")
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
        if self.reward_population_wait_penalty_weight < 0.0:
            raise ValueError("reward_population_wait_penalty_weight must be non-negative")
        if not 0.0 < self.low_throughput_percentile < 50.0:
            raise ValueError("low_throughput_percentile must be in (0, 50)")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not 0.0 <= self.target_bler < 1.0:
            raise ValueError("target_bler must be in [0, 1)")
        if self.link_adaptation_mode == "cqi_mcs_bler" and self.target_bler <= 0.0:
            raise ValueError("target_bler must be positive in cqi_mcs_bler mode")
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

        if self.reward_positive_scale < 0.0:
            raise ValueError("reward_positive_scale must be non-negative")

        reward_sum = (
            self.reward_throughput_weight
            + self.reward_fairness_weight
            + self.reward_service_weight
            + self.reward_deficit_service_weight
            + self.reward_pf_utility_weight
            + self.reward_low_throughput_weight
            + self.reward_urgency_service_weight
        )
        if abs(reward_sum - 1.0) > 1e-6:
            raise ValueError("positive reward weights must sum to 1")
        for name, value in (
            ("reward_pf_utility_weight", self.reward_pf_utility_weight),
            ("reward_low_throughput_weight", self.reward_low_throughput_weight),
            ("reward_urgency_service_weight", self.reward_urgency_service_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.reward_fairness_delta_weight < 0.0:
            raise ValueError("reward_fairness_delta_weight must be non-negative")
        if self.reward_pf_utility_delta_weight < 0.0:
            raise ValueError("reward_pf_utility_delta_weight must be non-negative")
        if self.fairness_delta_scale <= 0.0 or self.pf_utility_delta_scale <= 0.0:
            raise ValueError("fairness/PF delta scales must be positive")
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
