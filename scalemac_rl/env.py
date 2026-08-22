from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ScaleMacConfig
from .metrics import clip01, jain_fairness, safe_percentile
from .link_adaptation import (
    CQI_TABLE1_EFFICIENCY,
    bler_probability_from_cqi_mismatch,
    mcs_efficiency,
    mcs_modulation_order,
    required_cqi_for_mcs,
    select_mcs_from_reported_cqi,
)
from .projector import ProjectedGrant, project_action


# Standard 15-level CQI spectral-efficiency abstraction (3GPP CQI Table 1).
_CQI_EFFICIENCY = CQI_TABLE1_EFFICIENCY.astype(np.float32)


# Per-UE observation columns.
CQI = 0
QUEUE = 1
DEMAND = 2
EWMA_THROUGHPUT = 3
TIME_SINCE_SERVICE = 4
HARQ_PENDING = 5
HARQ_RETX_COUNT = 6
ELIGIBLE = 7
THROUGHPUT_DEFICIT = 8
SERVICE_DEFICIT = 9
CQI_RANK = 10
THROUGHPUT_DEFICIT_RANK = 11
WAIT_RANK = 12
THROUGHPUT_TO_MEAN = 13
WAIT_TO_DEADLINE = 14
LAST_PRB_SHARE = 15
OBSERVATION_FEATURES = 16


def observation_feature_count(config: ScaleMacConfig) -> int:
    """Return per-UE observation width for the selected feature ablation."""
    return (
        OBSERVATION_FEATURES
        + int(config.observation_include_csi_age)
        + int(config.observation_include_reported_cqi_trend)
    )


def deadline_risk_score(
    tail_waits: np.ndarray,
    *,
    target_slots: float,
    start_ratio: float,
) -> float:
    """Return a dense, non-saturating tail-delay risk score.

    The score is zero before ``start_ratio * target_slots``, equals one at the
    target, and continues growing logarithmically beyond the target. This keeps
    PPO able to distinguish moderately late and severely late states instead of
    clipping both to the same penalty.
    """
    if target_slots <= 0.0:
        raise ValueError("target_slots must be positive")
    if not 0.0 <= start_ratio < 1.0:
        raise ValueError("start_ratio must be in [0, 1)")
    waits = np.asarray(tail_waits, dtype=np.float64)
    if waits.size == 0:
        return 0.0
    start = start_ratio * target_slots
    span = max(target_slots - start, 1e-9)
    normalized = np.maximum((waits - start) / span, 0.0)
    # log1p(x) / log(2) maps x=1 (the target) to risk=1 while remaining
    # smooth and non-saturating for x>1.
    return float(np.mean(np.log1p(normalized) / np.log(2.0)))


@dataclass(slots=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ScaleMacDownlinkEnv:
    """Abstract single-cell downlink scheduler environment.

    This is a fast training surrogate, not a full 3GPP PHY/MAC implementation.
    Raw policy actions are projected to the required output contract:
    Top-K selected UEs and PRBs per selected UE.
    """

    def __init__(self, config: ScaleMacConfig | None = None):
        self.config = config or ScaleMacConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        # Keep channel innovations on a separate RNG stream so enabling Dynamic
        # CQI does not change the HARQ success/failure random sequence.
        self.channel_rng = np.random.default_rng(self.config.seed + 1_000_003)
        # CSI measurement noise has its own RNG stream so reporting realism does
        # not perturb either the channel process or HARQ success/failure draws.
        self.csi_rng = np.random.default_rng(self.config.seed + 2_000_003)

        n = self.config.num_ues
        self.slot = 0
        # ``cqi`` is the true instantaneous channel state used by the PHY.
        # ``reported_cqi`` is what the scheduler observes through CSI reporting.
        self.cqi = np.zeros(n, dtype=np.int16)
        self.reported_cqi = np.zeros(n, dtype=np.int16)
        self.cqi_anchor = np.zeros(n, dtype=np.float64)
        self._cqi_latent = np.zeros(n, dtype=np.float64)
        self.demand_factor = np.ones(n, dtype=np.float32)
        # Still metadata only in v0.11. Dynamic CQI is intentionally controlled
        # by explicit channel parameters rather than mobility at this stage.
        self.speed_mps = np.zeros(n, dtype=np.float32)
        self.queue_bytes = np.zeros(n, dtype=np.float64)
        self.queue_target_bytes = np.zeros(n, dtype=np.float64)
        self.ewma_throughput_bits = np.zeros(n, dtype=np.float64)
        self.cumulative_delivered_bits = np.zeros(n, dtype=np.float64)
        # Separate scheduling-opportunity accounting from successful-delivery throughput.
        # This lets the reward distinguish "was selected" from "delivered many bits".
        self.ewma_schedule_rate = np.zeros(n, dtype=np.float64)
        self.cumulative_schedule_count = np.zeros(n, dtype=np.float64)
        # Primary wait counter: slots since the last successful delivery.
        # A second counter records scheduling gaps even when HARQ transmission fails.
        self.time_since_service = np.zeros(n, dtype=np.int32)
        self.time_since_schedule = np.zeros(n, dtype=np.int32)
        self.harq_pending = np.zeros(n, dtype=bool)
        self.harq_retx_count = np.zeros(n, dtype=np.int16)
        self.eligible = np.ones(n, dtype=bool)
        self.last_grant = np.zeros(n, dtype=np.int32)
        self.last_success = np.zeros(n, dtype=bool)
        self._frozen_cqi: np.ndarray | None = None
        self._frozen_demand_factor: np.ndarray | None = None
        self._frozen_speed_mps: np.ndarray | None = None
        self._previous_cumulative_fairness = 0.0
        self._previous_pf_utility = 0.0
        self._last_cqi_mean_abs_change = 0.0
        self._last_cqi_changed_fraction = 0.0
        self._reported_cqi_generation_slot = 0
        self._previous_reported_cqi = np.zeros(n, dtype=np.int16)
        self._reported_cqi_trend = np.zeros(n, dtype=np.float32)
        self._pending_csi_reports: list[tuple[int, int, np.ndarray]] = []

        self.reset(seed=self.config.seed)

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (self.config.num_ues, observation_feature_count(self.config))

    @property
    def action_shape(self) -> tuple[int, int]:
        return (self.config.num_ues, 2)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.channel_rng = np.random.default_rng(seed + 1_000_003)
            self.csi_rng = np.random.default_rng(seed + 2_000_003)

        self.slot = 0
        if self.config.freeze_static_profiles:
            if self._frozen_cqi is None:
                profile_seed = (
                    self.config.static_profile_seed
                    if self.config.static_profile_seed is not None
                    else self.config.seed
                )
                profile_rng = np.random.default_rng(profile_seed)
                self._frozen_cqi = self._sample_cqi_profiles(profile_rng)
                self._frozen_demand_factor = self._sample_demand_profiles(profile_rng)
                self._frozen_speed_mps = profile_rng.choice(
                    np.asarray([0.0, 1.5, 5.0, 15.0, 25.0], dtype=np.float32),
                    size=self.config.num_ues,
                    replace=True,
                )
            self.cqi = self._frozen_cqi.copy()
            self.demand_factor = self._frozen_demand_factor.copy()
            self.speed_mps = self._frozen_speed_mps.copy()
        else:
            self.cqi = self._sample_cqi_profiles(self.rng)
            self.demand_factor = self._sample_demand_profiles(self.rng)
            self.speed_mps = self.rng.choice(
                np.asarray([0.0, 1.5, 5.0, 15.0, 25.0], dtype=np.float32),
                size=self.config.num_ues,
                replace=True,
            )

        # Preserve each UE's heterogeneous channel class as a long-term anchor.
        # Dynamic CQI starts exactly at that anchor on every reset.
        self.cqi_anchor = self.cqi.astype(np.float64).copy()
        self._cqi_latent = self.cqi_anchor.copy()
        self._last_cqi_mean_abs_change = 0.0
        self._last_cqi_changed_fraction = 0.0
        # Start every episode with an exact initial report. Staleness/error only
        # appears after the true channel begins evolving.
        self.reported_cqi = self.cqi.copy()
        self._previous_reported_cqi = self.reported_cqi.copy()
        self._reported_cqi_trend.fill(0.0)
        self._reported_cqi_generation_slot = 0
        self._pending_csi_reports.clear()

        self.queue_target_bytes = (
            self.config.full_buffer_base_bytes * self.demand_factor
        ).astype(np.float64)
        self.queue_bytes = self.queue_target_bytes.copy()
        self.ewma_throughput_bits.fill(0.0)
        self.cumulative_delivered_bits.fill(0.0)
        self.ewma_schedule_rate.fill(0.0)
        self.cumulative_schedule_count.fill(0.0)
        self.time_since_service.fill(0)
        self.time_since_schedule.fill(0)
        self.harq_pending.fill(False)
        self.harq_retx_count.fill(0)
        self.eligible.fill(True)
        self.last_grant.fill(0)
        self.last_success.fill(False)
        self._previous_cumulative_fairness = 0.0
        self._previous_pf_utility = 0.0

        observation = self._observation()
        info = {
            "slot": self.slot,
            "num_active_ues": self.config.num_ues,
            "num_prbs": self.config.num_prbs,
            "max_selected_ues": self.config.max_selected_ues,
            "cqi_mode": self.config.cqi_mode,
            "mean_cqi": float(np.mean(self.cqi)),
            "std_cqi": float(np.std(self.cqi)),
            "csi_report_mode": self.config.csi_report_mode,
            "mean_reported_cqi": float(np.mean(self.reported_cqi)),
            "mean_csi_abs_error": 0.0,
            "p95_csi_abs_error": 0.0,
            "csi_stale_fraction": 0.0,
            "csi_report_age_slots": 0.0,
            "csi_report_generated": 0.0,
            "csi_report_delivered": 0.0,
            "link_adaptation_mode": self.config.link_adaptation_mode,
            "mean_mcs_index": 0.0,
            "mean_modulation_order": 0.0,
            "mean_predicted_bler": float(self.config.target_bler),
            "observed_bler": 0.0,
            "spectral_efficiency_bps_hz": 0.0,
            "attempted_spectral_efficiency_bps_hz": 0.0,
            "harq_retransmission_attempts": 0.0,
            "harq_retransmission_fraction": 0.0,
        }
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        grant = project_action(
            action,
            eligible=self.eligible,
            harq_pending=self.harq_pending,
            harq_retx_count=self.harq_retx_count,
            time_since_service=self.time_since_service,
            num_prbs=self.config.num_prbs,
            max_selected_ues=self.config.max_selected_ues,
            safety_reserve_ues=self.config.safety_reserve_ues,
            safety_wait_threshold_ratio=self.config.safety_wait_threshold_ratio,
            starvation_threshold_slots=self.config.starvation_threshold_slots,
            selection_mode=self.config.scheduler_mode,
            force_harq_retransmissions=self.config.force_harq_retransmissions,
        )
        metrics = self._execute_grant(grant)
        reward, reward_breakdown = self._reward(metrics)
        metrics.update(reward_breakdown)

        self.slot += 1
        cqi_diagnostics = self._advance_cqi()
        metrics.update(cqi_diagnostics)
        csi_diagnostics = self._advance_csi_reporting()
        metrics.update(csi_diagnostics)
        terminated = self.slot >= self.config.episode_slots
        truncated = False
        observation = self._observation()
        info = {"slot": self.slot, **metrics}
        return observation, reward, terminated, truncated, info

    def _advance_cqi(self) -> dict[str, float | str]:
        """Advance the per-UE CQI process for the next scheduling slot.

        ``static`` is an exact no-op. ``correlated`` uses a mean-reverting
        latent process around the reset-time heterogeneous CQI anchor. The
        latent state is clipped to the valid CQI range and the quantized CQI
        is additionally rate-limited, preventing independent slot-to-slot jumps.
        """
        previous = self.cqi.copy()
        if (
            self.config.cqi_mode == "correlated"
            and self.slot % self.config.cqi_update_interval_slots == 0
        ):
            rho = self.config.cqi_temporal_correlation
            innovation = self.channel_rng.normal(
                loc=0.0,
                scale=self.config.cqi_innovation_std,
                size=self.config.num_ues,
            )
            self._cqi_latent = (
                self.cqi_anchor
                + rho * (self._cqi_latent - self.cqi_anchor)
                + innovation
            )
            self._cqi_latent = np.clip(self._cqi_latent, 1.0, 15.0)
            proposed = np.rint(self._cqi_latent).astype(np.int16)
            max_delta = int(self.config.cqi_max_delta_per_update)
            lower = np.maximum(previous.astype(np.int32) - max_delta, 1)
            upper = np.minimum(previous.astype(np.int32) + max_delta, 15)
            self.cqi = np.clip(proposed.astype(np.int32), lower, upper).astype(np.int16)

        absolute_change = np.abs(self.cqi.astype(np.int32) - previous.astype(np.int32))
        self._last_cqi_mean_abs_change = float(np.mean(absolute_change))
        self._last_cqi_changed_fraction = float(np.mean(absolute_change > 0))
        return {
            "cqi_mode": self.config.cqi_mode,
            "mean_cqi": float(np.mean(self.cqi)),
            "std_cqi": float(np.std(self.cqi)),
            "cqi_mean_abs_change": self._last_cqi_mean_abs_change,
            "cqi_changed_fraction": self._last_cqi_changed_fraction,
        }


    def _advance_csi_reporting(self) -> dict[str, float | str]:
        """Advance the abstract CSI measurement/reporting pipeline.

        The scheduler observes ``reported_cqi`` while transmission physics and
        the throughput oracle continue to use the true ``cqi``. ``perfect`` is
        an exact observation of the current true CQI. ``periodic`` creates one
        cell-wide report snapshot every configured period and delivers it after
        a configurable delay. Measurement noise is expressed in CQI-index units.
        """
        generated = 0.0
        delivered = 0.0
        if self.config.csi_report_mode == "perfect":
            previous_report = self.reported_cqi.copy()
            self.reported_cqi = self.cqi.copy()
            self._previous_reported_cqi = previous_report
            self._reported_cqi_trend = (
                self.reported_cqi.astype(np.float32) - previous_report.astype(np.float32)
            )
            self._reported_cqi_generation_slot = self.slot
            self._pending_csi_reports.clear()
            generated = 1.0
            delivered = 1.0
        else:
            if self.slot % self.config.csi_report_period_slots == 0:
                measurement = self.cqi.astype(np.float64)
                if self.config.csi_report_error_std > 0.0:
                    measurement = measurement + self.csi_rng.normal(
                        loc=0.0,
                        scale=self.config.csi_report_error_std,
                        size=self.config.num_ues,
                    )
                measurement = np.clip(np.rint(measurement), 1, 15).astype(np.int16)
                self._pending_csi_reports.append(
                    (
                        self.slot + self.config.csi_report_delay_slots,
                        self.slot,
                        measurement,
                    )
                )
                generated = 1.0

            ready = [item for item in self._pending_csi_reports if item[0] <= self.slot]
            if ready:
                # If several reports become deliverable together, the newest
                # measurement supersedes older snapshots.
                _, generation_slot, measurement = max(ready, key=lambda item: item[1])
                previous_report = self.reported_cqi.copy()
                self.reported_cqi = measurement.copy()
                self._previous_reported_cqi = previous_report
                self._reported_cqi_trend = (
                    self.reported_cqi.astype(np.float32) - previous_report.astype(np.float32)
                )
                self._reported_cqi_generation_slot = generation_slot
                self._pending_csi_reports = [
                    item for item in self._pending_csi_reports if item[0] > self.slot
                ]
                delivered = 1.0

        abs_error = np.abs(
            self.reported_cqi.astype(np.int32) - self.cqi.astype(np.int32)
        )
        return {
            "csi_report_mode": self.config.csi_report_mode,
            "mean_reported_cqi": float(np.mean(self.reported_cqi)),
            "mean_csi_abs_error": float(np.mean(abs_error)),
            "p95_csi_abs_error": float(np.percentile(abs_error, 95)),
            "csi_stale_fraction": float(np.mean(abs_error > 0)),
            "csi_report_age_slots": float(
                max(self.slot - self._reported_cqi_generation_slot, 0)
            ),
            "csi_report_generated": generated,
            "csi_report_delivered": delivered,
            "mean_abs_reported_cqi_trend": float(np.mean(np.abs(self._reported_cqi_trend))),
        }

    def _sample_cqi_profiles(self, rng: np.random.Generator) -> np.ndarray:
        n = self.config.num_ues
        low = int(round(n * self.config.low_cqi_fraction))
        medium = int(round(n * self.config.medium_cqi_fraction))
        high = n - low - medium
        values = np.concatenate(
            [
                rng.integers(1, 6, size=low),
                rng.integers(6, 11, size=medium),
                rng.integers(11, 16, size=high),
            ]
        ).astype(np.int16)
        rng.shuffle(values)
        return values

    def _sample_demand_profiles(self, rng: np.random.Generator) -> np.ndarray:
        n = self.config.num_ues
        low = int(round(n * self.config.low_demand_fraction))
        medium = int(round(n * self.config.medium_demand_fraction))
        high = n - low - medium
        values = np.concatenate(
            [
                np.full(low, 0.5, dtype=np.float32),
                np.full(medium, 1.0, dtype=np.float32),
                np.full(high, 2.0, dtype=np.float32),
            ]
        )
        rng.shuffle(values)
        return values

    def _observation(self) -> np.ndarray:
        max_queue = max(float(self.queue_target_bytes.max(initial=1.0)), 1.0)
        max_ewma = max(float(self.ewma_throughput_bits.max(initial=1.0)), 1.0)
        wait_denominator = max(self.config.starvation_threshold_slots, 1)
        retx_denominator = max(self.config.max_harq_retransmissions, 1)

        base = np.empty((self.config.num_ues, OBSERVATION_FEATURES), dtype=np.float32)
        observation = base
        observation[:, CQI] = self.reported_cqi / 15.0
        observation[:, QUEUE] = np.clip(self.queue_bytes / max_queue, 0.0, 1.0)
        observation[:, DEMAND] = self.demand_factor / 2.0
        observation[:, EWMA_THROUGHPUT] = np.clip(
            self.ewma_throughput_bits / max_ewma, 0.0, 1.0
        )
        observation[:, TIME_SINCE_SERVICE] = np.clip(
            self.time_since_service / wait_denominator, 0.0, 2.0
        )
        observation[:, HARQ_PENDING] = self.harq_pending.astype(np.float32)
        observation[:, HARQ_RETX_COUNT] = np.clip(
            self.harq_retx_count / retx_denominator, 0.0, 1.0
        )
        observation[:, ELIGIBLE] = self.eligible.astype(np.float32)
        observation[:, THROUGHPUT_DEFICIT] = self._throughput_deficit()
        expected_cycle = max(
            int(np.ceil(self.config.num_ues / max(self.config.max_selected_ues, 1))),
            1,
        )
        observation[:, SERVICE_DEFICIT] = np.clip(
            self.time_since_service / expected_cycle, 0.0, 2.0
        )

        demand_normalized_throughput = self.ewma_throughput_bits / np.maximum(
            self.demand_factor, 1e-6
        )
        observation[:, CQI_RANK] = self._percentile_rank(
            self.reported_cqi.astype(np.float64)
        )
        # High value means that the UE is relatively underserved.
        observation[:, THROUGHPUT_DEFICIT_RANK] = 1.0 - self._percentile_rank(
            demand_normalized_throughput
        )
        observation[:, WAIT_RANK] = self._percentile_rank(
            self.time_since_service.astype(np.float64)
        )
        eligible_values = demand_normalized_throughput[self.eligible]
        mean_throughput = float(eligible_values.mean()) if eligible_values.size else 0.0
        if mean_throughput <= 1e-9:
            observation[:, THROUGHPUT_TO_MEAN] = 0.0
        else:
            observation[:, THROUGHPUT_TO_MEAN] = np.clip(
                demand_normalized_throughput / mean_throughput / 2.0,
                0.0,
                1.0,
            )
        observation[:, WAIT_TO_DEADLINE] = np.clip(
            self.time_since_service / max(self.config.deadline_target_slots, 1e-9),
            0.0,
            2.0,
        )
        observation[:, LAST_PRB_SHARE] = np.clip(
            self.last_grant / max(self.config.num_prbs, 1), 0.0, 1.0
        )

        extras: list[np.ndarray] = []
        if self.config.observation_include_csi_age:
            age_slots = max(self.slot - self._reported_cqi_generation_slot, 0)
            age_scale = max(
                self.config.csi_report_period_slots + self.config.csi_report_delay_slots,
                1,
            )
            extras.append(
                np.full(
                    (self.config.num_ues, 1),
                    min(age_slots / age_scale, 2.0),
                    dtype=np.float32,
                )
            )
        if self.config.observation_include_reported_cqi_trend:
            extras.append(
                np.clip(self._reported_cqi_trend / 14.0, -1.0, 1.0)
                .astype(np.float32)[:, None]
            )
        if extras:
            return np.concatenate([base, *extras], axis=1).astype(np.float32, copy=False)
        return base

    @staticmethod
    def _percentile_rank(values: np.ndarray) -> np.ndarray:
        """Tie-aware percentile ranks in [0, 1]."""
        array = np.asarray(values, dtype=np.float64)
        n = int(array.size)
        if n <= 1:
            return np.zeros(n, dtype=np.float32)
        order = np.argsort(array, kind="mergesort")
        sorted_values = array[order]
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_values[end] == sorted_values[start]:
                end += 1
            average_rank = 0.5 * (start + end - 1)
            ranks[order[start:end]] = average_rank / (n - 1)
            start = end
        return ranks.astype(np.float32)

    def _throughput_deficit(self) -> np.ndarray:
        """Demand-normalized per-UE throughput deficit used as dense actor input."""
        normalized = self.ewma_throughput_bits / np.maximum(self.demand_factor, 1e-6)
        eligible_values = normalized[self.eligible]
        reference = float(eligible_values.mean()) if eligible_values.size else 0.0
        if reference <= 1e-9:
            return np.zeros(self.config.num_ues, dtype=np.float32)
        deficit = np.maximum((reference - normalized) / reference, 0.0)
        return np.clip(deficit, 0.0, 2.0).astype(np.float32)

    def _pf_utility(self) -> float:
        normalized = self.ewma_throughput_bits / np.maximum(self.demand_factor, 1e-6)
        scale = max(float(normalized.max(initial=1.0)), 1.0)
        return float(np.mean(np.log1p(normalized / scale)))

    def _pf_utility_score(self) -> float:
        # _pf_utility is bounded by log(2) because values are normalized by max.
        return clip01(self._pf_utility() / np.log(2.0))

    def _low_throughput_score(self) -> float:
        normalized = self.ewma_throughput_bits / np.maximum(self.demand_factor, 1e-6)
        eligible_values = normalized[self.eligible]
        if eligible_values.size == 0:
            return 0.0
        cell_mean = float(eligible_values.mean())
        if cell_mean <= 1e-9:
            return 0.0
        percentile = float(
            np.percentile(eligible_values, self.config.low_throughput_percentile)
        )
        return clip01(percentile / cell_mean)

    @staticmethod
    def _bits_per_prb(efficiency: np.ndarray | float) -> np.ndarray | float:
        # 12 subcarriers x 14 OFDM symbols with a simple 14% overhead abstraction.
        return 12.0 * 14.0 * efficiency * 0.86

    def _slot_oracle_expected_goodput_bits(self) -> float:
        """Expected-goodput upper bound under the active PHY abstraction.

        Legacy mode preserves the pre-v0.13 oracle. Link-adaptation mode gives
        the oracle perfect current CQI, selects MCS from true CQI, and applies
        the same mismatch-BLER model. The scheduler never receives this oracle.
        """
        eligible_cqi = self.cqi[self.eligible]
        if eligible_cqi.size == 0:
            return 0.0

        if self.config.link_adaptation_mode == "legacy_fixed_bler":
            eligible_efficiency = _CQI_EFFICIENCY[eligible_cqi - 1]
            k = min(
                self.config.max_selected_ues,
                eligible_efficiency.size,
                self.config.num_prbs,
            )
            strongest = np.sort(eligible_efficiency)[-k:][::-1]
            grants = np.ones(k, dtype=np.float64)
            grants[0] += self.config.num_prbs - k
            attempted = float(np.sum(grants * self._bits_per_prb(strongest)))
            success_probability = (
                1.0 - self.config.target_bler if self.config.harq_enabled else 1.0
            )
            return attempted * success_probability

        ideal_mcs = select_mcs_from_reported_cqi(
            eligible_cqi, cqi_backoff=self.config.link_adaptation_cqi_backoff
        )
        efficiency = mcs_efficiency(ideal_mcs)
        if self.config.harq_enabled:
            bler = bler_probability_from_cqi_mismatch(
                true_cqi=eligible_cqi,
                mcs_index=ideal_mcs,
                target_bler=self.config.target_bler,
                mismatch_slope=self.config.bler_mismatch_slope,
            )
        else:
            bler = np.zeros_like(efficiency, dtype=np.float64)
        expected_per_prb = self._bits_per_prb(efficiency) * (1.0 - bler)
        k = min(
            self.config.max_selected_ues,
            expected_per_prb.size,
            self.config.num_prbs,
        )
        strongest = np.sort(expected_per_prb)[-k:][::-1]
        grants = np.ones(k, dtype=np.float64)
        grants[0] += self.config.num_prbs - k
        return float(np.sum(grants * strongest))

    def _execute_grant(self, grant: ProjectedGrant) -> dict[str, Any]:
        n = self.config.num_ues
        selected = grant.selected_ues
        self.last_grant.fill(0)
        self.last_grant[selected] = grant.prbs
        self.last_success.fill(False)

        self.time_since_schedule += 1
        self.time_since_schedule[selected] = 0
        self.time_since_service += 1
        pre_service_wait = self.time_since_service.astype(np.float64).copy()

        pre_throughput_deficit = self._throughput_deficit().astype(np.float64)
        delivered_bits = np.zeros(n, dtype=np.float64)
        attempted_bits = np.zeros(n, dtype=np.float64)
        failed_transmissions = 0
        dropped_harq = 0
        selected_mcs = np.zeros(selected.size, dtype=np.int16)
        selected_mod_order = np.zeros(selected.size, dtype=np.int16)
        predicted_bler = np.zeros(selected.size, dtype=np.float64)
        pre_harq_pending = self.harq_pending.copy()
        harq_retransmission_attempts = int(np.sum(pre_harq_pending[selected])) if selected.size else 0

        if selected.size:
            if self.config.link_adaptation_mode == "legacy_fixed_bler":
                efficiency = _CQI_EFFICIENCY[self.cqi[selected] - 1]
                predicted_bler.fill(self.config.target_bler if self.config.harq_enabled else 0.0)
                selected_mcs.fill(-1)
                selected_mod_order.fill(0)
            else:
                selected_mcs = select_mcs_from_reported_cqi(
                    self.reported_cqi[selected],
                    cqi_backoff=self.config.link_adaptation_cqi_backoff,
                ).astype(np.int16)
                selected_mod_order = mcs_modulation_order(selected_mcs).astype(np.int16)
                efficiency = mcs_efficiency(selected_mcs).astype(np.float64)
                if self.config.harq_enabled:
                    predicted_bler = bler_probability_from_cqi_mismatch(
                        true_cqi=self.cqi[selected],
                        mcs_index=selected_mcs,
                        target_bler=self.config.target_bler,
                        mismatch_slope=self.config.bler_mismatch_slope,
                    ).astype(np.float64)

            attempted = grant.prbs.astype(np.float64) * self._bits_per_prb(efficiency)
            attempted_bits[selected] = attempted

            if self.config.harq_enabled:
                success = self.rng.random(selected.size) >= predicted_bler
            else:
                success = np.ones(selected.size, dtype=bool)

            successful_ues = selected[success]
            delivered_bits[successful_ues] = attempted[success]

            failed_ues = selected[~success]
            failed_transmissions = int(failed_ues.size)
            for ue in failed_ues:
                self.harq_retx_count[ue] += 1
                if self.harq_retx_count[ue] > self.config.max_harq_retransmissions:
                    self.harq_pending[ue] = False
                    self.harq_retx_count[ue] = 0
                    dropped_harq += 1
                else:
                    self.harq_pending[ue] = True

            self.harq_pending[successful_ues] = False
            self.harq_retx_count[successful_ues] = 0
            # A UE is only considered served when data is actually delivered.
            self.time_since_service[successful_ues] = 0
            self.last_success[successful_ues] = True

        # Full-buffer refill: every UE remains backlogged at its heterogeneous target.
        served_bytes = delivered_bits / 8.0
        self.queue_bytes = np.maximum(0.0, self.queue_bytes - served_bytes)
        self.queue_bytes = self.queue_target_bytes.copy()

        alpha = self.config.ewma_alpha
        self.ewma_throughput_bits = (
            (1.0 - alpha) * self.ewma_throughput_bits + alpha * delivered_bits
        )
        self.cumulative_delivered_bits += delivered_bits

        scheduled_indicator = np.zeros(n, dtype=np.float64)
        scheduled_indicator[selected] = 1.0
        self.ewma_schedule_rate = (
            (1.0 - alpha) * self.ewma_schedule_rate + alpha * scheduled_indicator
        )
        self.cumulative_schedule_count[selected] += 1.0

        cell_goodput_bits = float(delivered_bits.sum())
        oracle_goodput = self._slot_oracle_expected_goodput_bits()
        throughput_score = clip01(cell_goodput_bits / max(oracle_goodput, 1.0))
        cumulative_fairness = jain_fairness(self.cumulative_delivered_bits)
        short_term_fairness = jain_fairness(self.ewma_throughput_bits)
        # The cumulative KPI remains visible, while the reward receives a more
        # responsive fairness signal that can change within an episode.
        fairness_score = clip01(0.60 * cumulative_fairness + 0.40 * short_term_fairness)
        cumulative_schedule_fairness = jain_fairness(self.cumulative_schedule_count)
        short_term_schedule_fairness = jain_fairness(self.ewma_schedule_rate)
        # Same cumulative/short-term blend as throughput fairness, but over UE
        # scheduling frequency rather than delivered bits.
        schedule_fairness_score = clip01(
            0.60 * cumulative_schedule_fairness + 0.40 * short_term_schedule_fairness
        )
        fairness_delta = cumulative_fairness - self._previous_cumulative_fairness
        fairness_progress = float(
            np.tanh(self.config.fairness_delta_scale * fairness_delta)
        )
        pf_utility = self._pf_utility()
        pf_utility_score = self._pf_utility_score()
        low_throughput_score = self._low_throughput_score()
        pf_utility_delta = pf_utility - self._previous_pf_utility
        pf_utility_progress = float(
            np.tanh(self.config.pf_utility_delta_scale * pf_utility_delta)
        )
        successful_mask = delivered_bits > 0.0
        deficit_service_score = (
            clip01(float(np.mean(pre_throughput_deficit[successful_mask])))
            if np.any(successful_mask)
            else 0.0
        )
        pre_wait_pressure = np.clip(
            pre_service_wait / max(self.config.reference_deadline_target_slots, 1e-9),
            0.0,
            2.0,
        )
        urgency_service_score = (
            clip01(
                float(
                    np.mean(
                        0.5 * pre_throughput_deficit[successful_mask]
                        + 0.5 * np.minimum(pre_wait_pressure[successful_mask], 1.0)
                    )
                )
            )
            if np.any(successful_mask)
            else 0.0
        )
        self._previous_cumulative_fairness = cumulative_fairness
        self._previous_pf_utility = pf_utility

        starvation_mask = self.time_since_service >= self.config.starvation_threshold_slots
        scheduling_starvation_mask = (
            self.time_since_schedule >= self.config.starvation_threshold_slots
        )
        starvation_rate = float(starvation_mask.mean())
        scheduling_starvation_rate = float(scheduling_starvation_mask.mean())
        mean_wait_score = float(
            np.mean(
                np.clip(
                    self.time_since_service
                    / max(self.config.starvation_threshold_slots, 1),
                    0.0,
                    1.0,
                )
            )
        )
        near_deadline_rate = float(
            np.mean(
                self.time_since_service
                >= self.config.deadline_risk_start_ratio
                * self.config.deadline_target_slots
            )
        )
        delay_penalty = clip01(
            0.40 * starvation_rate
            + 0.35 * mean_wait_score
            + 0.25 * near_deadline_rate
        )
        service_score = clip01(1.0 - delay_penalty)

        # Dense risk for the worst-served 1% of UEs. It starts increasing before
        # the configured P99 deadline is crossed, giving PPO a less sparse signal.
        tail_count = max(1, int(np.ceil(0.01 * n)))
        tail_waits = np.partition(self.time_since_service, n - tail_count)[-tail_count:]
        deadline_risk = deadline_risk_score(
            tail_waits,
            target_slots=self.config.deadline_target_slots,
            start_ratio=self.config.deadline_risk_start_ratio,
        )
        reference_deadline_risk = deadline_risk_score(
            tail_waits,
            target_slots=self.config.reference_deadline_target_slots,
            start_ratio=self.config.deadline_risk_start_ratio,
        )
        tail_mean_wait_slots = float(np.mean(tail_waits))
        max_wait_slots = float(np.max(self.time_since_service, initial=0))
        scheduling_max_wait_slots = float(np.max(self.time_since_schedule, initial=0))
        max_wait_risk = deadline_risk_score(
            np.asarray([max_wait_slots], dtype=np.float64),
            target_slots=self.config.max_wait_target_slots,
            start_ratio=self.config.deadline_risk_start_ratio,
        )
        population_wait_risk = deadline_risk_score(
            self.time_since_service.astype(np.float64),
            target_slots=self.config.reference_deadline_target_slots,
            start_ratio=self.config.deadline_risk_start_ratio,
        )

        total_re = float(self.config.num_prbs * 12 * 14)
        attempted_spectral_efficiency = float(attempted_bits.sum() / max(total_re, 1.0))
        spectral_efficiency = float(cell_goodput_bits / max(total_re, 1.0))
        observed_bler = float(failed_transmissions / max(int(selected.size), 1))
        mean_predicted_bler = float(np.mean(predicted_bler)) if predicted_bler.size else 0.0
        mean_mcs_index = (
            float(np.mean(selected_mcs[selected_mcs >= 0]))
            if np.any(selected_mcs >= 0)
            else -1.0
        )
        mean_modulation_order = (
            float(np.mean(selected_mod_order[selected_mod_order > 0]))
            if np.any(selected_mod_order > 0)
            else 0.0
        )

        return {
            "selected_ues": selected.copy(),
            "prbs_per_selected_ue": grant.prbs.copy(),
            "prbs_per_ue": grant.prbs_per_ue.copy(),
            "cell_goodput_bits": cell_goodput_bits,
            "cell_attempted_bits": float(attempted_bits.sum()),
            "oracle_expected_goodput_bits": float(oracle_goodput),
            "link_adaptation_mode": self.config.link_adaptation_mode,
            "mean_mcs_index": mean_mcs_index,
            "mean_modulation_order": mean_modulation_order,
            "mean_predicted_bler": mean_predicted_bler,
            "observed_bler": observed_bler,
            "spectral_efficiency_bps_hz": spectral_efficiency,
            "attempted_spectral_efficiency_bps_hz": attempted_spectral_efficiency,
            "harq_retransmission_attempts": float(harq_retransmission_attempts),
            "harq_retransmission_fraction": float(
                harq_retransmission_attempts / max(int(selected.size), 1)
            ),
            "throughput_score": throughput_score,
            "throughput_normalized": throughput_score,  # backward-compatible alias
            "jain_fairness": cumulative_fairness,
            "short_term_jain_fairness": short_term_fairness,
            "fairness_score": fairness_score,
            "cumulative_schedule_fairness": cumulative_schedule_fairness,
            "short_term_schedule_fairness": short_term_schedule_fairness,
            "schedule_fairness_score": schedule_fairness_score,
            "throughput_deficit_mean": float(self._throughput_deficit().mean()),
            "deficit_service_score": deficit_service_score,
            "urgency_service_score": urgency_service_score,
            "low_throughput_score": low_throughput_score,
            "pf_utility_score": pf_utility_score,
            "fairness_delta": float(fairness_delta),
            "fairness_progress": fairness_progress,
            "pf_utility": pf_utility,
            "pf_utility_delta": float(pf_utility_delta),
            "pf_utility_progress": pf_utility_progress,
            # Primary starvation means no successful delivery for the threshold.
            "starvation_rate": starvation_rate,
            "delivery_starvation_rate": starvation_rate,
            "scheduling_starvation_rate": scheduling_starvation_rate,
            "mean_wait_slots": float(self.time_since_service.mean()),
            "max_wait_slots": max_wait_slots,
            "scheduling_mean_wait_slots": float(self.time_since_schedule.mean()),
            "scheduling_max_wait_slots": scheduling_max_wait_slots,
            "p95_wait_slots": safe_percentile(self.time_since_service, 95),
            "p99_wait_slots": safe_percentile(self.time_since_service, 99),
            "near_deadline_rate": near_deadline_rate,
            "max_wait_risk": max_wait_risk,
            "population_wait_risk": population_wait_risk,
            "tail_mean_wait_slots": tail_mean_wait_slots,
            "deadline_risk": deadline_risk,
            "reference_deadline_risk": reference_deadline_risk,
            "delay_penalty": delay_penalty,
            "service_score": service_score,
            "failed_transmissions": failed_transmissions,
            "harq_drops": dropped_harq,
            "forced_harq_count": grant.forced_harq_count,
            "forced_long_wait_count": grant.forced_long_wait_count,
            "forced_oldest_wait_count": grant.forced_oldest_wait_count,
            "selection_mode": grant.selection_mode,
            "safety_selected_count": grant.safety_selected_count,
            "scheduler_selected_count": grant.scheduler_selected_count,
            "scheduler_selection_fraction": (
                grant.scheduler_selected_count / max(int(grant.selected_ues.size), 1)
            ),
            "ppo_selected_count": grant.ppo_selected_count,
            "rule_selected_count": grant.rule_selected_count,
            # Backward-compatible aliases.
            "learned_selected_count": grant.learned_selected_count,
            "learned_selection_fraction": (
                grant.learned_selected_count / max(int(grant.selected_ues.size), 1)
            ),
            "harq_overflow_count": grant.harq_overflow_count,
            "prb_utilization": float(grant.prbs.sum() / self.config.num_prbs)
            if grant.prbs.size
            else 0.0,
        }

    def _reward(self, metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
        cfg = self.config
        positive_scale = cfg.reward_positive_scale
        throughput_component = (
            positive_scale * cfg.reward_throughput_weight * metrics["throughput_score"]
        )
        fairness_component = (
            positive_scale * cfg.reward_fairness_weight * metrics["fairness_score"]
        )
        schedule_fairness_component = (
            positive_scale
            * cfg.reward_schedule_fairness_weight
            * metrics["schedule_fairness_score"]
        )
        service_component = (
            positive_scale * cfg.reward_service_weight * metrics["service_score"]
        )
        deficit_service_component = (
            positive_scale
            * cfg.reward_deficit_service_weight
            * metrics["deficit_service_score"]
        )
        pf_utility_component = (
            positive_scale * cfg.reward_pf_utility_weight * metrics["pf_utility_score"]
        )
        low_throughput_component = (
            positive_scale
            * cfg.reward_low_throughput_weight
            * metrics["low_throughput_score"]
        )
        urgency_service_component = (
            positive_scale
            * cfg.reward_urgency_service_weight
            * metrics["urgency_service_score"]
        )
        fairness_progress_component = (
            cfg.reward_fairness_delta_weight * metrics["fairness_progress"]
        )
        pf_utility_progress_component = (
            cfg.reward_pf_utility_delta_weight * metrics["pf_utility_progress"]
        )

        tolerance = cfg.starvation_tolerance
        starvation_violation = clip01(
            (metrics["starvation_rate"] - tolerance) / max(1.0 - tolerance, 1e-9)
        )
        starvation_penalty = cfg.reward_starvation_penalty_weight * starvation_violation
        core_total = (
            throughput_component
            + fairness_component
            + schedule_fairness_component
            + service_component
            + deficit_service_component
            + pf_utility_component
            + low_throughput_component
            + urgency_service_component
            - starvation_penalty
        )
        shaped_core_total = (
            core_total + fairness_progress_component + pf_utility_progress_component
        )
        deadline_risk_penalty = (
            cfg.reward_deadline_risk_penalty_weight * float(metrics["deadline_risk"])
        )
        reference_deadline_risk_penalty = (
            cfg.reward_deadline_risk_penalty_weight
            * float(metrics["reference_deadline_risk"])
        )
        max_wait_risk_penalty = (
            cfg.reward_max_wait_risk_penalty_weight * float(metrics["max_wait_risk"])
        )
        population_wait_penalty = (
            cfg.reward_population_wait_penalty_weight
            * float(metrics["population_wait_risk"])
        )
        total = (
            shaped_core_total
            - deadline_risk_penalty
            - max_wait_risk_penalty
            - population_wait_penalty
        )
        final_target_total = (
            shaped_core_total
            - reference_deadline_risk_penalty
            - max_wait_risk_penalty
            - population_wait_penalty
        )

        return float(total), {
            "reward_total": float(total),
            "reward_core_total": float(core_total),
            "reward_shaped_core_total": float(shaped_core_total),
            "reward_final_target_total": float(final_target_total),
            "reward_throughput_component": float(throughput_component),
            "reward_fairness_component": float(fairness_component),
            "reward_schedule_fairness_component": float(schedule_fairness_component),
            "reward_service_component": float(service_component),
            "reward_deficit_service_component": float(deficit_service_component),
            "reward_pf_utility_component": float(pf_utility_component),
            "reward_low_throughput_component": float(low_throughput_component),
            "reward_urgency_service_component": float(urgency_service_component),
            "reward_fairness_progress_component": float(fairness_progress_component),
            "reward_pf_utility_progress_component": float(pf_utility_progress_component),
            "starvation_violation": float(starvation_violation),
            "reward_starvation_penalty": float(starvation_penalty),
            "reward_deadline_risk_penalty": float(deadline_risk_penalty),
            "reward_reference_deadline_risk_penalty": float(
                reference_deadline_risk_penalty
            ),
            "reward_max_wait_risk_penalty": float(max_wait_risk_penalty),
            "reward_population_wait_penalty": float(population_wait_penalty),
        }

    def render(self) -> str:
        return (
            f"slot={self.slot} active_ues={self.config.num_ues} "
            f"scheduled={int(np.count_nonzero(self.last_grant))} "
            f"prbs={int(self.last_grant.sum())}/{self.config.num_prbs} "
            f"fairness={jain_fairness(self.cumulative_delivered_bits):.4f} "
            f"starved_delivery={(self.time_since_service >= self.config.starvation_threshold_slots).sum()} "
            f"max_delivery_wait={int(self.time_since_service.max(initial=0))}"
        )
