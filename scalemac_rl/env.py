from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ScaleMacConfig
from .metrics import clip01, jain_fairness, safe_percentile
from .projector import ProjectedGrant, project_action


# Standard 15-level CQI spectral-efficiency abstraction.
_CQI_EFFICIENCY = np.asarray(
    [
        0.1523,
        0.2344,
        0.3770,
        0.6016,
        0.8770,
        1.1758,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,
    ],
    dtype=np.float32,
)

# Per-UE observation columns.
CQI = 0
QUEUE = 1
DEMAND = 2
EWMA_THROUGHPUT = 3
TIME_SINCE_SERVICE = 4
HARQ_PENDING = 5
HARQ_RETX_COUNT = 6
ELIGIBLE = 7
OBSERVATION_FEATURES = 8


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

        n = self.config.num_ues
        self.slot = 0
        self.cqi = np.zeros(n, dtype=np.int16)
        self.demand_factor = np.ones(n, dtype=np.float32)
        self.speed_mps = np.zeros(n, dtype=np.float32)  # metadata only in static-CQI MVP
        self.queue_bytes = np.zeros(n, dtype=np.float64)
        self.queue_target_bytes = np.zeros(n, dtype=np.float64)
        self.ewma_throughput_bits = np.zeros(n, dtype=np.float64)
        self.cumulative_delivered_bits = np.zeros(n, dtype=np.float64)
        self.time_since_service = np.zeros(n, dtype=np.int32)
        self.harq_pending = np.zeros(n, dtype=bool)
        self.harq_retx_count = np.zeros(n, dtype=np.int16)
        self.eligible = np.ones(n, dtype=bool)
        self.last_grant = np.zeros(n, dtype=np.int32)

        self.reset(seed=self.config.seed)

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (self.config.num_ues, OBSERVATION_FEATURES)

    @property
    def action_shape(self) -> tuple[int, int]:
        return (self.config.num_ues, 2)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.slot = 0
        self.cqi = self._sample_cqi_profiles()
        self.demand_factor = self._sample_demand_profiles()
        self.speed_mps = self.rng.choice(
            np.asarray([0.0, 1.5, 5.0, 15.0, 25.0], dtype=np.float32),
            size=self.config.num_ues,
            replace=True,
        )

        self.queue_target_bytes = (
            self.config.full_buffer_base_bytes * self.demand_factor
        ).astype(np.float64)
        self.queue_bytes = self.queue_target_bytes.copy()
        self.ewma_throughput_bits.fill(0.0)
        self.cumulative_delivered_bits.fill(0.0)
        self.time_since_service.fill(0)
        self.harq_pending.fill(False)
        self.harq_retx_count.fill(0)
        self.eligible.fill(True)
        self.last_grant.fill(0)

        observation = self._observation()
        info = {
            "slot": self.slot,
            "num_active_ues": self.config.num_ues,
            "num_prbs": self.config.num_prbs,
            "max_selected_ues": self.config.max_selected_ues,
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
        )
        metrics = self._execute_grant(grant)
        reward, reward_breakdown = self._reward(metrics)
        metrics.update(reward_breakdown)

        self.slot += 1
        terminated = self.slot >= self.config.episode_slots
        truncated = False
        observation = self._observation()
        info = {"slot": self.slot, **metrics}
        return observation, reward, terminated, truncated, info

    def _sample_cqi_profiles(self) -> np.ndarray:
        n = self.config.num_ues
        low = int(round(n * self.config.low_cqi_fraction))
        medium = int(round(n * self.config.medium_cqi_fraction))
        high = n - low - medium
        values = np.concatenate(
            [
                self.rng.integers(1, 6, size=low),
                self.rng.integers(6, 11, size=medium),
                self.rng.integers(11, 16, size=high),
            ]
        ).astype(np.int16)
        self.rng.shuffle(values)
        return values

    def _sample_demand_profiles(self) -> np.ndarray:
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
        self.rng.shuffle(values)
        return values

    def _observation(self) -> np.ndarray:
        max_queue = max(float(self.queue_target_bytes.max(initial=1.0)), 1.0)
        max_ewma = max(float(self.ewma_throughput_bits.max(initial=1.0)), 1.0)
        wait_denominator = max(self.config.starvation_threshold_slots, 1)
        retx_denominator = max(self.config.max_harq_retransmissions, 1)

        observation = np.empty(self.observation_shape, dtype=np.float32)
        observation[:, CQI] = self.cqi / 15.0
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
        return observation

    @staticmethod
    def _bits_per_prb(efficiency: np.ndarray | float) -> np.ndarray | float:
        # 12 subcarriers x 14 OFDM symbols with a simple 14% overhead abstraction.
        return 12.0 * 14.0 * efficiency * 0.86

    def _slot_oracle_expected_goodput_bits(self) -> float:
        """Upper-bound expected goodput for the current static CQI population.

        The oracle obeys the current Top-K and one-PRB-per-selected-UE contract,
        then gives all remaining PRBs to the strongest selected UE. It is used
        only to normalize throughput, not as a training baseline.
        """
        eligible_efficiency = _CQI_EFFICIENCY[self.cqi[self.eligible] - 1]
        if eligible_efficiency.size == 0:
            return 0.0
        k = min(
            self.config.max_selected_ues,
            eligible_efficiency.size,
            self.config.num_prbs,
        )
        strongest = np.sort(eligible_efficiency)[-k:][::-1]
        grants = np.ones(k, dtype=np.float64)
        grants[0] += self.config.num_prbs - k
        attempted = float(np.sum(grants * self._bits_per_prb(strongest)))
        success_probability = 1.0 - self.config.target_bler if self.config.harq_enabled else 1.0
        return attempted * success_probability

    def _execute_grant(self, grant: ProjectedGrant) -> dict[str, Any]:
        n = self.config.num_ues
        selected = grant.selected_ues
        self.last_grant.fill(0)
        self.last_grant[selected] = grant.prbs

        self.time_since_service += 1
        self.time_since_service[selected] = 0

        delivered_bits = np.zeros(n, dtype=np.float64)
        attempted_bits = np.zeros(n, dtype=np.float64)
        failed_transmissions = 0
        dropped_harq = 0

        if selected.size:
            efficiency = _CQI_EFFICIENCY[self.cqi[selected] - 1]
            attempted = grant.prbs.astype(np.float64) * self._bits_per_prb(efficiency)
            attempted_bits[selected] = attempted

            if self.config.harq_enabled:
                success = self.rng.random(selected.size) >= self.config.target_bler
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

        # Full-buffer refill: every UE remains backlogged at its heterogeneous target.
        served_bytes = delivered_bits / 8.0
        self.queue_bytes = np.maximum(0.0, self.queue_bytes - served_bytes)
        self.queue_bytes = self.queue_target_bytes.copy()

        alpha = self.config.ewma_alpha
        self.ewma_throughput_bits = (
            (1.0 - alpha) * self.ewma_throughput_bits + alpha * delivered_bits
        )
        self.cumulative_delivered_bits += delivered_bits

        cell_goodput_bits = float(delivered_bits.sum())
        oracle_goodput = self._slot_oracle_expected_goodput_bits()
        throughput_score = clip01(cell_goodput_bits / max(oracle_goodput, 1.0))
        fairness_score = jain_fairness(self.cumulative_delivered_bits)

        starvation_mask = self.time_since_service >= self.config.starvation_threshold_slots
        starvation_rate = float(starvation_mask.mean())
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
        delay_penalty = clip01(0.5 * starvation_rate + 0.5 * mean_wait_score)
        service_score = clip01(1.0 - delay_penalty)

        return {
            "selected_ues": selected.copy(),
            "prbs_per_selected_ue": grant.prbs.copy(),
            "prbs_per_ue": grant.prbs_per_ue.copy(),
            "cell_goodput_bits": cell_goodput_bits,
            "cell_attempted_bits": float(attempted_bits.sum()),
            "oracle_expected_goodput_bits": float(oracle_goodput),
            "throughput_score": throughput_score,
            "throughput_normalized": throughput_score,  # backward-compatible alias
            "jain_fairness": fairness_score,
            "fairness_score": fairness_score,
            "starvation_rate": starvation_rate,
            "mean_wait_slots": float(self.time_since_service.mean()),
            "p95_wait_slots": safe_percentile(self.time_since_service, 95),
            "p99_wait_slots": safe_percentile(self.time_since_service, 99),
            "delay_penalty": delay_penalty,
            "service_score": service_score,
            "failed_transmissions": failed_transmissions,
            "harq_drops": dropped_harq,
            "forced_harq_count": grant.forced_harq_count,
            "forced_long_wait_count": grant.forced_long_wait_count,
            "safety_selected_count": grant.safety_selected_count,
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
        throughput_component = cfg.reward_throughput_weight * metrics["throughput_score"]
        fairness_component = cfg.reward_fairness_weight * metrics["fairness_score"]
        service_component = cfg.reward_service_weight * metrics["service_score"]

        tolerance = cfg.starvation_tolerance
        starvation_violation = clip01(
            (metrics["starvation_rate"] - tolerance) / max(1.0 - tolerance, 1e-9)
        )
        starvation_penalty = cfg.reward_starvation_penalty_weight * starvation_violation
        total = throughput_component + fairness_component + service_component - starvation_penalty

        return float(total), {
            "reward_total": float(total),
            "reward_throughput_component": float(throughput_component),
            "reward_fairness_component": float(fairness_component),
            "reward_service_component": float(service_component),
            "starvation_violation": float(starvation_violation),
            "reward_starvation_penalty": float(starvation_penalty),
        }

    def render(self) -> str:
        return (
            f"slot={self.slot} active_ues={self.config.num_ues} "
            f"scheduled={int(np.count_nonzero(self.last_grant))} "
            f"prbs={int(self.last_grant.sum())}/{self.config.num_prbs} "
            f"fairness={jain_fairness(self.cumulative_delivered_bits):.4f} "
            f"starved={(self.time_since_service >= self.config.starvation_threshold_slots).sum()}"
        )
