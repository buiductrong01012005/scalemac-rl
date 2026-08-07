from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class ProjectedGrant:
    selected_ues: np.ndarray
    prbs: np.ndarray
    prbs_per_ue: np.ndarray
    forced_harq_count: int
    forced_long_wait_count: int
    forced_oldest_wait_count: int
    safety_selected_count: int
    learned_selected_count: int
    harq_overflow_count: int


def _largest_remainder_allocation(weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate an integer total proportionally while preserving the exact sum."""
    if total <= 0 or weights.size == 0:
        return np.zeros(weights.size, dtype=np.int32)

    clipped = np.clip(weights.astype(np.float64), 1e-9, None)
    raw = clipped / clipped.sum() * total
    floor = np.floor(raw).astype(np.int32)
    remainder = total - int(floor.sum())
    if remainder > 0:
        order = np.argsort(-(raw - floor), kind="stable")
        floor[order[:remainder]] += 1
    return floor


def project_action(
    action: np.ndarray,
    *,
    eligible: np.ndarray,
    harq_pending: np.ndarray,
    harq_retx_count: np.ndarray,
    time_since_service: np.ndarray,
    num_prbs: int,
    max_selected_ues: int,
    safety_reserve_ues: int = 0,
    safety_wait_threshold_ratio: float = 0.80,
    starvation_threshold_slots: int = 100,
) -> ProjectedGrant:
    """Convert policy scores into feasible Top-K grants with a safety reserve.

    ``action[:, 0]`` is the learned priority score and ``action[:, 1]`` is the
    PRB-demand score. HARQ retransmissions remain mandatory. Up to
    ``safety_reserve_ues`` total grants are then reserved for the longest-waiting
    eligible UEs. PPO selects the remaining grants from the candidate pool.
    """
    raw = np.asarray(action, dtype=np.float32)
    num_ues = eligible.size
    if raw.shape != (num_ues, 2):
        raise ValueError(f"action must have shape {(num_ues, 2)}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("action contains NaN or infinity")
    if not 0 <= safety_reserve_ues <= max_selected_ues:
        raise ValueError("safety_reserve_ues must be in [0, max_selected_ues]")
    if starvation_threshold_slots <= 0:
        raise ValueError("starvation_threshold_slots must be positive")
    if safety_wait_threshold_ratio < 0.0:
        raise ValueError("safety_wait_threshold_ratio must be non-negative")

    priority = raw[:, 0]
    demand = np.clip(raw[:, 1], 0.0, 1.0)
    eligible = eligible.astype(bool, copy=False)
    harq_bool = harq_pending.astype(bool, copy=False)

    mandatory = np.flatnonzero(eligible & harq_bool)
    if mandatory.size:
        mandatory_order = np.lexsort(
            (-time_since_service[mandatory], -harq_retx_count[mandatory])
        )
        mandatory = mandatory[mandatory_order]

    harq_overflow = max(0, int(mandatory.size - max_selected_ues))
    selected = list(mandatory[:max_selected_ues])
    forced_harq_count = len(selected)

    # HARQ may consume more than the nominal reserve. Otherwise the reserve is
    # filled completely with the oldest eligible UEs, even before they cross the
    # urgent-wait threshold. This produces a dense, proactive tail-delay guard.
    safety_target = min(max_selected_ues, max(safety_reserve_ues, forced_harq_count))
    safety_slots_left = max(0, safety_target - len(selected))
    forced_long_wait_count = 0
    forced_oldest_wait_count = 0
    if safety_slots_left > 0:
        threshold_slots = safety_wait_threshold_ratio * starvation_threshold_slots
        chosen = np.zeros(num_ues, dtype=bool)
        if selected:
            chosen[np.asarray(selected, dtype=np.int64)] = True
        oldest_candidates = np.flatnonzero(eligible & ~chosen & ~harq_bool)
        if oldest_candidates.size:
            order = np.argsort(-time_since_service[oldest_candidates], kind="stable")
            safety_ues = oldest_candidates[order[:safety_slots_left]]
            selected.extend(safety_ues.tolist())
            forced_oldest_wait_count = int(safety_ues.size)
            forced_long_wait_count = int(
                np.sum(time_since_service[safety_ues] >= threshold_slots)
            )

    safety_selected_count = len(selected)
    remaining = max_selected_ues - len(selected)
    learned_selected_count = 0
    if remaining > 0:
        chosen = np.zeros(num_ues, dtype=bool)
        if selected:
            chosen[np.asarray(selected, dtype=np.int64)] = True
        candidates = np.flatnonzero(eligible & ~chosen)
        if candidates.size:
            order = np.argsort(-priority[candidates], kind="stable")
            learned = candidates[order[:remaining]]
            selected.extend(learned.tolist())
            learned_selected_count = int(learned.size)

    selected_array = np.asarray(selected, dtype=np.int32)
    prbs_per_ue = np.zeros(num_ues, dtype=np.int32)
    if selected_array.size == 0:
        return ProjectedGrant(
            selected_ues=selected_array,
            prbs=np.empty(0, dtype=np.int32),
            prbs_per_ue=prbs_per_ue,
            forced_harq_count=0,
            forced_long_wait_count=0,
            forced_oldest_wait_count=0,
            safety_selected_count=0,
            learned_selected_count=0,
            harq_overflow_count=harq_overflow,
        )

    base = np.ones(selected_array.size, dtype=np.int32)
    remaining_prbs = num_prbs - int(base.sum())
    extra = _largest_remainder_allocation(demand[selected_array] + 1e-3, remaining_prbs)
    grants = base + extra
    prbs_per_ue[selected_array] = grants

    if int(grants.sum()) != num_prbs:
        raise RuntimeError("projector failed to conserve PRBs")

    return ProjectedGrant(
        selected_ues=selected_array,
        prbs=grants,
        prbs_per_ue=prbs_per_ue,
        forced_harq_count=forced_harq_count,
        forced_long_wait_count=forced_long_wait_count,
        forced_oldest_wait_count=forced_oldest_wait_count,
        safety_selected_count=safety_selected_count,
        learned_selected_count=learned_selected_count,
        harq_overflow_count=harq_overflow,
    )
