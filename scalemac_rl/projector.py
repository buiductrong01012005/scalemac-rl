from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class ProjectedGrant:
    selected_ues: np.ndarray
    prbs: np.ndarray
    prbs_per_ue: np.ndarray
    forced_harq_count: int
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
) -> ProjectedGrant:
    """Convert raw policy scores into feasible Top-K UE grants.

    action[:, 0] is the priority score and action[:, 1] is the PRB-demand score.
    Pending HARQ UEs are selected before new transmissions whenever capacity allows.
    """
    raw = np.asarray(action, dtype=np.float32)
    num_ues = eligible.size
    if raw.shape != (num_ues, 2):
        raise ValueError(f"action must have shape {(num_ues, 2)}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("action contains NaN or infinity")

    priority = raw[:, 0]
    demand = np.clip(raw[:, 1], 0.0, 1.0)
    eligible = eligible.astype(bool, copy=False)
    mandatory = np.flatnonzero(eligible & harq_pending.astype(bool, copy=False))

    # Most retransmissions first, then longest waiting UE.
    if mandatory.size:
        mandatory_order = np.lexsort(
            (
                -time_since_service[mandatory],
                -harq_retx_count[mandatory],
            )
        )
        mandatory = mandatory[mandatory_order]

    harq_overflow = max(0, int(mandatory.size - max_selected_ues))
    selected = list(mandatory[:max_selected_ues])
    forced_harq_count = len(selected)

    remaining = max_selected_ues - len(selected)
    if remaining > 0:
        candidates = np.flatnonzero(eligible & ~harq_pending.astype(bool, copy=False))
        if candidates.size:
            order = np.argsort(-priority[candidates], kind="stable")
            selected.extend(candidates[order[:remaining]].tolist())

    selected_array = np.asarray(selected, dtype=np.int32)
    prbs_per_ue = np.zeros(num_ues, dtype=np.int32)
    if selected_array.size == 0:
        return ProjectedGrant(
            selected_ues=selected_array,
            prbs=np.empty(0, dtype=np.int32),
            prbs_per_ue=prbs_per_ue,
            forced_harq_count=0,
            harq_overflow_count=harq_overflow,
        )

    # Give every selected UE one PRB, then distribute the remainder by demand.
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
        harq_overflow_count=harq_overflow,
    )
