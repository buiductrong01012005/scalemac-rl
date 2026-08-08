from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import (
    CQI,
    DEMAND,
    ELIGIBLE,
    EWMA_THROUGHPUT,
    HARQ_PENDING,
    SERVICE_DEFICIT,
    THROUGHPUT_DEFICIT,
    TIME_SINCE_SERVICE,
)


@dataclass(frozen=True, slots=True)
class CandidateDiagnostics:
    eligible_count: int
    candidate_count: int
    candidate_coverage: float
    harq_pending_count: int
    harq_retention_rate: float
    long_wait_count: int
    long_wait_retention_rate: float
    long_wait_missed_count: int


def build_candidate_mask(
    observation: np.ndarray,
    *,
    max_candidates: int,
    min_candidates: int,
    long_wait_threshold: float = 0.8,
) -> np.ndarray:
    """Build a deterministic safety-aware candidate set.

    Selection order is deliberately conservative:
    1. pending HARQ UEs;
    2. eligible UEs close to the starvation threshold;
    3. remaining UEs ranked by channel, waiting time, inverse service history,
       and heterogeneous demand.

    ``TIME_SINCE_SERVICE`` is normalized by the environment starvation threshold,
    so ``long_wait_threshold=0.8`` retains UEs that have consumed at least 80% of
    their waiting-time budget whenever capacity permits.
    """
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError("observation must have shape [num_ues, features]")
    num_ues = obs.shape[0]
    if not 1 <= min_candidates <= num_ues:
        raise ValueError("min_candidates must be in [1, num_ues]")
    if max_candidates < min_candidates:
        raise ValueError("max_candidates must be >= min_candidates")
    if long_wait_threshold < 0.0:
        raise ValueError("long_wait_threshold must be non-negative")

    capacity = min(max_candidates, num_ues)
    eligible = obs[:, ELIGIBLE] > 0.5
    harq = eligible & (obs[:, HARQ_PENDING] > 0.5)
    long_wait = eligible & ~harq & (obs[:, TIME_SINCE_SERVICE] >= long_wait_threshold)
    selected: list[int] = []

    mandatory = np.flatnonzero(harq)
    if mandatory.size:
        order = np.argsort(-obs[mandatory, TIME_SINCE_SERVICE], kind="stable")
        selected.extend(mandatory[order[:capacity]].tolist())

    remaining = capacity - len(selected)
    if remaining > 0:
        urgent = np.flatnonzero(long_wait)
        if urgent.size:
            order = np.argsort(-obs[urgent, TIME_SINCE_SERVICE], kind="stable")
            selected.extend(urgent[order[:remaining]].tolist())

    remaining = capacity - len(selected)
    if remaining > 0:
        already_selected = np.zeros(num_ues, dtype=bool)
        if selected:
            already_selected[np.asarray(selected, dtype=np.int64)] = True
        candidates = np.flatnonzero(eligible & ~already_selected)
        if candidates.size:
            cqi = obs[candidates, CQI]
            wait = np.clip(obs[candidates, TIME_SINCE_SERVICE], 0.0, 2.0) / 2.0
            demand = obs[candidates, DEMAND]
            inverse_history = 1.0 - np.clip(obs[candidates, EWMA_THROUGHPUT], 0.0, 1.0)
            throughput_deficit = np.clip(obs[candidates, THROUGHPUT_DEFICIT], 0.0, 2.0) / 2.0
            service_deficit = np.clip(obs[candidates, SERVICE_DEFICIT], 0.0, 2.0) / 2.0
            score = (
                0.30 * cqi
                + 0.20 * wait
                + 0.15 * inverse_history
                + 0.10 * demand
                + 0.15 * throughput_deficit
                + 0.10 * service_deficit
            )
            order = np.argsort(-score, kind="stable")
            selected.extend(candidates[order[:remaining]].tolist())

    if len(selected) < min_candidates:
        chosen = np.zeros(num_ues, dtype=bool)
        if selected:
            chosen[np.asarray(selected, dtype=np.int64)] = True
        backfill = np.flatnonzero(eligible & ~chosen)
        selected.extend(backfill[: min_candidates - len(selected)].tolist())

    mask = np.zeros(num_ues, dtype=bool)
    if selected:
        mask[np.asarray(selected, dtype=np.int64)] = True
    return mask



def build_all_eligible_mask(observation: np.ndarray) -> np.ndarray:
    """Return all eligible UEs for full-PPO experiments."""
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError("observation must have shape [num_ues, features]")
    mask = obs[:, ELIGIBLE] > 0.5
    if not np.any(mask):
        raise ValueError("observation contains no eligible UE")
    return mask.astype(bool, copy=False)

def candidate_diagnostics(
    observation: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    long_wait_threshold: float = 0.8,
) -> CandidateDiagnostics:
    obs = np.asarray(observation, dtype=np.float32)
    mask = np.asarray(candidate_mask, dtype=bool)
    if obs.ndim != 2 or mask.shape != (obs.shape[0],):
        raise ValueError("observation and candidate_mask shapes are inconsistent")

    eligible = obs[:, ELIGIBLE] > 0.5
    harq = eligible & (obs[:, HARQ_PENDING] > 0.5)
    long_wait = eligible & (obs[:, TIME_SINCE_SERVICE] >= long_wait_threshold)
    eligible_count = int(eligible.sum())
    candidate_count = int((mask & eligible).sum())
    harq_count = int(harq.sum())
    long_wait_count = int(long_wait.sum())
    retained_harq = int((mask & harq).sum())
    retained_long_wait = int((mask & long_wait).sum())

    return CandidateDiagnostics(
        eligible_count=eligible_count,
        candidate_count=candidate_count,
        candidate_coverage=candidate_count / max(eligible_count, 1),
        harq_pending_count=harq_count,
        harq_retention_rate=retained_harq / harq_count if harq_count else 1.0,
        long_wait_count=long_wait_count,
        long_wait_retention_rate=(retained_long_wait / long_wait_count if long_wait_count else 1.0),
        long_wait_missed_count=max(0, long_wait_count - retained_long_wait),
    )


def candidate_indices(candidate_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("candidate_mask must be one-dimensional")
    indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError("candidate_mask must select at least one UE")
    return indices


def gather_candidates(
    observation: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError("observation must have shape [num_ues, features]")
    indices = candidate_indices(candidate_mask)
    return np.ascontiguousarray(obs[indices]), indices


def gather_candidate_batch(
    observations: np.ndarray,
    candidate_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observations, dtype=np.float32)
    masks = np.asarray(candidate_masks, dtype=bool)
    if obs.ndim != 3 or masks.shape != obs.shape[:2]:
        raise ValueError("observations and candidate_masks shapes are inconsistent")
    counts = masks.sum(axis=1)
    if np.any(counts == 0) or not np.all(counts == counts[0]):
        raise ValueError("every batch item must select the same positive candidate count")
    indices = np.stack([np.flatnonzero(mask) for mask in masks], axis=0).astype(np.int64)
    compact = np.take_along_axis(obs, indices[..., None], axis=1)
    return np.ascontiguousarray(compact), indices


def scatter_candidate_action(
    compact_action: np.ndarray,
    candidate_indices_array: np.ndarray,
    *,
    num_ues: int,
) -> np.ndarray:
    compact = np.asarray(compact_action, dtype=np.float32)
    indices = np.asarray(candidate_indices_array, dtype=np.int64)
    if compact.ndim != 2 or compact.shape[1] != 2 or indices.shape != (compact.shape[0],):
        raise ValueError("compact_action and candidate indices shapes are inconsistent")
    full = np.empty((num_ues, 2), dtype=np.float32)
    full[:, 0] = -1.0
    full[:, 1] = 0.0
    full[indices] = compact
    return full


def scatter_candidate_action_batch(
    compact_actions: np.ndarray,
    candidate_indices_batch: np.ndarray,
    *,
    num_ues: int,
) -> np.ndarray:
    actions = np.asarray(compact_actions, dtype=np.float32)
    indices = np.asarray(candidate_indices_batch, dtype=np.int64)
    if actions.ndim != 3 or actions.shape[-1] != 2 or indices.shape != actions.shape[:2]:
        raise ValueError("compact_actions and candidate indices shapes are inconsistent")
    full = np.empty((actions.shape[0], num_ues, 2), dtype=np.float32)
    full[..., 0] = -1.0
    full[..., 1] = 0.0
    rows = np.arange(actions.shape[0])[:, None]
    full[rows, indices] = actions
    return full


def apply_candidate_mask(action: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    """Backward-compatible full-action masking helper."""
    masked = np.asarray(action, dtype=np.float32).copy()
    mask = np.asarray(candidate_mask, dtype=bool)
    if masked.shape != (mask.size, 2):
        raise ValueError("action and candidate_mask shapes are inconsistent")
    masked[~mask, 0] = -1.0
    masked[~mask, 1] = 0.0
    return masked
