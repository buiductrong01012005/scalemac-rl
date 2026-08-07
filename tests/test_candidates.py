import numpy as np

from scalemac_rl.candidates import apply_candidate_mask, build_candidate_mask
from scalemac_rl.env import ELIGIBLE, HARQ_PENDING, OBSERVATION_FEATURES, TIME_SINCE_SERVICE


def test_candidate_filter_keeps_harq_and_requested_size() -> None:
    observation = np.zeros((100, OBSERVATION_FEATURES), dtype=np.float32)
    observation[:, ELIGIBLE] = 1.0
    observation[:, TIME_SINCE_SERVICE] = np.linspace(0.0, 1.0, 100)
    observation[[3, 7, 9], HARQ_PENDING] = 1.0
    mask = build_candidate_mask(observation, max_candidates=32, min_candidates=16)
    assert mask.sum() == 32
    assert mask[3] and mask[7] and mask[9]


def test_apply_candidate_mask_suppresses_non_candidates() -> None:
    action = np.ones((10, 2), dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    mask[:4] = True
    masked = apply_candidate_mask(action, mask)
    assert np.all(masked[:4] == 1.0)
    assert np.all(masked[4:, 0] == -1.0)
    assert np.all(masked[4:, 1] == 0.0)

from scalemac_rl.candidates import (
    candidate_diagnostics,
    gather_candidate_batch,
    gather_candidates,
    scatter_candidate_action,
)


def test_candidate_filter_keeps_long_waiting_ues_when_capacity_allows() -> None:
    observation = np.zeros((20, OBSERVATION_FEATURES), dtype=np.float32)
    observation[:, ELIGIBLE] = 1.0
    observation[[4, 5, 6], TIME_SINCE_SERVICE] = [0.8, 1.0, 1.4]
    mask = build_candidate_mask(
        observation,
        max_candidates=8,
        min_candidates=4,
        long_wait_threshold=0.8,
    )
    assert mask[4] and mask[5] and mask[6]
    diagnostics = candidate_diagnostics(observation, mask, long_wait_threshold=0.8)
    assert diagnostics.long_wait_count == 3
    assert diagnostics.long_wait_retention_rate == 1.0


def test_compact_candidate_round_trip() -> None:
    observation = np.arange(10 * OBSERVATION_FEATURES, dtype=np.float32).reshape(10, -1)
    mask = np.zeros(10, dtype=bool)
    mask[[1, 4, 8]] = True
    compact, indices = gather_candidates(observation, mask)
    compact_action = np.ones((3, 2), dtype=np.float32)
    full_action = scatter_candidate_action(compact_action, indices, num_ues=10)
    assert compact.shape == (3, OBSERVATION_FEATURES)
    assert np.array_equal(indices, np.asarray([1, 4, 8]))
    assert np.all(full_action[indices] == 1.0)
    assert np.all(full_action[[0, 2, 3], 0] == -1.0)


def test_gather_candidate_batch_uses_equal_candidate_count() -> None:
    observations = np.zeros((2, 6, OBSERVATION_FEATURES), dtype=np.float32)
    masks = np.zeros((2, 6), dtype=bool)
    masks[0, [0, 2, 4]] = True
    masks[1, [1, 3, 5]] = True
    compact, indices = gather_candidate_batch(observations, masks)
    assert compact.shape == (2, 3, OBSERVATION_FEATURES)
    assert indices.shape == (2, 3)
