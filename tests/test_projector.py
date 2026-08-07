import numpy as np

from scalemac_rl.projector import project_action


def test_projector_preserves_budget_and_top_k() -> None:
    n = 100
    action = np.zeros((n, 2), dtype=np.float32)
    action[:, 0] = np.linspace(0.0, 1.0, n)
    action[:, 1] = 1.0
    pending = np.zeros(n, dtype=bool)
    pending[:3] = True

    result = project_action(
        action,
        eligible=np.ones(n, dtype=bool),
        harq_pending=pending,
        harq_retx_count=np.arange(n, dtype=np.int16),
        time_since_service=np.arange(n, dtype=np.int32),
        num_prbs=273,
        max_selected_ues=64,
    )

    assert result.selected_ues.size == 64
    assert int(result.prbs.sum()) == 273
    assert np.all(result.prbs >= 1)
    assert set(range(3)).issubset(set(result.selected_ues.tolist()))


def test_projector_reserves_long_wait_grants_and_keeps_learned_selection() -> None:
    n = 128
    action = np.zeros((n, 2), dtype=np.float32)
    action[:, 0] = np.linspace(1.0, 0.0, n)
    action[:, 1] = 0.5
    waits = np.zeros(n, dtype=np.int32)
    waits[-20:] = np.arange(81, 101)

    result = project_action(
        action,
        eligible=np.ones(n, dtype=bool),
        harq_pending=np.zeros(n, dtype=bool),
        harq_retx_count=np.zeros(n, dtype=np.int16),
        time_since_service=waits,
        num_prbs=273,
        max_selected_ues=64,
        safety_reserve_ues=16,
        safety_wait_threshold_ratio=0.8,
        starvation_threshold_slots=100,
    )

    assert result.safety_selected_count == 16
    assert result.forced_long_wait_count == 16
    assert result.learned_selected_count == 48
    assert set(np.arange(n - 16, n)).issubset(set(result.selected_ues.tolist()))
    assert int(result.prbs.sum()) == 273
