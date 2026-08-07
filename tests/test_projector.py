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
