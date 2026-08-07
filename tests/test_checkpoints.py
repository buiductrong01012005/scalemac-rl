from pathlib import Path

import pytest

from scalemac_rl.checkpoints import require_checkpoint


def test_require_checkpoint_returns_existing_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert require_checkpoint(checkpoint) == checkpoint


def test_missing_best_feasible_mentions_diagnostic_alternative(tmp_path: Path) -> None:
    best_reward = tmp_path / "best_reward.pt"
    best_reward.write_bytes(b"checkpoint")
    with pytest.raises(FileNotFoundError, match="No best_feasible.pt was created") as exc_info:
        require_checkpoint(tmp_path / "best_feasible.pt")
    assert "diagnostic" in str(exc_info.value)
    assert str(best_reward) in str(exc_info.value)
