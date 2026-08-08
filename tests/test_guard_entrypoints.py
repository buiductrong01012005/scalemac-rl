from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scalemac_rl.scripts import run_ppo_guard_ablation, train_ppo_guarded


def test_guard_ablation_forwards_ppo_actor_and_small_reserves(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    def fake_main() -> None:
        captured.extend(sys.argv[1:])

    checkpoint = tmp_path / "ppo.pt"
    monkeypatch.setattr(run_ppo_guard_ablation, "run_split_ablation_main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ppo_guard_ablation", "--ppo-checkpoint", str(checkpoint), "--seeds", "3"],
    )
    run_ppo_guard_ablation.main()

    assert captured[captured.index("--actor-checkpoint") + 1] == str(checkpoint)
    assert captured[captured.index("--rule-reserves") + 1] == "4,8,12,16"
    assert captured[captured.index("--seeds") + 1] == "3"


def test_guarded_training_starts_from_ppo_only_checkpoint(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    def fake_main() -> None:
        captured.extend(sys.argv[1:])

    checkpoint = tmp_path / "ppo_only.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(train_ppo_guarded, "train_single_seed_main", fake_main)
    monkeypatch.setattr(train_ppo_guarded, "_infer_hidden_dim", lambda _: 64)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo_guarded",
            "--reserve",
            "8",
            "--steps",
            "1024",
            "--start-checkpoint",
            str(checkpoint),
        ],
    )
    train_ppo_guarded.main()

    assert captured[captured.index("--scheduler-mode") + 1] == "hybrid"
    assert captured[captured.index("--safety-reserve-ues") + 1] == "8"
    assert captured[captured.index("--steps-per-stage") + 1] == "1024"
    assert captured[captured.index("--resume-checkpoint") + 1] == str(checkpoint)


def test_guarded_training_rejects_large_reserve(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_ppo_guarded", "--reserve", "24"])
    with pytest.raises(SystemExit):
        train_ppo_guarded.main()
