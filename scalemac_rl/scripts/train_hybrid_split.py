from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scalemac_rl.scripts.train_single_seed import main as train_single_seed_main


DEFAULT_BUDGET_STEPS = 120_064
DEFAULT_MILESTONES = "60160,120064"


def _has_option(arguments: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in arguments)


def main() -> None:
    """Fine-tune one fixed rule/PPO split from the same hybrid starting point.

    Use the no-training split ablation first. Then fine-tune only promising
    reserves so compute is not wasted on clearly dominated split points.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--reserve", type=int, required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_BUDGET_STEPS)
    parser.add_argument("--start-checkpoint", type=Path)
    known, remaining = parser.parse_known_args()

    if not 0 <= known.reserve <= 64:
        parser.error("reserve must be in [0, 64]")
    if known.steps <= 0 or known.steps % 256 != 0:
        parser.error("steps must be positive and divisible by 256")

    tag = f"hybrid_r{known.reserve:02d}_{known.steps // 1000}k"
    milestones = f"{known.steps // 2 // 256 * 256},{known.steps}"
    forwarded = list(remaining)
    defaults = [
        "--steps-per-stage", str(known.steps),
        "--safety-reserve-ues", str(known.reserve),
        "--validate-every", "64",
        "--checkpoint-every", "128",
        "--milestone-env-steps", milestones,
        "--output", f"artifacts/{tag}_latest.pt",
        "--best-feasible-output", f"artifacts/{tag}_best_feasible.pt",
        "--best-reward-output", f"artifacts/{tag}_best_reward.pt",
        "--best-tradeoff-output", f"artifacts/{tag}_best_tradeoff.pt",
        "--best-lowest-violation-output", f"artifacts/{tag}_best_lowest_violation.pt",
        "--checkpoint-dir", f"artifacts/checkpoints/{tag}",
        "--log-output", f"artifacts/{tag}_training.csv",
        "--validation-output", f"artifacts/{tag}_validation.csv",
        "--checkpoint-manifest-output", f"artifacts/{tag}_checkpoint_manifest.csv",
    ]

    if known.start_checkpoint is not None:
        defaults.extend(["--resume-checkpoint", str(known.start_checkpoint)])
    elif not _has_option(forwarded, "--resume-checkpoint") and not _has_option(
        forwarded, "--init-checkpoint"
    ):
        candidates = [
            Path("artifacts/hybrid_300k_best_tradeoff.pt"),
            Path("artifacts/hybrid_300k_best_feasible.pt"),
            Path("artifacts/best_tradeoff.pt"),
            Path("artifacts/best_lowest_violation.pt"),
        ]
        start = next((path for path in candidates if path.is_file()), None)
        if start is not None:
            defaults.extend(["--resume-checkpoint", str(start)])

    # train_single_seed inserts its own defaults before these arguments, so this
    # split-specific configuration wins while any forwarded CLI override wins last.
    sys.argv = [sys.argv[0], *defaults, *forwarded]
    train_single_seed_main()


if __name__ == "__main__":
    main()
