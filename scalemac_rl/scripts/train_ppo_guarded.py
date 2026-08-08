from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from scalemac_rl.scripts.train_single_seed import main as train_single_seed_main


DEFAULT_STEPS = 120_064


def _has_option(arguments: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in arguments)


def _infer_hidden_dim(checkpoint_path: Path) -> int | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    weight = state_dict.get("encoder.0.weight") if isinstance(state_dict, dict) else None
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return None
    return int(weight.shape[0])


def main() -> None:
    """Fine-tune an independently trained PPO actor with a small oldest-UE guard.

    The intended workflow is:
    1. run ``run_ppo_guard_ablation`` with fixed actor weights;
    2. select one or two non-dominated reserves, usually 4 or 8;
    3. fine-tune each reserve separately with this command.

    Starting from a PPO-only checkpoint preserves the actor's learned safety behavior,
    while the small guard protects rare tail-delay cases during fine-tuning.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--reserve", type=int, default=8)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--start-checkpoint", type=Path)
    known, remaining = parser.parse_known_args()

    if not 1 <= known.reserve <= 16:
        parser.error("reserve must be in [1, 16] for the small-guard experiment")
    if known.steps <= 0 or known.steps % 256 != 0:
        parser.error("steps must be positive and divisible by 256")

    tag = f"ppo_guard_r{known.reserve:02d}_{known.steps // 1000}k"
    halfway = max(256, (known.steps // 2 // 256) * 256)
    milestones = f"{halfway},{known.steps}"
    forwarded = list(remaining)
    defaults = [
        "--scheduler-mode", "hybrid",
        "--safety-reserve-ues", str(known.reserve),
        "--force-harq-retransmissions",
        "--steps-per-stage", str(known.steps),
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
            Path("artifacts/ppo_only_300k_best_tradeoff.pt"),
            Path("artifacts/ppo_only_300k_best_feasible.pt"),
            Path("artifacts/ppo_scratch_candidate128_best_tradeoff.pt"),
            Path("artifacts/ppo_scratch_candidate128_best_lowest_violation.pt"),
        ]
        start = next((path for path in candidates if path.is_file()), None)
        if start is None:
            parser.error(
                "no PPO-only checkpoint found; pass --start-checkpoint explicitly"
            )
        defaults.extend(["--resume-checkpoint", str(start)])

    start_path = known.start_checkpoint
    if start_path is None:
        try:
            resume_index = defaults.index("--resume-checkpoint")
            start_path = Path(defaults[resume_index + 1])
        except ValueError:
            start_path = None
    if start_path is not None and not _has_option(forwarded, "--hidden-dim"):
        hidden_dim = _infer_hidden_dim(start_path)
        if hidden_dim is not None:
            defaults.extend(["--hidden-dim", str(hidden_dim)])

    # train_single_seed injects its defaults before these arguments. These explicit
    # small-guard settings therefore override the generic single-seed defaults, while
    # user-supplied arguments in ``remaining`` win last.
    sys.argv = [sys.argv[0], *defaults, *forwarded]
    train_single_seed_main()


if __name__ == "__main__":
    main()
