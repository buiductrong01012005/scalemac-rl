from __future__ import annotations

import sys
from pathlib import Path

from scalemac_rl.scripts.train_single_seed import main as train_single_seed_main


BUDGET_STEPS = 300_032
MILESTONES = "100096,200192,300032"


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    """Fine-tune the hybrid PPO policy for approximately 300k env steps.

    The run keeps separate outputs, fixed-target trade-off checkpoint selection,
    and checkpoints near 100k, 200k, and 300k environment steps. Existing
    hybrid weights are reused when available; source archives never include
    generated artifacts.
    """
    defaults = [
        "--steps-per-stage", str(BUDGET_STEPS),
        "--validate-every", "64",
        "--checkpoint-every", "128",
        "--milestone-env-steps", MILESTONES,
        "--output", "artifacts/hybrid_300k_latest.pt",
        "--best-feasible-output", "artifacts/hybrid_300k_best_feasible.pt",
        "--best-reward-output", "artifacts/hybrid_300k_best_reward.pt",
        "--best-tradeoff-output", "artifacts/hybrid_300k_best_tradeoff.pt",
        "--best-lowest-violation-output", "artifacts/hybrid_300k_best_lowest_violation.pt",
        "--checkpoint-dir", "artifacts/checkpoints/hybrid_300k",
        "--log-output", "artifacts/hybrid_300k_training.csv",
        "--validation-output", "artifacts/hybrid_300k_validation.csv",
        "--checkpoint-manifest-output", "artifacts/hybrid_300k_checkpoint_manifest.csv",
    ]

    if not _has_option("--resume-checkpoint") and not _has_option("--init-checkpoint"):
        resume_candidates = [
            Path("artifacts/hybrid_300k_latest.pt"),
            Path("artifacts/best_tradeoff.pt"),
            Path("artifacts/best_lowest_violation.pt"),
            Path("artifacts/best_reward.pt"),
        ]
        resume = next((path for path in resume_candidates if path.is_file()), None)
        if resume is not None:
            defaults.extend(["--resume-checkpoint", str(resume)])

    # These options follow train_single_seed defaults, so the 300k settings win.
    sys.argv[1:1] = defaults
    train_single_seed_main()


if __name__ == "__main__":
    main()
