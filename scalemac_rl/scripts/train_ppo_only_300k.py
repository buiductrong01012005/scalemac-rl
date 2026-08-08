from __future__ import annotations

import sys

from scalemac_rl.scripts.train_ppo_from_scratch import main as train_from_scratch_main


BUDGET_STEPS = 300_032
MILESTONES = "100096,200192,300032"


def main() -> None:
    """Train candidate-128 PPO-only from random weights for about 300k steps."""
    defaults = [
        "--steps-per-stage", str(BUDGET_STEPS),
        "--validate-every", "64",
        "--checkpoint-every", "128",
        "--milestone-env-steps", MILESTONES,
        "--output", "artifacts/ppo_only_300k_latest.pt",
        "--best-feasible-output", "artifacts/ppo_only_300k_best_feasible.pt",
        "--best-reward-output", "artifacts/ppo_only_300k_best_reward.pt",
        "--best-tradeoff-output", "artifacts/ppo_only_300k_best_tradeoff.pt",
        "--best-lowest-violation-output", "artifacts/ppo_only_300k_best_lowest_violation.pt",
        "--checkpoint-dir", "artifacts/checkpoints/ppo_only_300k",
        "--log-output", "artifacts/ppo_only_300k_training.csv",
        "--validation-output", "artifacts/ppo_only_300k_validation.csv",
        "--checkpoint-manifest-output", "artifacts/ppo_only_300k_checkpoint_manifest.csv",
    ]
    # The underlying script still injects random initialization and ppo_only mode.
    sys.argv[1:1] = defaults
    train_from_scratch_main()


if __name__ == "__main__":
    main()
