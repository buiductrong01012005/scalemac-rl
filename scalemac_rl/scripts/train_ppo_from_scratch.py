from __future__ import annotations

import sys
from pathlib import Path

from scalemac_rl.scripts.train_ppo import main as train_ppo_main


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    """Train PPO from random weights with no rule-selected UE grants.

    Default mode exposes 128 heuristic candidates and lets PPO choose all Top-64
    grants. Pass ``--full-ues`` to expose all 1,200 UEs directly to the actor.
    The action projector only enforces Top-K and exact 273-PRB feasibility.
    """
    full_ues = "--full-ues" in sys.argv[1:]
    sys.argv = [arg for arg in sys.argv if arg != "--full-ues"]

    tag = "full1200" if full_ues else "candidate128"
    candidate_mode = "all" if full_ues else "heuristic"
    max_candidates = "1200" if full_ues else "128"

    defaults = [
        "--single-seed-upper-bound",
        "--freeze-static-profiles",
        "--curriculum", "1200",
        "--stage-p99-wait-limits", "50",
        "--steps-per-stage", "200192",
        "--workers", "1",
        "--rollout-steps", "256",
        "--episode-slots", "2000",
        "--candidate-mode", candidate_mode,
        "--max-candidates", max_candidates,
        "--scheduler-mode", "ppo_only",
        "--safety-reserve-ues", "0",
        "--no-force-harq-retransmissions",
        "--final-stage-p99-schedule", "80,65,55,50",
        "--fairness-target-schedule", "0.50,0.55,0.60",
        "--validation-seeds", "1701",
        "--validation-repeats", "1",
        "--validation-slots", "5000",
        "--validate-every", "64",
        "--rollback-patience", "2",
        "--checkpoint-every", "64",
        "--seed", "1701",
        "--fixed-profile-seed", "1701",
        "--deadline-risk-start-ratio", "0.60",
        "--deadline-risk-penalty-weight", "0.15",
        "--max-wait-risk-penalty-weight", "0.10",
        "--starvation-threshold-slots", "64",
        "--reward-throughput-weight", "0.45",
        "--reward-fairness-weight", "0.35",
        "--reward-service-weight", "0.15",
        "--reward-deficit-service-weight", "0.05",
        "--reward-fairness-delta-weight", "0.03",
        "--reward-pf-utility-delta-weight", "0.02",
        "--max-starvation-rate", "0",
        "--max-p99-wait-slots", "50",
        "--min-jain-fairness", "0.60",
        "--max-wait-slots", "60",
        "--init-checkpoint", "artifacts/__random_initialization__.pt",
        "--output", f"artifacts/ppo_scratch_{tag}_latest.pt",
        "--best-feasible-output", f"artifacts/ppo_scratch_{tag}_best_feasible.pt",
        "--best-reward-output", f"artifacts/ppo_scratch_{tag}_best_reward.pt",
        "--best-lowest-violation-output", f"artifacts/ppo_scratch_{tag}_best_lowest_violation.pt",
        "--checkpoint-dir", f"artifacts/checkpoints/ppo_scratch_{tag}",
        "--log-output", f"artifacts/ppo_scratch_{tag}_training.csv",
        "--validation-output", f"artifacts/ppo_scratch_{tag}_validation.csv",
        "--checkpoint-manifest-output", f"artifacts/ppo_scratch_{tag}_checkpoint_manifest.csv",
    ]

    # Defaults are inserted before user arguments, so explicit CLI options win.
    sys.argv[1:1] = defaults
    train_ppo_main()


if __name__ == "__main__":
    main()
