from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from scalemac_rl.scripts.train_ppo import main as train_ppo_main


BUDGET_STEPS = 300_032
MILESTONES = "100096,200192,300032"


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    name: str
    reward_weights: tuple[float, float, float, float, float, float, float]
    gamma: float
    gae_lambda: float
    clip_coef: float
    learning_rate: float
    learning_rate_end: float
    entropy_coef: float
    entropy_coef_end: float
    population_wait_penalty: float


PROFILES: dict[str, ExperimentProfile] = {
    # Main hypothesis: combine cell throughput with proportional-fair utility,
    # low-percentile throughput, and successful service of urgent UEs.
    "balanced": ExperimentProfile(
        name="balanced",
        reward_weights=(0.40, 0.15, 0.10, 0.05, 0.15, 0.10, 0.05),
        gamma=0.999,
        gae_lambda=0.97,
        clip_coef=0.10,
        learning_rate=1.0e-4,
        learning_rate_end=2.5e-5,
        entropy_coef=5.0e-3,
        entropy_coef_end=5.0e-4,
        population_wait_penalty=0.08,
    ),
    # Stronger bottom-UE pressure to test whether fairness and tail delay can be
    # improved without falling back to an external oldest-UE rule.
    "fairness": ExperimentProfile(
        name="fairness",
        reward_weights=(0.35, 0.15, 0.08, 0.05, 0.17, 0.15, 0.05),
        gamma=0.999,
        gae_lambda=0.97,
        clip_coef=0.10,
        learning_rate=7.5e-5,
        learning_rate_end=2.0e-5,
        entropy_coef=6.0e-3,
        entropy_coef_end=5.0e-4,
        population_wait_penalty=0.10,
    ),
    # Conservative optimizer control: same balanced reward, smaller updates.
    "stable": ExperimentProfile(
        name="stable",
        reward_weights=(0.40, 0.15, 0.10, 0.05, 0.15, 0.10, 0.05),
        gamma=0.999,
        gae_lambda=0.97,
        clip_coef=0.08,
        learning_rate=5.0e-5,
        learning_rate_end=1.0e-5,
        entropy_coef=3.0e-3,
        entropy_coef_end=3.0e-4,
        population_wait_penalty=0.08,
    ),
}


def _parse_wrapper_args(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    parsed, remaining = parser.parse_known_args(argv)
    return parsed.profile, remaining


def main() -> None:
    """Train a rule-free PPO scheduler over all 1,200 UEs.

    PPO controls UE ranking and PRB demand for every UE. The projector only
    enforces Top-64 selection, at least one PRB per selected UE, and exactly 273
    allocated PRBs. No candidate filter, HARQ override, or oldest-UE rule is used.
    """
    profile_name, remaining = _parse_wrapper_args(sys.argv[1:])
    profile = PROFILES[profile_name]
    sys.argv = [sys.argv[0], *remaining]

    (
        throughput,
        fairness,
        service,
        deficit,
        pf_utility,
        low_throughput,
        urgency,
    ) = profile.reward_weights
    tag = f"full_control_v2_{profile.name}"

    defaults = [
        "--single-seed-upper-bound",
        "--freeze-static-profiles",
        "--curriculum", "1200",
        "--stage-p99-wait-limits", "50",
        "--steps-per-stage", str(BUDGET_STEPS),
        "--workers", "1",
        "--rollout-steps", "256",
        "--episode-slots", "5000",
        "--candidate-mode", "all",
        "--max-candidates", "1200",
        "--scheduler-mode", "ppo_only",
        "--safety-reserve-ues", "0",
        "--no-force-harq-retransmissions",
        "--final-stage-p99-schedule", "100,80,65,50",
        "--fairness-target-schedule", "0.45,0.52,0.58,0.60",
        "--validation-seeds", "1701",
        "--validation-repeats", "1",
        "--validation-slots", "5000",
        "--validate-every", "64",
        "--rollback-patience", "3",
        "--checkpoint-every", "128",
        "--milestone-env-steps", MILESTONES,
        "--seed", "1701",
        "--fixed-profile-seed", "1701",
        "--hidden-dim", "96",
        "--lr", str(profile.learning_rate),
        "--lr-end", str(profile.learning_rate_end),
        "--gamma", str(profile.gamma),
        "--gae-lambda", str(profile.gae_lambda),
        "--clip-coef", str(profile.clip_coef),
        "--entropy-coef", str(profile.entropy_coef),
        "--entropy-coef-end", str(profile.entropy_coef_end),
        "--update-epochs", "4",
        "--minibatch-size", "8",
        "--target-kl", "0.02",
        "--deadline-risk-start-ratio", "0.60",
        "--deadline-risk-penalty-weight", "0.10",
        "--max-wait-risk-penalty-weight", "0.08",
        "--population-wait-penalty-weight", str(profile.population_wait_penalty),
        "--low-throughput-percentile", "10",
        "--starvation-threshold-slots", "64",
        "--reward-throughput-weight", str(throughput),
        "--reward-fairness-weight", str(fairness),
        "--reward-service-weight", str(service),
        "--reward-deficit-service-weight", str(deficit),
        "--reward-pf-utility-weight", str(pf_utility),
        "--reward-low-throughput-weight", str(low_throughput),
        "--reward-urgency-service-weight", str(urgency),
        "--reward-fairness-delta-weight", "0.02",
        "--reward-pf-utility-delta-weight", "0.03",
        "--max-starvation-rate", "0",
        "--max-p99-wait-slots", "50",
        "--min-jain-fairness", "0.60",
        "--min-throughput-score", "0.43",
        "--max-wait-slots", "60",
        "--starvation-multiplier", "8.0",
        "--wait-multiplier", "1.5",
        "--fairness-multiplier", "2.0",
        "--max-wait-multiplier", "1.5",
        "--lagrangian-lr", "0.05",
        "--init-checkpoint", "artifacts/__random_initialization__.pt",
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
    # Explicit user arguments remain last and therefore override defaults.
    sys.argv[1:1] = defaults
    train_ppo_main()


if __name__ == "__main__":
    main()
