from __future__ import annotations

import argparse
import subprocess
import sys


DEFAULT_PROFILES = "balanced,fairness,stable"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run short rule-free full-control PPO profile comparisons"
    )
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--steps", type=int, default=100_096)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    if not profiles:
        parser.error("at least one profile is required")
    if args.steps <= 0 or args.steps % 256 != 0:
        parser.error("steps must be positive and divisible by rollout length 256")

    for profile in profiles:
        tag = f"full_control_v2_tuning_{profile}_{args.steps}"
        command = [
            sys.executable,
            "-m",
            "scalemac_rl.scripts.train_full_control_ppo_v2",
            "--profile",
            profile,
            "--steps-per-stage",
            str(args.steps),
            "--device",
            args.device,
            "--milestone-env-steps",
            str(args.steps),
            "--run-dir", f"artifacts/runs/{tag}",
        ]
        if args.no_progress:
            command.append("--no-progress")
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
