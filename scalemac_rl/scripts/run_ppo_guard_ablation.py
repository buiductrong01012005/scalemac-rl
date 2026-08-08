from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scalemac_rl.scripts.run_rule_ppo_split_ablation import main as run_split_ablation_main


def main() -> None:
    """Measure how much a small safety guard helps one independently trained PPO actor.

    The PPO checkpoint is held fixed. Only the number of rule-selected grants changes,
    so the resulting curve isolates the benefit and cost of adding a small guard to a
    policy that previously learned without rules.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--rule-reserves", default="4,8,12,16")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ppo_guard_ablation.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/ppo_guard_ablation_manifest.csv"),
    )
    known, remaining = parser.parse_known_args()

    forwarded = [
        "--actor-checkpoint",
        str(known.ppo_checkpoint),
        "--rule-reserves",
        known.rule_reserves,
        "--output",
        str(known.output),
        "--manifest-output",
        str(known.manifest_output),
        *remaining,
    ]
    sys.argv = [sys.argv[0], *forwarded]
    run_split_ablation_main()


if __name__ == "__main__":
    main()
