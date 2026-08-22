from __future__ import annotations

import argparse
from pathlib import Path

from scalemac_rl.schedule_fairness_analysis import build_schedule_fairness_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 20 schedule-frequency fairness reward analysis")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = build_schedule_fairness_analysis(
        plan_path=args.plan,
        round_dir=args.round_dir,
        output_dir=args.output_dir,
    )
    for name, path in paths.items():
        print(f"saved {name}: {path}")


if __name__ == "__main__":
    main()
