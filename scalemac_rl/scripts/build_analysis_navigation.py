from __future__ import annotations

import argparse
from pathlib import Path

from scalemac_rl.analysis_navigation import build_reward_analysis_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the linked reward-analysis index")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("docs/analysis/reward_study"),
    )
    args = parser.parse_args()
    output = build_reward_analysis_index(args.root)
    print(output)


if __name__ == "__main__":
    main()
