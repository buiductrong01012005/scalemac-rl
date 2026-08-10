from __future__ import annotations

import argparse
from pathlib import Path

from scalemac_rl.reproducibility_analysis import build_reproducibility_analysis
from scalemac_rl.reward_study import RewardStudyPlan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Round 09 reproducibility report")
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/reproducibility/round_09_tjs_repeatability.json"),
    )
    parser.add_argument(
        "--round-dir",
        type=Path,
        default=Path("artifacts/runs/reward_study/round_09_reproducibility_diagnostic"),
    )
    args = parser.parse_args()
    plan = RewardStudyPlan.from_json(args.plan)
    output = Path(str(plan.analysis["output"]))
    print(build_reproducibility_analysis(plan=plan, round_dir=args.round_dir, output_path=output))


if __name__ == "__main__":
    main()
