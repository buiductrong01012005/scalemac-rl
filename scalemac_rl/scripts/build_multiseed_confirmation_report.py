from __future__ import annotations

import argparse
from pathlib import Path

from scalemac_rl.multiseed_analysis import build_multiseed_confirmation_analysis
from scalemac_rl.reward_study import RewardStudyPlan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Round 08 multi-seed confirmation HTML/CSV report")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs/reward_study"))
    args = parser.parse_args()
    plan = RewardStudyPlan.from_json(args.plan)
    round_dir = args.output_root / plan.round_id
    output = Path(str(plan.analysis["output"]))
    print(build_multiseed_confirmation_analysis(plan=plan, round_dir=round_dir, output_path=output))


if __name__ == "__main__":
    main()
