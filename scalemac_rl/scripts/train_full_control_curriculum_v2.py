from __future__ import annotations

import sys

from scalemac_rl.scripts.train_full_control_ppo_v2 import main as full_control_main


STEPS_PER_STAGE = 75_008
TOTAL_BUDGET = STEPS_PER_STAGE * 4


def main() -> None:
    """Run the same rule-free PPO through a 128->256->600->1200 UE curriculum."""
    defaults = [
        "--curriculum", "128,256,600,1200",
        "--stage-p99-wait-limits", "80,80,70,50",
        "--steps-per-stage", str(STEPS_PER_STAGE),
        "--no-single-seed-upper-bound",
        "--validation-seeds", "1701",
        "--milestone-env-steps", "75008,150016,225024,300032",
        "--run-dir", "artifacts/runs/full_control_v2_curriculum",
    ]
    sys.argv[1:1] = defaults
    full_control_main()


if __name__ == "__main__":
    main()
