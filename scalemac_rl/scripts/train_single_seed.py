from __future__ import annotations

import sys
from pathlib import Path

from scalemac_rl.scripts.train_ppo import main as train_ppo_main


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    """Run the long single-profile upper-bound experiment.

    Defaults intentionally overfit one frozen 1,200-UE CQI/demand profile so the
    project can measure how far the current actor/projector design can optimize a
    known scenario. This is not a generalization experiment.
    """
    defaults = [
        "--single-seed-upper-bound",
        "--freeze-static-profiles",
        "--curriculum", "1200",
        "--steps-per-stage", "524288",
        "--workers", "1",
        "--rollout-steps", "256",
        "--episode-slots", "2000",
        "--max-candidates", "128",
        "--safety-reserve-ues", "16",
        "--stage-p99-wait-limits", "50",
        "--final-stage-p99-schedule", "80,65,55,50",
        "--validation-seeds", "1701",
        "--validation-repeats", "1",
        "--validation-slots", "5000",
        "--validate-every", "32",
        "--rollback-patience", "2",
        "--checkpoint-every", "64",
        "--seed", "1701",
        "--fixed-profile-seed", "1701",
        "--deadline-risk-start-ratio", "0.60",
        "--deadline-risk-penalty-weight", "0.15",
    ]

    # Continue from the strongest v0.5 1,200-UE checkpoint when it exists.
    resume = Path("artifacts/checkpoints/best_stage_1200.pt")
    if not _has_option("--resume-checkpoint") and not _has_option("--init-checkpoint"):
        if resume.is_file():
            defaults.extend(["--resume-checkpoint", str(resume)])
        else:
            defaults.extend(["--init-checkpoint", "artifacts/pf_imitation.pt"])

    # Defaults are inserted before user arguments so explicit CLI options win.
    sys.argv[1:1] = defaults
    train_ppo_main()


if __name__ == "__main__":
    main()
