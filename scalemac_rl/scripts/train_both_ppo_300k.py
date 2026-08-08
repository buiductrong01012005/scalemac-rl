from __future__ import annotations

import subprocess
import sys


def main() -> None:
    """Run hybrid fine-tuning and PPO-only random training sequentially."""
    commands = [
        [sys.executable, "-m", "scalemac_rl.scripts.train_hybrid_300k"],
        [sys.executable, "-m", "scalemac_rl.scripts.train_ppo_only_300k"],
    ]
    for command in commands:
        print(f"running: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
