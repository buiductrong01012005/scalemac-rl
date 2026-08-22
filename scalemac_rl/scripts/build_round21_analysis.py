from __future__ import annotations

import argparse
from pathlib import Path

from scalemac_rl.round21_analysis import build_round21_analysis


def main() -> None:
    p = argparse.ArgumentParser(description='Build Round 21 pure-PPO analysis')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--round-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--diagnostic-slots', type=int, default=512)
    args = p.parse_args()
    paths = build_round21_analysis(
        plan_path=args.plan,
        round_dir=args.round_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        diagnostic_slots=args.diagnostic_slots,
    )
    for name, path in paths.items():
        print(f'saved {name}: {path}')


if __name__ == '__main__':
    main()
