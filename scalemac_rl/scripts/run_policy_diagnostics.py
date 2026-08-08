from __future__ import annotations

import argparse
from pathlib import Path

from scalemac_rl.policy_diagnostics import (
    discover_cases,
    parse_modes,
    resolve_device,
    run_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic mean-action and stochastic Beta-action PPO behaviour "
            "without changing reward, environment, or architecture"
        )
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path(
            "artifacts/runs/reward_study/round_02_throughput_jain_sweep"
        ),
    )
    parser.add_argument(
        "--cases",
        default="t0375_j0625,t025_j075",
        help="comma-separated case folders under --study-root",
    )
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument(
        "--modes",
        default="deterministic,stochastic",
        help="deterministic, stochastic, or both",
    )
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--profile-seed", type=int, default=1701)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--tie-epsilon", type=float, default=1e-6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/runs/reward_study/round_03_policy_diagnostics"
        ),
    )
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=Path(
            "docs/analysis/reward_study/round_03/policy_diagnostics.html"
        ),
    )
    args = parser.parse_args()

    if args.slots <= 0:
        parser.error("--slots must be positive")
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.window_size <= 0:
        parser.error("--window-size must be positive")
    if args.tie_epsilon < 0.0:
        parser.error("--tie-epsilon must be non-negative")

    try:
        modes = parse_modes(args.modes)
        case_ids = [value.strip() for value in args.cases.split(",") if value.strip()]
        cases = discover_cases(
            study_root=args.study_root,
            case_ids=case_ids,
            checkpoint_name=args.checkpoint,
        )
        device = resolve_device(args.device)
        paths = run_diagnostics(
            cases=cases,
            modes=modes,
            device=device,
            slots=args.slots,
            first_seed=args.seed,
            seeds=args.seeds,
            profile_seed=args.profile_seed,
            window_size=args.window_size,
            tie_epsilon=args.tie_epsilon,
            output_root=args.output_root,
            docs_output=args.docs_output,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return

    for label, path in paths.items():
        print(f"saved {label}: {path}")


if __name__ == "__main__":
    main()
