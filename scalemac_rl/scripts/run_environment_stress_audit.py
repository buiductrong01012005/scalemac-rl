from __future__ import annotations

import argparse
import json
from pathlib import Path

from scalemac_rl.environment_stress_audit import (
    DEFAULT_STRESS_SEEDS,
    KEY_SEEDS,
    run_environment_stress_audit,
    write_rows,
)


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round 17C environment stress audit using Oracle and PF only"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/analysis/optimization/round_17c"),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_STRESS_SEEDS,
        help="comma-separated environment seeds",
    )
    parser.add_argument("--slots", type=int, default=5000)
    args = parser.parse_args()

    result = run_environment_stress_audit(seeds=args.seeds, slots=args.slots)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_rows(args.output_dir / "environment_stress_metrics.csv", result.metrics)
    write_rows(args.output_dir / "environment_stress_summary.csv", result.summary)
    if result.key_seed_positions:
        write_rows(
            args.output_dir / "environment_stress_key_seed_positions.csv",
            result.key_seed_positions,
        )

    oracle_rows = [row for row in result.metrics if row["policy"] == "oracle"]
    pf_rows = [row for row in result.metrics if row["policy"] == "pf"]
    payload = {
        "round": "17C",
        "purpose": "test whether the three PPO study seeds are atypically easy/hard environment realizations",
        "seeds": list(args.seeds),
        "key_seeds_present": [seed for seed in KEY_SEEDS if seed in args.seeds],
        "slots_per_policy_seed": args.slots,
        "evaluation_cases": len(result.metrics),
        "training_cases": 0,
        "oracle_all_zero_starvation": all(int(row["zero_starvation"]) for row in oracle_rows),
        "oracle_all_service_feasible_under_64": all(
            int(row["service_feasible_under_64"]) for row in oracle_rows
        ),
        "pf_zero_starvation_seeds": sum(int(row["zero_starvation"]) for row in pf_rows),
        "interpretation_rule": (
            "Use per-metric hardness percentiles rather than a synthetic scalar difficulty score. "
            "Percentiles near 50 are typical; values near 100 are among the harder seeds for that metric."
        ),
    }
    (args.output_dir / "environment_stress_decision.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(json.dumps(payload, indent=2))
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
