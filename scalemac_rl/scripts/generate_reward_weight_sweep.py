from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from scalemac_rl.reward_study import POSITIVE_COMPONENTS


def _parse_components(value: str) -> list[str]:
    components = [item.strip() for item in value.split(",") if item.strip()]
    if len(components) < 2:
        raise argparse.ArgumentTypeError("at least two components are required")
    unknown = sorted(set(components) - set(POSITIVE_COMPONENTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown positive components: {', '.join(unknown)}"
        )
    if len(components) != len(set(components)):
        raise argparse.ArgumentTypeError("components must be unique")
    return components


def _simplex_weights(component_count: int, divisions: int) -> list[tuple[int, ...]]:
    values: list[tuple[int, ...]] = []
    for candidate in itertools.product(range(divisions + 1), repeat=component_count):
        if sum(candidate) == divisions:
            values.append(candidate)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a normalized positive-reward simplex sweep plan"
    )
    parser.add_argument(
        "--components",
        type=_parse_components,
        default=["throughput", "fairness"],
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="grid step; must divide 1 exactly, e.g. 0.05, 0.1, 0.2",
    )
    parser.add_argument("--positive-scale", type=float, default=1.0)
    parser.add_argument(
        "--base-plan",
        type=Path,
        default=Path("configs/reward_study/round_02_cumulative_equal.json"),
        help="copy common settings and optional fixed penalty/delta weights from a case",
    )
    parser.add_argument("--base-case", default="r3_add_delay_equal")
    parser.add_argument("--round-id", default="round_03_weight_sweep")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/reward_study/round_03_weight_sweep.json"),
    )
    parser.add_argument("--max-cases", type=int, default=200)
    args = parser.parse_args()

    if args.step <= 0.0 or args.step > 1.0:
        parser.error("--step must be in (0, 1]")
    divisions = round(1.0 / args.step)
    if abs(divisions * args.step - 1.0) > 1e-9:
        parser.error("--step must divide 1 exactly")
    if args.positive_scale < 0.0:
        parser.error("--positive-scale must be non-negative")

    try:
        base = json.loads(args.base_plan.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    base_case: dict[str, Any] | None = next(
        (case for case in base.get("cases", []) if case.get("id") == args.base_case),
        None,
    )
    if base_case is None:
        parser.error(f"base case not found: {args.base_case}")

    grid = _simplex_weights(len(args.components), divisions)
    # Exclude zero-weight endpoints only when more than two components are swept;
    # the component-screen round already covers pure single-component cases.
    if len(args.components) > 2:
        grid = [candidate for candidate in grid if all(value > 0 for value in candidate)]
    if len(grid) > args.max_cases:
        parser.error(
            f"grid contains {len(grid)} cases, above --max-cases {args.max_cases}; use a larger step"
        )

    cases: list[dict[str, Any]] = []
    for integers in grid:
        weights = {
            component: integer / divisions
            for component, integer in zip(args.components, integers, strict=True)
        }
        suffix = "_".join(
            f"{component[:3]}{int(round(weight * 100)):03d}"
            for component, weight in weights.items()
        )
        cases.append(
            {
                "id": f"sweep_{suffix}",
                "label": ", ".join(
                    f"{component}={weight:.2f}" for component, weight in weights.items()
                ),
                "hypothesis": "Generated simplex sweep around retained reward components.",
                "positive_scale": args.positive_scale,
                "positive_weights": weights,
                "delta_weights": base_case.get("delta_weights", {}),
                "penalty_weights": base_case.get("penalty_weights", {}),
            }
        )

    payload = {
        "study_id": base.get("study_id", "full_control_reward_study"),
        "round_id": args.round_id,
        "description": (
            f"Generated simplex sweep for {', '.join(args.components)} at step {args.step}. "
            f"Penalty/delta terms copied from {args.base_case}."
        ),
        "common": base.get("common", {}),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
