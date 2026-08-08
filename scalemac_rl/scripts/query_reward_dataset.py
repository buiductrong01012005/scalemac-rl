from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from scalemac_rl.reward_study import safe_float


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query previously evaluated reward weights by KPI constraints"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/runs/reward_study/reward_weight_dataset.csv"),
    )
    parser.add_argument("--min-fairness", type=float, default=0.0)
    parser.add_argument("--max-starvation", type=float, default=1.0)
    parser.add_argument("--max-p99", type=float, default=float("inf"))
    parser.add_argument("--max-wait", type=float, default=float("inf"))
    parser.add_argument(
        "--objective",
        choices=["goodput", "fairness", "balanced"],
        default="goodput",
    )
    parser.add_argument("--pareto-only", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/reward_study/query_results.csv"),
    )
    args = parser.parse_args()

    try:
        rows = _read(args.dataset)
    except FileNotFoundError:
        parser.error(f"dataset does not exist: {args.dataset}")
    selected = [
        row
        for row in rows
        if safe_float(row, "final_jain_fairness") >= args.min_fairness
        and safe_float(row, "max_starvation_rate") <= args.max_starvation
        and safe_float(row, "max_p99_wait_slots") <= args.max_p99
        and safe_float(row, "max_wait_slots") <= args.max_wait
        and (not args.pareto_only or str(row.get("pareto_front", "")).lower() == "true")
    ]

    def score(row: dict[str, str]) -> tuple[float, ...]:
        goodput = safe_float(row, "mean_goodput_bits_per_slot")
        fairness = safe_float(row, "final_jain_fairness")
        starvation = safe_float(row, "max_starvation_rate")
        p99 = safe_float(row, "max_p99_wait_slots")
        max_wait = safe_float(row, "max_wait_slots")
        if args.objective == "goodput":
            return (goodput, fairness, -starvation, -p99, -max_wait)
        if args.objective == "fairness":
            return (fairness, goodput, -starvation, -p99, -max_wait)
        normalized_goodput = goodput / 200_000.0
        delay_score = min(1.0, 50.0 / max(p99, 1.0))
        wait_score = min(1.0, 60.0 / max(max_wait, 1.0))
        safety_score = max(0.0, 1.0 - starvation)
        balanced = (
            max(normalized_goodput, 1e-9)
            * max(fairness, 1e-9)
            * max(delay_score, 1e-9)
            * max(wait_score, 1e-9)
            * max(safety_score, 1e-9)
        ) ** 0.2
        return (balanced, goodput, fairness)

    selected.sort(key=score, reverse=True)
    selected = selected[: max(args.limit, 0)]
    _write(args.output, selected)
    if not selected:
        print("no reward configuration matched the requested KPI constraints")
        return
    print("case_id | goodput | fairness | starvation | max_p99 | max_wait | pareto")
    for row in selected:
        print(
            f"{row.get('case_id', '')} | "
            f"{safe_float(row, 'mean_goodput_bits_per_slot'):,.1f} | "
            f"{safe_float(row, 'final_jain_fairness'):.4f} | "
            f"{safe_float(row, 'max_starvation_rate'):.6f} | "
            f"{safe_float(row, 'max_p99_wait_slots'):.1f} | "
            f"{safe_float(row, 'max_wait_slots'):.1f} | "
            f"{row.get('pareto_front', '')}"
        )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
