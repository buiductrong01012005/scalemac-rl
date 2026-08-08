from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from scalemac_rl.reporting import markdown_report_path, write_csv, write_markdown


_KPI_FIELDS = (
    "mean_reward",
    "mean_goodput_bits_per_slot",
    "final_jain_fairness",
    "mean_starvation_rate",
    "max_starvation_rate",
    "final_p99_wait_slots",
    "max_p99_wait_slots",
    "final_max_wait_slots",
    "max_wait_slots",
)
_MATCH_FIELDS = (
    "checkpoint_sha256",
    "evaluation_protocol_hash",
    "scenario_hash",
    "scheduler_runtime_hash",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in _MATCH_FIELDS)


def compare_rows(
    standalone_rows: list[dict[str, str]],
    unified_rows: list[dict[str, str]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    unified_by_key = {_key(row): row for row in unified_rows if row.get("checkpoint_sha256")}
    results: list[dict[str, Any]] = []
    for standalone in standalone_rows:
        key = _key(standalone)
        unified = unified_by_key.get(key)
        if unified is None:
            results.append(
                {
                    "standalone_method": standalone.get("method", ""),
                    "unified_method": "",
                    "checkpoint_sha256": standalone.get("checkpoint_sha256", ""),
                    "scenario_hash": standalone.get("scenario_hash", ""),
                    "matched": False,
                    "consistent": False,
                    "max_absolute_delta": "",
                    "reason": "no row with matching checkpoint/protocol/scenario/runtime hashes",
                }
            )
            continue
        deltas = {
            field: abs(float(standalone[field]) - float(unified[field]))
            for field in _KPI_FIELDS
        }
        maximum = max(deltas.values(), default=0.0)
        result: dict[str, Any] = {
            "standalone_method": standalone.get("method", ""),
            "unified_method": unified.get("method", ""),
            "checkpoint_sha256": standalone.get("checkpoint_sha256", ""),
            "scenario_hash": standalone.get("scenario_hash", ""),
            "matched": True,
            "consistent": maximum <= tolerance,
            "max_absolute_delta": maximum,
            "reason": "" if maximum <= tolerance else "KPI mismatch",
        }
        result.update({f"delta_{field}": value for field, value in deltas.items()})
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that standalone evaluate_ppo and unified cross-scheduler "
            "evaluation produce the same KPI row for a checkpoint."
        )
    )
    parser.add_argument("standalone_csv", type=Path)
    parser.add_argument("unified_csv", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation_consistency.csv"),
    )
    args = parser.parse_args()
    if args.tolerance < 0.0:
        parser.error("tolerance must be non-negative")

    results = compare_rows(
        _read(args.standalone_csv),
        _read(args.unified_csv),
        tolerance=args.tolerance,
    )
    if not results:
        parser.error("standalone CSV contains no rows")
    write_csv(args.output, results)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL evaluation consistency check",
        description=(
            "KPI equality is checked only after checkpoint, protocol, scenario, "
            "and scheduler-runtime hashes match. Inference timing is excluded."
        ),
        rows=results,
    )
    for row in results:
        print(
            f"checkpoint={row['checkpoint_sha256'][:12]} "
            f"matched={row['matched']} consistent={row['consistent']} "
            f"max_delta={row['max_absolute_delta']}"
        )
    print(f"saved: {args.output}")
    if not all(bool(row["consistent"]) for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
