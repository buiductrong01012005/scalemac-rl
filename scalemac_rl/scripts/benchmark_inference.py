from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.checkpoints import require_checkpoint
from scalemac_rl.candidates import (
    build_candidate_mask,
    gather_candidates,
    scatter_candidate_action,
)
from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.projector import project_action
from scalemac_rl.reporting import markdown_report_path, write_csv, write_markdown


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive comma-separated numbers")
    return values


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive comma-separated integers")
    return values


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_us": float(array.mean()),
        "p50_us": float(np.percentile(array, 50)),
        "p95_us": float(np.percentile(array, 95)),
        "p99_us": float(np.percentile(array, 99)),
        "max_us": float(array.max()),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Component and candidate-count latency benchmark")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--candidate-counts", type=_parse_ints, default=[64, 128, 256])
    parser.add_argument("--max-candidates", type=int, default=None, help="Backward-compatible single-count override")
    parser.add_argument("--long-wait-threshold", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--deadlines-us", type=_parse_floats, default=[500.0, 1000.0])
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference_benchmark.csv"))
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.torch_threads <= 0:
        parser.error("warmup must be non-negative; repeats and torch threads must be positive")
    if args.max_candidates is not None:
        args.candidate_counts = [args.max_candidates]
    torch.set_num_threads(args.torch_threads)
    device = _resolve_device(args.device)
    try:
        checkpoint_path = require_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SharedSetActorCritic(
        input_dim=checkpoint.get("input_dim", 8), hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = ScaleMacConfig(
        num_ues=args.num_ues,
        max_selected_ues=min(64, args.num_ues),
        episode_slots=1,
    )
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=777)
    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for requested_count in args.candidate_counts:
            candidate_count = min(
                max(requested_count, config.max_selected_ues),
                config.num_ues,
            )
            mask = build_candidate_mask(
                observation,
                max_candidates=candidate_count,
                min_candidates=config.max_selected_ues,
                long_wait_threshold=args.long_wait_threshold,
            )
            compact_observation, indices = gather_candidates(observation, mask)

            for _ in range(args.warmup):
                x = torch.from_numpy(compact_observation).to(device)
                compact_action = model.deterministic_action(x).action.cpu().numpy()
                action = scatter_candidate_action(compact_action, indices, num_ues=config.num_ues)
                _ = project_action(
                    action,
                    eligible=env.eligible,
                    harq_pending=env.harq_pending,
                    harq_retx_count=env.harq_retx_count,
                    time_since_service=env.time_since_service,
                    num_prbs=config.num_prbs,
                    max_selected_ues=config.max_selected_ues,
                )
            _sync(device)

            timings: dict[str, list[float]] = {
                "candidate_filter": [],
                "candidate_gather": [],
                "tensor_conversion": [],
                "encoder_context": [],
                "actor_head": [],
                "scatter_and_project": [],
                "end_to_end": [],
            }

            for _ in range(args.repeats):
                total_start = perf_counter_ns()

                start = perf_counter_ns()
                current_mask = build_candidate_mask(
                    observation,
                    max_candidates=candidate_count,
                    min_candidates=config.max_selected_ues,
                    long_wait_threshold=args.long_wait_threshold,
                )
                timings["candidate_filter"].append((perf_counter_ns() - start) / 1000.0)

                start = perf_counter_ns()
                current_compact, current_indices = gather_candidates(observation, current_mask)
                timings["candidate_gather"].append((perf_counter_ns() - start) / 1000.0)

                start = perf_counter_ns()
                x = torch.from_numpy(current_compact).to(device)
                _sync(device)
                timings["tensor_conversion"].append((perf_counter_ns() - start) / 1000.0)

                start = perf_counter_ns()
                actor_features, _ = model.encode_features(x)
                _sync(device)
                timings["encoder_context"].append((perf_counter_ns() - start) / 1000.0)

                start = perf_counter_ns()
                compact_action_tensor = model.actor(actor_features).clamp(1e-4, 1.0 - 1e-4)
                compact_action = compact_action_tensor.squeeze(0).cpu().numpy()
                _sync(device)
                timings["actor_head"].append((perf_counter_ns() - start) / 1000.0)

                start = perf_counter_ns()
                action = scatter_candidate_action(
                    compact_action,
                    current_indices,
                    num_ues=config.num_ues,
                )
                _ = project_action(
                    action,
                    eligible=env.eligible,
                    harq_pending=env.harq_pending,
                    harq_retx_count=env.harq_retx_count,
                    time_since_service=env.time_since_service,
                    num_prbs=config.num_prbs,
                    max_selected_ues=config.max_selected_ues,
                )
                timings["scatter_and_project"].append((perf_counter_ns() - start) / 1000.0)
                timings["end_to_end"].append((perf_counter_ns() - total_start) / 1000.0)

            for component, values in timings.items():
                row: dict[str, object] = {
                    "component": component,
                    "checkpoint": args.checkpoint.name,
                    "num_ues": args.num_ues,
                    "max_candidates": candidate_count,
                    "device": str(device),
                    "torch_threads": args.torch_threads,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    **_summary(values),
                }
                for deadline in args.deadlines_us:
                    row[f"deadline_miss_rate_{deadline:g}us"] = float(
                        np.mean(np.asarray(values) > deadline)
                    )
                rows.append(row)
                if component == "end_to_end":
                    print(
                        f"candidates={candidate_count:3d} mean={row['mean_us']:.1f}us "
                        f"p99={row['p99_us']:.1f}us max={row['max_us']:.1f}us"
                    )

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL component inference benchmark",
        description=(
            "Compact candidate-set inference is profiled by pipeline component and candidate count. "
            "Results are machine-specific and do not prove 5G-LENA real-time compliance."
        ),
        rows=rows,
        notes=(
            "The deployed actor path skips Beta distribution and critic computation.",
            "Use the same torch thread count when comparing machines.",
        ),
    )
    print(f"saved: {args.output}")
    print(f"saved: {markdown_report_path(args.output)}")


if __name__ == "__main__":
    main()
