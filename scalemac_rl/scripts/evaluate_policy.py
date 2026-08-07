from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.models import SharedSetPolicy
from scalemac_rl.reporting import (
    sibling_with_stem,
    summarize_by_group,
    write_csv,
    write_markdown,
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _evaluate_once(
    *,
    policy: SharedSetPolicy,
    device: torch.device,
    config: ScaleMacConfig,
    seed: int,
    checkpoint_name: str,
) -> dict[str, Any]:
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=seed)

    metrics: dict[str, list[float]] = {
        "reward_total": [],
        "cell_goodput_bits": [],
        "throughput_score": [],
        "fairness_score": [],
        "service_score": [],
        "starvation_rate": [],
        "reward_throughput_component": [],
        "reward_fairness_component": [],
        "reward_service_component": [],
        "reward_starvation_penalty": [],
    }
    inference_us: list[float] = []
    final_info: dict[str, Any] = {}

    with torch.inference_mode():
        while True:
            x = torch.from_numpy(observation).to(device)
            start_ns = perf_counter_ns()
            action = policy(x).cpu().numpy()
            inference_us.append((perf_counter_ns() - start_ns) / 1000.0)

            observation, _, terminated, truncated, final_info = env.step(action)
            for name in metrics:
                metrics[name].append(float(final_info[name]))
            if terminated or truncated:
                break

    return {
        "policy": checkpoint_name,
        "seed": seed,
        "num_ues": config.num_ues,
        "slots": config.episode_slots,
        "device": str(device),
        "mean_reward": mean(metrics["reward_total"]),
        "mean_goodput_bits_per_slot": mean(metrics["cell_goodput_bits"]),
        "mean_throughput_score": mean(metrics["throughput_score"]),
        "final_jain_fairness": float(final_info["jain_fairness"]),
        "mean_fairness_score": mean(metrics["fairness_score"]),
        "mean_service_score": mean(metrics["service_score"]),
        "mean_starvation_rate": mean(metrics["starvation_rate"]),
        "final_p99_wait_slots": float(final_info["p99_wait_slots"]),
        "mean_reward_throughput_component": mean(metrics["reward_throughput_component"]),
        "mean_reward_fairness_component": mean(metrics["reward_fairness_component"]),
        "mean_reward_service_component": mean(metrics["reward_service_component"]),
        "mean_reward_starvation_penalty": mean(metrics["reward_starvation_penalty"]),
        "mean_inference_us": mean(inference_us),
        "p95_inference_us": float(np.percentile(inference_us, 95)),
        "p99_inference_us": float(np.percentile(inference_us, 99)),
        "max_inference_us": max(inference_us),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101, help="first evaluation seed")
    parser.add_argument("--seeds", type=int, default=1, help="number of consecutive seeds")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.csv"))
    args = parser.parse_args()

    if args.seeds <= 0:
        parser.error("--seeds must be positive")

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = SharedSetPolicy(
        input_dim=checkpoint.get("input_dim", 8),
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    config = ScaleMacConfig(num_ues=args.num_ues, episode_slots=args.slots)
    config.validate()

    rows = [
        _evaluate_once(
            policy=policy,
            device=device,
            config=config,
            seed=args.seed + offset,
            checkpoint_name=args.checkpoint.name,
        )
        for offset in range(args.seeds)
    ]

    for row in rows:
        print(
            f"seed={row['seed']} reward={row['mean_reward']:.6f} "
            f"goodput={row['mean_goodput_bits_per_slot']:.1f} "
            f"fairness={row['final_jain_fairness']:.6f} "
            f"starvation={row['mean_starvation_rate']:.6f} "
            f"p99_inference_us={row['p99_inference_us']:.1f}"
        )

    write_csv(args.output, rows)
    write_markdown(
        args.output.with_suffix(".md"),
        title="ScaleMAC-RL policy evaluation",
        description=(
            "Evaluation results use mean goodput across all slots and expose each normalized reward component. "
            "Inference timing measures the neural policy call and tensor transfer in this Python surrogate."
        ),
        rows=rows,
        notes=(
            "The old `mean_last_slot_goodput_bits` label has been removed; goodput is now averaged over the full episode.",
            "Inference timing is machine-dependent and is not yet a 5G-LENA real-time guarantee.",
        ),
    )

    summary_csv = sibling_with_stem(args.output, "_summary", ".csv")
    summary_md = sibling_with_stem(args.output, "_summary", ".md")
    numeric_fields = [
        "mean_reward",
        "mean_goodput_bits_per_slot",
        "mean_throughput_score",
        "final_jain_fairness",
        "mean_fairness_score",
        "mean_service_score",
        "mean_starvation_rate",
        "final_p99_wait_slots",
        "mean_reward_throughput_component",
        "mean_reward_fairness_component",
        "mean_reward_service_component",
        "mean_reward_starvation_penalty",
        "mean_inference_us",
        "p95_inference_us",
        "p99_inference_us",
        "max_inference_us",
    ]
    summary = summarize_by_group(rows, group_key="policy", numeric_fields=numeric_fields)
    write_csv(summary_csv, summary)
    write_markdown(
        summary_md,
        title="ScaleMAC-RL policy evaluation summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )

    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_suffix('.md')}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
