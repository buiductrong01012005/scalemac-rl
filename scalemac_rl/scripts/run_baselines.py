from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.reporting import (
    markdown_report_path,
    sibling_with_stem,
    summarize_by_group,
    write_csv,
    write_markdown,
)
from scalemac_rl.schedulers import (
    MaxCqiScheduler,
    ProportionalFairScheduler,
    RoundRobinScheduler,
)


_TRACKED_METRICS = (
    "reward_total",
    "cell_goodput_bits",
    "throughput_score",
    "fairness_score",
    "short_term_jain_fairness",
    "service_score",
    "starvation_rate",
    "scheduling_starvation_rate",
    "p99_wait_slots",
    "max_wait_slots",
    "reward_throughput_component",
    "reward_fairness_component",
    "reward_service_component",
    "reward_starvation_penalty",
)


def evaluate(name: str, scheduler: Any, config: ScaleMacConfig, seed: int) -> dict[str, float | int | str]:
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=seed)
    scheduler.reset()

    history: dict[str, list[float]] = {metric: [] for metric in _TRACKED_METRICS}
    final_info: dict[str, Any] = {}

    while True:
        action = scheduler.act(observation)
        observation, _, terminated, truncated, final_info = env.step(action)
        for metric in _TRACKED_METRICS:
            history[metric].append(float(final_info[metric]))
        if terminated or truncated:
            break

    return {
        "scheduler": name,
        "seed": seed,
        "num_ues": config.num_ues,
        "slots": config.episode_slots,
        "mean_reward": mean(history["reward_total"]),
        "mean_goodput_bits_per_slot": mean(history["cell_goodput_bits"]),
        "mean_throughput_score": mean(history["throughput_score"]),
        "final_jain_fairness": float(final_info["jain_fairness"]),
        "mean_fairness_score": mean(history["fairness_score"]),
        "mean_short_term_jain_fairness": mean(history["short_term_jain_fairness"]),
        "mean_service_score": mean(history["service_score"]),
        "mean_starvation_rate": mean(history["starvation_rate"]),
        "mean_scheduling_starvation_rate": mean(history["scheduling_starvation_rate"]),
        "final_p99_wait_slots": float(final_info["p99_wait_slots"]),
        "final_max_wait_slots": float(final_info["max_wait_slots"]),
        "max_wait_slots": max(history["max_wait_slots"]),
        "mean_reward_throughput_component": mean(history["reward_throughput_component"]),
        "mean_reward_fairness_component": mean(history["reward_fairness_component"]),
        "mean_reward_service_component": mean(history["reward_service_component"]),
        "mean_reward_starvation_penalty": mean(history["reward_starvation_penalty"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baselines.csv"))
    args = parser.parse_args()

    if args.seeds <= 0:
        parser.error("--seeds must be positive")

    config = ScaleMacConfig(
        num_ues=args.num_ues, episode_slots=args.slots,
        scheduler_mode="ppo_only", force_harq_retransmissions=False,
    )
    config.validate()

    factories = {
        "rr": lambda: RoundRobinScheduler(config.max_selected_ues),
        "max_cqi": MaxCqiScheduler,
        "pf": ProportionalFairScheduler,
    }

    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        for name, factory in factories.items():
            row = evaluate(name, factory(), config, seed)
            rows.append(row)
            print(
                f"{name:8s} seed={seed} reward={row['mean_reward']:.4f} "
                f"goodput={row['mean_goodput_bits_per_slot']:.1f} "
                f"throughput_score={row['mean_throughput_score']:.4f} "
                f"fairness={row['final_jain_fairness']:.4f} "
                f"starvation={row['mean_starvation_rate']:.4f} "
                f"max_wait={row['max_wait_slots']:.1f}"
            )

    raw_md = markdown_report_path(args.output)
    summary_csv = sibling_with_stem(args.output, "_summary", ".csv")
    summary_md = markdown_report_path(args.output, suffix="_summary")

    write_csv(args.output, rows)
    write_markdown(
        raw_md,
        title="ScaleMAC-RL baseline runs",
        description=(
            "Per-seed results with normalized reward components. Throughput, fairness, "
            "and service scores are bounded to [0, 1]; starvation is reported as a separate constraint penalty."
        ),
        rows=rows,
        notes=(
            "Goodput is the mean successfully delivered bits per slot.",
            "A high reward with severe starvation should no longer be possible because starvation is penalized separately.",
        ),
    )

    numeric_fields = [
        "mean_reward",
        "mean_goodput_bits_per_slot",
        "mean_throughput_score",
        "final_jain_fairness",
        "mean_fairness_score",
        "mean_short_term_jain_fairness",
        "mean_service_score",
        "mean_starvation_rate",
        "mean_scheduling_starvation_rate",
        "final_p99_wait_slots",
        "final_max_wait_slots",
        "max_wait_slots",
        "mean_reward_throughput_component",
        "mean_reward_fairness_component",
        "mean_reward_service_component",
        "mean_reward_starvation_penalty",
    ]
    summary = summarize_by_group(
        rows,
        group_key="scheduler",
        numeric_fields=numeric_fields,
    )
    write_csv(summary_csv, summary)
    write_markdown(
        summary_md,
        title="ScaleMAC-RL baseline summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )

    print(f"saved: {args.output}")
    print(f"saved: {raw_md}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
