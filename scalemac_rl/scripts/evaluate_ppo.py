from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scalemac_rl import ScaleMacConfig
from scalemac_rl.constraints import ServiceConstraints
from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.reporting import (
    markdown_report_path,
    sibling_with_stem,
    summarize_by_group,
    write_csv,
    write_markdown,
)
from scalemac_rl.rl_evaluation import evaluate_actor_critic


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--long-wait-threshold", type=float, default=0.8)
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo_evaluation.csv"))
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SharedSetActorCritic(
        input_dim=checkpoint.get("input_dim", 8),
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    config = ScaleMacConfig(
        num_ues=args.num_ues,
        max_selected_ues=min(64, args.num_ues),
        episode_slots=args.slots,
    )
    config.validate()
    constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
    )
    constraints.validate()

    rows = [
        evaluate_actor_critic(
            model=model,
            device=device,
            config=config,
            seed=args.seed + offset,
            name=args.checkpoint.stem,
            max_candidates=args.max_candidates,
            long_wait_threshold=args.long_wait_threshold,
            constraints=constraints,
        )
        for offset in range(args.seeds)
    ]
    for row in rows:
        print(
            f"seed={row['seed']} reward={row['mean_reward']:.6f} "
            f"goodput={row['mean_goodput_bits_per_slot']:.1f} "
            f"fairness={row['final_jain_fairness']:.6f} "
            f"starvation={row['mean_starvation_rate']:.6f} "
            f"p99_inference_us={row['p99_inference_us']:.1f} "
            f"feasible={row['constraint_feasible']}"
        )

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL PPO evaluation",
        description="Deterministic evaluation of the PPO actor after curriculum fine-tuning.",
        rows=rows,
        notes=("This is fast-surrogate evaluation, not 5G-LENA validation.",),
    )
    summary = summarize_by_group(
        rows,
        group_key="method",
        numeric_fields=[
            "mean_reward",
            "mean_goodput_bits_per_slot",
            "mean_throughput_score",
            "final_jain_fairness",
            "mean_fairness_score",
            "mean_service_score",
            "mean_starvation_rate",
            "max_starvation_rate",
            "final_p99_wait_slots",
            "max_p99_wait_slots",
            "mean_candidate_coverage",
            "mean_harq_retention_rate",
            "mean_long_wait_retention_rate",
            "mean_long_wait_missed_count",
            "mean_inference_us",
            "p95_inference_us",
            "p99_inference_us",
            "max_inference_us",
        ],
    )
    summary_csv = sibling_with_stem(args.output, "_summary", ".csv")
    write_csv(summary_csv, summary)
    write_markdown(
        markdown_report_path(args.output, suffix="_summary"),
        title="ScaleMAC-RL PPO evaluation summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )
    print(f"saved: {args.output}")
    print(f"saved: {markdown_report_path(args.output)}")
    print(f"saved: {summary_csv}")
    print(f"saved: {markdown_report_path(args.output, suffix="_summary")}")


if __name__ == "__main__":
    main()
