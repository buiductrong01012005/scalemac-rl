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
from scalemac_rl.rl_evaluation import evaluate_actor_critic, evaluate_scheduler
from scalemac_rl.schedulers import MaxCqiScheduler, ProportionalFairScheduler, RoundRobinScheduler


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired baseline-versus-PPO evaluation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--long-wait-threshold", type=float, default=0.8)
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/paired_evaluation.csv"))
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SharedSetActorCritic(
        input_dim=checkpoint.get("input_dim", 8), hidden_dim=checkpoint["hidden_dim"]
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

    rows = []
    for offset in range(args.seeds):
        seed = args.seed + offset
        rows.extend(
            [
                evaluate_scheduler(
                    scheduler=RoundRobinScheduler(config.max_selected_ues),
                    config=config,
                    seed=seed,
                    name="rr",
                    constraints=constraints,
                ),
                evaluate_scheduler(
                    scheduler=MaxCqiScheduler(), config=config, seed=seed, name="max_cqi", constraints=constraints
                ),
                evaluate_scheduler(
                    scheduler=ProportionalFairScheduler(), config=config, seed=seed, name="pf", constraints=constraints
                ),
                evaluate_actor_critic(
                    model=model,
                    device=device,
                    config=config,
                    seed=seed,
                    name="ppo",
                    max_candidates=args.max_candidates,
                    long_wait_threshold=args.long_wait_threshold,
                    constraints=constraints,
                ),
            ]
        )
        print(f"completed paired seed={seed}")

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL paired evaluation",
        description=(
            "RR, Max-CQI, PF, and PPO are evaluated with the same scenario seeds. "
            "This reduces variation from CQI profiles and HARQ random streams."
        ),
        rows=rows,
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
        ],
    )
    summary_csv = sibling_with_stem(args.output, "_summary", ".csv")
    summary_md = markdown_report_path(args.output, suffix="_summary")
    write_csv(summary_csv, summary)
    write_markdown(
        summary_md,
        title="ScaleMAC-RL paired evaluation summary",
        description=f"Mean and sample standard deviation across {args.seeds} paired seed(s).",
        rows=summary,
    )
    print(f"saved: {args.output}")
    print(f"saved: {markdown_report_path(args.output)}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
