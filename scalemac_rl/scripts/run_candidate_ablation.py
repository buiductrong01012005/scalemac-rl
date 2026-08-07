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


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("candidate counts must be positive integers")
    return values


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-count performance ablation")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--candidate-counts", type=_parse_ints, default=[64, 128, 256])
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--long-wait-threshold", type=float, default=0.8)
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidate_ablation.csv"))
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SharedSetActorCritic(
        input_dim=checkpoint.get("input_dim", 8), hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
    )
    constraints.validate()
    config = ScaleMacConfig(
        num_ues=args.num_ues,
        max_selected_ues=min(64, args.num_ues),
        episode_slots=args.slots,
    )
    config.validate()

    rows = []
    for candidate_count in args.candidate_counts:
        effective = min(max(candidate_count, config.max_selected_ues), config.num_ues)
        for offset in range(args.seeds):
            row = evaluate_actor_critic(
                model=model,
                device=device,
                config=config,
                seed=args.seed + offset,
                name=f"candidates_{effective}",
                max_candidates=effective,
                long_wait_threshold=args.long_wait_threshold,
                constraints=constraints,
            )
            row["candidate_setting"] = effective
            rows.append(row)
        print(f"completed candidate_count={effective}")

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL candidate-count performance ablation",
        description=(
            "The same checkpoint is evaluated with compact candidate sets of different sizes "
            "to expose the throughput/fairness/service trade-off."
        ),
        rows=rows,
    )
    summary = summarize_by_group(
        rows,
        group_key="candidate_setting",
        numeric_fields=[
            "mean_reward",
            "mean_goodput_bits_per_slot",
            "final_jain_fairness",
            "mean_starvation_rate",
            "max_starvation_rate",
            "max_p99_wait_slots",
            "mean_candidate_coverage",
            "mean_harq_retention_rate",
            "mean_long_wait_retention_rate",
            "mean_long_wait_missed_count",
            "mean_inference_us",
            "p99_inference_us",
        ],
    )
    summary_path = sibling_with_stem(args.output, "_summary", ".csv")
    write_csv(summary_path, summary)
    write_markdown(
        markdown_report_path(args.output, suffix="_summary"),
        title="ScaleMAC-RL candidate-count ablation summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )
    print(f"saved: {args.output}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
