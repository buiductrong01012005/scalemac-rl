from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scalemac_rl import ScaleMacConfig
from scalemac_rl.checkpoints import require_checkpoint
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
    parser.add_argument("--safety-reserve-ues", type=int, default=None)
    parser.add_argument("--long-wait-threshold", type=float, default=None)
    parser.add_argument("--fixed-profile-seed", type=int, default=None)
    parser.add_argument("--freeze-static-profiles", action="store_true")
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidate_ablation.csv"))
    args = parser.parse_args()

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
    training = checkpoint.get("training", {})
    safety_reserve = int(
        args.safety_reserve_ues
        if args.safety_reserve_ues is not None
        else training.get("safety_reserve_ues", 16)
    )
    long_wait_threshold = float(
        args.long_wait_threshold
        if args.long_wait_threshold is not None
        else training.get("long_wait_threshold", 0.8)
    )
    freeze_static_profiles = bool(
        args.freeze_static_profiles or training.get("freeze_static_profiles", False)
    )
    fixed_profile_seed = (
        args.fixed_profile_seed
        if args.fixed_profile_seed is not None
        else training.get("fixed_profile_seed")
    )
    deadline_risk_start_ratio = float(training.get("deadline_risk_start_ratio", 0.60))
    deadline_risk_penalty_weight = float(
        training.get("deadline_risk_penalty_weight", 0.15)
    )
    constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
    )
    constraints.validate()
    config = ScaleMacConfig(
        num_ues=args.num_ues,
        max_selected_ues=min(64, args.num_ues),
        episode_slots=args.slots,
        safety_reserve_ues=min(safety_reserve, min(64, args.num_ues) - 1),
        safety_wait_threshold_ratio=long_wait_threshold,
        freeze_static_profiles=freeze_static_profiles,
        static_profile_seed=fixed_profile_seed,
        deadline_target_slots=args.max_p99_wait_slots,
        deadline_risk_start_ratio=deadline_risk_start_ratio,
        reward_deadline_risk_penalty_weight=deadline_risk_penalty_weight,
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
                long_wait_threshold=long_wait_threshold,
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
            "mean_safety_selected_count",
            "mean_forced_oldest_wait_count",
            "mean_learned_selected_count",
            "mean_learned_selection_fraction",
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
