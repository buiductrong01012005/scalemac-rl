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
    parser.add_argument("--min-jain-fairness", type=float, default=0.60)
    parser.add_argument("--max-wait-slots", type=float, default=60.0)
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
        input_dim=10, hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_compatible_state_dict(checkpoint["model_state_dict"], strict=True)
    training = checkpoint.get("training", {})
    scheduler_mode = str(training.get("scheduler_mode", "hybrid"))
    force_harq = bool(training.get("force_harq_retransmissions", True))
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
    max_wait_risk_penalty_weight = float(
        training.get("max_wait_risk_penalty_weight", 0.10)
    )
    starvation_threshold_slots = int(training.get("starvation_threshold_slots", 64))
    reward_throughput_weight = float(training.get("reward_throughput_weight", 0.50))
    reward_fairness_weight = float(training.get("reward_fairness_weight", 0.35))
    reward_schedule_fairness_weight = float(training.get("reward_schedule_fairness_weight", 0.0))
    reward_service_weight = float(training.get("reward_service_weight", 0.15))
    reward_deficit_service_weight = float(training.get("reward_deficit_service_weight", 0.0))
    reward_fairness_delta_weight = float(training.get("reward_fairness_delta_weight", 0.0))
    reward_pf_utility_delta_weight = float(training.get("reward_pf_utility_delta_weight", 0.0))
    constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
        min_jain_fairness=args.min_jain_fairness,
        max_wait_slots=args.max_wait_slots,
    )
    constraints.validate()
    config = ScaleMacConfig(
        num_ues=args.num_ues,
        max_selected_ues=min(64, args.num_ues),
        episode_slots=args.slots,
        scheduler_mode=scheduler_mode,
        force_harq_retransmissions=force_harq,
        safety_reserve_ues=(
            0 if scheduler_mode == "ppo_only"
            else min(safety_reserve, min(64, args.num_ues))
        ),
        safety_wait_threshold_ratio=long_wait_threshold,
        freeze_static_profiles=freeze_static_profiles,
        static_profile_seed=fixed_profile_seed,
        deadline_target_slots=args.max_p99_wait_slots,
        deadline_risk_start_ratio=deadline_risk_start_ratio,
        reward_deadline_risk_penalty_weight=deadline_risk_penalty_weight,
        reward_max_wait_risk_penalty_weight=max_wait_risk_penalty_weight,
        starvation_threshold_slots=starvation_threshold_slots,
        reward_throughput_weight=reward_throughput_weight,
        reward_fairness_weight=reward_fairness_weight,
        reward_schedule_fairness_weight=reward_schedule_fairness_weight,
        reward_service_weight=reward_service_weight,
        reward_deficit_service_weight=reward_deficit_service_weight,
        reward_fairness_delta_weight=reward_fairness_delta_weight,
        reward_pf_utility_delta_weight=reward_pf_utility_delta_weight,
        max_wait_target_slots=args.max_wait_slots,
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
                candidate_mode="heuristic",
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
            "final_max_wait_slots",
            "max_wait_slots",
            "max_scheduling_wait_slots",
            "mean_near_deadline_rate",
            "mean_scheduling_starvation_rate",
            "mean_candidate_coverage",
            "mean_harq_retention_rate",
            "mean_long_wait_retention_rate",
            "mean_long_wait_missed_count",
            "mean_safety_selected_count",
            "mean_forced_oldest_wait_count",
            "mean_scheduler_selected_count",
            "mean_scheduler_selection_fraction",
            "mean_ppo_selected_count",
            "mean_rule_selected_count",
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
