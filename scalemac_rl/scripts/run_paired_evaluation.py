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
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-mode", choices=["heuristic", "all"], default=None)
    parser.add_argument(
        "--scheduler-mode", choices=["hybrid", "ppo_only", "rule_only"], default=None
    )
    parser.add_argument(
        "--force-harq-retransmissions", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--safety-reserve-ues", type=int, default=None)
    parser.add_argument("--long-wait-threshold", type=float, default=None)
    parser.add_argument("--fixed-profile-seed", type=int, default=None)
    parser.add_argument("--freeze-static-profiles", action="store_true")
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--min-jain-fairness", type=float, default=0.60)
    parser.add_argument("--max-wait-slots", type=float, default=60.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/paired_evaluation.csv"))
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
    max_candidates = int(args.max_candidates or training.get("max_candidates", 128))
    candidate_mode = str(args.candidate_mode or training.get("candidate_mode", "heuristic"))
    scheduler_mode = str(args.scheduler_mode or training.get("scheduler_mode", "hybrid"))
    force_harq = bool(
        training.get("force_harq_retransmissions", True)
        if args.force_harq_retransmissions is None
        else args.force_harq_retransmissions
    )
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
    baseline_config = ScaleMacConfig(
        num_ues=args.num_ues,
        max_selected_ues=min(64, args.num_ues),
        episode_slots=args.slots,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        safety_reserve_ues=0,
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
    baseline_config.validate()
    ppo_config = ScaleMacConfig(
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
    ppo_config.validate()
    constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
        min_jain_fairness=args.min_jain_fairness,
        max_wait_slots=args.max_wait_slots,
    )
    constraints.validate()

    rows = []
    for offset in range(args.seeds):
        seed = args.seed + offset
        rows.extend(
            [
                evaluate_scheduler(
                    scheduler=RoundRobinScheduler(baseline_config.max_selected_ues),
                    config=baseline_config,
                    seed=seed,
                    name="rr",
                    constraints=constraints,
                ),
                evaluate_scheduler(
                    scheduler=MaxCqiScheduler(), config=baseline_config, seed=seed, name="max_cqi", constraints=constraints
                ),
                evaluate_scheduler(
                    scheduler=ProportionalFairScheduler(), config=baseline_config, seed=seed, name="pf", constraints=constraints
                ),
                evaluate_actor_critic(
                    model=model,
                    device=device,
                    config=ppo_config,
                    seed=seed,
                    name=f"ppo_{scheduler_mode}",
                    max_candidates=(args.num_ues if candidate_mode == "all" else max_candidates),
                    candidate_mode=candidate_mode,
                    long_wait_threshold=long_wait_threshold,
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
            "mean_short_term_jain_fairness",
            "mean_service_score",
            "mean_deadline_risk",
            "mean_tail_mean_wait_slots",
            "mean_starvation_rate",
            "max_starvation_rate",
            "final_p99_wait_slots",
            "max_p99_wait_slots",
            "final_max_wait_slots",
            "max_wait_slots",
            "max_scheduling_wait_slots",
            "mean_near_deadline_rate",
            "mean_scheduling_starvation_rate",
            "mean_safety_selected_count",
            "mean_forced_oldest_wait_count",
            "mean_scheduler_selected_count",
            "mean_scheduler_selection_fraction",
            "mean_ppo_selected_count",
            "mean_rule_selected_count",
            "mean_learned_selected_count",
            "mean_learned_selection_fraction",
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
