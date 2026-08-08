from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
from scalemac_rl.schedulers import (
    MaxCqiScheduler,
    ProportionalFairScheduler,
    RoundRobinScheduler,
    RuleOnlyScheduler,
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _load_model(path: Path, device: torch.device) -> tuple[SharedSetActorCritic, dict[str, Any]]:
    checkpoint_path = require_checkpoint(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SharedSetActorCritic(input_dim=10, hidden_dim=checkpoint["hidden_dim"]).to(device)
    model.load_compatible_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _learned_config(
    *,
    checkpoint: dict[str, Any],
    num_ues: int,
    slots: int,
    fixed_profile_seed: int,
    scheduler_mode: str,
    candidate_mode: str,
) -> tuple[ScaleMacConfig, int, float]:
    training = checkpoint.get("training", {})
    reserve = int(training.get("safety_reserve_ues", 16))
    if scheduler_mode == "ppo_only":
        reserve = 0
    config = ScaleMacConfig(
        num_ues=num_ues,
        max_selected_ues=min(64, num_ues),
        episode_slots=slots,
        scheduler_mode=scheduler_mode,
        force_harq_retransmissions=bool(
            training.get("force_harq_retransmissions", scheduler_mode == "hybrid")
        ),
        safety_reserve_ues=min(reserve, min(64, num_ues)),
        safety_wait_threshold_ratio=float(training.get("long_wait_threshold", 0.8)),
        freeze_static_profiles=True,
        static_profile_seed=fixed_profile_seed,
        deadline_target_slots=50.0,
        reference_deadline_target_slots=50.0,
        deadline_risk_start_ratio=float(training.get("deadline_risk_start_ratio", 0.60)),
        reward_deadline_risk_penalty_weight=float(
            training.get("deadline_risk_penalty_weight", 0.15)
        ),
        reward_max_wait_risk_penalty_weight=float(
            training.get("max_wait_risk_penalty_weight", 0.10)
        ),
        starvation_threshold_slots=int(training.get("starvation_threshold_slots", 64)),
        reward_throughput_weight=float(training.get("reward_throughput_weight", 0.45)),
        reward_fairness_weight=float(training.get("reward_fairness_weight", 0.35)),
        reward_service_weight=float(training.get("reward_service_weight", 0.15)),
        reward_deficit_service_weight=float(
            training.get("reward_deficit_service_weight", 0.05)
        ),
        reward_fairness_delta_weight=float(
            training.get("reward_fairness_delta_weight", 0.03)
        ),
        reward_pf_utility_delta_weight=float(
            training.get("reward_pf_utility_delta_weight", 0.02)
        ),
        max_wait_target_slots=60.0,
    )
    config.validate()
    max_candidates = num_ues if candidate_mode == "all" else int(
        training.get("max_candidates", 128)
    )
    return config, max_candidates, float(training.get("long_wait_threshold", 0.8))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate rule, PPO, candidate-filter, and projector contributions"
    )
    parser.add_argument("--hybrid-checkpoint", type=Path)
    parser.add_argument("--ppo-candidate-checkpoint", type=Path)
    parser.add_argument("--ppo-full-checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/scheduler_attribution.csv")
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    constraints = ServiceConstraints(
        max_starvation_rate=0.0,
        max_p99_wait_slots=50.0,
        min_jain_fairness=0.60,
        max_wait_slots=60.0,
    )
    base = ScaleMacConfig(
        num_ues=args.num_ues,
        max_selected_ues=min(64, args.num_ues),
        episode_slots=args.slots,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        safety_reserve_ues=0,
        freeze_static_profiles=True,
        static_profile_seed=args.seed,
        reward_throughput_weight=0.45,
        reward_fairness_weight=0.35,
        reward_service_weight=0.15,
        reward_deficit_service_weight=0.05,
        reward_fairness_delta_weight=0.03,
        reward_pf_utility_delta_weight=0.02,
        starvation_threshold_slots=64,
        deadline_target_slots=50.0,
        reference_deadline_target_slots=50.0,
        max_wait_target_slots=60.0,
    )
    rule_config = ScaleMacConfig(**base.to_dict())
    rule_config.scheduler_mode = "rule_only"
    rule_config.force_harq_retransmissions = True
    rule_config.safety_reserve_ues = min(64, args.num_ues)

    learned: list[tuple[str, Path, str, str]] = []
    if args.hybrid_checkpoint:
        learned.append(("hybrid_ppo", args.hybrid_checkpoint, "hybrid", "heuristic"))
    if args.ppo_candidate_checkpoint:
        learned.append(
            ("ppo_only_candidate128", args.ppo_candidate_checkpoint, "ppo_only", "heuristic")
        )
    if args.ppo_full_checkpoint:
        learned.append(("ppo_from_scratch_full1200", args.ppo_full_checkpoint, "ppo_only", "all"))

    rows: list[dict[str, Any]] = []
    for offset in range(args.seeds):
        seed = args.seed + offset
        rows.extend(
            [
                evaluate_scheduler(
                    scheduler=RoundRobinScheduler(base.max_selected_ues),
                    config=base,
                    seed=seed,
                    name="rr",
                    constraints=constraints,
                ),
                evaluate_scheduler(
                    scheduler=ProportionalFairScheduler(),
                    config=base,
                    seed=seed,
                    name="pf",
                    constraints=constraints,
                ),
                evaluate_scheduler(
                    scheduler=MaxCqiScheduler(),
                    config=base,
                    seed=seed,
                    name="max_cqi",
                    constraints=constraints,
                ),
                evaluate_scheduler(
                    scheduler=RuleOnlyScheduler(),
                    config=rule_config,
                    seed=seed,
                    name="rule_only",
                    constraints=constraints,
                ),
            ]
        )
        for name, path, scheduler_mode, candidate_mode in learned:
            model, checkpoint = _load_model(path, device)
            config, max_candidates, long_wait = _learned_config(
                checkpoint=checkpoint,
                num_ues=args.num_ues,
                slots=args.slots,
                fixed_profile_seed=args.seed,
                scheduler_mode=scheduler_mode,
                candidate_mode=candidate_mode,
            )
            rows.append(
                evaluate_actor_critic(
                    model=model,
                    device=device,
                    config=config,
                    seed=seed,
                    name=name,
                    max_candidates=max_candidates,
                    candidate_mode=candidate_mode,
                    long_wait_threshold=long_wait,
                    constraints=constraints,
                )
            )
        print(f"completed attribution seed={seed}")

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL scheduler attribution",
        description=(
            "RR, PF, Max-CQI, rule-only, hybrid PPO, candidate PPO-only, and full PPO "
            "are compared under identical scenario seeds when their checkpoints are supplied."
        ),
        rows=rows,
    )
    summary = summarize_by_group(
        rows,
        group_key="method",
        numeric_fields=[
            "mean_reward",
            "mean_goodput_bits_per_slot",
            "final_jain_fairness",
            "mean_starvation_rate",
            "max_starvation_rate",
            "final_p99_wait_slots",
            "max_p99_wait_slots",
            "max_wait_slots",
            "mean_safety_selected_count",
            "mean_scheduler_selected_count",
            "mean_ppo_selected_count",
            "mean_rule_selected_count",
        ],
    )
    summary_path = sibling_with_stem(args.output, "_summary", ".csv")
    write_csv(summary_path, summary)
    write_markdown(
        markdown_report_path(args.output, suffix="_summary"),
        title="ScaleMAC-RL scheduler attribution summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )
    print(f"saved: {args.output}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
