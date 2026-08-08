from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scalemac_rl.checkpoints import require_checkpoint
from scalemac_rl.evaluation_protocol import (
    UnifiedEvaluationProtocol,
    learned_provenance,
    load_policy_checkpoint,
    resolve_policy_runtime,
)
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


def _manifest_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = (
        "method",
        "seed",
        "evaluation_protocol_version",
        "evaluation_protocol_hash",
        "evaluation_reward_version",
        "scenario_hash",
        "scheduler_runtime_hash",
        "scheduler_runtime_json",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_type",
        "checkpoint_tag",
        "checkpoint_training_reward_version",
        "checkpoint_training_reward_signature",
        "checkpoint_observation_features",
        "evaluation_observation_features",
        "compatibility_adapter_applied",
    )
    return [{key: row.get(key, "") for key in keys} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one PPO checkpoint under the same fixed protocol used by "
            "scheduler attribution. Checkpoint training reward weights are provenance only."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--profile-seed", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-mode", choices=["heuristic", "all"], default=None)
    parser.add_argument(
        "--scheduler-mode", choices=["hybrid", "ppo_only", "rule_only"], default=None
    )
    parser.add_argument(
        "--force-harq-retransmissions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--safety-reserve-ues", type=int, default=None)
    parser.add_argument("--long-wait-threshold", type=float, default=None)
    # Backward-compatible alias. The unified evaluator always freezes one static
    # profile; this flag is accepted so old command lines continue to work.
    parser.add_argument("--fixed-profile-seed", type=int, default=None)
    parser.add_argument("--freeze-static-profiles", action="store_true")
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--min-jain-fairness", type=float, default=0.60)
    parser.add_argument("--max-wait-slots", type=float, default=60.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo_evaluation.csv"))
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/ppo_evaluation_manifest.csv"),
    )
    args = parser.parse_args()

    if args.seeds <= 0:
        parser.error("seeds must be positive")
    profile_seed = (
        args.profile_seed
        if args.profile_seed is not None
        else args.fixed_profile_seed
        if args.fixed_profile_seed is not None
        else args.seed
    )

    device = _resolve_device(args.device)
    try:
        checkpoint_path = require_checkpoint(args.checkpoint)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    model, checkpoint = load_policy_checkpoint(checkpoint_path, device)
    runtime = resolve_policy_runtime(
        checkpoint,
        num_ues=args.num_ues,
        scheduler_mode=args.scheduler_mode,
        candidate_mode=args.candidate_mode,
        max_candidates=args.max_candidates,
        safety_reserve_ues=args.safety_reserve_ues,
        force_harq_retransmissions=args.force_harq_retransmissions,
        long_wait_threshold=args.long_wait_threshold,
    )
    protocol = UnifiedEvaluationProtocol(
        num_ues=args.num_ues,
        slots=args.slots,
        profile_seed=profile_seed,
        max_starvation_rate=args.max_starvation_rate,
        p99_wait_target_slots=args.max_p99_wait_slots,
        min_jain_fairness=args.min_jain_fairness,
        max_wait_target_slots=args.max_wait_slots,
    )
    protocol.validate()
    config = protocol.build_config(
        scheduler_mode=runtime.scheduler_mode,
        safety_reserve_ues=runtime.safety_reserve_ues,
        force_harq_retransmissions=runtime.force_harq_retransmissions,
        safety_wait_threshold_ratio=runtime.long_wait_threshold,
    )
    constraints = protocol.constraints()

    rows: list[dict[str, object]] = []
    for offset in range(args.seeds):
        seed = args.seed + offset
        row = evaluate_actor_critic(
            model=model,
            device=device,
            config=config,
            seed=seed,
            name=args.checkpoint.stem,
            max_candidates=runtime.max_candidates,
            candidate_mode=runtime.candidate_mode,
            long_wait_threshold=runtime.long_wait_threshold,
            constraints=constraints,
        )
        row.update(
            learned_provenance(
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                protocol=protocol,
                runtime=runtime,
                rollout_seed=seed,
            )
        )
        rows.append(row)
        print(
            f"seed={row['seed']} reward={row['mean_reward']:.6f} "
            f"goodput={row['mean_goodput_bits_per_slot']:.1f} "
            f"fairness={row['final_jain_fairness']:.6f} "
            f"starvation={row['mean_starvation_rate']:.6f} "
            f"max_wait={row['max_wait_slots']:.1f} "
            f"p99_inference_us={row['p99_inference_us']:.1f} "
            f"feasible={row['constraint_feasible']}"
        )

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL unified PPO evaluation",
        description=(
            "Deterministic PPO evaluation under the fixed unified protocol also "
            "used by scheduler attribution."
        ),
        rows=rows,
        notes=(
            "Checkpoint training reward fields are provenance only; all methods use the same evaluation reward and constraints.",
            f"Evaluation protocol hash: {protocol.protocol_hash}",
            "This is fast-surrogate evaluation, not 5G-LENA validation.",
        ),
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
            "mean_deficit_service_score",
            "mean_fairness_progress",
            "mean_pf_utility_progress",
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
            "p95_inference_us",
            "p99_inference_us",
            "max_inference_us",
        ],
    )
    summary_csv = sibling_with_stem(args.output, "_summary", ".csv")
    write_csv(summary_csv, summary)
    write_markdown(
        markdown_report_path(args.output, suffix="_summary"),
        title="ScaleMAC-RL unified PPO evaluation summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )
    write_csv(args.manifest_output, _manifest_rows(rows))
    write_markdown(
        markdown_report_path(args.manifest_output),
        title="ScaleMAC-RL unified PPO evaluation manifest",
        description="Hashes and provenance required to reproduce this evaluation.",
        rows=_manifest_rows(rows),
    )
    print(f"protocol_hash={protocol.protocol_hash}")
    print(f"checkpoint_sha256={rows[0]['checkpoint_sha256']}")
    print(f"saved: {args.output}")
    print(f"saved: {markdown_report_path(args.output)}")
    print(f"saved: {summary_csv}")
    print(f"saved: {markdown_report_path(args.output, suffix='_summary')}")
    print(f"saved: {args.manifest_output}")


if __name__ == "__main__":
    main()
