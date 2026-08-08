from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from scalemac_rl.checkpoints import require_checkpoint
from scalemac_rl.evaluation_protocol import (
    UnifiedEvaluationProtocol,
    classical_provenance,
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


def _manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _classical_row(
    *,
    scheduler: Any,
    config: Any,
    seed: int,
    name: str,
    protocol: UnifiedEvaluationProtocol,
) -> dict[str, Any]:
    row = evaluate_scheduler(
        scheduler=scheduler,
        config=config,
        seed=seed,
        name=name,
        constraints=protocol.constraints(),
    )
    row.update(
        classical_provenance(
            protocol=protocol,
            scheduler_name=name,
            scheduler_mode=config.scheduler_mode,
            rollout_seed=seed,
        )
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare classical, rule-only, hybrid PPO, and PPO-only schedulers "
            "under one fixed environment/KPI/reward protocol."
        )
    )
    parser.add_argument("--hybrid-checkpoint", type=Path)
    parser.add_argument("--ppo-candidate-checkpoint", type=Path)
    parser.add_argument("--ppo-full-checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--profile-seed", type=int, default=None)
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--min-jain-fairness", type=float, default=0.60)
    parser.add_argument("--max-wait-slots", type=float, default=60.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/scheduler_attribution.csv")
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/scheduler_attribution_manifest.csv"),
    )
    args = parser.parse_args()

    if args.seeds <= 0:
        parser.error("seeds must be positive")
    profile_seed = args.seed if args.profile_seed is None else args.profile_seed
    device = _resolve_device(args.device)
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

    baseline_config = protocol.build_config(
        scheduler_mode="ppo_only",
        safety_reserve_ues=0,
        force_harq_retransmissions=False,
    )
    rule_config = protocol.build_config(
        scheduler_mode="rule_only",
        safety_reserve_ues=min(64, args.num_ues),
        force_harq_retransmissions=True,
    )

    learned_specs: list[tuple[str, Path, str, str]] = []
    if args.hybrid_checkpoint:
        learned_specs.append(
            ("hybrid_ppo", args.hybrid_checkpoint, "hybrid", "heuristic")
        )
    if args.ppo_candidate_checkpoint:
        learned_specs.append(
            (
                "ppo_only_candidate128",
                args.ppo_candidate_checkpoint,
                "ppo_only",
                "heuristic",
            )
        )
    if args.ppo_full_checkpoint:
        learned_specs.append(
            (
                "ppo_from_scratch_full1200",
                args.ppo_full_checkpoint,
                "ppo_only",
                "all",
            )
        )

    loaded: list[
        tuple[str, Path, Any, dict[str, Any], Any]
    ] = []
    for name, path, scheduler_mode, candidate_mode in learned_specs:
        try:
            checkpoint_path = require_checkpoint(path)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        model, checkpoint = load_policy_checkpoint(checkpoint_path, device)
        runtime = resolve_policy_runtime(
            checkpoint,
            num_ues=args.num_ues,
            scheduler_mode=scheduler_mode,
            candidate_mode=candidate_mode,
        )
        loaded.append((name, checkpoint_path, model, checkpoint, runtime))

    rows: list[dict[str, Any]] = []
    for offset in range(args.seeds):
        seed = args.seed + offset
        rows.extend(
            [
                _classical_row(
                    scheduler=RoundRobinScheduler(baseline_config.max_selected_ues),
                    config=baseline_config,
                    seed=seed,
                    name="rr",
                    protocol=protocol,
                ),
                _classical_row(
                    scheduler=ProportionalFairScheduler(),
                    config=baseline_config,
                    seed=seed,
                    name="pf",
                    protocol=protocol,
                ),
                _classical_row(
                    scheduler=MaxCqiScheduler(),
                    config=baseline_config,
                    seed=seed,
                    name="max_cqi",
                    protocol=protocol,
                ),
                _classical_row(
                    scheduler=RuleOnlyScheduler(),
                    config=rule_config,
                    seed=seed,
                    name="rule_only",
                    protocol=protocol,
                ),
            ]
        )
        for name, checkpoint_path, model, checkpoint, runtime in loaded:
            config = protocol.build_config(
                scheduler_mode=runtime.scheduler_mode,
                safety_reserve_ues=runtime.safety_reserve_ues,
                force_harq_retransmissions=runtime.force_harq_retransmissions,
                safety_wait_threshold_ratio=runtime.long_wait_threshold,
            )
            row = evaluate_actor_critic(
                model=model,
                device=device,
                config=config,
                seed=seed,
                name=name,
                max_candidates=runtime.max_candidates,
                candidate_mode=runtime.candidate_mode,
                long_wait_threshold=runtime.long_wait_threshold,
                constraints=protocol.constraints(),
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
        print(f"completed unified evaluation seed={seed}")

    protocol_hashes = {str(row["evaluation_protocol_hash"]) for row in rows}
    if protocol_hashes != {protocol.protocol_hash}:
        raise RuntimeError("methods were evaluated under inconsistent protocol hashes")
    for seed in range(args.seed, args.seed + args.seeds):
        scenario_hashes = {
            str(row["scenario_hash"]) for row in rows if int(row["seed"]) == seed
        }
        if len(scenario_hashes) != 1:
            raise RuntimeError(f"methods used inconsistent scenario hashes for seed {seed}")

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL unified scheduler evaluation",
        description=(
            "RR, PF, Max-CQI, rule-only, hybrid PPO, candidate PPO-only, and full "
            "PPO are evaluated under the same static profile, HARQ randomness, "
            "KPI definitions, reward, constraints, and projector contract."
        ),
        rows=rows,
        notes=(
            f"Evaluation protocol hash: {protocol.protocol_hash}",
            "Checkpoint training rewards are logged for provenance and do not alter the evaluation environment.",
            "A standalone evaluate_ppo run with the same arguments must produce identical KPI values and hashes for the same checkpoint.",
        ),
    )
    summary = summarize_by_group(
        rows,
        group_key="method",
        numeric_fields=[
            "mean_reward",
            "mean_core_reward",
            "mean_goodput_bits_per_slot",
            "mean_throughput_score",
            "final_jain_fairness",
            "mean_fairness_score",
            "mean_starvation_rate",
            "max_starvation_rate",
            "final_p99_wait_slots",
            "max_p99_wait_slots",
            "final_max_wait_slots",
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
        title="ScaleMAC-RL unified scheduler evaluation summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )
    manifest_rows = _manifest_rows(rows)
    write_csv(args.manifest_output, manifest_rows)
    write_markdown(
        markdown_report_path(args.manifest_output),
        title="ScaleMAC-RL unified evaluation manifest",
        description="Protocol, scenario, runtime, and checkpoint hashes for every row.",
        rows=manifest_rows,
    )
    print(f"protocol_hash={protocol.protocol_hash}")
    print(f"saved: {args.output}")
    print(f"saved: {summary_path}")
    print(f"saved: {args.manifest_output}")


if __name__ == "__main__":
    main()
