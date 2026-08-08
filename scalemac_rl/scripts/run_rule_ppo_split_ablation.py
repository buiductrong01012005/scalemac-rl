from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from scalemac_rl.checkpoints import require_checkpoint
from scalemac_rl.evaluation_protocol import (
    PolicyRuntime,
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
from scalemac_rl.rule_attribution import (
    add_rule_lift_deltas,
    parse_rule_reserves,
    same_actor_curve,
    summarize_rule_dependency,
)
from scalemac_rl.schedulers import (
    MaxCqiScheduler,
    ProportionalFairScheduler,
    RoundRobinScheduler,
    RuleOnlyScheduler,
)
from scalemac_rl.tradeoff import annotate_tradeoff_scores


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


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
    row.update(
        {
            "ablation_family": "classical_baseline",
            "same_actor_weights": False,
            "target_rule_reserve_ues": (
                64 if name == "rule_only_uniform_prb" else 0
            ),
            "rule_selection_contract": (
                "rule selects all UEs; uniform PRB demand"
                if name == "rule_only_uniform_prb"
                else "classical scheduler"
            ),
            "candidate_filter_contract": "not applicable",
        }
    )
    return row


def _runtime(
    checkpoint: dict[str, Any],
    *,
    num_ues: int,
    scheduler_mode: str,
    reserve: int,
    force_harq: bool,
) -> PolicyRuntime:
    return resolve_policy_runtime(
        checkpoint,
        num_ues=num_ues,
        scheduler_mode=scheduler_mode,
        candidate_mode="heuristic",
        max_candidates=128,
        safety_reserve_ues=reserve,
        force_harq_retransmissions=force_harq,
    )


def _learned_row(
    *,
    name: str,
    model: Any,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    runtime: PolicyRuntime,
    protocol: UnifiedEvaluationProtocol,
    seed: int,
    same_actor_weights: bool,
    target_rule_reserve_ues: int,
    rule_selection_contract: str,
    ablation_family: str,
    device: torch.device,
) -> dict[str, Any]:
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
    row.update(
        {
            "ablation_family": ablation_family,
            "same_actor_weights": same_actor_weights,
            "target_rule_reserve_ues": target_rule_reserve_ues,
            "rule_selection_contract": rule_selection_contract,
            "candidate_filter_contract": (
                "all eligible UEs" if runtime.candidate_mode == "all" else f"heuristic Top-{runtime.max_candidates}"
            ),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hold one actor fixed and vary how many UE selections are made by rules. "
            "This quantifies whether the rule is lifting the PPO policy."
        )
    )
    parser.add_argument("--actor-checkpoint", "--hybrid-checkpoint", dest="actor_checkpoint", type=Path, required=True)
    parser.add_argument("--ppo-only-checkpoint", type=Path)
    parser.add_argument("--rule-reserves", default="8,16,24,32,48,64")
    parser.add_argument(
        "--include-all-ues-ablation",
        action="store_true",
        help="also remove the heuristic 128-candidate filter while keeping actor weights fixed",
    )
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--profile-seed", type=int, default=None)
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--min-jain-fairness", type=float, default=0.60)
    parser.add_argument("--max-wait-slots", type=float, default=60.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rule_ppo_split_ablation.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/rule_ppo_split_ablation_manifest.csv"),
    )
    args = parser.parse_args()

    if args.seeds <= 0:
        parser.error("seeds must be positive")
    try:
        reserves = parse_rule_reserves(args.rule_reserves)
    except ValueError as exc:
        parser.error(str(exc))

    device = _resolve_device(args.device)
    profile_seed = args.seed if args.profile_seed is None else args.profile_seed
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

    actor_path = require_checkpoint(args.actor_checkpoint)
    actor_model, actor_checkpoint = load_policy_checkpoint(actor_path, device)

    ppo_only_loaded: tuple[Path, Any, dict[str, Any], PolicyRuntime] | None = None
    if args.ppo_only_checkpoint:
        ppo_path = require_checkpoint(args.ppo_only_checkpoint)
        ppo_model, ppo_checkpoint = load_policy_checkpoint(ppo_path, device)
        ppo_runtime = _runtime(
            ppo_checkpoint,
            num_ues=args.num_ues,
            scheduler_mode="ppo_only",
            reserve=0,
            force_harq=False,
        )
        ppo_only_loaded = (ppo_path, ppo_model, ppo_checkpoint, ppo_runtime)

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
                    name="rule_only_uniform_prb",
                    protocol=protocol,
                ),
            ]
        )

        # Same actor, no rules at all: the causal reference for rule lift.
        ppo_same_runtime = _runtime(
            actor_checkpoint,
            num_ues=args.num_ues,
            scheduler_mode="ppo_only",
            reserve=0,
            force_harq=False,
        )
        rows.append(
            _learned_row(
                name="ppo_same_weights",
                model=actor_model,
                checkpoint_path=actor_path,
                checkpoint=actor_checkpoint,
                runtime=ppo_same_runtime,
                protocol=protocol,
                seed=seed,
                same_actor_weights=True,
                target_rule_reserve_ues=0,
                rule_selection_contract="PPO selects all 64 UEs; no forced HARQ",
                ablation_family="same_actor_rule_split",
                device=device,
            )
        )

        # Isolate the smallest rule contribution: mandatory HARQ only.
        harq_runtime = _runtime(
            actor_checkpoint,
            num_ues=args.num_ues,
            scheduler_mode="hybrid",
            reserve=0,
            force_harq=True,
        )
        if args.include_all_ues_ablation:
            ppo_all_runtime = resolve_policy_runtime(
                actor_checkpoint,
                num_ues=args.num_ues,
                scheduler_mode="ppo_only",
                candidate_mode="all",
                max_candidates=args.num_ues,
                safety_reserve_ues=0,
                force_harq_retransmissions=False,
            )
            rows.append(
                _learned_row(
                    name="ppo_same_weights_all1200",
                    model=actor_model,
                    checkpoint_path=actor_path,
                    checkpoint=actor_checkpoint,
                    runtime=ppo_all_runtime,
                    protocol=protocol,
                    seed=seed,
                    same_actor_weights=True,
                    target_rule_reserve_ues=0,
                    rule_selection_contract="PPO selects all 64 UEs; no forced HARQ",
                    ablation_family="same_actor_candidate_filter_ablation",
                    device=device,
                )
            )

        rows.append(
            _learned_row(
                name="hybrid_harq_only",
                model=actor_model,
                checkpoint_path=actor_path,
                checkpoint=actor_checkpoint,
                runtime=harq_runtime,
                protocol=protocol,
                seed=seed,
                same_actor_weights=True,
                target_rule_reserve_ues=0,
                rule_selection_contract="Rule forces pending HARQ only; PPO fills remaining grants",
                ablation_family="same_actor_rule_split",
                device=device,
            )
        )

        for reserve in reserves:
            runtime = _runtime(
                actor_checkpoint,
                num_ues=args.num_ues,
                scheduler_mode="hybrid",
                reserve=reserve,
                force_harq=True,
            )
            rows.append(
                _learned_row(
                    name=f"hybrid_rule_{reserve:02d}",
                    model=actor_model,
                    checkpoint_path=actor_path,
                    checkpoint=actor_checkpoint,
                    runtime=runtime,
                    protocol=protocol,
                    seed=seed,
                    same_actor_weights=True,
                    target_rule_reserve_ues=reserve,
                    rule_selection_contract=(
                        f"Rule selects up to {reserve} UEs; PPO selects the rest and predicts PRB demand"
                    ),
                    ablation_family="same_actor_rule_split",
                    device=device,
                )
            )

        if args.include_all_ues_ablation:
            checkpoint_runtime = resolve_policy_runtime(
                actor_checkpoint,
                num_ues=args.num_ues,
                scheduler_mode="hybrid",
                candidate_mode="all",
                max_candidates=args.num_ues,
            )
            rows.append(
                _learned_row(
                    name=f"hybrid_rule_{checkpoint_runtime.safety_reserve_ues:02d}_all1200",
                    model=actor_model,
                    checkpoint_path=actor_path,
                    checkpoint=actor_checkpoint,
                    runtime=checkpoint_runtime,
                    protocol=protocol,
                    seed=seed,
                    same_actor_weights=True,
                    target_rule_reserve_ues=checkpoint_runtime.safety_reserve_ues,
                    rule_selection_contract=(
                        f"Rule selects up to {checkpoint_runtime.safety_reserve_ues} UEs; PPO sees all eligible UEs"
                    ),
                    ablation_family="same_actor_candidate_filter_ablation",
                    device=device,
                )
            )

        if ppo_only_loaded is not None:
            ppo_path, ppo_model, ppo_checkpoint, ppo_runtime = ppo_only_loaded
            rows.append(
                _learned_row(
                    name="ppo_only_independently_trained",
                    model=ppo_model,
                    checkpoint_path=ppo_path,
                    checkpoint=ppo_checkpoint,
                    runtime=ppo_runtime,
                    protocol=protocol,
                    seed=seed,
                    same_actor_weights=False,
                    target_rule_reserve_ues=0,
                    rule_selection_contract="Independently trained PPO selects all 64 UEs",
                    ablation_family="independent_policy_reference",
                    device=device,
                )
            )
        print(f"completed rule/PPO split ablation seed={seed}")

    rows = annotate_tradeoff_scores(
        rows,
        max_starvation_rate=args.max_starvation_rate,
        group_key="seed",
    )
    rows = add_rule_lift_deltas(rows)

    write_csv(args.output, rows)
    write_markdown(
        markdown_report_path(args.output),
        title="ScaleMAC-RL rule/PPO split ablation",
        description=(
            "The actor weights are held fixed while rule-selected UE grants are varied. "
            "This isolates rule dependence from PPO retraining effects."
        ),
        rows=rows,
        notes=(
            "ppo_same_weights disables projector safety rules but retains the heuristic 128-candidate filter.",
            "--include-all-ues-ablation additionally tests the same actor with the candidate filter removed; this is out-of-distribution for a candidate-128 checkpoint.",
            "hybrid_harq_only forces pending HARQ but has no fixed oldest-UE reserve.",
            "hybrid_rule_64 uses rule-based UE selection for all grants, while the actor still supplies PRB-demand scores.",
            "This is a dependency ablation, not a fair final comparison of policies trained separately at every split.",
        ),
    )

    numeric_fields = [
        "mean_reward",
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "mean_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "mean_rule_selected_count",
        "mean_ppo_selected_count",
        "balanced_score",
        "worst_kpi_gap",
        "rule_lift_goodput",
        "rule_lift_fairness",
        "rule_lift_p99_reduction",
        "rule_lift_max_wait_reduction",
        "rule_lift_starvation_reduction",
        "rule_lift_balanced_score",
    ]
    summary = summarize_by_group(rows, group_key="method", numeric_fields=numeric_fields)
    summary_path = sibling_with_stem(args.output, "_summary", ".csv")
    write_csv(summary_path, summary)
    write_markdown(
        markdown_report_path(args.output, suffix="_summary"),
        title="ScaleMAC-RL rule/PPO split ablation summary",
        description=f"Mean and sample standard deviation across {args.seeds} seed(s).",
        rows=summary,
    )

    curve = same_actor_curve(rows)
    curve_path = sibling_with_stem(args.output, "_same_actor_curve", ".csv")
    write_csv(curve_path, curve)
    write_markdown(
        markdown_report_path(args.output, suffix="_same_actor_curve"),
        title="ScaleMAC-RL same-actor rule-dependency curve",
        description="Seed-level fixed-weight curve from zero rule grants to all-rule UE selection.",
        rows=curve,
    )

    dependency = summarize_rule_dependency(rows)
    dependency_path = sibling_with_stem(args.output, "_dependency", ".csv")
    write_csv(dependency_path, dependency)
    write_markdown(
        markdown_report_path(args.output, suffix="_dependency"),
        title="ScaleMAC-RL mean rule-dependency curve",
        description=(
            "Positive rule_lift values mean the rule improved the KPI relative to the same actor with rules disabled."
        ),
        rows=dependency,
    )

    manifest_keys = (
        "method",
        "seed",
        "ablation_family",
        "same_actor_weights",
        "target_rule_reserve_ues",
        "rule_selection_contract",
        "candidate_filter_contract",
        "evaluation_protocol_hash",
        "scenario_hash",
        "scheduler_runtime_hash",
        "scheduler_runtime_json",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_tag",
        "checkpoint_training_reward_version",
        "checkpoint_observation_features",
        "compatibility_adapter_applied",
    )
    manifest = [{key: row.get(key, "") for key in manifest_keys} for row in rows]
    write_csv(args.manifest_output, manifest)
    write_markdown(
        markdown_report_path(args.manifest_output),
        title="ScaleMAC-RL rule/PPO split ablation manifest",
        description="Protocol, scenario, runtime, and checkpoint identity for every ablation row.",
        rows=manifest,
    )

    print(f"protocol_hash={protocol.protocol_hash}")
    print(f"saved: {args.output}")
    print(f"saved: {summary_path}")
    print(f"saved: {curve_path}")
    print(f"saved: {dependency_path}")
    print(f"saved: {args.manifest_output}")


if __name__ == "__main__":
    main()
