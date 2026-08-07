from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.candidates import (
    build_candidate_mask,
    candidate_diagnostics,
    gather_candidate_batch,
    scatter_candidate_action_batch,
)
from scalemac_rl.constraints import LagrangeController, ServiceConstraints, validation_feasible
from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.reporting import markdown_report_path, write_csv, write_markdown
from scalemac_rl.rl_evaluation import evaluate_actor_critic


@dataclass(slots=True)
class PpoHyperparameters:
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    update_epochs: int
    minibatch_size: int
    target_kl: float


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _parse_int_list(value: str) -> list[int]:
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be comma-separated integers") from exc
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all values must be positive")
    return items


def _parse_float_list(value: str) -> list[float]:
    try:
        items = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be comma-separated numbers") from exc
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all values must be positive")
    return items


def _candidate_masks(
    observations: np.ndarray,
    max_candidates: int,
    min_candidates: int,
    long_wait_threshold: float,
) -> np.ndarray:
    return np.stack(
        [
            build_candidate_mask(
                observation,
                max_candidates=max_candidates,
                min_candidates=min_candidates,
                long_wait_threshold=long_wait_threshold,
            )
            for observation in observations
        ],
        axis=0,
    )


def _ppo_update(
    *,
    model: SharedSetActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    candidate_masks: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    hyper: PpoHyperparameters,
) -> dict[str, float]:
    batch_size = observations.shape[0]
    indices = np.arange(batch_size)
    losses: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(hyper.update_epochs):
        np.random.shuffle(indices)
        stop_early = False
        for start in range(0, batch_size, hyper.minibatch_size):
            mb = torch.as_tensor(
                indices[start : start + hyper.minibatch_size], device=observations.device
            )
            output = model.get_action_and_value(
                observations[mb], candidate_masks[mb], action=actions[mb]
            )
            log_ratio = output.log_prob - old_log_probs[mb]
            ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
            mb_adv = advantages[mb]
            policy_loss = torch.maximum(
                -mb_adv * ratio,
                -mb_adv
                * torch.clamp(ratio, 1.0 - hyper.clip_coef, 1.0 + hyper.clip_coef),
            ).mean()
            value_loss = 0.5 * torch.mean((output.value - returns[mb]) ** 2)
            entropy = output.entropy.mean()
            loss = policy_loss + hyper.value_coef * value_loss - hyper.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyper.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean((ratio - 1.0) - log_ratio).abs()
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > hyper.clip_coef).float()
                )
            losses["policy_loss"].append(float(policy_loss.item()))
            losses["value_loss"].append(float(value_loss.item()))
            losses["entropy"].append(float(entropy.item()))
            losses["approx_kl"].append(float(approx_kl.item()))
            losses["clip_fraction"].append(float(clip_fraction.item()))
            if hyper.target_kl > 0.0 and float(approx_kl.item()) > hyper.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    return {name: mean(values) if values else 0.0 for name, values in losses.items()}


def _snapshot(
    model: SharedSetActorCritic,
    optimizer: torch.optim.Optimizer,
    controller: LagrangeController,
) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "controller": copy.deepcopy(asdict(controller)),
    }


def _restore(
    snapshot: dict[str, Any],
    model: SharedSetActorCritic,
    optimizer: torch.optim.Optimizer,
    controller: LagrangeController,
) -> None:
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    for key, value in snapshot["controller"].items():
        setattr(controller, key, value)


def _checkpoint_payload(
    *,
    model: SharedSetActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    initialized_from: str,
    global_env_steps: int,
    update_index: int,
    stage_index: int,
    num_ues: int,
    controller: LagrangeController,
    constraints: ServiceConstraints,
    tag: str,
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "checkpoint_type": "hybrid_safety_constrained_ppo_actor_critic",
        "checkpoint_tag": tag,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": 8,
        "hidden_dim": args.hidden_dim,
        "curriculum": args.curriculum,
        "initialized_from": initialized_from,
        "training": {
            "global_env_steps": global_env_steps,
            "update": update_index,
            "stage": stage_index,
            "num_ues": num_ues,
            "steps_per_stage": args.steps_per_stage,
            "workers": args.workers,
            "rollout_steps": args.rollout_steps,
            "max_candidates": args.max_candidates,
            "safety_reserve_ues": args.safety_reserve_ues,
            "long_wait_threshold": args.long_wait_threshold,
            "stage_p99_wait_limits": args.stage_p99_wait_limits,
            "final_stage_p99_schedule": args.final_stage_p99_schedule,
            "validation_repeats": args.validation_repeats,
            "single_seed_upper_bound": args.single_seed_upper_bound,
            "freeze_static_profiles": args.freeze_static_profiles,
            "fixed_profile_seed": args.fixed_profile_seed,
            "deadline_risk_start_ratio": args.deadline_risk_start_ratio,
            "deadline_risk_penalty_weight": args.deadline_risk_penalty_weight,
            "reference_deadline_target_slots": args.max_p99_wait_slots,
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_coef": args.clip_coef,
            "device": str(next(model.parameters()).device),
        },
        "constraints": asdict(constraints),
        "lagrange_controller": asdict(controller),
        "validation": validation,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)



def _active_p99_limit(
    *,
    stage_index: int,
    stage_count: int,
    stage_env_steps: int,
    steps_per_stage: int,
    default_limit: float,
    final_stage_schedule: list[float],
) -> float:
    """Return the progressively tightened P99 target for the current update."""
    if stage_index != stage_count or not final_stage_schedule:
        return float(default_limit)
    progress = min(max(stage_env_steps, 0), max(steps_per_stage - 1, 0))
    segment = min(
        len(final_stage_schedule) - 1,
        int(progress * len(final_stage_schedule) / max(steps_per_stage, 1)),
    )
    return float(final_stage_schedule[segment])


def _load_initial_state(
    *,
    model: SharedSetActorCritic,
    optimizer: torch.optim.Optimizer,
    controller: LagrangeController,
    init_checkpoint: Path,
    resume_checkpoint: Path | None,
    device: torch.device,
) -> str:
    """Load either a full PPO checkpoint or an imitation actor checkpoint."""
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for key, value in checkpoint.get("lagrange_controller", {}).items():
            if hasattr(controller, key):
                setattr(controller, key, value)
        return str(resume_checkpoint)

    if init_checkpoint.is_file():
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        checkpoint_type = str(checkpoint.get("checkpoint_type", ""))
        if "ppo_actor_critic" in checkpoint_type:
            model.load_state_dict(state_dict)
        else:
            model.load_imitation_state_dict(state_dict)
        return str(init_checkpoint)
    return "random"

def _validation_seed_list(
    base_seeds: list[int],
    stage_index: int,
    repeats: int,
    *,
    fixed_seed_mode: bool = False,
) -> list[int]:
    if fixed_seed_mode:
        return [seed for _ in range(repeats) for seed in base_seeds]
    stage_offset = stage_index * 100_000
    return [
        seed + stage_offset + repeat * 10_000
        for repeat in range(repeats)
        for seed in base_seeds
    ]


def _validate(
    *,
    model: SharedSetActorCritic,
    device: torch.device,
    num_ues: int,
    slots: int,
    seeds: list[int],
    max_candidates: int,
    safety_reserve_ues: int,
    long_wait_threshold: float,
    constraints: ServiceConstraints,
    freeze_static_profiles: bool,
    fixed_profile_seed: int | None,
    deadline_risk_start_ratio: float,
    deadline_risk_penalty_weight: float,
    reference_deadline_target_slots: float,
    update_index: int,
    stage_index: int,
    global_env_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = ScaleMacConfig(
        num_ues=num_ues,
        num_prbs=273,
        max_selected_ues=min(64, num_ues, 273),
        episode_slots=slots,
        safety_reserve_ues=min(safety_reserve_ues, min(64, num_ues, 273)),
        safety_wait_threshold_ratio=long_wait_threshold,
        freeze_static_profiles=freeze_static_profiles,
        static_profile_seed=fixed_profile_seed,
        deadline_target_slots=constraints.max_p99_wait_slots,
        reference_deadline_target_slots=reference_deadline_target_slots,
        deadline_risk_start_ratio=deadline_risk_start_ratio,
        reward_deadline_risk_penalty_weight=deadline_risk_penalty_weight,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row = evaluate_actor_critic(
            model=model,
            device=device,
            config=config,
            seed=seed,
            name="ppo_validation",
            max_candidates=max_candidates,
            long_wait_threshold=long_wait_threshold,
            constraints=constraints,
        )
        row.update(
            {"update": update_index, "stage": stage_index, "global_env_steps": global_env_steps}
        )
        rows.append(row)

    worst_starvation = max(float(row["max_starvation_rate"]) for row in rows)
    worst_wait = max(float(row["max_p99_wait_slots"]) for row in rows)
    starvation_excess, wait_excess = constraints.excesses(
        starvation_rate=worst_starvation, p99_wait_slots=worst_wait
    )
    summary: dict[str, Any] = {
        "update": update_index,
        "stage": stage_index,
        "num_ues": num_ues,
        "global_env_steps": global_env_steps,
        "validation_episodes": len(rows),
        "mean_reward": mean(float(row["mean_reward"]) for row in rows),
        "mean_core_reward": mean(float(row["mean_core_reward"]) for row in rows),
        "mean_final_target_reward": mean(
            float(row["mean_final_target_reward"]) for row in rows
        ),
        "mean_goodput_bits_per_slot": mean(
            float(row["mean_goodput_bits_per_slot"]) for row in rows
        ),
        "mean_jain_fairness": mean(float(row["final_jain_fairness"]) for row in rows),
        "worst_starvation_rate": worst_starvation,
        "worst_p99_wait_slots": worst_wait,
        "validation_starvation_excess": starvation_excess,
        "validation_wait_excess": wait_excess,
        "total_constraint_excess": starvation_excess + wait_excess,
        "mean_candidate_coverage": mean(float(row["mean_candidate_coverage"]) for row in rows),
        "mean_long_wait_retention_rate": mean(
            float(row["mean_long_wait_retention_rate"]) for row in rows
        ),
        "mean_harq_retention_rate": mean(
            float(row["mean_harq_retention_rate"]) for row in rows
        ),
        "mean_deadline_risk": mean(float(row["mean_deadline_risk"]) for row in rows),
        "mean_reference_deadline_risk": mean(
            float(row["mean_reference_deadline_risk"]) for row in rows
        ),
        "mean_reference_deadline_penalty": mean(
            float(row["mean_reward_reference_deadline_risk_penalty"]) for row in rows
        ),
        "mean_tail_mean_wait_slots": mean(
            float(row["mean_tail_mean_wait_slots"]) for row in rows
        ),
        "mean_safety_selected_count": mean(
            float(row["mean_safety_selected_count"]) for row in rows
        ),
        "mean_oldest_selected_count": mean(
            float(row["mean_forced_oldest_wait_count"]) for row in rows
        ),
        "mean_learned_selected_count": mean(
            float(row["mean_learned_selected_count"]) for row in rows
        ),
        "mean_learned_selection_fraction": mean(
            float(row["mean_learned_selection_fraction"]) for row in rows
        ),
        "constraint_feasible": validation_feasible(rows, constraints),
        "rolled_back": False,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid safety-reserve constrained curriculum PPO for ScaleMAC-RL"
    )
    parser.add_argument("--init-checkpoint", type=Path, default=Path("artifacts/pf_imitation.pt"))
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--curriculum", type=_parse_int_list, default=[128, 256, 600, 1200])
    parser.add_argument("--steps-per-stage", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episode-slots", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--safety-reserve-ues", type=int, default=16)
    parser.add_argument("--long-wait-threshold", type=float, default=0.8)
    parser.add_argument("--freeze-static-profiles", action="store_true")
    parser.add_argument("--fixed-profile-seed", type=int, default=None)
    parser.add_argument("--single-seed-upper-bound", action="store_true")
    parser.add_argument("--deadline-risk-start-ratio", type=float, default=0.60)
    parser.add_argument("--deadline-risk-penalty-weight", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument(
        "--stage-p99-wait-limits", type=_parse_float_list, default=[80.0, 80.0, 80.0, 50.0]
    )
    parser.add_argument(
        "--final-stage-p99-schedule", type=_parse_float_list, default=[80.0, 65.0, 55.0, 50.0]
    )
    parser.add_argument("--starvation-multiplier", type=float, default=5.0)
    parser.add_argument("--wait-multiplier", type=float, default=1.0)
    parser.add_argument("--lagrangian-lr", type=float, default=0.10)
    parser.add_argument("--validation-lagrangian-scale", type=float, default=1.0)
    parser.add_argument("--max-lagrange-multiplier", type=float, default=50.0)
    parser.add_argument("--validation-seeds", type=_parse_int_list, default=[9001, 9002, 9003])
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--validation-slots", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=16)
    parser.add_argument("--rollback-patience", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show a tqdm progress bar with elapsed time and ETA",
    )
    parser.add_argument("--progress-refresh-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest.pt"))
    parser.add_argument("--best-feasible-output", type=Path, default=Path("artifacts/best_feasible.pt"))
    parser.add_argument("--best-reward-output", type=Path, default=Path("artifacts/best_reward.pt"))
    parser.add_argument(
        "--best-lowest-violation-output",
        type=Path,
        default=Path("artifacts/best_lowest_violation.pt"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--log-output", type=Path, default=Path("artifacts/ppo_training.csv"))
    parser.add_argument("--validation-output", type=Path, default=Path("artifacts/ppo_validation.csv"))
    args = parser.parse_args()

    if args.steps_per_stage <= 0 or args.workers <= 0 or args.rollout_steps <= 0:
        parser.error("steps, workers, and rollout length must be positive")
    if args.max_candidates < 64:
        parser.error("max_candidates must be at least the Top-K value 64")
    if not 0 <= args.safety_reserve_ues < 64:
        parser.error("safety_reserve_ues must be in [0, 63] so PPO keeps learned grants")
    if len(args.stage_p99_wait_limits) != len(args.curriculum):
        parser.error("stage-p99-wait-limits must contain one value per curriculum stage")
    if args.validation_slots <= 0 or args.validation_repeats <= 0:
        parser.error("validation slots and repeats must be positive")
    if args.fixed_profile_seed is not None and args.fixed_profile_seed < 0:
        parser.error("fixed-profile-seed must be non-negative")
    if not 0.0 <= args.deadline_risk_start_ratio < 1.0:
        parser.error("deadline-risk-start-ratio must be in [0, 1)")
    if args.deadline_risk_penalty_weight < 0.0:
        parser.error("deadline-risk-penalty-weight must be non-negative")
    if args.single_seed_upper_bound:
        if args.workers != 1:
            parser.error("single-seed-upper-bound requires --workers 1")
        if len(args.curriculum) != 1:
            parser.error("single-seed-upper-bound requires one curriculum stage, e.g. --curriculum 1200")
        if len(args.validation_seeds) != 1:
            parser.error("single-seed-upper-bound requires exactly one validation seed")
        args.freeze_static_profiles = True
        if args.fixed_profile_seed is None:
            args.fixed_profile_seed = args.seed
    if args.validate_every <= 0 or args.rollback_patience <= 0 or args.checkpoint_every <= 0:
        parser.error("validation, rollback, and checkpoint intervals must be positive")

    final_constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
    )
    final_constraints.validate()
    controller = LagrangeController(
        starvation_multiplier=args.starvation_multiplier,
        wait_multiplier=args.wait_multiplier,
        learning_rate=args.lagrangian_lr,
        max_multiplier=args.max_lagrange_multiplier,
    )
    controller.validate()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    model = SharedSetActorCritic(input_dim=8, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    try:
        initialized_from = _load_initial_state(
            model=model,
            optimizer=optimizer,
            controller=controller,
            init_checkpoint=args.init_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
            device=device,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    hyper = PpoHyperparameters(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        target_kl=args.target_kl,
    )

    log_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    validation_summary_rows: list[dict[str, Any]] = []
    global_env_steps = 0
    update_index = 0
    best_feasible_reward = float("-inf")
    best_reward = float("-inf")
    best_lowest_violation_score: tuple[float, float] | None = None
    best_lowest_violation_saved = False
    best_feasible_saved = False
    target_num_ues = args.curriculum[-1]
    last_validation: dict[str, Any] | None = None
    training_started_at = perf_counter()
    total_requested_steps = args.steps_per_stage * len(args.curriculum)
    progress = tqdm(
        total=total_requested_steps,
        desc="ScaleMAC PPO",
        unit="step",
        unit_scale=True,
        dynamic_ncols=True,
        mininterval=args.progress_refresh_seconds,
        disable=not args.progress,
    )

    for stage_index, num_ues in enumerate(args.curriculum, start=1):
        max_selected = min(64, num_ues, 273)
        max_candidates = min(max(args.max_candidates, max_selected), num_ues)
        safety_reserve = min(args.safety_reserve_ues, max_selected - 1)
        stage_default_p99_limit = args.stage_p99_wait_limits[stage_index - 1]
        initial_p99_limit = _active_p99_limit(
            stage_index=stage_index,
            stage_count=len(args.curriculum),
            stage_env_steps=0,
            steps_per_stage=args.steps_per_stage,
            default_limit=stage_default_p99_limit,
            final_stage_schedule=args.final_stage_p99_schedule,
        )
        stage_seed = args.seed if args.single_seed_upper_bound else args.seed + stage_index * 10_000
        profile_seed = (
            args.fixed_profile_seed
            if args.fixed_profile_seed is not None
            else (args.seed if args.single_seed_upper_bound else None)
        )
        config = ScaleMacConfig(
            num_ues=num_ues,
            num_prbs=273,
            max_selected_ues=max_selected,
            episode_slots=args.episode_slots,
            safety_reserve_ues=safety_reserve,
            safety_wait_threshold_ratio=args.long_wait_threshold,
            freeze_static_profiles=args.freeze_static_profiles or args.single_seed_upper_bound,
            static_profile_seed=profile_seed,
            deadline_target_slots=initial_p99_limit,
            reference_deadline_target_slots=args.max_p99_wait_slots,
            deadline_risk_start_ratio=args.deadline_risk_start_ratio,
            reward_deadline_risk_penalty_weight=args.deadline_risk_penalty_weight,
            seed=stage_seed,
        )
        config.validate()
        envs = [ScaleMacDownlinkEnv(config) for _ in range(args.workers)]
        observations = np.stack(
            [env.reset(seed=config.seed + worker)[0] for worker, env in enumerate(envs)], axis=0
        )
        stage_env_steps = 0
        stage_best_feasible_reward = float("-inf")
        stage_best_feasible_snapshot: dict[str, Any] | None = None
        stage_best_candidate_snapshot: dict[str, Any] | None = None
        stage_best_candidate_score: tuple[float, float] | None = None
        consecutive_infeasible = 0

        while stage_env_steps < args.steps_per_stage:
            active_p99_limit = _active_p99_limit(
                stage_index=stage_index,
                stage_count=len(args.curriculum),
                stage_env_steps=stage_env_steps,
                steps_per_stage=args.steps_per_stage,
                default_limit=stage_default_p99_limit,
                final_stage_schedule=args.final_stage_p99_schedule,
            )
            active_constraints = ServiceConstraints(
                max_starvation_rate=args.max_starvation_rate,
                max_p99_wait_slots=active_p99_limit,
            )
            active_constraints.validate()
            config.deadline_target_slots = active_p99_limit
            for env in envs:
                env.config.deadline_target_slots = active_p99_limit

            obs_buffer: list[np.ndarray] = []
            action_buffer: list[np.ndarray] = []
            compact_mask_buffer: list[np.ndarray] = []
            logprob_buffer: list[np.ndarray] = []
            value_buffer: list[np.ndarray] = []
            reward_buffer: list[np.ndarray] = []
            done_buffer: list[np.ndarray] = []
            metric_window: dict[str, list[float]] = {
                key: []
                for key in (
                    "base_reward", "core_reward", "final_target_reward",
                    "constrained_reward", "constraint_penalty",
                    "deadline_risk_penalty", "reference_deadline_risk_penalty",
                    "deadline_risk", "reference_deadline_risk", "tail_mean_wait",
                    "throughput", "fairness", "service", "starvation", "p99_wait",
                    "starvation_excess", "wait_excess", "goodput", "candidate_coverage",
                    "harq_retention", "long_wait_retention", "long_wait_missed",
                    "safety_selected", "oldest_selected", "learned_selected", "learned_fraction",
                )
            }

            for _ in range(args.rollout_steps):
                masks = _candidate_masks(
                    observations, max_candidates, max_selected, args.long_wait_threshold
                )
                compact_observations, candidate_indices = gather_candidate_batch(observations, masks)
                compact_masks = np.ones(compact_observations.shape[:2], dtype=bool)
                obs_tensor = torch.from_numpy(compact_observations).to(device)
                mask_tensor = torch.from_numpy(compact_masks).to(device)
                with torch.no_grad():
                    output = model.get_action_and_value(obs_tensor, mask_tensor)
                compact_actions = output.action.cpu().numpy()
                full_actions = scatter_candidate_action_batch(
                    compact_actions, candidate_indices, num_ues=num_ues
                )

                next_observations: list[np.ndarray] = []
                rewards = np.zeros(args.workers, dtype=np.float32)
                dones = np.zeros(args.workers, dtype=np.float32)
                for worker, env in enumerate(envs):
                    diagnostics = candidate_diagnostics(
                        observations[worker], masks[worker],
                        long_wait_threshold=args.long_wait_threshold,
                    )
                    next_obs, base_reward, terminated, truncated, info = env.step(
                        full_actions[worker]
                    )
                    done = terminated or truncated
                    starvation_excess, wait_excess = active_constraints.excesses(
                        starvation_rate=float(info["starvation_rate"]),
                        p99_wait_slots=float(info["p99_wait_slots"]),
                    )
                    constrained_reward, constraint_penalty = controller.adjusted_reward(
                        base_reward,
                        starvation_excess=starvation_excess,
                        wait_excess=wait_excess,
                    )
                    rewards[worker] = constrained_reward
                    dones[worker] = float(done)
                    values = {
                        "base_reward": float(info["reward_total"]),
                        "core_reward": float(info["reward_core_total"]),
                        "final_target_reward": float(info["reward_final_target_total"]),
                        "constrained_reward": constrained_reward,
                        "constraint_penalty": constraint_penalty,
                        "deadline_risk_penalty": float(info["reward_deadline_risk_penalty"]),
                        "reference_deadline_risk_penalty": float(
                            info["reward_reference_deadline_risk_penalty"]
                        ),
                        "deadline_risk": float(info["deadline_risk"]),
                        "reference_deadline_risk": float(info["reference_deadline_risk"]),
                        "tail_mean_wait": float(info["tail_mean_wait_slots"]),
                        "throughput": float(info["throughput_score"]),
                        "fairness": float(info["fairness_score"]),
                        "service": float(info["service_score"]),
                        "starvation": float(info["starvation_rate"]),
                        "p99_wait": float(info["p99_wait_slots"]),
                        "starvation_excess": starvation_excess,
                        "wait_excess": wait_excess,
                        "goodput": float(info["cell_goodput_bits"]),
                        "candidate_coverage": diagnostics.candidate_coverage,
                        "harq_retention": diagnostics.harq_retention_rate,
                        "long_wait_retention": diagnostics.long_wait_retention_rate,
                        "long_wait_missed": float(diagnostics.long_wait_missed_count),
                        "safety_selected": float(info["safety_selected_count"]),
                        "oldest_selected": float(info["forced_oldest_wait_count"]),
                        "learned_selected": float(info["learned_selected_count"]),
                        "learned_fraction": float(info["learned_selection_fraction"]),
                    }
                    for key, value in values.items():
                        metric_window[key].append(value)
                    if done:
                        next_obs, _ = env.reset()
                    next_observations.append(next_obs)

                obs_buffer.append(compact_observations.copy())
                action_buffer.append(compact_actions.copy())
                compact_mask_buffer.append(compact_masks.copy())
                logprob_buffer.append(output.log_prob.cpu().numpy())
                value_buffer.append(output.value.cpu().numpy())
                reward_buffer.append(rewards)
                done_buffer.append(dones)
                observations = np.stack(next_observations, axis=0)

            with torch.no_grad():
                next_masks_full = _candidate_masks(
                    observations, max_candidates, max_selected, args.long_wait_threshold
                )
                next_compact_obs, _ = gather_candidate_batch(observations, next_masks_full)
                next_obs_tensor = torch.from_numpy(next_compact_obs).to(device)
                next_compact_masks = torch.ones(
                    next_compact_obs.shape[:2], dtype=torch.bool, device=device
                )
                next_value = model.get_action_and_value(
                    next_obs_tensor, next_compact_masks, deterministic=True
                ).value.cpu().numpy()

            rewards_np = np.asarray(reward_buffer, dtype=np.float32)
            dones_np = np.asarray(done_buffer, dtype=np.float32)
            values_np = np.asarray(value_buffer, dtype=np.float32)
            advantages_np = np.zeros_like(rewards_np)
            last_gae = np.zeros(args.workers, dtype=np.float32)
            for t in reversed(range(args.rollout_steps)):
                following_value = next_value if t == args.rollout_steps - 1 else values_np[t + 1]
                nonterminal = 1.0 - dones_np[t]
                delta = rewards_np[t] + args.gamma * following_value * nonterminal - values_np[t]
                last_gae = delta + args.gamma * args.gae_lambda * nonterminal * last_gae
                advantages_np[t] = last_gae
            returns_np = advantages_np + values_np

            candidate_count = max_candidates
            flat_obs = torch.from_numpy(
                np.asarray(obs_buffer).reshape(-1, candidate_count, 8)
            ).to(device)
            flat_actions = torch.from_numpy(
                np.asarray(action_buffer).reshape(-1, candidate_count, 2)
            ).to(device)
            flat_masks = torch.from_numpy(
                np.asarray(compact_mask_buffer).reshape(-1, candidate_count)
            ).to(device)
            flat_logprobs = torch.from_numpy(np.asarray(logprob_buffer).reshape(-1)).to(device)
            flat_returns = torch.from_numpy(returns_np.reshape(-1)).to(device)
            flat_advantages = torch.from_numpy(advantages_np.reshape(-1)).to(device)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (
                flat_advantages.std(unbiased=False) + 1e-8
            )
            update_metrics = _ppo_update(
                model=model, optimizer=optimizer, observations=flat_obs,
                actions=flat_actions, candidate_masks=flat_masks,
                old_log_probs=flat_logprobs, returns=flat_returns,
                advantages=flat_advantages, hyper=hyper,
            )
            controller.update(
                mean_starvation_excess=mean(metric_window["starvation_excess"]),
                mean_wait_excess=mean(metric_window["wait_excess"]),
            )
            collected = args.rollout_steps * args.workers
            stage_env_steps += collected
            global_env_steps += collected
            update_index += 1
            row = {
                "update": update_index,
                "stage": stage_index,
                "num_ues": num_ues,
                "stage_env_steps": stage_env_steps,
                "global_env_steps": global_env_steps,
                "workers": args.workers,
                "max_candidates": max_candidates,
                "safety_reserve_ues": safety_reserve,
                "stage_max_p99_wait_slots": active_constraints.max_p99_wait_slots,
                "mean_reward": mean(metric_window["base_reward"]),
                "mean_core_reward": mean(metric_window["core_reward"]),
                "mean_final_target_reward": mean(metric_window["final_target_reward"]),
                "mean_training_reward": mean(metric_window["constrained_reward"]),
                "mean_constrained_reward": mean(metric_window["constrained_reward"]),
                "mean_constraint_penalty": mean(metric_window["constraint_penalty"]),
                "mean_deadline_risk_penalty": mean(metric_window["deadline_risk_penalty"]),
                "mean_reference_deadline_risk_penalty": mean(
                    metric_window["reference_deadline_risk_penalty"]
                ),
                "mean_deadline_risk": mean(metric_window["deadline_risk"]),
                "mean_reference_deadline_risk": mean(
                    metric_window["reference_deadline_risk"]
                ),
                "mean_tail_mean_wait_slots": mean(metric_window["tail_mean_wait"]),
                "mean_goodput_bits_per_slot": mean(metric_window["goodput"]),
                "mean_throughput_score": mean(metric_window["throughput"]),
                "mean_fairness_score": mean(metric_window["fairness"]),
                "mean_service_score": mean(metric_window["service"]),
                "mean_starvation_rate": mean(metric_window["starvation"]),
                "max_starvation_rate": max(metric_window["starvation"]),
                "mean_p99_wait_slots": mean(metric_window["p99_wait"]),
                "max_p99_wait_slots": max(metric_window["p99_wait"]),
                "mean_starvation_excess": mean(metric_window["starvation_excess"]),
                "mean_wait_excess": mean(metric_window["wait_excess"]),
                "starvation_multiplier": controller.starvation_multiplier,
                "wait_multiplier": controller.wait_multiplier,
                "mean_candidate_coverage": mean(metric_window["candidate_coverage"]),
                "mean_harq_retention_rate": mean(metric_window["harq_retention"]),
                "mean_long_wait_retention_rate": mean(metric_window["long_wait_retention"]),
                "max_long_wait_missed_count": max(metric_window["long_wait_missed"]),
                "mean_safety_selected_count": mean(metric_window["safety_selected"]),
                "mean_oldest_selected_count": mean(metric_window["oldest_selected"]),
                "mean_learned_selected_count": mean(metric_window["learned_selected"]),
                "mean_learned_selection_fraction": mean(metric_window["learned_fraction"]),
                "elapsed_seconds": perf_counter() - training_started_at,
                "steps_per_second": global_env_steps
                / max(perf_counter() - training_started_at, 1e-9),
                "eta_seconds": (
                    max(total_requested_steps - global_env_steps, 0)
                    / max(
                        global_env_steps
                        / max(perf_counter() - training_started_at, 1e-9),
                        1e-9,
                    )
                ),
                **update_metrics,
                "device": str(device),
            }
            log_rows.append(row)
            progress.update(collected)
            progress.set_postfix(
                stage=f"{stage_index}/{len(args.curriculum)}",
                ues=num_ues,
                core=f"{row['mean_core_reward']:.3f}",
                final=f"{row['mean_final_target_reward']:.3f}",
                train=f"{row['mean_training_reward']:.3f}",
                p99=f"{row['mean_p99_wait_slots']:.1f}",
                goodput=f"{row['mean_goodput_bits_per_slot'] / 1000.0:.1f}k",
            )
            if not args.progress:
                print(
                    f"stage={stage_index}/{len(args.curriculum)} ues={num_ues:4d} "
                    f"steps={stage_env_steps:7d}/{args.steps_per_stage} "
                    f"core={row['mean_core_reward']:.4f} "
                    f"final={row['mean_final_target_reward']:.4f} "
                    f"train={row['mean_training_reward']:.4f} "
                    f"p99={row['mean_p99_wait_slots']:.1f}"
                )

            should_validate = update_index % args.validate_every == 0 or stage_env_steps >= args.steps_per_stage
            if should_validate:
                seeds = _validation_seed_list(
                    args.validation_seeds,
                    stage_index,
                    args.validation_repeats,
                    fixed_seed_mode=args.single_seed_upper_bound,
                )
                detailed, summary = _validate(
                    model=model, device=device, num_ues=num_ues,
                    slots=args.validation_slots, seeds=seeds,
                    max_candidates=max_candidates,
                    safety_reserve_ues=safety_reserve,
                    long_wait_threshold=args.long_wait_threshold,
                    constraints=active_constraints,
                    freeze_static_profiles=config.freeze_static_profiles,
                    fixed_profile_seed=config.static_profile_seed,
                    deadline_risk_start_ratio=args.deadline_risk_start_ratio,
                    deadline_risk_penalty_weight=args.deadline_risk_penalty_weight,
                    reference_deadline_target_slots=args.max_p99_wait_slots,
                    update_index=update_index, stage_index=stage_index,
                    global_env_steps=global_env_steps,
                )
                # Held-out failures now directly strengthen the dual controller.
                controller.update(
                    mean_starvation_excess=(
                        args.validation_lagrangian_scale
                        * float(summary["validation_starvation_excess"])
                    ),
                    mean_wait_excess=(
                        args.validation_lagrangian_scale
                        * float(summary["validation_wait_excess"])
                    ),
                )
                summary["starvation_multiplier_after_validation"] = controller.starvation_multiplier
                summary["wait_multiplier_after_validation"] = controller.wait_multiplier

                score = (
                    float(summary["total_constraint_excess"]),
                    -float(summary["mean_reward"]),
                )
                if stage_best_candidate_score is None or score < stage_best_candidate_score:
                    stage_best_candidate_score = score
                    stage_best_candidate_snapshot = _snapshot(model, optimizer, controller)
                    stage_path = args.checkpoint_dir / f"best_stage_{num_ues}.pt"
                    _save_checkpoint(
                        stage_path,
                        _checkpoint_payload(
                            model=model, optimizer=optimizer, args=args,
                            initialized_from=initialized_from,
                            global_env_steps=global_env_steps, update_index=update_index,
                            stage_index=stage_index, num_ues=num_ues,
                            controller=controller, constraints=active_constraints,
                            tag=f"best_stage_{num_ues}", validation=summary,
                        ),
                    )

                if bool(summary["constraint_feasible"]):
                    consecutive_infeasible = 0
                    if float(summary["mean_reward"]) > stage_best_feasible_reward:
                        stage_best_feasible_reward = float(summary["mean_reward"])
                        stage_best_feasible_snapshot = _snapshot(model, optimizer, controller)
                        _save_checkpoint(
                            args.checkpoint_dir / f"best_feasible_stage_{num_ues}.pt",
                            _checkpoint_payload(
                                model=model, optimizer=optimizer, args=args,
                                initialized_from=initialized_from,
                                global_env_steps=global_env_steps, update_index=update_index,
                                stage_index=stage_index, num_ues=num_ues,
                                controller=controller, constraints=active_constraints,
                                tag=f"best_feasible_stage_{num_ues}", validation=summary,
                            ),
                        )
                else:
                    consecutive_infeasible += 1
                    if (
                        stage_best_feasible_snapshot is not None
                        and consecutive_infeasible >= args.rollback_patience
                    ):
                        _restore(stage_best_feasible_snapshot, model, optimizer, controller)
                        observations = np.stack(
                            [
                                env.reset(seed=config.seed + worker + update_index)[0]
                                for worker, env in enumerate(envs)
                            ],
                            axis=0,
                        )
                        summary["rolled_back"] = True
                        consecutive_infeasible = 0

                validation_rows.extend(detailed)
                validation_summary_rows.append(summary)
                last_validation = summary
                validation_message = (
                    f"validation ues={num_ues} active_reward={summary['mean_reward']:.4f} "
                    f"final_reward={summary['mean_final_target_reward']:.4f} "
                    f"fairness={summary['mean_jain_fairness']:.4f} "
                    f"worst_starvation={summary['worst_starvation_rate']:.4f} "
                    f"worst_p99={summary['worst_p99_wait_slots']:.1f} "
                    f"feasible={summary['constraint_feasible']} rollback={summary['rolled_back']}"
                )
                if args.progress:
                    progress.write(validation_message)
                else:
                    print(validation_message)

                if num_ues == target_num_ues:
                    payload = _checkpoint_payload(
                        model=model, optimizer=optimizer, args=args,
                        initialized_from=initialized_from,
                        global_env_steps=global_env_steps, update_index=update_index,
                        stage_index=stage_index, num_ues=num_ues,
                        controller=controller, constraints=final_constraints,
                        tag="validation", validation=summary,
                    )
                    final_starvation_excess, final_wait_excess = final_constraints.excesses(
                        starvation_rate=float(summary["worst_starvation_rate"]),
                        p99_wait_slots=float(summary["worst_p99_wait_slots"]),
                    )
                    summary["final_target_starvation_excess"] = final_starvation_excess
                    summary["final_target_wait_excess"] = final_wait_excess
                    summary["final_target_total_constraint_excess"] = (
                        final_starvation_excess + final_wait_excess
                    )
                    final_score = (
                        float(summary["final_target_total_constraint_excess"]),
                        -float(summary["mean_final_target_reward"]),
                    )
                    if (
                        best_lowest_violation_score is None
                        or final_score < best_lowest_violation_score
                    ):
                        best_lowest_violation_score = final_score
                        best_lowest_violation_saved = True
                        lowest_payload = copy.deepcopy(payload)
                        lowest_payload["checkpoint_tag"] = "best_lowest_violation"
                        _save_checkpoint(args.best_lowest_violation_output, lowest_payload)
                    if float(summary["mean_final_target_reward"]) > best_reward:
                        best_reward = float(summary["mean_final_target_reward"])
                        payload["checkpoint_tag"] = "best_reward"
                        _save_checkpoint(args.best_reward_output, payload)
                    final_feasible = (
                        float(summary["worst_starvation_rate"])
                        <= final_constraints.max_starvation_rate + 1e-12
                        and float(summary["worst_p99_wait_slots"])
                        <= final_constraints.max_p99_wait_slots + 1e-12
                    )
                    if (
                        final_feasible
                        and float(summary["mean_final_target_reward"]) > best_feasible_reward
                    ):
                        best_feasible_reward = float(summary["mean_final_target_reward"])
                        best_feasible_saved = True
                        payload["checkpoint_tag"] = "best_feasible"
                        _save_checkpoint(args.best_feasible_output, payload)

            if update_index % args.checkpoint_every == 0:
                _save_checkpoint(
                    args.checkpoint_dir / f"ppo_update_{update_index:05d}.pt",
                    _checkpoint_payload(
                        model=model, optimizer=optimizer, args=args,
                        initialized_from=initialized_from,
                        global_env_steps=global_env_steps, update_index=update_index,
                        stage_index=stage_index, num_ues=num_ues,
                        controller=controller, constraints=active_constraints,
                        tag="periodic", validation=last_validation,
                    ),
                )

        # Start the next curriculum stage from the safest available policy.
        if stage_best_feasible_snapshot is not None:
            _restore(stage_best_feasible_snapshot, model, optimizer, controller)
            print(f"stage {num_ues}: continuing from best feasible checkpoint")
        elif stage_best_candidate_snapshot is not None:
            _restore(stage_best_candidate_snapshot, model, optimizer, controller)
            print(f"stage {num_ues}: no feasible checkpoint; continuing from lowest-violation checkpoint")

    progress.close()
    total_elapsed_seconds = perf_counter() - training_started_at

    latest_payload = _checkpoint_payload(
        model=model, optimizer=optimizer, args=args,
        initialized_from=initialized_from,
        global_env_steps=global_env_steps, update_index=update_index,
        stage_index=len(args.curriculum), num_ues=args.curriculum[-1],
        controller=controller, constraints=final_constraints,
        tag="latest", validation=last_validation,
    )
    _save_checkpoint(args.output, latest_payload)

    write_csv(args.log_output, log_rows)
    write_markdown(
        markdown_report_path(args.log_output),
        title="ScaleMAC-RL hybrid safety-reserve constrained PPO training",
        description=(
            "Long curriculum PPO run with a deterministic safety reserve, learned UE selection, "
            "validation-driven Lagrange updates, rollback, and per-stage checkpoint selection."
        ),
        rows=log_rows,
        notes=(
            f"Initialized from: `{initialized_from}`",
            f"Curriculum UE stages: {args.curriculum}",
            f"Environment steps per stage: {args.steps_per_stage}",
            f"Candidate pool: {args.max_candidates}; safety reserve: {args.safety_reserve_ues}; Top-K: 64",
            f"Stage P99 limits: {args.stage_p99_wait_limits}; fixed comparison target: {args.max_p99_wait_slots}",
            f"Total wall-clock training time: {total_elapsed_seconds:.1f} seconds",
            "Workers are vectorized environments in one process, not distributed RL.",
            "This remains the fast surrogate, not 5G-LENA.",
        ),
    )
    write_csv(args.validation_output, validation_rows)
    write_markdown(
        markdown_report_path(args.validation_output),
        title="ScaleMAC-RL repeated held-out PPO validation",
        description="Per-episode validation used for dual updates, rollback, and checkpoint selection.",
        rows=validation_rows,
    )
    validation_summary_output = args.validation_output.with_name(
        f"{args.validation_output.stem}_summary.csv"
    )
    write_csv(validation_summary_output, validation_summary_rows)
    write_markdown(
        markdown_report_path(validation_summary_output),
        title="ScaleMAC-RL held-out PPO validation summary",
        description="Worst-case constraints, grant attribution, and rollback status at each validation.",
        rows=validation_summary_rows,
    )

    print(f"saved: {args.output}")
    print(f"saved: {args.best_reward_output}")
    if best_lowest_violation_saved:
        print(f"saved: {args.best_lowest_violation_output}")
    if best_feasible_saved:
        print(f"saved: {args.best_feasible_output}")
    else:
        print("warning: no final-stage validation checkpoint satisfied the official constraints")
    print(f"saved: {args.log_output}")
    print(f"saved: {args.validation_output}")
    print(f"saved: {validation_summary_output}")
    print(
        f"training_time={total_elapsed_seconds:.1f}s "
        f"throughput={global_env_steps / max(total_elapsed_seconds, 1e-9):.1f} env_steps/s"
    )


if __name__ == "__main__":
    main()
