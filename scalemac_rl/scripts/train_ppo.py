from __future__ import annotations

import argparse
import copy
import math
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
from scalemac_rl.env import OBSERVATION_FEATURES
from scalemac_rl.candidates import (
    build_all_eligible_mask,
    build_candidate_mask,
    candidate_diagnostics,
    gather_candidate_batch,
    scatter_candidate_action_batch,
)
from scalemac_rl.constraints import LagrangeController, ServiceConstraints, validation_feasible
from scalemac_rl.models import RecurrentSharedSetActorCritic, SharedSetActorCritic
from scalemac_rl.reporting import markdown_report_path, write_csv, write_markdown
from scalemac_rl.rl_evaluation import evaluate_actor_critic
from scalemac_rl.reproducibility import (
    collect_runtime_fingerprint,
    numpy_global_rng_sha256,
    tensor_mapping_sha256,
    torch_cpu_rng_sha256,
    write_runtime_metadata,
)
from scalemac_rl.tradeoff import validation_tradeoff_metrics


PolicyModel = SharedSetActorCritic | RecurrentSharedSetActorCritic


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


def _set_beta_concentration(model: PolicyModel, concentration: float) -> None:
    """Set both Beta action heads to the requested total concentration."""
    if concentration <= 2.0:
        raise ValueError("Beta concentration must be greater than 2")
    raw = math.log(math.expm1(concentration - 2.0))
    with torch.no_grad():
        model.raw_concentration.fill_(raw)


def _current_beta_concentration(model: PolicyModel) -> tuple[float, float]:
    values = torch.nn.functional.softplus(model.raw_concentration.detach()) + 2.0
    return float(values[0].item()), float(values[1].item())


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
    candidate_mode: str = "heuristic",
) -> np.ndarray:
    if candidate_mode == "all":
        return np.stack(
            [build_all_eligible_mask(observation) for observation in observations],
            axis=0,
        )
    if candidate_mode != "heuristic":
        raise ValueError("candidate_mode must be heuristic or all")
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
    model: PolicyModel,
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


def _rppo_update(
    *,
    model: RecurrentSharedSetActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    candidate_masks: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    dones: torch.Tensor,
    hidden_states: torch.Tensor,
    sequence_length: int,
    minibatch_sequences: int,
    hyper: PpoHyperparameters,
) -> dict[str, float]:
    """PPO update over contiguous recurrent sequences using truncated BPTT.

    Input tensors use rollout-major shapes [T,W,...]. Sequence boundaries never
    mix workers, and the hidden state recorded before each action initializes the
    corresponding truncated sequence.
    """
    rollout_steps, workers = observations.shape[:2]
    if rollout_steps % sequence_length != 0:
        raise ValueError(
            "rollout_steps must be divisible by recurrent sequence length"
        )
    sequences_per_worker = rollout_steps // sequence_length
    sequence_refs = [
        (worker, start)
        for worker in range(workers)
        for start in range(0, rollout_steps, sequence_length)
    ]
    order = np.arange(len(sequence_refs))
    losses: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(hyper.update_epochs):
        np.random.shuffle(order)
        stop_early = False
        for batch_start in range(0, len(order), minibatch_sequences):
            batch_ids = order[batch_start : batch_start + minibatch_sequences]
            obs_batch = []
            action_batch = []
            mask_batch = []
            old_logprob_batch = []
            return_batch = []
            advantage_batch = []
            done_batch = []
            hidden_batch = []
            for seq_index in batch_ids:
                worker, start = sequence_refs[int(seq_index)]
                stop = start + sequence_length
                obs_batch.append(observations[start:stop, worker])
                action_batch.append(actions[start:stop, worker])
                mask_batch.append(candidate_masks[start:stop, worker])
                old_logprob_batch.append(old_log_probs[start:stop, worker])
                return_batch.append(returns[start:stop, worker])
                advantage_batch.append(advantages[start:stop, worker])
                done_batch.append(dones[start:stop, worker])
                hidden_batch.append(hidden_states[start, worker])

            obs_mb = torch.stack(obs_batch, dim=0)
            actions_mb = torch.stack(action_batch, dim=0)
            masks_mb = torch.stack(mask_batch, dim=0)
            old_logprobs_mb = torch.stack(old_logprob_batch, dim=0)
            returns_mb = torch.stack(return_batch, dim=0)
            advantages_mb = torch.stack(advantage_batch, dim=0)
            dones_mb = torch.stack(done_batch, dim=0)
            initial_hidden_mb = torch.stack(hidden_batch, dim=0)

            log_prob, entropy, value, _ = model.evaluate_sequence(
                obs_mb, masks_mb, actions_mb, initial_hidden_mb, dones_mb
            )
            log_ratio = log_prob - old_logprobs_mb
            ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
            policy_loss = torch.maximum(
                -advantages_mb * ratio,
                -advantages_mb
                * torch.clamp(
                    ratio, 1.0 - hyper.clip_coef, 1.0 + hyper.clip_coef
                ),
            ).mean()
            value_loss = 0.5 * torch.mean((value - returns_mb) ** 2)
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                + hyper.value_coef * value_loss
                - hyper.entropy_coef * entropy_loss
            )

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
            losses["entropy"].append(float(entropy_loss.item()))
            losses["approx_kl"].append(float(approx_kl.item()))
            losses["clip_fraction"].append(float(clip_fraction.item()))
            if hyper.target_kl > 0.0 and float(approx_kl.item()) > hyper.target_kl:
                stop_early = True
                break
        if stop_early:
            break
    return {name: mean(values) if values else 0.0 for name, values in losses.items()}


def _snapshot(
    model: PolicyModel,
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
    model: PolicyModel,
    optimizer: torch.optim.Optimizer,
    controller: LagrangeController,
) -> None:
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])
    for key, value in snapshot["controller"].items():
        setattr(controller, key, value)


def _checkpoint_payload(
    *,
    model: PolicyModel,
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
        "checkpoint_type": (
            f"{args.scheduler_mode}_constrained_rppo_actor_critic"
            if args.policy_architecture == "recurrent"
            else f"{args.scheduler_mode}_constrained_ppo_actor_critic"
        ),
        "checkpoint_tag": tag,
        "policy_architecture": args.policy_architecture,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_dim": OBSERVATION_FEATURES,
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
            "policy_architecture": args.policy_architecture,
            "recurrent_seq_len": args.recurrent_seq_len,
            "recurrent_minibatch_sequences": args.recurrent_minibatch_sequences,
            "max_candidates": args.max_candidates,
            "candidate_mode": args.candidate_mode,
            "scheduler_mode": args.scheduler_mode,
            "force_harq_retransmissions": args.force_harq_retransmissions,
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
            "max_wait_risk_penalty_weight": args.max_wait_risk_penalty_weight,
            "population_wait_penalty_weight": args.population_wait_penalty_weight,
            "low_throughput_percentile": args.low_throughput_percentile,
            "starvation_threshold_slots": args.starvation_threshold_slots,
            "reward_positive_scale": args.reward_positive_scale,
            "reward_throughput_weight": args.reward_throughput_weight,
            "reward_fairness_weight": args.reward_fairness_weight,
            "reward_service_weight": args.reward_service_weight,
            "reward_deficit_service_weight": args.reward_deficit_service_weight,
            "reward_pf_utility_weight": args.reward_pf_utility_weight,
            "reward_low_throughput_weight": args.reward_low_throughput_weight,
            "reward_urgency_service_weight": args.reward_urgency_service_weight,
            "reward_fairness_delta_weight": args.reward_fairness_delta_weight,
            "reward_pf_utility_delta_weight": args.reward_pf_utility_delta_weight,
            "reward_starvation_penalty_weight": args.reward_starvation_penalty_weight,
            "fairness_target_schedule": args.fairness_target_schedule,
            "reference_deadline_target_slots": args.max_p99_wait_slots,
            "max_wait_target_slots": args.max_wait_slots,
            "tradeoff_target_throughput_score": args.min_throughput_score,
            "milestone_env_steps": args.milestone_env_steps,
            "training_budget_env_steps": args.steps_per_stage * len(args.curriculum),
            "learning_rate": args.lr,
            "learning_rate_end": args.lr_end,
            "entropy_coef": args.entropy_coef,
            "entropy_coef_end": args.entropy_coef_end,
            "beta_concentration_start": args.beta_concentration_start,
            "beta_concentration_end": args.beta_concentration_end,
            "freeze_beta_concentration": args.freeze_beta_concentration,
            "beta_concentration_schedule_managed": (
                args.freeze_beta_concentration
                or abs(args.beta_concentration_end - args.beta_concentration_start) > 1e-12
            ),
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


def _active_fairness_target(
    *,
    stage_index: int,
    stage_count: int,
    stage_env_steps: int,
    steps_per_stage: int,
    final_target: float,
    schedule: list[float],
) -> float:
    """Progressively tighten fairness on the final curriculum stage."""
    if stage_index != stage_count or not schedule:
        return float(final_target)
    progress = min(max(stage_env_steps, 0), max(steps_per_stage - 1, 0))
    segment = min(
        len(schedule) - 1,
        int(progress * len(schedule) / max(steps_per_stage, 1)),
    )
    return float(schedule[segment])


def _load_initial_state(
    *,
    model: PolicyModel,
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
        checkpoint_input_dim = int(checkpoint.get("input_dim", OBSERVATION_FEATURES))
        model.load_compatible_state_dict(checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint and checkpoint_input_dim == OBSERVATION_FEATURES:
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
            model.load_compatible_state_dict(state_dict, strict=True)
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
    model: PolicyModel,
    device: torch.device,
    num_ues: int,
    slots: int,
    seeds: list[int],
    max_candidates: int,
    candidate_mode: str,
    scheduler_mode: str,
    force_harq_retransmissions: bool,
    safety_reserve_ues: int,
    long_wait_threshold: float,
    constraints: ServiceConstraints,
    freeze_static_profiles: bool,
    fixed_profile_seed: int | None,
    cqi_mode: str,
    cqi_temporal_correlation: float,
    cqi_innovation_std: float,
    cqi_update_interval_slots: int,
    cqi_max_delta_per_update: int,
    csi_report_mode: str,
    csi_report_period_slots: int,
    csi_report_delay_slots: int,
    csi_report_error_std: float,
    link_adaptation_mode: str,
    link_adaptation_cqi_backoff: int,
    bler_mismatch_slope: float,
    deadline_risk_start_ratio: float,
    deadline_risk_penalty_weight: float,
    reference_deadline_target_slots: float,
    starvation_threshold_slots: int,
    reward_positive_scale: float,
    reward_throughput_weight: float,
    reward_fairness_weight: float,
    reward_service_weight: float,
    reward_deficit_service_weight: float,
    reward_pf_utility_weight: float,
    reward_low_throughput_weight: float,
    reward_urgency_service_weight: float,
    reward_fairness_delta_weight: float,
    reward_pf_utility_delta_weight: float,
    reward_starvation_penalty_weight: float,
    max_wait_target_slots: float,
    max_wait_risk_penalty_weight: float,
    population_wait_penalty_weight: float,
    low_throughput_percentile: float,
    update_index: int,
    stage_index: int,
    global_env_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = ScaleMacConfig(
        num_ues=num_ues,
        num_prbs=273,
        max_selected_ues=min(64, num_ues, 273),
        episode_slots=slots,
        scheduler_mode=scheduler_mode,
        force_harq_retransmissions=force_harq_retransmissions,
        safety_reserve_ues=min(safety_reserve_ues, min(64, num_ues, 273)),
        safety_wait_threshold_ratio=long_wait_threshold,
        freeze_static_profiles=freeze_static_profiles,
        static_profile_seed=fixed_profile_seed,
        cqi_mode=cqi_mode,
        cqi_temporal_correlation=cqi_temporal_correlation,
        cqi_innovation_std=cqi_innovation_std,
        cqi_update_interval_slots=cqi_update_interval_slots,
        cqi_max_delta_per_update=cqi_max_delta_per_update,
        csi_report_mode=csi_report_mode,
        csi_report_period_slots=csi_report_period_slots,
        csi_report_delay_slots=csi_report_delay_slots,
        csi_report_error_std=csi_report_error_std,
        link_adaptation_mode=link_adaptation_mode,
        link_adaptation_cqi_backoff=link_adaptation_cqi_backoff,
        bler_mismatch_slope=bler_mismatch_slope,
        deadline_target_slots=constraints.max_p99_wait_slots,
        reference_deadline_target_slots=reference_deadline_target_slots,
        deadline_risk_start_ratio=deadline_risk_start_ratio,
        reward_deadline_risk_penalty_weight=deadline_risk_penalty_weight,
        starvation_threshold_slots=starvation_threshold_slots,
        reward_positive_scale=reward_positive_scale,
        reward_throughput_weight=reward_throughput_weight,
        reward_fairness_weight=reward_fairness_weight,
        reward_service_weight=reward_service_weight,
        reward_deficit_service_weight=reward_deficit_service_weight,
        reward_pf_utility_weight=reward_pf_utility_weight,
        reward_low_throughput_weight=reward_low_throughput_weight,
        reward_urgency_service_weight=reward_urgency_service_weight,
        reward_fairness_delta_weight=reward_fairness_delta_weight,
        reward_pf_utility_delta_weight=reward_pf_utility_delta_weight,
        reward_starvation_penalty_weight=reward_starvation_penalty_weight,
        max_wait_target_slots=max_wait_target_slots,
        reward_max_wait_risk_penalty_weight=max_wait_risk_penalty_weight,
        reward_population_wait_penalty_weight=population_wait_penalty_weight,
        low_throughput_percentile=low_throughput_percentile,
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row = evaluate_actor_critic(
            model=model,
            device=device,
            config=config,
            seed=seed,
            name="ppo_validation",
            max_candidates=(num_ues if candidate_mode == "all" else max_candidates),
            candidate_mode=candidate_mode,
            long_wait_threshold=long_wait_threshold,
            constraints=constraints,
        )
        row.update(
            {"update": update_index, "stage": stage_index, "global_env_steps": global_env_steps}
        )
        rows.append(row)

    worst_starvation = max(float(row["max_starvation_rate"]) for row in rows)
    worst_wait = max(float(row["max_p99_wait_slots"]) for row in rows)
    worst_max_wait = max(float(row["max_wait_slots"]) for row in rows)
    minimum_fairness = min(float(row["final_jain_fairness"]) for row in rows)
    (
        starvation_excess,
        wait_excess,
        fairness_excess,
        max_wait_excess,
    ) = constraints.all_excesses(
        starvation_rate=worst_starvation,
        p99_wait_slots=worst_wait,
        jain_fairness=minimum_fairness,
        max_wait_slots=worst_max_wait,
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
        "mean_spectral_efficiency_bps_hz": mean(
            float(row["mean_spectral_efficiency_bps_hz"]) for row in rows
        ),
        "mean_observed_bler": mean(float(row["mean_observed_bler"]) for row in rows),
        "mean_predicted_bler": mean(float(row["mean_predicted_bler"]) for row in rows),
        "mean_mcs_index": mean(float(row["mean_mcs_index"]) for row in rows),
        "mean_harq_retransmission_fraction": mean(
            float(row["mean_harq_retransmission_fraction"]) for row in rows
        ),
        "mean_cqi": mean(float(row["mean_cqi"]) for row in rows),
        "mean_cqi_std": mean(float(row["mean_cqi_std"]) for row in rows),
        "mean_cqi_abs_change_per_slot": mean(
            float(row["mean_cqi_abs_change_per_slot"]) for row in rows
        ),
        "mean_cqi_changed_fraction": mean(
            float(row["mean_cqi_changed_fraction"]) for row in rows
        ),
        "mean_throughput_score": mean(
            float(row["mean_throughput_score"]) for row in rows
        ),
        "mean_jain_fairness": mean(float(row["final_jain_fairness"]) for row in rows),
        "minimum_jain_fairness": minimum_fairness,
        "worst_starvation_rate": worst_starvation,
        "worst_p99_wait_slots": worst_wait,
        "worst_max_wait_slots": worst_max_wait,
        "validation_starvation_excess": starvation_excess,
        "validation_wait_excess": wait_excess,
        "validation_fairness_excess": fairness_excess,
        "validation_max_wait_excess": max_wait_excess,
        "total_constraint_excess": (
            starvation_excess + wait_excess + fairness_excess + max_wait_excess
        ),
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
        "mean_scheduler_selected_count": mean(
            float(row["mean_scheduler_selected_count"]) for row in rows
        ),
        "mean_scheduler_selection_fraction": mean(
            float(row["mean_scheduler_selection_fraction"]) for row in rows
        ),
        "mean_ppo_selected_count": mean(
            float(row["mean_ppo_selected_count"]) for row in rows
        ),
        "mean_rule_selected_count": mean(
            float(row["mean_rule_selected_count"]) for row in rows
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
        description="Attribution-aware constrained PPO for ScaleMAC-RL"
    )
    parser.add_argument("--init-checkpoint", type=Path, default=Path("artifacts/pf_imitation.pt"))
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--curriculum", type=_parse_int_list, default=[128, 256, 600, 1200])
    parser.add_argument("--steps-per-stage", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episode-slots", type=int, default=500)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument(
        "--candidate-mode", choices=["heuristic", "all"], default="heuristic",
        help="heuristic uses a reduced candidate pool; all exposes every UE",
    )
    parser.add_argument(
        "--scheduler-mode", choices=["hybrid", "ppo_only", "rule_only"],
        default="hybrid",
    )
    parser.add_argument(
        "--force-harq-retransmissions", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--safety-reserve-ues", type=int, default=16)
    parser.add_argument("--long-wait-threshold", type=float, default=0.8)
    parser.add_argument("--freeze-static-profiles", action="store_true")
    parser.add_argument("--fixed-profile-seed", type=int, default=None)
    parser.add_argument("--cqi-mode", choices=["static", "correlated"], default="static")
    parser.add_argument("--cqi-temporal-correlation", type=float, default=0.97)
    parser.add_argument("--cqi-innovation-std", type=float, default=0.35)
    parser.add_argument("--cqi-update-interval-slots", type=int, default=1)
    parser.add_argument("--cqi-max-delta-per-update", type=int, default=1)
    parser.add_argument("--csi-report-mode", choices=["perfect", "periodic"], default="perfect")
    parser.add_argument("--csi-report-period-slots", type=int, default=1)
    parser.add_argument("--csi-report-delay-slots", type=int, default=0)
    parser.add_argument("--csi-report-error-std", type=float, default=0.0)
    parser.add_argument(
        "--link-adaptation-mode",
        choices=["legacy_fixed_bler", "cqi_mcs_bler"],
        default="legacy_fixed_bler",
    )
    parser.add_argument("--link-adaptation-cqi-backoff", type=int, default=0)
    parser.add_argument("--bler-mismatch-slope", type=float, default=1.5)
    parser.add_argument("--single-seed-upper-bound", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deadline-risk-start-ratio", type=float, default=0.60)
    parser.add_argument("--deadline-risk-penalty-weight", type=float, default=0.15)
    parser.add_argument("--max-wait-risk-penalty-weight", type=float, default=0.10)
    parser.add_argument("--population-wait-penalty-weight", type=float, default=0.0)
    parser.add_argument("--low-throughput-percentile", type=float, default=10.0)
    parser.add_argument("--starvation-threshold-slots", type=int, default=64)
    parser.add_argument("--reward-positive-scale", type=float, default=1.0)
    parser.add_argument("--reward-throughput-weight", type=float, default=0.45)
    parser.add_argument("--reward-fairness-weight", type=float, default=0.35)
    parser.add_argument("--reward-service-weight", type=float, default=0.15)
    parser.add_argument("--reward-deficit-service-weight", type=float, default=0.05)
    parser.add_argument("--reward-pf-utility-weight", type=float, default=0.0)
    parser.add_argument("--reward-low-throughput-weight", type=float, default=0.0)
    parser.add_argument("--reward-urgency-service-weight", type=float, default=0.0)
    parser.add_argument("--reward-fairness-delta-weight", type=float, default=0.03)
    parser.add_argument("--reward-pf-utility-delta-weight", type=float, default=0.02)
    parser.add_argument("--reward-starvation-penalty-weight", type=float, default=0.50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--policy-architecture",
        choices=["feedforward", "recurrent"],
        default="feedforward",
        help="feedforward PPO or per-UE shared-GRU recurrent PPO",
    )
    parser.add_argument(
        "--recurrent-seq-len",
        type=int,
        default=16,
        help="truncated-BPTT sequence length used only by recurrent PPO",
    )
    parser.add_argument(
        "--recurrent-minibatch-sequences",
        type=int,
        default=4,
        help="number of contiguous recurrent sequences per PPO minibatch",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-end", type=float, default=None,
        help="linearly anneal Adam learning rate to this value; defaults to --lr",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument(
        "--entropy-coef-end", type=float, default=None,
        help="linearly anneal entropy coefficient to this value; defaults to --entropy-coef",
    )
    parser.add_argument(
        "--beta-concentration-start",
        type=float,
        default=20.0,
        help="initial total concentration of each bounded Beta action head",
    )
    parser.add_argument(
        "--beta-concentration-end",
        type=float,
        default=None,
        help=(
            "linearly anneal Beta concentration to this value; when provided, "
            "the concentration is schedule-managed rather than learned"
        ),
    )
    parser.add_argument(
        "--freeze-beta-concentration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep Beta concentration fixed/scheduled instead of optimizing it",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-starvation-rate", type=float, default=0.0)
    parser.add_argument("--max-p99-wait-slots", type=float, default=50.0)
    parser.add_argument("--min-jain-fairness", type=float, default=0.60)
    parser.add_argument(
        "--min-throughput-score",
        type=float,
        default=0.45,
        help="soft fixed-target throughput score used only for best_tradeoff selection",
    )
    parser.add_argument("--max-wait-slots", type=float, default=60.0)
    parser.add_argument(
        "--stage-p99-wait-limits", type=_parse_float_list, default=[80.0, 80.0, 80.0, 50.0]
    )
    parser.add_argument(
        "--final-stage-p99-schedule", type=_parse_float_list, default=[80.0, 65.0, 55.0, 50.0]
    )
    parser.add_argument(
        "--fairness-target-schedule", type=_parse_float_list,
        default=[0.50, 0.55, 0.60],
    )
    parser.add_argument("--starvation-multiplier", type=float, default=5.0)
    parser.add_argument("--wait-multiplier", type=float, default=1.0)
    parser.add_argument("--fairness-multiplier", type=float, default=1.0)
    parser.add_argument("--max-wait-multiplier", type=float, default=1.0)
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
        "--milestone-env-steps",
        type=_parse_int_list,
        default=[],
        help="comma-separated environment-step milestones saved in the checkpoint directory",
    )
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
        "--best-tradeoff-output",
        type=Path,
        default=Path("artifacts/best_tradeoff.pt"),
    )
    parser.add_argument(
        "--best-lowest-violation-output",
        type=Path,
        default=Path("artifacts/best_lowest_violation.pt"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--log-output", type=Path, default=Path("artifacts/ppo_training.csv"))
    parser.add_argument("--validation-output", type=Path, default=Path("artifacts/ppo_validation.csv"))
    parser.add_argument(
        "--runtime-metadata-output",
        type=Path,
        default=None,
        help="optional JSON runtime/RNG/model fingerprint for reproducibility diagnostics",
    )
    parser.add_argument(
        "--checkpoint-manifest-output",
        type=Path,
        default=Path("artifacts/checkpoint_manifest.csv"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("docs/reports"),
        help="directory for generated Markdown reports",
    )
    args = parser.parse_args()

    if args.steps_per_stage <= 0 or args.workers <= 0 or args.rollout_steps <= 0:
        parser.error("steps, workers, and rollout length must be positive")
    if args.recurrent_seq_len <= 0 or args.recurrent_minibatch_sequences <= 0:
        parser.error("recurrent sequence settings must be positive")
    if args.policy_architecture == "recurrent":
        if args.candidate_mode != "all":
            parser.error("recurrent policy currently requires --candidate-mode all to preserve UE memory identity")
        if args.rollout_steps % args.recurrent_seq_len != 0:
            parser.error("recurrent policy requires rollout-steps divisible by recurrent-seq-len")
    if args.max_candidates < 64 and args.candidate_mode != "all":
        parser.error("max_candidates must be at least the Top-K value 64")
    if not 0 <= args.safety_reserve_ues <= 64:
        parser.error("safety_reserve_ues must be in [0, 64]")
    if args.scheduler_mode == "ppo_only" and args.safety_reserve_ues != 0:
        parser.error("ppo_only requires --safety-reserve-ues 0")
    if args.scheduler_mode == "rule_only" and args.safety_reserve_ues not in {0, 64}:
        parser.error("rule_only uses a full 64-UE rule reserve")
    if len(args.stage_p99_wait_limits) != len(args.curriculum):
        parser.error("stage-p99-wait-limits must contain one value per curriculum stage")
    if args.validation_slots <= 0 or args.validation_repeats <= 0:
        parser.error("validation slots and repeats must be positive")
    if args.fixed_profile_seed is not None and args.fixed_profile_seed < 0:
        parser.error("fixed-profile-seed must be non-negative")
    if not 0.0 <= args.cqi_temporal_correlation < 1.0:
        parser.error("cqi-temporal-correlation must be in [0, 1)")
    if args.cqi_innovation_std < 0.0:
        parser.error("cqi-innovation-std must be non-negative")
    if args.cqi_update_interval_slots <= 0:
        parser.error("cqi-update-interval-slots must be positive")
    if args.cqi_max_delta_per_update <= 0:
        parser.error("cqi-max-delta-per-update must be positive")
    if args.csi_report_period_slots <= 0:
        parser.error("csi-report-period-slots must be positive")
    if args.csi_report_delay_slots < 0:
        parser.error("csi-report-delay-slots must be non-negative")
    if args.csi_report_error_std < 0.0:
        parser.error("csi-report-error-std must be non-negative")
    if not 0.0 <= args.deadline_risk_start_ratio < 1.0:
        parser.error("deadline-risk-start-ratio must be in [0, 1)")
    if args.deadline_risk_penalty_weight < 0.0:
        parser.error("deadline-risk-penalty-weight must be non-negative")
    if args.max_wait_risk_penalty_weight < 0.0:
        parser.error("max-wait-risk-penalty-weight must be non-negative")
    if args.population_wait_penalty_weight < 0.0:
        parser.error("population-wait-penalty-weight must be non-negative")
    if not 0.0 < args.low_throughput_percentile < 50.0:
        parser.error("low-throughput-percentile must be in (0, 50)")
    if not 0.0 < args.min_throughput_score <= 1.0:
        parser.error("min-throughput-score must be in (0, 1]")
    if args.starvation_threshold_slots <= 0:
        parser.error("starvation-threshold-slots must be positive")
    if args.reward_positive_scale < 0.0:
        parser.error("reward-positive-scale must be non-negative")
    if args.reward_starvation_penalty_weight < 0.0:
        parser.error("reward-starvation-penalty-weight must be non-negative")
    reward_weight_sum = (
        args.reward_throughput_weight
        + args.reward_fairness_weight
        + args.reward_service_weight
        + args.reward_deficit_service_weight
        + args.reward_pf_utility_weight
        + args.reward_low_throughput_weight
        + args.reward_urgency_service_weight
    )
    if abs(reward_weight_sum - 1.0) > 1e-6:
        parser.error("positive reward weights must sum to 1")
    for name in (
        "reward_pf_utility_weight",
        "reward_low_throughput_weight",
        "reward_urgency_service_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"{name.replace('_', '-')} must be non-negative")
    if args.lr_end is None:
        args.lr_end = args.lr
    if args.entropy_coef_end is None:
        args.entropy_coef_end = args.entropy_coef
    if args.beta_concentration_end is None:
        args.beta_concentration_end = args.beta_concentration_start
    if args.beta_concentration_start <= 2.0 or args.beta_concentration_end <= 2.0:
        parser.error("Beta concentrations must be greater than 2")
    if args.lr <= 0.0 or args.lr_end <= 0.0:
        parser.error("learning rates must be positive")
    if args.entropy_coef < 0.0 or args.entropy_coef_end < 0.0:
        parser.error("entropy coefficients must be non-negative")
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
    if any(step <= 0 for step in args.milestone_env_steps):
        parser.error("milestone environment steps must be positive")
    args.milestone_env_steps = sorted(set(args.milestone_env_steps))

    final_constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
        min_jain_fairness=args.min_jain_fairness,
        max_wait_slots=args.max_wait_slots,
    )
    final_constraints.validate()
    controller = LagrangeController(
        starvation_multiplier=args.starvation_multiplier,
        wait_multiplier=args.wait_multiplier,
        fairness_multiplier=args.fairness_multiplier,
        max_wait_multiplier=args.max_wait_multiplier,
        learning_rate=args.lagrangian_lr,
        max_multiplier=args.max_lagrange_multiplier,
    )
    controller.validate()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    model: PolicyModel
    if args.policy_architecture == "recurrent":
        model = RecurrentSharedSetActorCritic(
            input_dim=OBSERVATION_FEATURES,
            hidden_dim=args.hidden_dim,
            initial_concentration=args.beta_concentration_start,
        ).to(device)
    else:
        model = SharedSetActorCritic(
            input_dim=OBSERVATION_FEATURES,
            hidden_dim=args.hidden_dim,
            initial_concentration=args.beta_concentration_start,
        ).to(device)
    if args.runtime_metadata_output is not None:
        write_runtime_metadata(
            args.runtime_metadata_output,
            {
                "runtime": collect_runtime_fingerprint(),
                "run": {
                    "seed": args.seed,
                    "fixed_profile_seed": args.fixed_profile_seed,
                    "validation_seeds": args.validation_seeds,
                    "device": str(device),
                    "policy_architecture": args.policy_architecture,
                    "recurrent_seq_len": args.recurrent_seq_len,
                    "recurrent_minibatch_sequences": args.recurrent_minibatch_sequences,
                    "cqi_mode": args.cqi_mode,
                    "cqi_temporal_correlation": args.cqi_temporal_correlation,
                    "cqi_innovation_std": args.cqi_innovation_std,
                    "cqi_update_interval_slots": args.cqi_update_interval_slots,
                    "cqi_max_delta_per_update": args.cqi_max_delta_per_update,
                    "csi_report_mode": args.csi_report_mode,
                    "csi_report_period_slots": args.csi_report_period_slots,
                    "csi_report_delay_slots": args.csi_report_delay_slots,
                    "csi_report_error_std": args.csi_report_error_std,
                    "link_adaptation_mode": args.link_adaptation_mode,
                    "link_adaptation_cqi_backoff": args.link_adaptation_cqi_backoff,
                    "bler_mismatch_slope": args.bler_mismatch_slope,
                },
                "rng_after_seed": {
                    "numpy": numpy_global_rng_sha256(),
                    "torch_cpu": torch_cpu_rng_sha256(),
                },
                "initial_model_parameter_sha256": tensor_mapping_sha256(model.state_dict()),
            },
        )
    schedule_managed_concentration = (
        args.freeze_beta_concentration
        or abs(args.beta_concentration_end - args.beta_concentration_start) > 1e-12
    )
    if schedule_managed_concentration:
        model.raw_concentration.requires_grad_(False)
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
    if schedule_managed_concentration:
        _set_beta_concentration(model, args.beta_concentration_start)
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
    checkpoint_manifest_rows: list[dict[str, Any]] = []
    global_env_steps = 0
    update_index = 0
    best_feasible_reward = float("-inf")
    best_reward = float("-inf")
    best_tradeoff_score: tuple[bool, float, float, float] | None = None
    best_tradeoff_saved = False
    best_lowest_violation_score: tuple[float, float, float, float, float] | None = None
    best_lowest_violation_saved = False
    best_feasible_saved = False
    saved_milestones: set[int] = set()
    target_num_ues = args.curriculum[-1]
    last_validation: dict[str, Any] | None = None
    training_started_at = perf_counter()
    total_requested_steps = args.steps_per_stage * len(args.curriculum)
    total_requested_updates = max(
        math.ceil(total_requested_steps / max(args.rollout_steps * args.workers, 1)),
        1,
    )
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
        max_candidates = (
            num_ues
            if args.candidate_mode == "all"
            else min(max(args.max_candidates, max_selected), num_ues)
        )
        if args.scheduler_mode == "ppo_only":
            safety_reserve = 0
        elif args.scheduler_mode == "rule_only":
            safety_reserve = max_selected
        else:
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
            scheduler_mode=args.scheduler_mode,
            force_harq_retransmissions=args.force_harq_retransmissions,
            safety_reserve_ues=safety_reserve,
            safety_wait_threshold_ratio=args.long_wait_threshold,
            freeze_static_profiles=args.freeze_static_profiles or args.single_seed_upper_bound,
            static_profile_seed=profile_seed,
            cqi_mode=args.cqi_mode,
            cqi_temporal_correlation=args.cqi_temporal_correlation,
            cqi_innovation_std=args.cqi_innovation_std,
            cqi_update_interval_slots=args.cqi_update_interval_slots,
            cqi_max_delta_per_update=args.cqi_max_delta_per_update,
            csi_report_mode=args.csi_report_mode,
            csi_report_period_slots=args.csi_report_period_slots,
            csi_report_delay_slots=args.csi_report_delay_slots,
            csi_report_error_std=args.csi_report_error_std,
            link_adaptation_mode=args.link_adaptation_mode,
            link_adaptation_cqi_backoff=args.link_adaptation_cqi_backoff,
            bler_mismatch_slope=args.bler_mismatch_slope,
            deadline_target_slots=initial_p99_limit,
            reference_deadline_target_slots=args.max_p99_wait_slots,
            deadline_risk_start_ratio=args.deadline_risk_start_ratio,
            reward_deadline_risk_penalty_weight=args.deadline_risk_penalty_weight,
            reward_max_wait_risk_penalty_weight=args.max_wait_risk_penalty_weight,
            starvation_threshold_slots=args.starvation_threshold_slots,
            reward_positive_scale=args.reward_positive_scale,
            reward_throughput_weight=args.reward_throughput_weight,
            reward_fairness_weight=args.reward_fairness_weight,
            reward_service_weight=args.reward_service_weight,
            reward_deficit_service_weight=args.reward_deficit_service_weight,
            reward_pf_utility_weight=args.reward_pf_utility_weight,
            reward_low_throughput_weight=args.reward_low_throughput_weight,
            reward_urgency_service_weight=args.reward_urgency_service_weight,
            reward_fairness_delta_weight=args.reward_fairness_delta_weight,
            reward_pf_utility_delta_weight=args.reward_pf_utility_delta_weight,
            reward_starvation_penalty_weight=args.reward_starvation_penalty_weight,
            max_wait_target_slots=args.max_wait_slots,
            reward_population_wait_penalty_weight=args.population_wait_penalty_weight,
            low_throughput_percentile=args.low_throughput_percentile,
            seed=stage_seed,
        )
        config.validate()
        envs = [ScaleMacDownlinkEnv(config) for _ in range(args.workers)]
        observations = np.stack(
            [env.reset(seed=config.seed + worker)[0] for worker, env in enumerate(envs)], axis=0
        )
        recurrent_hidden: torch.Tensor | None = None
        if isinstance(model, RecurrentSharedSetActorCritic):
            recurrent_hidden = model.initial_state(
                args.workers, num_ues, device=device
            )
        stage_env_steps = 0
        stage_best_feasible_reward = float("-inf")
        stage_best_feasible_snapshot: dict[str, Any] | None = None
        stage_best_candidate_snapshot: dict[str, Any] | None = None
        stage_best_candidate_score: tuple[float, float, float, float, float] | None = None
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
            active_max_wait_limit = active_p99_limit + max(
                args.max_wait_slots - args.max_p99_wait_slots, 0.0
            )
            active_fairness_target = _active_fairness_target(
                stage_index=stage_index,
                stage_count=len(args.curriculum),
                stage_env_steps=stage_env_steps,
                steps_per_stage=args.steps_per_stage,
                final_target=args.min_jain_fairness,
                schedule=args.fairness_target_schedule,
            )
            active_constraints = ServiceConstraints(
                max_starvation_rate=args.max_starvation_rate,
                max_p99_wait_slots=active_p99_limit,
                min_jain_fairness=active_fairness_target,
                max_wait_slots=active_max_wait_limit,
            )
            active_constraints.validate()
            config.deadline_target_slots = active_p99_limit
            for env in envs:
                env.config.deadline_target_slots = active_p99_limit

            # Linear schedules are based on the global requested budget so each
            # curriculum stage continues the same optimization schedule.
            progress_fraction = min(
                global_env_steps / max(total_requested_steps, 1), 1.0
            )
            current_lr = args.lr + progress_fraction * (args.lr_end - args.lr)
            current_entropy_coef = (
                args.entropy_coef
                + progress_fraction * (args.entropy_coef_end - args.entropy_coef)
            )
            if schedule_managed_concentration:
                beta_progress_fraction = min(
                    update_index / max(total_requested_updates - 1, 1), 1.0
                )
                scheduled_concentration = (
                    args.beta_concentration_start
                    + beta_progress_fraction
                    * (args.beta_concentration_end - args.beta_concentration_start)
                )
                _set_beta_concentration(model, scheduled_concentration)
            current_beta_priority, current_beta_demand = _current_beta_concentration(model)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            hyper.entropy_coef = current_entropy_coef

            obs_buffer: list[np.ndarray] = []
            action_buffer: list[np.ndarray] = []
            compact_mask_buffer: list[np.ndarray] = []
            logprob_buffer: list[np.ndarray] = []
            value_buffer: list[np.ndarray] = []
            reward_buffer: list[np.ndarray] = []
            done_buffer: list[np.ndarray] = []
            hidden_buffer: list[np.ndarray] = []
            metric_window: dict[str, list[float]] = {
                key: []
                for key in (
                    "base_reward", "core_reward", "final_target_reward",
                    "constrained_reward", "constraint_penalty",
                    "deadline_risk_penalty", "reference_deadline_risk_penalty",
                    "max_wait_risk_penalty", "population_wait_penalty", "deadline_risk",
                    "reference_deadline_risk", "max_wait_risk", "population_wait_risk",
                    "tail_mean_wait",
                    "throughput", "fairness", "jain_fairness", "short_fairness",
                    "deficit_service", "urgency_service", "low_throughput",
                    "pf_utility_score", "fairness_progress", "pf_utility_progress",
                    "service", "starvation", "scheduling_starvation",
                    "p99_wait", "max_wait", "near_deadline_rate",
                    "starvation_excess", "wait_excess", "fairness_excess",
                    "max_wait_excess", "goodput", "mean_cqi", "std_cqi",
                    "cqi_mean_abs_change", "cqi_changed_fraction",
                    "mean_reported_cqi", "csi_abs_error", "csi_p95_abs_error",
                    "csi_stale_fraction", "csi_report_age",
                    "csi_report_generated", "csi_report_delivered",
                    "mcs_index", "modulation_order", "predicted_bler", "observed_bler",
                    "spectral_efficiency", "attempted_spectral_efficiency",
                    "harq_retransmission_attempts", "harq_retransmission_fraction",
                    "candidate_coverage", "harq_retention", "long_wait_retention", "long_wait_missed",
                    "safety_selected", "oldest_selected", "scheduler_selected",
                    "scheduler_fraction", "ppo_selected", "rule_selected",
                    "learned_selected", "learned_fraction",
                )
            }

            for _ in range(args.rollout_steps):
                masks = _candidate_masks(
                    observations, max_candidates, max_selected, args.long_wait_threshold,
                    args.candidate_mode,
                )
                compact_observations, candidate_indices = gather_candidate_batch(observations, masks)
                compact_masks = np.ones(compact_observations.shape[:2], dtype=bool)
                obs_tensor = torch.from_numpy(compact_observations).to(device)
                mask_tensor = torch.from_numpy(compact_masks).to(device)
                with torch.no_grad():
                    if isinstance(model, RecurrentSharedSetActorCritic):
                        assert recurrent_hidden is not None
                        hidden_buffer.append(recurrent_hidden.cpu().numpy().copy())
                        output = model.get_action_and_value(
                            obs_tensor, mask_tensor, recurrent_hidden
                        )
                        next_recurrent_hidden = output.hidden_state
                    else:
                        output = model.get_action_and_value(obs_tensor, mask_tensor)
                        next_recurrent_hidden = None
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
                    (
                        starvation_excess,
                        wait_excess,
                        fairness_excess,
                        max_wait_excess,
                    ) = active_constraints.all_excesses(
                        starvation_rate=float(info["starvation_rate"]),
                        p99_wait_slots=float(info["p99_wait_slots"]),
                        jain_fairness=float(info["fairness_score"]),
                        max_wait_slots=float(info["max_wait_slots"]),
                    )
                    constrained_reward, constraint_penalty = controller.adjusted_reward(
                        base_reward,
                        starvation_excess=starvation_excess,
                        wait_excess=wait_excess,
                        fairness_excess=fairness_excess,
                        max_wait_excess=max_wait_excess,
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
                        "max_wait_risk_penalty": float(
                            info["reward_max_wait_risk_penalty"]
                        ),
                        "population_wait_penalty": float(
                            info["reward_population_wait_penalty"]
                        ),
                        "deadline_risk": float(info["deadline_risk"]),
                        "reference_deadline_risk": float(info["reference_deadline_risk"]),
                        "max_wait_risk": float(info["max_wait_risk"]),
                        "population_wait_risk": float(info["population_wait_risk"]),
                        "tail_mean_wait": float(info["tail_mean_wait_slots"]),
                        "throughput": float(info["throughput_score"]),
                        "fairness": float(info["fairness_score"]),
                        "jain_fairness": float(info["jain_fairness"]),
                        "short_fairness": float(info["short_term_jain_fairness"]),
                        "deficit_service": float(info["deficit_service_score"]),
                        "urgency_service": float(info["urgency_service_score"]),
                        "low_throughput": float(info["low_throughput_score"]),
                        "pf_utility_score": float(info["pf_utility_score"]),
                        "fairness_progress": float(info["fairness_progress"]),
                        "pf_utility_progress": float(info["pf_utility_progress"]),
                        "service": float(info["service_score"]),
                        "starvation": float(info["starvation_rate"]),
                        "scheduling_starvation": float(
                            info["scheduling_starvation_rate"]
                        ),
                        "p99_wait": float(info["p99_wait_slots"]),
                        "max_wait": float(info["max_wait_slots"]),
                        "near_deadline_rate": float(info["near_deadline_rate"]),
                        "starvation_excess": starvation_excess,
                        "wait_excess": wait_excess,
                        "fairness_excess": fairness_excess,
                        "max_wait_excess": max_wait_excess,
                        "goodput": float(info["cell_goodput_bits"]),
                        "mean_cqi": float(info["mean_cqi"]),
                        "std_cqi": float(info["std_cqi"]),
                        "cqi_mean_abs_change": float(info["cqi_mean_abs_change"]),
                        "cqi_changed_fraction": float(info["cqi_changed_fraction"]),
                        "mean_reported_cqi": float(info["mean_reported_cqi"]),
                        "csi_abs_error": float(info["mean_csi_abs_error"]),
                        "csi_p95_abs_error": float(info["p95_csi_abs_error"]),
                        "csi_stale_fraction": float(info["csi_stale_fraction"]),
                        "csi_report_age": float(info["csi_report_age_slots"]),
                        "csi_report_generated": float(info["csi_report_generated"]),
                        "csi_report_delivered": float(info["csi_report_delivered"]),
                        "mcs_index": float(info["mean_mcs_index"]),
                        "modulation_order": float(info["mean_modulation_order"]),
                        "predicted_bler": float(info["mean_predicted_bler"]),
                        "observed_bler": float(info["observed_bler"]),
                        "spectral_efficiency": float(info["spectral_efficiency_bps_hz"]),
                        "attempted_spectral_efficiency": float(
                            info["attempted_spectral_efficiency_bps_hz"]
                        ),
                        "harq_retransmission_attempts": float(
                            info["harq_retransmission_attempts"]
                        ),
                        "harq_retransmission_fraction": float(
                            info["harq_retransmission_fraction"]
                        ),
                        "candidate_coverage": diagnostics.candidate_coverage,
                        "harq_retention": diagnostics.harq_retention_rate,
                        "long_wait_retention": diagnostics.long_wait_retention_rate,
                        "long_wait_missed": float(diagnostics.long_wait_missed_count),
                        "safety_selected": float(info["safety_selected_count"]),
                        "oldest_selected": float(info["forced_oldest_wait_count"]),
                        "scheduler_selected": float(info["scheduler_selected_count"]),
                        "scheduler_fraction": float(info["scheduler_selection_fraction"]),
                        "ppo_selected": float(info["ppo_selected_count"]),
                        "rule_selected": float(info["rule_selected_count"]),
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
                if isinstance(model, RecurrentSharedSetActorCritic):
                    assert next_recurrent_hidden is not None
                    keep = torch.from_numpy(1.0 - dones).to(
                        device=device, dtype=next_recurrent_hidden.dtype
                    ).view(args.workers, 1, 1)
                    recurrent_hidden = next_recurrent_hidden * keep

            with torch.no_grad():
                next_masks_full = _candidate_masks(
                    observations, max_candidates, max_selected, args.long_wait_threshold,
                    args.candidate_mode,
                )
                next_compact_obs, _ = gather_candidate_batch(observations, next_masks_full)
                next_obs_tensor = torch.from_numpy(next_compact_obs).to(device)
                next_compact_masks = torch.ones(
                    next_compact_obs.shape[:2], dtype=torch.bool, device=device
                )
                if isinstance(model, RecurrentSharedSetActorCritic):
                    assert recurrent_hidden is not None
                    next_value = model.get_action_and_value(
                        next_obs_tensor,
                        next_compact_masks,
                        recurrent_hidden,
                        deterministic=True,
                    ).value.cpu().numpy()
                else:
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
            obs_array = np.asarray(obs_buffer)
            action_array = np.asarray(action_buffer)
            mask_array = np.asarray(compact_mask_buffer)
            logprob_array = np.asarray(logprob_buffer)
            advantages_norm = (advantages_np - advantages_np.mean()) / (
                advantages_np.std() + 1e-8
            )
            if isinstance(model, RecurrentSharedSetActorCritic):
                recurrent_obs = torch.from_numpy(obs_array).to(device)
                recurrent_actions = torch.from_numpy(action_array).to(device)
                recurrent_masks = torch.from_numpy(mask_array).to(device)
                recurrent_logprobs = torch.from_numpy(logprob_array).to(device)
                recurrent_returns = torch.from_numpy(returns_np).to(device)
                recurrent_advantages = torch.from_numpy(advantages_norm).to(device)
                recurrent_dones = torch.from_numpy(dones_np).to(device)
                recurrent_hidden_states = torch.from_numpy(
                    np.asarray(hidden_buffer)
                ).to(device)
                update_metrics = _rppo_update(
                    model=model,
                    optimizer=optimizer,
                    observations=recurrent_obs,
                    actions=recurrent_actions,
                    candidate_masks=recurrent_masks,
                    old_log_probs=recurrent_logprobs,
                    returns=recurrent_returns,
                    advantages=recurrent_advantages,
                    dones=recurrent_dones,
                    hidden_states=recurrent_hidden_states,
                    sequence_length=args.recurrent_seq_len,
                    minibatch_sequences=args.recurrent_minibatch_sequences,
                    hyper=hyper,
                )
            else:
                flat_obs = torch.from_numpy(
                    obs_array.reshape(-1, candidate_count, OBSERVATION_FEATURES)
                ).to(device)
                flat_actions = torch.from_numpy(
                    action_array.reshape(-1, candidate_count, 2)
                ).to(device)
                flat_masks = torch.from_numpy(
                    mask_array.reshape(-1, candidate_count)
                ).to(device)
                flat_logprobs = torch.from_numpy(logprob_array.reshape(-1)).to(device)
                flat_returns = torch.from_numpy(returns_np.reshape(-1)).to(device)
                flat_advantages = torch.from_numpy(advantages_norm.reshape(-1)).to(device)
                update_metrics = _ppo_update(
                    model=model, optimizer=optimizer, observations=flat_obs,
                    actions=flat_actions, candidate_masks=flat_masks,
                    old_log_probs=flat_logprobs, returns=flat_returns,
                    advantages=flat_advantages, hyper=hyper,
                )
            controller.update(
                mean_starvation_excess=mean(metric_window["starvation_excess"]),
                mean_wait_excess=mean(metric_window["wait_excess"]),
                mean_fairness_excess=mean(metric_window["fairness_excess"]),
                mean_max_wait_excess=mean(metric_window["max_wait_excess"]),
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
                "stage_min_jain_fairness": active_constraints.min_jain_fairness,
                "stage_max_wait_slots": active_constraints.max_wait_slots,
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
                "mean_max_wait_risk_penalty": mean(
                    metric_window["max_wait_risk_penalty"]
                ),
                "mean_population_wait_penalty": mean(
                    metric_window["population_wait_penalty"]
                ),
                "mean_deadline_risk": mean(metric_window["deadline_risk"]),
                "mean_reference_deadline_risk": mean(
                    metric_window["reference_deadline_risk"]
                ),
                "mean_max_wait_risk": mean(metric_window["max_wait_risk"]),
                "mean_population_wait_risk": mean(metric_window["population_wait_risk"]),
                "mean_tail_mean_wait_slots": mean(metric_window["tail_mean_wait"]),
                "mean_goodput_bits_per_slot": mean(metric_window["goodput"]),
                "mean_cqi": mean(metric_window["mean_cqi"]),
                "mean_cqi_std": mean(metric_window["std_cqi"]),
                "mean_cqi_abs_change_per_slot": mean(metric_window["cqi_mean_abs_change"]),
                "mean_cqi_changed_fraction": mean(metric_window["cqi_changed_fraction"]),
                "mean_reported_cqi": mean(metric_window["mean_reported_cqi"]),
                "mean_csi_abs_error": mean(metric_window["csi_abs_error"]),
                "max_p95_csi_abs_error": max(metric_window["csi_p95_abs_error"]),
                "mean_csi_stale_fraction": mean(metric_window["csi_stale_fraction"]),
                "mean_csi_report_age_slots": mean(metric_window["csi_report_age"]),
                "mean_csi_report_generated_rate": mean(metric_window["csi_report_generated"]),
                "mean_csi_report_delivered_rate": mean(metric_window["csi_report_delivered"]),
                "mean_mcs_index": mean(metric_window["mcs_index"]),
                "mean_modulation_order": mean(metric_window["modulation_order"]),
                "mean_predicted_bler": mean(metric_window["predicted_bler"]),
                "mean_observed_bler": mean(metric_window["observed_bler"]),
                "mean_spectral_efficiency_bps_hz": mean(metric_window["spectral_efficiency"]),
                "mean_attempted_spectral_efficiency_bps_hz": mean(
                    metric_window["attempted_spectral_efficiency"]
                ),
                "mean_harq_retransmission_attempts": mean(
                    metric_window["harq_retransmission_attempts"]
                ),
                "mean_harq_retransmission_fraction": mean(
                    metric_window["harq_retransmission_fraction"]
                ),
                "mean_throughput_score": mean(metric_window["throughput"]),
                "mean_fairness_score": mean(metric_window["fairness"]),
                "mean_jain_fairness": mean(metric_window["jain_fairness"]),
                "mean_short_term_jain_fairness": mean(metric_window["short_fairness"]),
                "mean_deficit_service_score": mean(metric_window["deficit_service"]),
                "mean_urgency_service_score": mean(metric_window["urgency_service"]),
                "mean_low_throughput_score": mean(metric_window["low_throughput"]),
                "mean_pf_utility_score": mean(metric_window["pf_utility_score"]),
                "mean_fairness_progress": mean(metric_window["fairness_progress"]),
                "mean_pf_utility_progress": mean(metric_window["pf_utility_progress"]),
                "mean_service_score": mean(metric_window["service"]),
                "mean_starvation_rate": mean(metric_window["starvation"]),
                "max_starvation_rate": max(metric_window["starvation"]),
                "mean_scheduling_starvation_rate": mean(
                    metric_window["scheduling_starvation"]
                ),
                "mean_near_deadline_rate": mean(metric_window["near_deadline_rate"]),
                "mean_p99_wait_slots": mean(metric_window["p99_wait"]),
                "max_p99_wait_slots": max(metric_window["p99_wait"]),
                "mean_max_wait_slots": mean(metric_window["max_wait"]),
                "max_wait_slots": max(metric_window["max_wait"]),
                "mean_starvation_excess": mean(metric_window["starvation_excess"]),
                "mean_wait_excess": mean(metric_window["wait_excess"]),
                "mean_fairness_excess": mean(metric_window["fairness_excess"]),
                "mean_max_wait_excess": mean(metric_window["max_wait_excess"]),
                "starvation_multiplier": controller.starvation_multiplier,
                "wait_multiplier": controller.wait_multiplier,
                "fairness_multiplier": controller.fairness_multiplier,
                "max_wait_multiplier": controller.max_wait_multiplier,
                "mean_candidate_coverage": mean(metric_window["candidate_coverage"]),
                "mean_harq_retention_rate": mean(metric_window["harq_retention"]),
                "mean_long_wait_retention_rate": mean(metric_window["long_wait_retention"]),
                "max_long_wait_missed_count": max(metric_window["long_wait_missed"]),
                "mean_safety_selected_count": mean(metric_window["safety_selected"]),
                "mean_oldest_selected_count": mean(metric_window["oldest_selected"]),
                "mean_scheduler_selected_count": mean(metric_window["scheduler_selected"]),
                "mean_scheduler_selection_fraction": mean(metric_window["scheduler_fraction"]),
                "mean_ppo_selected_count": mean(metric_window["ppo_selected"]),
                "mean_rule_selected_count": mean(metric_window["rule_selected"]),
                "learning_rate": current_lr,
                "entropy_coef": current_entropy_coef,
                "beta_concentration_priority": current_beta_priority,
                "beta_concentration_demand": current_beta_demand,
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
                fair=f"{row['mean_jain_fairness']:.3f}",
                p99=f"{row['mean_p99_wait_slots']:.1f}",
                maxwait=f"{row['mean_max_wait_slots']:.0f}",
                starve=f"{row['mean_starvation_rate']:.3f}",
                goodput=f"{row['mean_goodput_bits_per_slot'] / 1000.0:.1f}k",
            )
            if not args.progress:
                print(
                    f"stage={stage_index}/{len(args.curriculum)} ues={num_ues:4d} "
                    f"steps={stage_env_steps:7d}/{args.steps_per_stage} "
                    f"core={row['mean_core_reward']:.4f} "
                    f"final={row['mean_final_target_reward']:.4f} "
                    f"train={row['mean_training_reward']:.4f} "
                    f"fair={row['mean_jain_fairness']:.4f} "
                    f"p99={row['mean_p99_wait_slots']:.1f} "
                    f"maxwait={row['mean_max_wait_slots']:.1f} "
                    f"starvation={row['mean_starvation_rate']:.4f}"
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
                    candidate_mode=args.candidate_mode,
                    scheduler_mode=args.scheduler_mode,
                    force_harq_retransmissions=args.force_harq_retransmissions,
                    safety_reserve_ues=safety_reserve,
                    long_wait_threshold=args.long_wait_threshold,
                    constraints=active_constraints,
                    freeze_static_profiles=config.freeze_static_profiles,
                    fixed_profile_seed=config.static_profile_seed,
                    cqi_mode=config.cqi_mode,
                    cqi_temporal_correlation=config.cqi_temporal_correlation,
                    cqi_innovation_std=config.cqi_innovation_std,
                    cqi_update_interval_slots=config.cqi_update_interval_slots,
                    cqi_max_delta_per_update=config.cqi_max_delta_per_update,
                    csi_report_mode=config.csi_report_mode,
                    csi_report_period_slots=config.csi_report_period_slots,
                    csi_report_delay_slots=config.csi_report_delay_slots,
                    csi_report_error_std=config.csi_report_error_std,
                    link_adaptation_mode=config.link_adaptation_mode,
                    link_adaptation_cqi_backoff=config.link_adaptation_cqi_backoff,
                    bler_mismatch_slope=config.bler_mismatch_slope,
                    deadline_risk_start_ratio=args.deadline_risk_start_ratio,
                    deadline_risk_penalty_weight=args.deadline_risk_penalty_weight,
                    reference_deadline_target_slots=args.max_p99_wait_slots,
                    starvation_threshold_slots=args.starvation_threshold_slots,
                    reward_positive_scale=args.reward_positive_scale,
                    reward_throughput_weight=args.reward_throughput_weight,
                    reward_fairness_weight=args.reward_fairness_weight,
                    reward_service_weight=args.reward_service_weight,
                    reward_deficit_service_weight=args.reward_deficit_service_weight,
                    reward_pf_utility_weight=args.reward_pf_utility_weight,
                    reward_low_throughput_weight=args.reward_low_throughput_weight,
                    reward_urgency_service_weight=args.reward_urgency_service_weight,
                    reward_fairness_delta_weight=args.reward_fairness_delta_weight,
                    reward_pf_utility_delta_weight=args.reward_pf_utility_delta_weight,
                    reward_starvation_penalty_weight=args.reward_starvation_penalty_weight,
                    max_wait_target_slots=args.max_wait_slots,
                    max_wait_risk_penalty_weight=args.max_wait_risk_penalty_weight,
                    population_wait_penalty_weight=args.population_wait_penalty_weight,
                    low_throughput_percentile=args.low_throughput_percentile,
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
                    mean_fairness_excess=(
                        args.validation_lagrangian_scale
                        * float(summary["validation_fairness_excess"])
                    ),
                    mean_max_wait_excess=(
                        args.validation_lagrangian_scale
                        * float(summary["validation_max_wait_excess"])
                    ),
                )
                summary["starvation_multiplier_after_validation"] = controller.starvation_multiplier
                summary["wait_multiplier_after_validation"] = controller.wait_multiplier
                summary["fairness_multiplier_after_validation"] = controller.fairness_multiplier
                summary["max_wait_multiplier_after_validation"] = controller.max_wait_multiplier

                score = (
                    float(summary["validation_starvation_excess"]),
                    float(summary["validation_max_wait_excess"]),
                    float(summary["validation_wait_excess"]),
                    float(summary["validation_fairness_excess"]),
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

                # Freeze the exact validated model before any rollback. Previous
                # versions could save a later validation label with rolled-back weights.
                validation_payload = _checkpoint_payload(
                    model=model, optimizer=optimizer, args=args,
                    initialized_from=initialized_from,
                    global_env_steps=global_env_steps, update_index=update_index,
                    stage_index=stage_index, num_ues=num_ues,
                    controller=controller, constraints=final_constraints,
                    tag="validation", validation=summary,
                )

                if num_ues == target_num_ues:
                    (
                        final_starvation_excess,
                        final_wait_excess,
                        final_fairness_excess,
                        final_max_wait_excess,
                    ) = final_constraints.all_excesses(
                        starvation_rate=float(summary["worst_starvation_rate"]),
                        p99_wait_slots=float(summary["worst_p99_wait_slots"]),
                        jain_fairness=float(summary["minimum_jain_fairness"]),
                        max_wait_slots=float(summary["worst_max_wait_slots"]),
                    )
                    summary["final_target_starvation_excess"] = final_starvation_excess
                    summary["final_target_wait_excess"] = final_wait_excess
                    summary["final_target_fairness_excess"] = final_fairness_excess
                    summary["final_target_max_wait_excess"] = final_max_wait_excess
                    summary["final_target_total_constraint_excess"] = (
                        final_starvation_excess
                        + final_wait_excess
                        + final_fairness_excess
                        + final_max_wait_excess
                    )
                    tradeoff_metrics = validation_tradeoff_metrics(
                        mean_throughput_score=float(summary["mean_throughput_score"]),
                        minimum_jain_fairness=float(summary["minimum_jain_fairness"]),
                        worst_starvation_rate=float(summary["worst_starvation_rate"]),
                        worst_p99_wait_slots=float(summary["worst_p99_wait_slots"]),
                        worst_max_wait_slots=float(summary["worst_max_wait_slots"]),
                        target_throughput_score=float(args.min_throughput_score),
                        target_jain_fairness=float(final_constraints.min_jain_fairness),
                        target_starvation_rate=float(final_constraints.max_starvation_rate),
                        target_p99_wait_slots=float(final_constraints.max_p99_wait_slots),
                        target_max_wait_slots=float(final_constraints.max_wait_slots),
                    )
                    summary.update(tradeoff_metrics)
                    current_tradeoff_score = (
                        not bool(tradeoff_metrics["target_starvation_feasible"]),
                        float(tradeoff_metrics["target_worst_kpi_gap"]),
                        -float(tradeoff_metrics["target_balanced_score"]),
                        -float(summary["mean_goodput_bits_per_slot"]),
                    )
                    if best_tradeoff_score is None or current_tradeoff_score < best_tradeoff_score:
                        best_tradeoff_score = current_tradeoff_score
                        best_tradeoff_saved = True
                        tradeoff_payload = copy.deepcopy(validation_payload)
                        tradeoff_payload["checkpoint_tag"] = "best_tradeoff"
                        _save_checkpoint(args.best_tradeoff_output, tradeoff_payload)
                        checkpoint_manifest_rows.append({
                            "checkpoint": str(args.best_tradeoff_output),
                            "tag": "best_tradeoff",
                            "update": update_index,
                            "global_env_steps": global_env_steps,
                            "selection_reason": "minimize_worst_target_gap_then_maximize_balanced_score",
                            "starvation_excess": final_starvation_excess,
                            "max_wait_excess": final_max_wait_excess,
                            "p99_wait_excess": final_wait_excess,
                            "fairness_excess": final_fairness_excess,
                            "target_worst_kpi_gap": float(tradeoff_metrics["target_worst_kpi_gap"]),
                            "target_balanced_score": float(tradeoff_metrics["target_balanced_score"]),
                            "final_target_reward": float(summary["mean_final_target_reward"]),
                        })
                    final_score = (
                        float(final_starvation_excess),
                        float(final_max_wait_excess),
                        float(final_wait_excess),
                        float(final_fairness_excess),
                        -float(summary["mean_final_target_reward"]),
                    )
                    if (
                        best_lowest_violation_score is None
                        or final_score < best_lowest_violation_score
                    ):
                        best_lowest_violation_score = final_score
                        best_lowest_violation_saved = True
                        lowest_payload = copy.deepcopy(validation_payload)
                        lowest_payload["checkpoint_tag"] = "best_lowest_violation"
                        _save_checkpoint(args.best_lowest_violation_output, lowest_payload)
                        checkpoint_manifest_rows.append({
                            "checkpoint": str(args.best_lowest_violation_output),
                            "tag": "best_lowest_violation",
                            "update": update_index,
                            "global_env_steps": global_env_steps,
                            "selection_reason": "lexicographic_final_constraints",
                            "starvation_excess": final_starvation_excess,
                            "max_wait_excess": final_max_wait_excess,
                            "p99_wait_excess": final_wait_excess,
                            "fairness_excess": final_fairness_excess,
                            "final_target_reward": float(summary["mean_final_target_reward"]),
                        })
                    if float(summary["mean_final_target_reward"]) > best_reward:
                        best_reward = float(summary["mean_final_target_reward"])
                        reward_payload = copy.deepcopy(validation_payload)
                        reward_payload["checkpoint_tag"] = "best_reward"
                        _save_checkpoint(args.best_reward_output, reward_payload)
                        checkpoint_manifest_rows.append({
                            "checkpoint": str(args.best_reward_output),
                            "tag": "best_reward",
                            "update": update_index,
                            "global_env_steps": global_env_steps,
                            "selection_reason": "highest_final_target_reward",
                            "starvation_excess": final_starvation_excess,
                            "max_wait_excess": final_max_wait_excess,
                            "p99_wait_excess": final_wait_excess,
                            "fairness_excess": final_fairness_excess,
                            "final_target_reward": float(summary["mean_final_target_reward"]),
                        })
                    final_feasible = all(
                        excess <= 1e-12
                        for excess in (
                            final_starvation_excess,
                            final_wait_excess,
                            final_fairness_excess,
                            final_max_wait_excess,
                        )
                    )
                    if (
                        final_feasible
                        and float(summary["mean_final_target_reward"]) > best_feasible_reward
                    ):
                        best_feasible_reward = float(summary["mean_final_target_reward"])
                        best_feasible_saved = True
                        feasible_payload = copy.deepcopy(validation_payload)
                        feasible_payload["checkpoint_tag"] = "best_feasible"
                        _save_checkpoint(args.best_feasible_output, feasible_payload)
                        checkpoint_manifest_rows.append({
                            "checkpoint": str(args.best_feasible_output),
                            "tag": "best_feasible",
                            "update": update_index,
                            "global_env_steps": global_env_steps,
                            "selection_reason": "all_final_constraints_feasible",
                            "starvation_excess": final_starvation_excess,
                            "max_wait_excess": final_max_wait_excess,
                            "p99_wait_excess": final_wait_excess,
                            "fairness_excess": final_fairness_excess,
                            "final_target_reward": float(summary["mean_final_target_reward"]),
                        })

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
                    f"fairness={summary['minimum_jain_fairness']:.4f} "
                    f"worst_starvation={summary['worst_starvation_rate']:.4f} "
                    f"worst_p99={summary['worst_p99_wait_slots']:.1f} "
                    f"worst_max_wait={summary['worst_max_wait_slots']:.1f} "
                    f"feasible={summary['constraint_feasible']} rollback={summary['rolled_back']}"
                )
                if args.progress:
                    progress.write(validation_message)
                else:
                    print(validation_message)


            for milestone in args.milestone_env_steps:
                if milestone in saved_milestones or global_env_steps < milestone:
                    continue
                milestone_path = args.checkpoint_dir / f"milestone_{milestone:07d}.pt"
                _save_checkpoint(
                    milestone_path,
                    _checkpoint_payload(
                        model=model, optimizer=optimizer, args=args,
                        initialized_from=initialized_from,
                        global_env_steps=global_env_steps, update_index=update_index,
                        stage_index=stage_index, num_ues=num_ues,
                        controller=controller, constraints=active_constraints,
                        tag=f"milestone_{milestone}", validation=last_validation,
                    ),
                )
                saved_milestones.add(milestone)
                checkpoint_manifest_rows.append({
                    "checkpoint": str(milestone_path),
                    "tag": f"milestone_{milestone}",
                    "update": update_index,
                    "global_env_steps": global_env_steps,
                    "selection_reason": "requested_environment_step_milestone",
                })

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
        markdown_report_path(args.log_output, docs_dir=args.report_dir),
        title=f"ScaleMAC-RL {args.scheduler_mode} constrained PPO training",
        description=(
            "Constraint-aware PPO run with configurable full-UE or candidate inputs, "
            "validation-driven Lagrange updates, rollback, and checkpoint selection."
        ),
        rows=log_rows,
        notes=(
            f"Initialized from: `{initialized_from}`",
            f"Curriculum UE stages: {args.curriculum}",
            f"Environment steps per stage: {args.steps_per_stage}",
            f"Candidate mode: {args.candidate_mode}; pool: {args.max_candidates}; safety reserve: {args.safety_reserve_ues}; Top-K: 64",
            f"Stage P99 limits: {args.stage_p99_wait_limits}; fixed P99 target: {args.max_p99_wait_slots}",
            f"Fairness target: Jain >= {args.min_jain_fairness}; maximum successful-delivery wait: {args.max_wait_slots} slots",
            f"Starvation definition: no successful delivery for >= {args.starvation_threshold_slots} slots",
            (
                "Reward weights: "
                f"throughput={args.reward_throughput_weight}, "
                f"fairness={args.reward_fairness_weight}, "
                f"PF={args.reward_pf_utility_weight}, "
                f"low-throughput={args.reward_low_throughput_weight}, "
                f"service={args.reward_service_weight}, "
                f"deficit={args.reward_deficit_service_weight}, "
                f"urgency={args.reward_urgency_service_weight}"
            ),
            f"LR schedule: {args.lr} -> {args.lr_end}; entropy: {args.entropy_coef} -> {args.entropy_coef_end}; "
            f"Beta concentration: {args.beta_concentration_start} -> {args.beta_concentration_end} "
            f"({'managed' if schedule_managed_concentration else 'learned'})",
            f"Total wall-clock training time: {total_elapsed_seconds:.1f} seconds",
            "best_tradeoff minimizes the largest fixed-target KPI gap, then maximizes the geometric balanced score.",
            "Workers are vectorized environments in one process, not distributed RL.",
            "This remains the fast surrogate, not 5G-LENA.",
        ),
    )
    write_csv(args.validation_output, validation_rows)
    write_markdown(
        markdown_report_path(args.validation_output, docs_dir=args.report_dir),
        title="ScaleMAC-RL repeated held-out PPO validation",
        description="Per-episode validation used for dual updates, rollback, and checkpoint selection.",
        rows=validation_rows,
    )
    validation_summary_output = args.validation_output.with_name(
        f"{args.validation_output.stem}_summary.csv"
    )
    write_csv(validation_summary_output, validation_summary_rows)
    write_markdown(
        markdown_report_path(validation_summary_output, docs_dir=args.report_dir),
        title="ScaleMAC-RL held-out PPO validation summary",
        description="Worst-case constraints, grant attribution, and rollback status at each validation.",
        rows=validation_summary_rows,
    )
    write_csv(args.checkpoint_manifest_output, checkpoint_manifest_rows)
    write_markdown(
        markdown_report_path(args.checkpoint_manifest_output, docs_dir=args.report_dir),
        title="ScaleMAC-RL checkpoint selection manifest",
        description=(
            "Checkpoint provenance. Lowest-violation ranking is lexicographic: "
            "successful-delivery starvation, maximum wait, P99 wait, fairness, then reward. "
            "best_tradeoff minimizes the fixed-target worst KPI gap before balanced score and goodput."
        ),
        rows=checkpoint_manifest_rows,
    )

    print(f"saved: {args.output}")
    print(f"saved: {args.best_reward_output}")
    if best_tradeoff_saved:
        print(f"saved: {args.best_tradeoff_output}")
    if best_lowest_violation_saved:
        print(f"saved: {args.best_lowest_violation_output}")
    if best_feasible_saved:
        print(f"saved: {args.best_feasible_output}")
    else:
        print("warning: no final-stage validation checkpoint satisfied the official constraints")
    print(f"saved: {args.log_output}")
    print(f"saved: {args.validation_output}")
    print(f"saved: {validation_summary_output}")
    print(f"saved: {args.checkpoint_manifest_output}")
    print(
        f"training_time={total_elapsed_seconds:.1f}s "
        f"throughput={global_env_steps / max(total_elapsed_seconds, 1e-9):.1f} env_steps/s"
    )


if __name__ == "__main__":
    main()
