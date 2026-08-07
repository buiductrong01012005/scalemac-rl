from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn

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
            mb = torch.as_tensor(indices[start : start + hyper.minibatch_size], device=observations.device)
            output = model.get_action_and_value(
                observations[mb],
                candidate_masks[mb],
                action=actions[mb],
            )
            log_ratio = output.log_prob - old_log_probs[mb]
            ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))

            mb_adv = advantages[mb]
            policy_loss_1 = -mb_adv * ratio
            policy_loss_2 = -mb_adv * torch.clamp(
                ratio,
                1.0 - hyper.clip_coef,
                1.0 + hyper.clip_coef,
            )
            policy_loss = torch.maximum(policy_loss_1, policy_loss_2).mean()
            value_loss = 0.5 * torch.mean((output.value - returns[mb]) ** 2)
            entropy = output.entropy.mean()
            loss = policy_loss + hyper.value_coef * value_loss - hyper.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyper.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean((ratio - 1.0) - log_ratio).abs()
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > hyper.clip_coef).float())
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
        "checkpoint_type": "constrained_ppo_actor_critic",
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
            "long_wait_threshold": args.long_wait_threshold,
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


def _validate(
    *,
    model: SharedSetActorCritic,
    device: torch.device,
    num_ues: int,
    slots: int,
    seeds: list[int],
    max_candidates: int,
    long_wait_threshold: float,
    constraints: ServiceConstraints,
    update_index: int,
    stage_index: int,
    global_env_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = ScaleMacConfig(
        num_ues=num_ues,
        num_prbs=273,
        max_selected_ues=min(64, num_ues, 273),
        episode_slots=slots,
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
            {
                "update": update_index,
                "stage": stage_index,
                "global_env_steps": global_env_steps,
            }
        )
        rows.append(row)

    summary: dict[str, Any] = {
        "update": update_index,
        "stage": stage_index,
        "num_ues": num_ues,
        "global_env_steps": global_env_steps,
        "validation_seeds": len(rows),
        "mean_reward": mean(float(row["mean_reward"]) for row in rows),
        "mean_goodput_bits_per_slot": mean(
            float(row["mean_goodput_bits_per_slot"]) for row in rows
        ),
        "mean_jain_fairness": mean(float(row["final_jain_fairness"]) for row in rows),
        "worst_starvation_rate": max(float(row["max_starvation_rate"]) for row in rows),
        "worst_p99_wait_slots": max(float(row["max_p99_wait_slots"]) for row in rows),
        "mean_candidate_coverage": mean(float(row["mean_candidate_coverage"]) for row in rows),
        "mean_long_wait_retention_rate": mean(
            float(row["mean_long_wait_retention_rate"]) for row in rows
        ),
        "mean_harq_retention_rate": mean(
            float(row["mean_harq_retention_rate"]) for row in rows
        ),
        "constraint_feasible": validation_feasible(rows, constraints),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained curriculum PPO for ScaleMAC-RL")
    parser.add_argument("--init-checkpoint", type=Path, default=Path("artifacts/pf_imitation.pt"))
    parser.add_argument("--curriculum", type=_parse_int_list, default=[128, 256, 600, 1200])
    parser.add_argument("--steps-per-stage", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episode-slots", type=int, default=250)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--long-wait-threshold", type=float, default=0.8)
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
    parser.add_argument("--starvation-multiplier", type=float, default=5.0)
    parser.add_argument("--wait-multiplier", type=float, default=1.0)
    parser.add_argument("--lagrangian-lr", type=float, default=0.10)
    parser.add_argument("--max-lagrange-multiplier", type=float, default=50.0)
    parser.add_argument("--validation-seeds", type=_parse_int_list, default=[9001, 9002, 9003])
    parser.add_argument("--validation-slots", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/latest.pt"))
    parser.add_argument("--best-feasible-output", type=Path, default=Path("artifacts/best_feasible.pt"))
    parser.add_argument("--best-reward-output", type=Path, default=Path("artifacts/best_reward.pt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--log-output", type=Path, default=Path("artifacts/ppo_training.csv"))
    parser.add_argument("--validation-output", type=Path, default=Path("artifacts/ppo_validation.csv"))
    args = parser.parse_args()

    if args.steps_per_stage <= 0 or args.workers <= 0 or args.rollout_steps <= 0:
        parser.error("steps, workers, and rollout length must be positive")
    if args.max_candidates < 1 or args.validate_every <= 0 or args.checkpoint_every <= 0:
        parser.error("candidate count and update intervals must be positive")
    if args.validation_slots <= 0:
        parser.error("validation slots must be positive")

    constraints = ServiceConstraints(
        max_starvation_rate=args.max_starvation_rate,
        max_p99_wait_slots=args.max_p99_wait_slots,
    )
    constraints.validate()
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
    initialized_from = "random"
    if args.init_checkpoint.exists():
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_imitation_state_dict(checkpoint["model_state_dict"])
        initialized_from = str(args.init_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    best_feasible_saved = False
    target_num_ues = args.curriculum[-1]
    last_validation: dict[str, Any] | None = None

    for stage_index, num_ues in enumerate(args.curriculum, start=1):
        max_selected = min(64, num_ues, 273)
        max_candidates = min(max(args.max_candidates, max_selected), num_ues)
        config = ScaleMacConfig(
            num_ues=num_ues,
            num_prbs=273,
            max_selected_ues=max_selected,
            episode_slots=args.episode_slots,
            seed=args.seed + stage_index * 10_000,
        )
        config.validate()
        envs = [ScaleMacDownlinkEnv(config) for _ in range(args.workers)]
        observations = np.stack(
            [env.reset(seed=config.seed + worker)[0] for worker, env in enumerate(envs)],
            axis=0,
        )
        stage_env_steps = 0

        while stage_env_steps < args.steps_per_stage:
            obs_buffer: list[np.ndarray] = []
            action_buffer: list[np.ndarray] = []
            compact_mask_buffer: list[np.ndarray] = []
            logprob_buffer: list[np.ndarray] = []
            value_buffer: list[np.ndarray] = []
            reward_buffer: list[np.ndarray] = []
            done_buffer: list[np.ndarray] = []
            metric_window: dict[str, list[float]] = {
                "base_reward": [],
                "constrained_reward": [],
                "constraint_penalty": [],
                "throughput": [],
                "fairness": [],
                "service": [],
                "starvation": [],
                "p99_wait": [],
                "starvation_excess": [],
                "wait_excess": [],
                "goodput": [],
                "candidate_coverage": [],
                "harq_retention": [],
                "long_wait_retention": [],
                "long_wait_missed": [],
            }

            for _ in range(args.rollout_steps):
                masks = _candidate_masks(
                    observations,
                    max_candidates,
                    max_selected,
                    args.long_wait_threshold,
                )
                compact_observations, candidate_indices = gather_candidate_batch(observations, masks)
                compact_masks = np.ones(compact_observations.shape[:2], dtype=bool)
                obs_tensor = torch.from_numpy(compact_observations).to(device)
                mask_tensor = torch.from_numpy(compact_masks).to(device)
                with torch.no_grad():
                    output = model.get_action_and_value(obs_tensor, mask_tensor)
                compact_actions = output.action.cpu().numpy()
                full_actions = scatter_candidate_action_batch(
                    compact_actions,
                    candidate_indices,
                    num_ues=num_ues,
                )

                next_observations: list[np.ndarray] = []
                rewards = np.zeros(args.workers, dtype=np.float32)
                dones = np.zeros(args.workers, dtype=np.float32)
                for worker, env in enumerate(envs):
                    diagnostics = candidate_diagnostics(
                        observations[worker],
                        masks[worker],
                        long_wait_threshold=args.long_wait_threshold,
                    )
                    next_obs, base_reward, terminated, truncated, info = env.step(full_actions[worker])
                    done = terminated or truncated
                    starvation_excess, wait_excess = constraints.excesses(
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
                    metric_window["base_reward"].append(float(info["reward_total"]))
                    metric_window["constrained_reward"].append(constrained_reward)
                    metric_window["constraint_penalty"].append(constraint_penalty)
                    metric_window["throughput"].append(float(info["throughput_score"]))
                    metric_window["fairness"].append(float(info["fairness_score"]))
                    metric_window["service"].append(float(info["service_score"]))
                    metric_window["starvation"].append(float(info["starvation_rate"]))
                    metric_window["p99_wait"].append(float(info["p99_wait_slots"]))
                    metric_window["starvation_excess"].append(starvation_excess)
                    metric_window["wait_excess"].append(wait_excess)
                    metric_window["goodput"].append(float(info["cell_goodput_bits"]))
                    metric_window["candidate_coverage"].append(diagnostics.candidate_coverage)
                    metric_window["harq_retention"].append(diagnostics.harq_retention_rate)
                    metric_window["long_wait_retention"].append(
                        diagnostics.long_wait_retention_rate
                    )
                    metric_window["long_wait_missed"].append(
                        float(diagnostics.long_wait_missed_count)
                    )
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
                    observations,
                    max_candidates,
                    max_selected,
                    args.long_wait_threshold,
                )
                next_compact_obs, _ = gather_candidate_batch(observations, next_masks_full)
                next_obs_tensor = torch.from_numpy(next_compact_obs).to(device)
                next_compact_masks = torch.ones(
                    next_compact_obs.shape[:2], dtype=torch.bool, device=device
                )
                next_value = model.get_action_and_value(
                    next_obs_tensor,
                    next_compact_masks,
                    deterministic=True,
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
                model=model,
                optimizer=optimizer,
                observations=flat_obs,
                actions=flat_actions,
                candidate_masks=flat_masks,
                old_log_probs=flat_logprobs,
                returns=flat_returns,
                advantages=flat_advantages,
                hyper=hyper,
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
                "mean_reward": mean(metric_window["base_reward"]),
                "mean_constrained_reward": mean(metric_window["constrained_reward"]),
                "mean_constraint_penalty": mean(metric_window["constraint_penalty"]),
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
                **update_metrics,
                "device": str(device),
            }
            log_rows.append(row)
            print(
                f"stage={stage_index}/{len(args.curriculum)} ues={num_ues:4d} "
                f"steps={stage_env_steps:6d}/{args.steps_per_stage} "
                f"reward={row['mean_reward']:.4f} constrained={row['mean_constrained_reward']:.4f} "
                f"starvation={row['mean_starvation_rate']:.4f} p99={row['mean_p99_wait_slots']:.1f} "
                f"lambda=({controller.starvation_multiplier:.2f},{controller.wait_multiplier:.2f})"
            )

            should_validate = (
                update_index % args.validate_every == 0
                or stage_env_steps >= args.steps_per_stage
            )
            if should_validate:
                seed_offset = stage_index * 100_000
                seeds = [seed + seed_offset for seed in args.validation_seeds]
                detailed, summary = _validate(
                    model=model,
                    device=device,
                    num_ues=num_ues,
                    slots=args.validation_slots,
                    seeds=seeds,
                    max_candidates=max_candidates,
                    long_wait_threshold=args.long_wait_threshold,
                    constraints=constraints,
                    update_index=update_index,
                    stage_index=stage_index,
                    global_env_steps=global_env_steps,
                )
                validation_rows.extend(detailed)
                validation_summary_rows.append(summary)
                last_validation = summary
                print(
                    f"validation ues={num_ues} reward={summary['mean_reward']:.4f} "
                    f"fairness={summary['mean_jain_fairness']:.4f} "
                    f"worst_starvation={summary['worst_starvation_rate']:.4f} "
                    f"worst_p99={summary['worst_p99_wait_slots']:.1f} "
                    f"feasible={summary['constraint_feasible']}"
                )

                if num_ues == target_num_ues:
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        initialized_from=initialized_from,
                        global_env_steps=global_env_steps,
                        update_index=update_index,
                        stage_index=stage_index,
                        num_ues=num_ues,
                        controller=controller,
                        constraints=constraints,
                        tag="validation",
                        validation=summary,
                    )
                    if float(summary["mean_reward"]) > best_reward:
                        best_reward = float(summary["mean_reward"])
                        payload["checkpoint_tag"] = "best_reward"
                        _save_checkpoint(args.best_reward_output, payload)
                    if bool(summary["constraint_feasible"]) and float(summary["mean_reward"]) > best_feasible_reward:
                        best_feasible_reward = float(summary["mean_reward"])
                        best_feasible_saved = True
                        payload["checkpoint_tag"] = "best_feasible"
                        _save_checkpoint(args.best_feasible_output, payload)

            if update_index % args.checkpoint_every == 0:
                periodic = args.checkpoint_dir / f"ppo_update_{update_index:05d}.pt"
                _save_checkpoint(
                    periodic,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        initialized_from=initialized_from,
                        global_env_steps=global_env_steps,
                        update_index=update_index,
                        stage_index=stage_index,
                        num_ues=num_ues,
                        controller=controller,
                        constraints=constraints,
                        tag="periodic",
                        validation=last_validation,
                    ),
                )

    latest_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        args=args,
        initialized_from=initialized_from,
        global_env_steps=global_env_steps,
        update_index=update_index,
        stage_index=len(args.curriculum),
        num_ues=args.curriculum[-1],
        controller=controller,
        constraints=constraints,
        tag="latest",
        validation=last_validation,
    )
    _save_checkpoint(args.output, latest_payload)

    write_csv(args.log_output, log_rows)
    write_markdown(
        markdown_report_path(args.log_output),
        title="ScaleMAC-RL constrained curriculum PPO training",
        description=(
            "PPO fine-tuning with compact candidate sets, normalized base reward, "
            "Lagrangian service constraints, candidate-retention diagnostics, and held-out validation."
        ),
        rows=log_rows,
        notes=(
            f"Initialized from: `{initialized_from}`",
            f"Curriculum UE stages: {args.curriculum}",
            f"Constraints: starvation <= {constraints.max_starvation_rate}, P99 wait <= {constraints.max_p99_wait_slots} slots",
            "Workers are vectorized rollout environments in one Python process, not distributed RL.",
            "This remains the fast surrogate, not 5G-LENA.",
        ),
    )
    write_csv(args.validation_output, validation_rows)
    write_markdown(
        markdown_report_path(args.validation_output),
        title="ScaleMAC-RL held-out PPO validation",
        description="Per-seed validation used for feasibility-first checkpoint selection.",
        rows=validation_rows,
    )
    validation_summary_output = args.validation_output.with_name(
        f"{args.validation_output.stem}_summary.csv"
    )
    write_csv(validation_summary_output, validation_summary_rows)
    write_markdown(
        markdown_report_path(validation_summary_output),
        title="ScaleMAC-RL held-out PPO validation summary",
        description="Worst-case constraint checks and mean performance at every validation event.",
        rows=validation_summary_rows,
    )

    print(f"saved: {args.output}")
    print(f"saved: {args.best_reward_output}")
    if best_feasible_saved:
        print(f"saved: {args.best_feasible_output}")
    else:
        print("warning: no final-stage validation checkpoint satisfied all constraints")
    print(f"saved: {args.log_output}")
    print(f"saved: {args.validation_output}")
    print(f"saved: {validation_summary_output}")


if __name__ == "__main__":
    main()
