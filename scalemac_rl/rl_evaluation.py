from __future__ import annotations

from statistics import mean
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch

from .candidates import (
    build_candidate_mask,
    candidate_diagnostics,
    gather_candidates,
    scatter_candidate_action,
)
from .config import ScaleMacConfig
from .constraints import ServiceConstraints
from .env import ScaleMacDownlinkEnv
from .models import SharedSetActorCritic
from .schedulers.base import Scheduler


_METRIC_KEYS = (
    "reward_total",
    "reward_core_total",
    "reward_final_target_total",
    "cell_goodput_bits",
    "throughput_score",
    "fairness_score",
    "service_score",
    "starvation_rate",
    "p99_wait_slots",
    "reward_throughput_component",
    "reward_fairness_component",
    "reward_service_component",
    "reward_starvation_penalty",
    "reward_deadline_risk_penalty",
    "reward_reference_deadline_risk_penalty",
    "deadline_risk",
    "reference_deadline_risk",
    "tail_mean_wait_slots",
    "forced_harq_count",
    "forced_long_wait_count",
    "forced_oldest_wait_count",
    "safety_selected_count",
    "learned_selected_count",
    "learned_selection_fraction",
)


def summarize_episode(
    *,
    name: str,
    seed: int,
    config: ScaleMacConfig,
    metrics: dict[str, list[float]],
    final_info: dict[str, Any],
    inference_us: list[float] | None = None,
    candidate_metrics: dict[str, list[float]] | None = None,
    constraints: ServiceConstraints | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": name,
        "seed": seed,
        "num_ues": config.num_ues,
        "slots": config.episode_slots,
        "mean_reward": mean(metrics["reward_total"]),
        "mean_core_reward": mean(metrics["reward_core_total"]),
        "mean_final_target_reward": mean(metrics["reward_final_target_total"]),
        "mean_goodput_bits_per_slot": mean(metrics["cell_goodput_bits"]),
        "mean_throughput_score": mean(metrics["throughput_score"]),
        "final_jain_fairness": float(final_info["jain_fairness"]),
        "mean_fairness_score": mean(metrics["fairness_score"]),
        "mean_service_score": mean(metrics["service_score"]),
        "mean_starvation_rate": mean(metrics["starvation_rate"]),
        "max_starvation_rate": max(metrics["starvation_rate"]),
        "final_p99_wait_slots": float(final_info["p99_wait_slots"]),
        "max_p99_wait_slots": max(metrics["p99_wait_slots"]),
        "mean_reward_throughput_component": mean(metrics["reward_throughput_component"]),
        "mean_reward_fairness_component": mean(metrics["reward_fairness_component"]),
        "mean_reward_service_component": mean(metrics["reward_service_component"]),
        "mean_reward_starvation_penalty": mean(metrics["reward_starvation_penalty"]),
        "mean_reward_deadline_risk_penalty": mean(
            metrics["reward_deadline_risk_penalty"]
        ),
        "mean_reward_reference_deadline_risk_penalty": mean(
            metrics["reward_reference_deadline_risk_penalty"]
        ),
        "mean_deadline_risk": mean(metrics["deadline_risk"]),
        "mean_reference_deadline_risk": mean(metrics["reference_deadline_risk"]),
        "mean_tail_mean_wait_slots": mean(metrics["tail_mean_wait_slots"]),
        "mean_forced_harq_count": mean(metrics["forced_harq_count"]),
        "mean_forced_long_wait_count": mean(metrics["forced_long_wait_count"]),
        "mean_forced_oldest_wait_count": mean(metrics["forced_oldest_wait_count"]),
        "mean_safety_selected_count": mean(metrics["safety_selected_count"]),
        "mean_learned_selected_count": mean(metrics["learned_selected_count"]),
        "mean_learned_selection_fraction": mean(metrics["learned_selection_fraction"]),
    }
    if candidate_metrics:
        for key, values in candidate_metrics.items():
            row[f"mean_{key}"] = mean(values) if values else 0.0
            if key == "long_wait_missed_count":
                row["max_long_wait_missed_count"] = max(values) if values else 0.0
    if constraints is not None:
        starvation_excess, wait_excess = constraints.excesses(
            starvation_rate=row["max_starvation_rate"],
            p99_wait_slots=row["max_p99_wait_slots"],
        )
        row.update(
            {
                "constraint_max_starvation_rate": constraints.max_starvation_rate,
                "constraint_max_p99_wait_slots": constraints.max_p99_wait_slots,
                "starvation_constraint_excess": starvation_excess,
                "wait_constraint_excess": wait_excess,
                "constraint_feasible": constraints.feasible(
                    starvation_rate=row["max_starvation_rate"],
                    p99_wait_slots=row["max_p99_wait_slots"],
                ),
            }
        )
    if inference_us:
        row.update(
            {
                "mean_inference_us": mean(inference_us),
                "p95_inference_us": float(np.percentile(inference_us, 95)),
                "p99_inference_us": float(np.percentile(inference_us, 99)),
                "max_inference_us": max(inference_us),
            }
        )
    return row


def evaluate_actor_critic(
    *,
    model: SharedSetActorCritic,
    device: torch.device,
    config: ScaleMacConfig,
    seed: int,
    name: str = "ppo",
    max_candidates: int = 256,
    long_wait_threshold: float = 0.8,
    constraints: ServiceConstraints | None = None,
) -> dict[str, Any]:
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=seed)
    metrics = {key: [] for key in _METRIC_KEYS}
    candidate_metrics: dict[str, list[float]] = {
        "candidate_count": [],
        "candidate_coverage": [],
        "harq_retention_rate": [],
        "long_wait_retention_rate": [],
        "long_wait_missed_count": [],
    }
    inference_us: list[float] = []
    final_info: dict[str, Any] = {}
    model.eval()
    candidate_count = min(max(max_candidates, config.max_selected_ues), config.num_ues)

    with torch.inference_mode():
        while True:
            mask = build_candidate_mask(
                observation,
                max_candidates=candidate_count,
                min_candidates=config.max_selected_ues,
                long_wait_threshold=long_wait_threshold,
            )
            diagnostics = candidate_diagnostics(
                observation,
                mask,
                long_wait_threshold=long_wait_threshold,
            )
            compact_observation, indices = gather_candidates(observation, mask)
            start_ns = perf_counter_ns()
            x = torch.from_numpy(compact_observation).to(device)
            compact_action = model.deterministic_action(x).action.cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_us.append((perf_counter_ns() - start_ns) / 1000.0)
            action = scatter_candidate_action(
                compact_action,
                indices,
                num_ues=config.num_ues,
            )
            observation, _, terminated, truncated, final_info = env.step(action)
            for key in _METRIC_KEYS:
                metrics[key].append(float(final_info[key]))
            candidate_metrics["candidate_count"].append(float(diagnostics.candidate_count))
            candidate_metrics["candidate_coverage"].append(diagnostics.candidate_coverage)
            candidate_metrics["harq_retention_rate"].append(diagnostics.harq_retention_rate)
            candidate_metrics["long_wait_retention_rate"].append(
                diagnostics.long_wait_retention_rate
            )
            candidate_metrics["long_wait_missed_count"].append(
                float(diagnostics.long_wait_missed_count)
            )
            if terminated or truncated:
                break

    row = summarize_episode(
        name=name,
        seed=seed,
        config=config,
        metrics=metrics,
        final_info=final_info,
        inference_us=inference_us,
        candidate_metrics=candidate_metrics,
        constraints=constraints,
    )
    row["device"] = str(device)
    row["max_candidates"] = candidate_count
    row["long_wait_threshold"] = long_wait_threshold
    return row


def evaluate_scheduler(
    *,
    scheduler: Scheduler,
    config: ScaleMacConfig,
    seed: int,
    name: str,
    constraints: ServiceConstraints | None = None,
) -> dict[str, Any]:
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=seed)
    scheduler.reset()
    metrics = {key: [] for key in _METRIC_KEYS}
    final_info: dict[str, Any] = {}
    while True:
        observation, _, terminated, truncated, final_info = env.step(scheduler.act(observation))
        for key in _METRIC_KEYS:
            metrics[key].append(float(final_info[key]))
        if terminated or truncated:
            break
    return summarize_episode(
        name=name,
        seed=seed,
        config=config,
        metrics=metrics,
        final_info=final_info,
        constraints=constraints,
    )
