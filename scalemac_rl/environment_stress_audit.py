from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np

from .config import ScaleMacConfig
from .oracle_sanity import evaluate_policy, service_aware_oracle_action
from .schedulers import ProportionalFairScheduler


DEFAULT_STRESS_SEEDS: tuple[int, ...] = (
    101, 301, 501, 701, 901, 1101, 1301, 1501, 1701,
    1901, 2101, 2301, 2501, 2701, 2901, 3101, 3401, 3701,
)
KEY_SEEDS: tuple[int, ...] = (1701, 2701, 3701)


@dataclass(frozen=True, slots=True)
class StressAuditResult:
    metrics: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    key_seed_positions: list[dict[str, Any]]


def stress_config(seed: int, slots: int) -> ScaleMacConfig:
    """Build the locked Round-17C radio environment.

    Reward weights are kept at the current service40 anchor for consistency,
    although Oracle/PF actions do not use reward values for their decisions.
    """
    return ScaleMacConfig(
        num_ues=1200,
        num_prbs=273,
        max_selected_ues=64,
        episode_slots=slots,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        freeze_static_profiles=True,
        static_profile_seed=seed,
        seed=seed,
        cqi_mode="correlated",
        cqi_temporal_correlation=0.97,
        cqi_innovation_std=0.35,
        cqi_update_interval_slots=1,
        cqi_max_delta_per_update=1,
        csi_report_mode="periodic",
        csi_report_period_slots=4,
        csi_report_delay_slots=2,
        csi_report_error_std=0.0,
        observation_include_csi_age=False,
        observation_include_reported_cqi_trend=False,
        link_adaptation_mode="cqi_mcs_bler",
        link_adaptation_cqi_backoff=0,
        bler_mismatch_slope=1.5,
        starvation_threshold_slots=64,
        reward_throughput_weight=0.30,
        reward_fairness_weight=0.30,
        reward_service_weight=0.40,
        reward_deficit_service_weight=0.0,
        reward_pf_utility_weight=0.0,
        reward_low_throughput_weight=0.0,
        reward_urgency_service_weight=0.0,
        reward_fairness_delta_weight=0.0,
        reward_pf_utility_delta_weight=0.0,
        reward_starvation_penalty_weight=0.0,
        reward_deadline_risk_penalty_weight=0.0,
        reward_max_wait_risk_penalty_weight=0.0,
        reward_population_wait_penalty_weight=0.0,
    )


def _pf_action_factory():
    pf = ProportionalFairScheduler()
    reset = getattr(pf, "reset", None)
    if callable(reset):
        reset()

    def action(env, obs):
        return pf.act(obs)

    return action


def run_environment_stress_audit(
    *,
    seeds: Iterable[int] = DEFAULT_STRESS_SEEDS,
    slots: int = 5000,
) -> StressAuditResult:
    seeds = tuple(int(seed) for seed in seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("stress-audit seeds must be unique")
    if slots <= 0:
        raise ValueError("slots must be positive")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        config = stress_config(seed, slots)
        rows.append(
            evaluate_policy(
                name="oracle",
                config=config,
                seed=seed,
                action_fn=lambda env, obs: service_aware_oracle_action(env),
            )
        )
        rows.append(
            evaluate_policy(
                name="pf",
                config=config,
                seed=seed,
                action_fn=_pf_action_factory(),
            )
        )

    summary = summarize_stress_rows(rows)
    key_positions = key_seed_positions(rows, seeds=seeds)
    return StressAuditResult(metrics=rows, summary=summary, key_seed_positions=key_positions)


def _percentile_rank(values: list[float], value: float, *, higher_is_harder: bool) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if higher_is_harder:
        return float(np.mean(arr <= value) * 100.0)
    return float(np.mean(arr >= value) * 100.0)


def summarize_stress_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for policy in ("oracle", "pf"):
        group = [row for row in rows if row["policy"] == policy]
        if not group:
            continue
        record: dict[str, Any] = {
            "policy": policy,
            "seeds": len(group),
            "zero_starvation_seeds": sum(int(row["zero_starvation"]) for row in group),
            "service_feasible_seeds": sum(int(row["service_feasible_under_64"]) for row in group),
        }
        for key in (
            "mean_goodput_bits_per_slot",
            "mean_spectral_efficiency_bps_hz",
            "final_jain_fairness",
            "max_starvation_rate",
            "max_p99_wait_slots",
            "max_wait_slots",
            "mean_observed_bler",
            "mean_harq_retransmission_fraction",
        ):
            values = [float(row[key]) for row in group]
            record[key + "_mean"] = mean(values)
            record[key + "_std"] = stdev(values) if len(values) > 1 else 0.0
            record[key + "_min"] = min(values)
            record[key + "_max"] = max(values)
        summary.append(record)
    return summary


def key_seed_positions(rows: list[dict[str, Any]], *, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    difficulty_metrics = (
        ("mean_goodput_bits_per_slot", False),
        ("mean_observed_bler", True),
        ("max_starvation_rate", True),
        ("max_p99_wait_slots", True),
        ("max_wait_slots", True),
    )
    key_set = set(KEY_SEEDS).intersection(seeds)
    for policy in ("oracle", "pf"):
        group = [row for row in rows if row["policy"] == policy]
        by_seed = {int(row["seed"]): row for row in group}
        for seed in sorted(key_set):
            row = by_seed[seed]
            record: dict[str, Any] = {"policy": policy, "seed": seed, "population_seeds": len(group)}
            for metric, higher_is_harder in difficulty_metrics:
                values = [float(item[metric]) for item in group]
                value = float(row[metric])
                record[metric] = value
                record[metric + "_hardness_percentile"] = _percentile_rank(
                    values, value, higher_is_harder=higher_is_harder
                )
            out.append(record)
    return out


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty rows")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
