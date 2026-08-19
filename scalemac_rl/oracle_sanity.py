from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
import torch

from .config import ScaleMacConfig
from .env import ScaleMacDownlinkEnv
from .link_adaptation import (
    bler_probability_from_cqi_mismatch,
    mcs_efficiency,
    select_mcs_from_reported_cqi,
)
from .models import SharedSetActorCritic
from .schedulers import ProportionalFairScheduler


def service_aware_oracle_action(env: ScaleMacDownlinkEnv) -> np.ndarray:
    """Current-state clairvoyant sanity policy.

    Uses true CQI and exact service-wait state.  It is intentionally not a
    deployable scheduler and does not know the future; its purpose is to test
    whether a difficult seed remains service-feasible under privileged state.
    Oldest-service-first dominates UE selection, while true-CQI expected rate
    breaks ties and attracts the extra PRBs.
    """
    true_cqi = env.cqi.astype(np.int16, copy=False)
    mcs = select_mcs_from_reported_cqi(
        true_cqi, cqi_backoff=env.config.link_adaptation_cqi_backoff
    )
    eff = mcs_efficiency(mcs).astype(np.float64)
    if env.config.harq_enabled:
        bler = bler_probability_from_cqi_mismatch(
            true_cqi=true_cqi,
            mcs_index=mcs,
            target_bler=env.config.target_bler,
            mismatch_slope=env.config.bler_mismatch_slope,
        )
    else:
        bler = np.zeros_like(eff)
    expected_rate = eff * (1.0 - bler)
    rate_norm = expected_rate / max(float(expected_rate.max(initial=1.0)), 1e-9)

    wait = env.time_since_service.astype(np.float64)
    # A one-slot wait difference dominates the channel tie-break, so the policy
    # first protects service feasibility and only then exploits true CQI.
    priority = wait + 1.0e-3 * rate_norm
    priority /= max(float(priority.max(initial=1.0)), 1.0)

    action = np.empty((env.config.num_ues, 2), dtype=np.float32)
    action[:, 0] = priority.astype(np.float32)
    action[:, 1] = np.clip(0.15 + 0.85 * rate_norm, 0.0, 1.0).astype(np.float32)
    return action


def load_feedforward_policy(path: Path, device: torch.device) -> SharedSetActorCritic:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if str(checkpoint.get("policy_architecture", "feedforward")) != "feedforward":
        raise ValueError(f"expected feed-forward PPO checkpoint: {path}")
    model = SharedSetActorCritic(
        input_dim=int(checkpoint.get("input_dim", 16)),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_compatible_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def evaluate_policy(
    *,
    name: str,
    config: ScaleMacConfig,
    seed: int,
    action_fn: Callable[[ScaleMacDownlinkEnv, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    env = ScaleMacDownlinkEnv(config)
    obs, _ = env.reset(seed=seed)
    goodput, se, bler, retx = [], [], [], []
    starvation, p99, maxwait, jfi = [], [], [], []
    while True:
        action = action_fn(env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        goodput.append(float(info["cell_goodput_bits"]))
        se.append(float(info["spectral_efficiency_bps_hz"]))
        bler.append(float(info["observed_bler"]))
        retx.append(float(info["harq_retransmission_fraction"]))
        starvation.append(float(info["starvation_rate"]))
        p99.append(float(info["p99_wait_slots"]))
        maxwait.append(float(info["max_wait_slots"]))
        jfi.append(float(info["jain_fairness"]))
        if terminated or truncated:
            break
    return {
        "policy": name,
        "seed": seed,
        "mean_goodput_bits_per_slot": mean(goodput),
        "mean_spectral_efficiency_bps_hz": mean(se),
        "final_jain_fairness": jfi[-1],
        "max_starvation_rate": max(starvation),
        "max_p99_wait_slots": max(p99),
        "max_wait_slots": max(maxwait),
        "mean_observed_bler": mean(bler),
        "mean_harq_retransmission_fraction": mean(retx),
        "zero_starvation": int(max(starvation) <= 1e-12),
        "service_feasible_under_64": int(max(starvation) <= 1e-12 and max(p99) < 64.0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
