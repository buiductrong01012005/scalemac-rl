from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import torch

from .config import ScaleMacConfig
from .env import ScaleMacDownlinkEnv
from .evaluation_protocol import load_policy_checkpoint
from .reporting import write_csv


_EVALUATION_MODES = {"deterministic", "stochastic"}


@dataclass(frozen=True, slots=True)
class DiagnosticCase:
    case_id: str
    label: str
    case_dir: Path
    checkpoint_path: Path
    run_config: dict[str, Any]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def parse_modes(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    modes = tuple(item.strip() for item in raw if item.strip())
    if not modes:
        raise ValueError("at least one evaluation mode is required")
    unknown = sorted(set(modes) - _EVALUATION_MODES)
    if unknown:
        raise ValueError(f"unknown evaluation modes: {', '.join(unknown)}")
    return modes


def discover_cases(
    *,
    study_root: Path,
    case_ids: Iterable[str],
    checkpoint_name: str,
) -> list[DiagnosticCase]:
    cases: list[DiagnosticCase] = []
    for raw_case_id in case_ids:
        case_id = raw_case_id.strip()
        if not case_id:
            continue
        case_dir = study_root / case_id
        run_config_path = case_dir / "run_config.json"
        checkpoint_path = case_dir / checkpoint_name
        if not run_config_path.is_file():
            raise FileNotFoundError(f"missing run config: {run_config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        case_payload = run_config.get("case", {})
        cases.append(
            DiagnosticCase(
                case_id=case_id,
                label=str(case_payload.get("label", case_id)),
                case_dir=case_dir,
                checkpoint_path=checkpoint_path,
                run_config=run_config,
            )
        )
    if not cases:
        raise ValueError("no diagnostic cases were selected")
    return cases


def _config_from_run_config(
    payload: dict[str, Any],
    *,
    episode_slots: int,
    seed: int,
    profile_seed: int,
) -> ScaleMacConfig:
    architecture = payload.get("architecture", {})
    common = payload.get("effective_common", payload.get("common", {}))
    case = payload.get("case", {})
    positive = case.get("positive_weights", {})
    delta = case.get("delta_weights", {})
    penalties = case.get("penalty_weights", {})

    config = ScaleMacConfig(
        num_ues=int(architecture.get("num_ues", 1200)),
        num_prbs=int(architecture.get("num_prbs", 273)),
        max_selected_ues=int(architecture.get("top_k", 64)),
        episode_slots=episode_slots,
        scheduler_mode="ppo_only",
        force_harq_retransmissions=False,
        safety_reserve_ues=0,
        freeze_static_profiles=True,
        static_profile_seed=profile_seed,
        starvation_threshold_slots=int(common.get("starvation_threshold_slots", 64)),
        deadline_target_slots=float(common.get("deadline_target_slots", 50.0)),
        reference_deadline_target_slots=float(
            common.get("reference_deadline_target_slots", 50.0)
        ),
        max_wait_target_slots=float(common.get("max_wait_target_slots", 60.0)),
        deadline_risk_start_ratio=float(common.get("deadline_risk_start_ratio", 0.60)),
        low_throughput_percentile=float(common.get("low_throughput_percentile", 10.0)),
        reward_positive_scale=float(case.get("positive_scale", 1.0)),
        reward_throughput_weight=float(positive.get("throughput", 0.0)),
        reward_fairness_weight=float(positive.get("fairness", 0.0)),
        reward_schedule_fairness_weight=float(positive.get("schedule_fairness", 0.0)),
        reward_service_weight=float(positive.get("service", 0.0)),
        reward_deficit_service_weight=float(positive.get("deficit_service", 0.0)),
        reward_pf_utility_weight=float(positive.get("pf_utility", 0.0)),
        reward_low_throughput_weight=float(positive.get("low_throughput", 0.0)),
        reward_urgency_service_weight=float(positive.get("urgency_service", 0.0)),
        reward_fairness_delta_weight=float(delta.get("fairness", 0.0)),
        reward_pf_utility_delta_weight=float(delta.get("pf_utility", 0.0)),
        reward_starvation_penalty_weight=float(penalties.get("starvation", 0.0)),
        reward_deadline_risk_penalty_weight=float(
            penalties.get("deadline_risk", 0.0)
        ),
        reward_max_wait_risk_penalty_weight=float(
            penalties.get("max_wait_risk", 0.0)
        ),
        reward_population_wait_penalty_weight=float(
            penalties.get("population_wait", 0.0)
        ),
        cqi_mode=str(common.get("cqi_mode", "static")),
        cqi_temporal_correlation=float(common.get("cqi_temporal_correlation", 0.97)),
        cqi_innovation_std=float(common.get("cqi_innovation_std", 0.35)),
        cqi_update_interval_slots=int(common.get("cqi_update_interval_slots", 1)),
        cqi_max_delta_per_update=int(common.get("cqi_max_delta_per_update", 1)),
        csi_report_mode=str(common.get("csi_report_mode", "perfect")),
        csi_report_period_slots=int(common.get("csi_report_period_slots", 1)),
        csi_report_delay_slots=int(common.get("csi_report_delay_slots", 0)),
        csi_report_error_std=float(common.get("csi_report_error_std", 0.0)),
        observation_include_csi_age=bool(common.get("observation_include_csi_age", False)),
        observation_include_reported_cqi_trend=bool(
            common.get("observation_include_reported_cqi_trend", False)
        ),
        link_adaptation_mode=str(common.get("link_adaptation_mode", "legacy_fixed_bler")),
        link_adaptation_cqi_backoff=int(common.get("link_adaptation_cqi_backoff", 0)),
        bler_mismatch_slope=float(common.get("bler_mismatch_slope", 1.5)),
        seed=seed,
    )
    config.validate()
    return config


def _gini(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or float(array.sum()) <= 0.0:
        return 0.0
    array = np.sort(array)
    n = array.size
    indices = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(indices * array) / np.sum(array) - (n + 1)) / n)


def _top_share(values: np.ndarray, count: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    if total <= 0.0:
        return 0.0
    k = min(max(int(count), 1), array.size)
    return float(np.partition(array, array.size - k)[-k:].sum() / total)


def _priority_metrics(
    scores: np.ndarray,
    *,
    top_k: int,
    tie_epsilon: float,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < top_k:
        raise ValueError("priority scores must be a 1-D array with at least top_k values")
    ordered = np.sort(values)[::-1]
    cutoff = float(ordered[top_k - 1])
    next_value = float(ordered[top_k]) if values.size > top_k else cutoff
    near_cutoff = int(np.sum(np.abs(values - cutoff) <= tie_epsilon))
    rounded_unique = int(np.unique(np.round(values, decimals=6)).size)
    return {
        "priority_mean": float(values.mean()),
        "priority_std": float(values.std()),
        "priority_min": float(values.min()),
        "priority_max": float(values.max()),
        "top_k_cutoff": cutoff,
        "top_k_margin": cutoff - next_value,
        "near_cutoff_count": float(near_cutoff),
        "near_cutoff_fraction": float(near_cutoff / values.size),
        "unique_priority_rounded_1e6": float(rounded_unique),
    }


def _window_unique(counter: np.ndarray) -> int:
    return int(np.count_nonzero(counter))


def evaluate_case_mode(
    *,
    case: DiagnosticCase,
    mode: str,
    device: torch.device,
    slots: int,
    seed: int,
    profile_seed: int,
    window_size: int,
    tie_epsilon: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if mode not in _EVALUATION_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    config = _config_from_run_config(
        case.run_config,
        episode_slots=slots,
        seed=seed,
        profile_seed=profile_seed,
    )
    model, checkpoint = load_policy_checkpoint(case.checkpoint_path, device)
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    concentration = (
        torch.nn.functional.softplus(model.raw_concentration).detach().cpu().numpy()
        + 2.0
    )
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=seed)

    selected_counts = np.zeros(config.num_ues, dtype=np.int64)
    success_counts = np.zeros(config.num_ues, dtype=np.int64)
    selected_window_counts = np.zeros(config.num_ues, dtype=np.int32)
    success_window_counts = np.zeros(config.num_ues, dtype=np.int32)
    selected_history: deque[np.ndarray] = deque()
    success_history: deque[np.ndarray] = deque()

    slot_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    reward_values: list[float] = []
    goodput_values: list[float] = []
    starvation_values: list[float] = []
    p99_values: list[float] = []
    max_wait_values: list[float] = []
    overlap_values: list[float] = []
    entropy_values: list[float] = []
    previous_selected = np.empty(0, dtype=np.int32)
    final_info: dict[str, Any] = {}

    candidate_mask = torch.ones(config.num_ues, dtype=torch.bool, device=device)
    with torch.inference_mode():
        while True:
            tensor_observation = torch.from_numpy(observation).to(device)
            if mode == "deterministic":
                policy_output = model.deterministic_action(tensor_observation)
                action_tensor = policy_output.action
                mean_action_tensor = policy_output.mean_action
                entropy = 0.0
            else:
                policy_output = model.get_action_and_value(
                    tensor_observation,
                    candidate_mask,
                    deterministic=False,
                )
                action_tensor = policy_output.action
                mean_action_tensor = policy_output.mean_action
                entropy = float(policy_output.entropy.detach().cpu().item())
            action = action_tensor.detach().cpu().numpy()
            mean_action = mean_action_tensor.detach().cpu().numpy()

            action_priority = _priority_metrics(
                action[:, 0], top_k=config.max_selected_ues, tie_epsilon=tie_epsilon
            )
            mean_priority = _priority_metrics(
                mean_action[:, 0], top_k=config.max_selected_ues, tie_epsilon=tie_epsilon
            )

            observation, reward, terminated, truncated, final_info = env.step(action)
            selected = np.asarray(final_info["selected_ues"], dtype=np.int32)
            successful = np.flatnonzero(env.last_success).astype(np.int32)
            selected_counts[selected] += 1
            success_counts[successful] += 1

            selected_history.append(selected)
            selected_window_counts[selected] += 1
            success_history.append(successful)
            success_window_counts[successful] += 1
            if len(selected_history) > window_size:
                expired = selected_history.popleft()
                selected_window_counts[expired] -= 1
                expired_success = success_history.popleft()
                success_window_counts[expired_success] -= 1

            if previous_selected.size:
                overlap = float(
                    np.intersect1d(previous_selected, selected, assume_unique=False).size
                    / max(config.max_selected_ues, 1)
                )
            else:
                overlap = 0.0
            previous_selected = selected.copy()
            overlap_values.append(overlap)
            entropy_values.append(entropy)

            reward_values.append(float(reward))
            goodput_values.append(float(final_info["cell_goodput_bits"]))
            starvation_values.append(float(final_info["starvation_rate"]))
            p99_values.append(float(final_info["p99_wait_slots"]))
            max_wait_values.append(float(final_info["max_wait_slots"]))

            slot = int(final_info["slot"])
            slot_row: dict[str, Any] = {
                "case_id": case.case_id,
                "case_label": case.label,
                "checkpoint": case.checkpoint_path.name,
                "mode": mode,
                "seed": seed,
                "slot": slot,
                "reward": float(reward),
                "goodput_bits": float(final_info["cell_goodput_bits"]),
                "jain_fairness": float(final_info["jain_fairness"]),
                "starvation_rate": float(final_info["starvation_rate"]),
                "p99_wait_slots": float(final_info["p99_wait_slots"]),
                "max_wait_slots": float(final_info["max_wait_slots"]),
                "selected_count": int(selected.size),
                "successful_count": int(successful.size),
                "selected_overlap_previous": overlap,
                "unique_selected_last_window": _window_unique(selected_window_counts),
                "unique_success_last_window": _window_unique(success_window_counts),
                "policy_entropy": entropy,
                "beta_concentration_priority": float(concentration[0]),
                "beta_concentration_demand": float(concentration[1]),
                **{f"action_{key}": value for key, value in action_priority.items()},
                **{f"mean_{key}": value for key, value in mean_priority.items()},
            }
            slot_rows.append(slot_row)

            if slot % window_size == 0 or terminated or truncated:
                window_rows.append(
                    {
                        "case_id": case.case_id,
                        "case_label": case.label,
                        "mode": mode,
                        "seed": seed,
                        "window_end_slot": slot,
                        "window_slots": len(selected_history),
                        "unique_selected_ues": _window_unique(selected_window_counts),
                        "unique_successful_ues": _window_unique(success_window_counts),
                        "never_selected_ues_so_far": int(np.sum(selected_counts == 0)),
                        "never_successful_ues_so_far": int(np.sum(success_counts == 0)),
                        "selection_top64_share_so_far": _top_share(
                            selected_counts, config.max_selected_ues
                        ),
                        "selection_gini_so_far": _gini(selected_counts),
                        "success_gini_so_far": _gini(success_counts),
                    }
                )
            if terminated or truncated:
                break

    ue_rows = [
        {
            "case_id": case.case_id,
            "case_label": case.label,
            "mode": mode,
            "seed": seed,
            "ue_id": ue_id,
            "selected_slots": int(selected_counts[ue_id]),
            "successful_slots": int(success_counts[ue_id]),
            "selection_fraction": float(selected_counts[ue_id] / slots),
            "success_fraction": float(success_counts[ue_id] / slots),
            "final_delivered_bits": float(env.cumulative_delivered_bits[ue_id]),
            "final_wait_slots": int(env.time_since_service[ue_id]),
            "cqi": int(env.cqi[ue_id]),
            "demand_factor": float(env.demand_factor[ue_id]),
        }
        for ue_id in range(config.num_ues)
    ]

    summary = {
        "case_id": case.case_id,
        "case_label": case.label,
        "checkpoint": str(case.checkpoint_path),
        "checkpoint_tag": str(checkpoint.get("checkpoint_tag", "unknown")),
        "mode": mode,
        "seed": seed,
        "profile_seed": profile_seed,
        "slots": slots,
        "num_ues": config.num_ues,
        "top_k": config.max_selected_ues,
        "starvation_threshold_slots": config.starvation_threshold_slots,
        "mean_reward": mean(reward_values),
        "mean_goodput_bits_per_slot": mean(goodput_values),
        "final_jain_fairness": float(final_info["jain_fairness"]),
        "mean_starvation_rate": mean(starvation_values),
        "max_starvation_rate": max(starvation_values),
        "max_p99_wait_slots": max(p99_values),
        "max_wait_slots": max(max_wait_values),
        "unique_selected_ues_episode": int(np.count_nonzero(selected_counts)),
        "unique_successful_ues_episode": int(np.count_nonzero(success_counts)),
        "never_selected_ues_episode": int(np.sum(selected_counts == 0)),
        "never_successful_ues_episode": int(np.sum(success_counts == 0)),
        "mean_unique_selected_last_window": mean(
            float(row["unique_selected_last_window"]) for row in slot_rows
        ),
        "min_unique_selected_last_window": min(
            int(row["unique_selected_last_window"]) for row in slot_rows
        ),
        "mean_unique_success_last_window": mean(
            float(row["unique_success_last_window"]) for row in slot_rows
        ),
        "min_unique_success_last_window": min(
            int(row["unique_success_last_window"]) for row in slot_rows
        ),
        "selection_top64_share": _top_share(selected_counts, config.max_selected_ues),
        "success_top64_share": _top_share(success_counts, config.max_selected_ues),
        "selection_gini": _gini(selected_counts),
        "success_gini": _gini(success_counts),
        "mean_selected_overlap_previous": mean(overlap_values),
        "mean_action_priority_std": mean(
            float(row["action_priority_std"]) for row in slot_rows
        ),
        "mean_mean_priority_std": mean(
            float(row["mean_priority_std"]) for row in slot_rows
        ),
        "mean_action_top_k_margin": mean(
            float(row["action_top_k_margin"]) for row in slot_rows
        ),
        "mean_mean_top_k_margin": mean(
            float(row["mean_top_k_margin"]) for row in slot_rows
        ),
        "mean_action_near_cutoff_count": mean(
            float(row["action_near_cutoff_count"]) for row in slot_rows
        ),
        "mean_mean_near_cutoff_count": mean(
            float(row["mean_near_cutoff_count"]) for row in slot_rows
        ),
        "mean_policy_entropy": mean(entropy_values),
        "beta_concentration_priority": float(concentration[0]),
        "beta_concentration_demand": float(concentration[1]),
    }
    return summary, slot_rows, window_rows, ue_rows


def _fmt(value: Any, decimals: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 1000:
            return f"{float(value):,.1f}"
        return f"{float(value):.{decimals}f}"
    return str(value)


def build_html_report(
    *,
    output: Path,
    summaries: list[dict[str, Any]],
    cases: list[DiagnosticCase],
    window_size: int,
    tie_epsilon: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_rows = []
    for case in cases:
        actual = case.run_config.get("case", {}).get("actual_coefficients", {})
        weights_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(case.label)}</strong></td>"
            f"<td>{float(actual.get('coef_throughput', 0.0)):.3f}</td>"
            f"<td>{float(actual.get('coef_fairness', 0.0)):.3f}</td>"
            f"<td>{html.escape(case.checkpoint_path.name)}</td>"
            "</tr>"
        )

    summary_rows = []
    for row in summaries:
        summary_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(row['case_label']))}</strong></td>"
            f"<td>{html.escape(str(row['mode']))}</td>"
            f"<td>{_fmt(row['mean_goodput_bits_per_slot'], 1)}</td>"
            f"<td>{_fmt(row['final_jain_fairness'])}</td>"
            f"<td>{100.0 * float(row['max_starvation_rate']):.2f}%</td>"
            f"<td>{_fmt(row['max_p99_wait_slots'], 0)}</td>"
            f"<td>{_fmt(row['max_wait_slots'], 0)}</td>"
            f"<td>{int(row['unique_selected_ues_episode'])}</td>"
            f"<td>{_fmt(row.get('mean_unique_selected_last_window', 0.0), 1)}</td>"
            f"<td>{_fmt(row.get('mean_unique_success_last_window', 0.0), 1)}</td>"
            f"<td>{int(row['never_selected_ues_episode'])}</td>"
            f"<td>{100.0 * float(row['selection_top64_share']):.1f}%</td>"
            f"<td>{_fmt(row['mean_mean_top_k_margin'], 8)}</td>"
            "</tr>"
        )

    pair_sections = []
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summaries:
        by_case.setdefault(str(row["case_id"]), {})[str(row["mode"])] = row
    for case in cases:
        modes = by_case.get(case.case_id, {})
        deterministic = modes.get("deterministic")
        stochastic = modes.get("stochastic")
        if deterministic is None or stochastic is None:
            continue
        starvation_gap = 100.0 * (
            float(deterministic["max_starvation_rate"])
            - float(stochastic["max_starvation_rate"])
        )
        coverage_gap = int(stochastic["unique_selected_ues_episode"]) - int(
            deterministic["unique_selected_ues_episode"]
        )
        deterministic_window_success = float(
            deterministic["mean_unique_success_last_window"]
        )
        stochastic_window_success = float(
            stochastic["mean_unique_success_last_window"]
        )
        likely_tie = (
            float(deterministic["max_starvation_rate"]) > 0.20
            and stochastic_window_success > deterministic_window_success + 100.0
            and float(deterministic["mean_mean_top_k_margin"]) < 1e-4
        )
        verdict = (
            "Dấu hiệu mạnh của deterministic coverage collapse: mean priority có ranh giới Top-K gần hòa, "
            "và sampling giúp nhiều UE khác nhau truyền thành công trong mỗi cửa sổ 64 slot."
            if likely_tie
            else "Chưa đủ bằng chứng để quy nguyên nhân chính cho Top-K tie; cần xem CSV theo slot, cửa sổ 64 slot và từng UE."
        )
        pair_sections.append(
            "<div class='case-card'>"
            f"<h3>{html.escape(case.label)}</h3>"
            f"<p>Stochastic phục vụ thêm <strong>{coverage_gap}</strong> UE khác nhau trong episode. "
            f"Chênh lệch starvation cực đại deterministic − stochastic là <strong>{starvation_gap:+.2f} điểm phần trăm</strong>.</p>"
            f"<p>{verdict}</p>"
            "</div>"
        )

    content = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScaleMAC-RL – Policy diagnostics</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;background:#f4f7fb;color:#172033;line-height:1.58;margin:0}}
main{{max-width:1280px;margin:auto;padding:28px 18px 60px}}header{{background:linear-gradient(135deg,#18264a,#315efb);color:white;border-radius:18px;padding:28px}}
.card,.case-card{{background:white;border:1px solid #dfe5ee;border-radius:15px;padding:20px;margin:16px 0;box-shadow:0 7px 20px rgba(23,32,51,.05)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dfe5ee;text-align:left;vertical-align:top}}th{{background:#eef2f8}}
.good{{background:#e9f7ef;padding:14px;border-radius:10px}}.warn{{background:#fff5d8;padding:14px;border-radius:10px}}code{{background:#eef1f7;padding:2px 6px;border-radius:5px}}
</style></head><body><main>
<header><h1>Round 03: chẩn đoán deterministic và stochastic policy</h1>
<p>Mục tiêu: xác định fairness thấp đến từ reward, hay từ priority gần hòa khiến deterministic Top-K chọn lặp một nhóm UE.</p></header>
<section class="card"><h2>Cấu hình được kiểm tra</h2>
<table><thead><tr><th>Case</th><th>Throughput weight</th><th>Jain weight</th><th>Checkpoint</th></tr></thead><tbody>{''.join(weights_rows)}</tbody></table>
<p>Mỗi checkpoint chạy hai lần trên cùng seed và static UE profile: <strong>deterministic</strong> dùng mean action; <strong>stochastic</strong> lấy mẫu từ Beta policy như trong rollout train.</p></section>
<section class="card"><h2>Kết quả tổng hợp</h2><table><thead><tr><th>Case</th><th>Mode</th><th>Goodput</th><th>Jain</th><th>Max starvation</th><th>Worst P99 wait</th><th>Max wait</th><th>UE từng được chọn</th><th>TB UE được chọn / 64 slot</th><th>TB UE thành công / 64 slot</th><th>UE chưa từng được chọn</th><th>Top-64 selection share</th><th>Mean Top-64 margin</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></section>
<section class="card"><h2>Cách hiểu các chẩn đoán</h2>
<ul><li><strong>UE từng được chọn:</strong> tổng số UE khác nhau đã xuất hiện ít nhất một lần trong Top-64.</li>
<li><strong>Top-64 selection share:</strong> tỷ lệ toàn bộ lượt cấp lịch thuộc về 64 UE được chọn nhiều nhất. Gần 100% nghĩa là policy gần như chỉ lặp cùng một nhóm.</li>
<li><strong>Top-64 margin:</strong> chênh lệch priority giữa UE xếp thứ 64 và 65. Gần 0 nghĩa là ranh giới Top-K rất dễ bị quyết định bởi cách phá hòa.</li>
<li><strong>Unique success last {window_size} slots:</strong> có bao nhiêu UE khác nhau truyền thành công trong cửa sổ {window_size} action gần nhất. Starvation threshold cũng là {window_size} slot trong thí nghiệm này.</li></ul>
<p>Score được coi là gần hòa nếu cách cutoff không quá <code>{tie_epsilon:g}</code>.</p></section>
<section class="card"><h2>So sánh theo từng case</h2>{''.join(pair_sections) if pair_sections else '<p>Chưa có đủ cả hai mode cho cùng case.</p>'}</section>
<section class="card"><h2>File chi tiết</h2><ul><li><code>summary.csv</code>: một dòng cho mỗi case/mode/seed.</li><li><code>slot_diagnostics.csv</code>: KPI, priority spread và coverage ở từng slot.</li><li><code>window_diagnostics.csv</code>: coverage theo cửa sổ 64 slot.</li><li><code>ue_selection.csv</code>: số lần từng UE được chọn và truyền thành công.</li><li><code>manifest.json</code>: checkpoint và tham số chạy.</li></ul></section>
</main></body></html>"""
    output.write_text(content, encoding="utf-8")


def run_diagnostics(
    *,
    cases: list[DiagnosticCase],
    modes: tuple[str, ...],
    device: torch.device,
    slots: int,
    first_seed: int,
    seeds: int,
    profile_seed: int,
    window_size: int,
    tie_epsilon: float,
    output_root: Path,
    docs_output: Path | None = None,
) -> dict[str, Path]:
    if slots <= 0 or seeds <= 0 or window_size <= 0:
        raise ValueError("slots, seeds, and window_size must be positive")
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    ue_rows: list[dict[str, Any]] = []
    for case in cases:
        for mode in modes:
            for offset in range(seeds):
                seed = first_seed + offset
                print(
                    f"diagnose case={case.case_id} mode={mode} seed={seed} slots={slots}"
                )
                summary, slots_out, windows_out, ues_out = evaluate_case_mode(
                    case=case,
                    mode=mode,
                    device=device,
                    slots=slots,
                    seed=seed,
                    profile_seed=profile_seed,
                    window_size=window_size,
                    tie_epsilon=tie_epsilon,
                )
                summaries.append(summary)
                slot_rows.extend(slots_out)
                window_rows.extend(windows_out)
                ue_rows.extend(ues_out)

    paths = {
        "summary": output_root / "summary.csv",
        "slot": output_root / "slot_diagnostics.csv",
        "window": output_root / "window_diagnostics.csv",
        "ue": output_root / "ue_selection.csv",
        "manifest": output_root / "manifest.json",
        "html": output_root / "analysis.html",
    }
    write_csv(paths["summary"], summaries)
    write_csv(paths["slot"], slot_rows)
    write_csv(paths["window"], window_rows)
    write_csv(paths["ue"], ue_rows)
    paths["manifest"].write_text(
        json.dumps(
            {
                "purpose": "diagnose deterministic Top-K collapse before changing reward or architecture",
                "cases": [
                    {
                        "case_id": case.case_id,
                        "label": case.label,
                        "checkpoint": str(case.checkpoint_path),
                    }
                    for case in cases
                ],
                "modes": list(modes),
                "device": str(device),
                "slots": slots,
                "first_seed": first_seed,
                "seeds": seeds,
                "profile_seed": profile_seed,
                "window_size": window_size,
                "tie_epsilon": tie_epsilon,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    build_html_report(
        output=paths["html"],
        summaries=summaries,
        cases=cases,
        window_size=window_size,
        tie_epsilon=tie_epsilon,
    )
    if docs_output is not None:
        build_html_report(
            output=docs_output,
            summaries=summaries,
            cases=cases,
            window_size=window_size,
            tie_epsilon=tie_epsilon,
        )
        paths["docs_html"] = docs_output
    return paths
