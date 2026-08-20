from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from .reward_study import RewardStudyPlan, read_csv_rows, safe_float


SERVICE_STARVATION_LIMIT = 0.0
SERVICE_P99_LIMIT = 50.0
SERVICE_MAX_WAIT_LIMIT = 60.0
_TOL = 1e-12


def _service_feasible(row: Mapping[str, Any]) -> bool:
    return (
        safe_float(row, "max_starvation_rate") <= SERVICE_STARVATION_LIMIT + _TOL
        and safe_float(row, "max_p99_wait_slots") <= SERVICE_P99_LIMIT + _TOL
        and safe_float(row, "max_wait_slots") <= SERVICE_MAX_WAIT_LIMIT + _TOL
    )


def _full_collapse(row: Mapping[str, Any]) -> bool:
    return (
        safe_float(row, "max_starvation_rate") >= 0.5
        or safe_float(row, "max_p99_wait_slots") >= 4999.0
        or safe_float(row, "max_wait_slots") >= 4999.0
    )


def _avg(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [safe_float(row, key) for row in rows if str(row.get(key, "")).strip() != ""]
    return mean(values) if values else 0.0


def _max(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [safe_float(row, key) for row in rows if str(row.get(key, "")).strip() != ""]
    return max(values) if values else 0.0


def _summary(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile(case, common: Mapping[str, Any]) -> str:
    explicit = str(common.get("audit_profile", "")).strip()
    if explicit:
        return explicit
    large = int(common.get("rollout_steps", 256)) > 256
    value_clip = float(common.get("value_clip_coef", 0.0)) > 0.0
    if large and value_clip:
        return "large_sampling_critic_stabilized"
    if large:
        return "large_sampling"
    if value_clip:
        return "critic_stabilized"
    return "baseline"


def _factor_levels(profile: str) -> tuple[str, str]:
    sampling = "large" if "large_sampling" in profile else "standard"
    critic = "critic_stabilized" if "critic_stabilized" in profile else "standard"
    return sampling, critic


def _latest_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "latest_goodput_bits_per_slot": safe_float(row, "mean_goodput_bits_per_slot"),
        "latest_spectral_efficiency_bps_hz": safe_float(row, "mean_spectral_efficiency_bps_hz"),
        "latest_jain_fairness": safe_float(row, "final_jain_fairness"),
        "latest_starvation_rate": safe_float(row, "max_starvation_rate"),
        "latest_p99_wait_slots": safe_float(row, "max_p99_wait_slots"),
        "latest_max_wait_slots": safe_float(row, "max_wait_slots"),
        "latest_observed_bler": safe_float(row, "mean_observed_bler"),
        "latest_harq_retx_fraction": safe_float(row, "mean_harq_retransmission_fraction"),
        "latest_service_feasible": int(_service_feasible(row)),
        "latest_full_collapse": int(_full_collapse(row)),
    }


def build_ppo_root_cause_analysis(
    *, plan: RewardStudyPlan, round_dir: Path, output_path: Path
) -> Path:
    case_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    for case in plan.cases:
        case_dir = round_dir / case.case_id
        training = read_csv_rows(case_dir / "training.csv")
        validations = read_csv_rows(case_dir / "validation.csv")
        if not training or not validations:
            raise ValueError(f"missing training/validation rows for {case.case_id}")

        common = dict(plan.common)
        common.update(case.common_overrides)
        profile = _profile(case, common)
        sampling, critic = _factor_levels(profile)
        seed = int(common.get("seed", 1701))
        tail_count = max(1, len(training) // 5)
        tail = training[-tail_count:]
        latest = validations[-1]
        feasible = [row for row in validations if _service_feasible(row)]
        ever_feasible = bool(feasible)
        latest_feasible = _service_feasible(latest)

        row: dict[str, Any] = {
            "case_id": case.case_id,
            "profile": profile,
            "sampling_factor": sampling,
            "critic_factor": critic,
            "seed": seed,
            "rollout_steps": int(common.get("rollout_steps", 256)),
            "minibatch_size": int(common.get("minibatch_size", 8)),
            "value_coef": float(common.get("value_coef", 0.5)),
            "value_clip_coef": float(common.get("value_clip_coef", 0.0)),
            "environment_steps_requested": int(common.get("environment_steps", 0)),
            "training_updates": len(training),
            "validation_points": len(validations),
            "ever_service_feasible": int(ever_feasible),
            "latest_service_feasible": int(latest_feasible),
            "learn_then_drift": int(ever_feasible and not latest_feasible),
            "first_feasible_global_env_steps": (
                min(int(float(r.get("global_env_steps", 0))) for r in feasible)
                if feasible else -1
            ),
            "last_feasible_global_env_steps": (
                max(int(float(r.get("global_env_steps", 0))) for r in feasible)
                if feasible else -1
            ),
            "mean_approx_kl": _avg(training, "approx_kl"),
            "tail_mean_approx_kl": _avg(tail, "approx_kl"),
            "max_approx_kl": _max(training, "max_approx_kl"),
            "mean_clip_fraction": _avg(training, "clip_fraction"),
            "tail_mean_clip_fraction": _avg(tail, "clip_fraction"),
            "mean_grad_clip_fraction": _avg(training, "grad_clip_fraction"),
            "max_grad_norm_preclip": _max(training, "max_grad_norm_preclip"),
            "mean_ppo_early_stop_rate": _avg(training, "ppo_early_stop"),
            "tail_ppo_early_stop_rate": _avg(tail, "ppo_early_stop"),
            "mean_minibatches_processed": _avg(training, "ppo_minibatches_processed"),
            "mean_sample_reuse": _avg(training, "ppo_sample_reuse"),
            "mean_value_loss": _avg(training, "value_loss"),
            "tail_mean_value_loss": _avg(tail, "value_loss"),
            "max_value_loss": _max(training, "value_loss"),
            "mean_value_explained_variance": _avg(training, "value_explained_variance_preupdate"),
            "tail_value_explained_variance": _avg(tail, "value_explained_variance_preupdate"),
            "mean_value_rmse": _avg(training, "value_rmse_preupdate"),
            "tail_value_rmse": _avg(tail, "value_rmse_preupdate"),
            "mean_actor_grad_norm_probe": _avg(training, "actor_grad_norm_probe"),
            "mean_critic_grad_norm_probe": _avg(training, "critic_grad_norm_probe"),
            "mean_critic_to_actor_grad_ratio_probe": _avg(training, "critic_to_actor_grad_ratio_probe"),
            "tail_critic_to_actor_grad_ratio_probe": _avg(tail, "critic_to_actor_grad_ratio_probe"),
            "mean_value_clip_fraction": _avg(training, "value_clip_fraction"),
            "mean_steps_per_second": _avg(tail, "steps_per_second"),
            "max_ratio": _max(training, "max_ratio"),
            "max_abs_log_ratio": _max(training, "max_abs_log_ratio"),
        }
        row.update(_latest_payload(latest))
        case_rows.append(row)

        for validation in validations:
            trajectory_rows.append(
                {
                    "case_id": case.case_id,
                    "profile": profile,
                    "sampling_factor": sampling,
                    "critic_factor": critic,
                    "seed": seed,
                    "update": int(float(validation.get("update", 0))),
                    "global_env_steps": int(float(validation.get("global_env_steps", 0))),
                    "goodput_bits_per_slot": safe_float(validation, "mean_goodput_bits_per_slot"),
                    "jain_fairness": safe_float(validation, "final_jain_fairness"),
                    "starvation_rate": safe_float(validation, "max_starvation_rate"),
                    "p99_wait_slots": safe_float(validation, "max_p99_wait_slots"),
                    "max_wait_slots": safe_float(validation, "max_wait_slots"),
                    "service_feasible": int(_service_feasible(validation)),
                    "is_latest": int(validation is latest),
                }
            )

    profile_order = ["baseline", "large_sampling", "critic_stabilized", "large_sampling_critic_stabilized"]
    summary_rows: list[dict[str, Any]] = []
    for profile in profile_order:
        subset = [row for row in case_rows if row["profile"] == profile]
        if not subset:
            continue
        entry: dict[str, Any] = {
            "profile": profile,
            "seeds": len(subset),
            "sampling_factor": subset[0]["sampling_factor"],
            "critic_factor": subset[0]["critic_factor"],
            "rollout_steps": subset[0]["rollout_steps"],
            "minibatch_size": subset[0]["minibatch_size"],
            "value_clip_coef": subset[0]["value_clip_coef"],
            "latest_service_feasible_seeds": sum(int(r["latest_service_feasible"]) for r in subset),
            "ever_service_feasible_seeds": sum(int(r["ever_service_feasible"]) for r in subset),
            "learn_then_drift_seeds": sum(int(r["learn_then_drift"]) for r in subset),
            "latest_full_collapse_seeds": sum(int(r["latest_full_collapse"]) for r in subset),
        }
        for metric in (
            "latest_goodput_bits_per_slot",
            "latest_jain_fairness",
            "latest_starvation_rate",
            "latest_p99_wait_slots",
            "tail_mean_approx_kl",
            "tail_mean_clip_fraction",
            "mean_grad_clip_fraction",
            "mean_ppo_early_stop_rate",
            "tail_mean_value_loss",
            "tail_value_explained_variance",
            "tail_critic_to_actor_grad_ratio_probe",
            "mean_steps_per_second",
        ):
            avg, sd = _summary([float(r[metric]) for r in subset])
            entry[f"mean_{metric}"] = avg
            entry[f"std_{metric}"] = sd
        summary_rows.append(entry)

    ranking_rows = sorted(
        summary_rows,
        key=lambda r: (
            -int(r["latest_service_feasible_seeds"]),
            -int(r["ever_service_feasible_seeds"]),
            int(r["latest_full_collapse_seeds"]),
            float(r["mean_latest_starvation_rate"]),
            float(r["mean_latest_p99_wait_slots"]),
            -float(r["mean_latest_jain_fairness"]),
        ),
    )
    for rank, row in enumerate(ranking_rows, 1):
        row["rank"] = rank

    # 2x2 factor effects. These are descriptive averages across the two paired levels.
    factor_rows: list[dict[str, Any]] = []
    for metric in (
        "latest_service_feasible",
        "latest_starvation_rate",
        "latest_p99_wait_slots",
        "latest_jain_fairness",
        "latest_goodput_bits_per_slot",
        "tail_mean_approx_kl",
        "tail_mean_value_loss",
        "tail_value_explained_variance",
        "tail_critic_to_actor_grad_ratio_probe",
    ):
        standard_sampling = [float(r[metric]) for r in case_rows if r["sampling_factor"] == "standard"]
        large_sampling = [float(r[metric]) for r in case_rows if r["sampling_factor"] == "large"]
        standard_critic = [float(r[metric]) for r in case_rows if r["critic_factor"] == "standard"]
        value_clip = [float(r[metric]) for r in case_rows if r["critic_factor"] == "critic_stabilized"]
        factor_rows.append({
            "metric": metric,
            "standard_sampling_mean": mean(standard_sampling),
            "large_sampling_mean": mean(large_sampling),
            "large_minus_standard_sampling": mean(large_sampling) - mean(standard_sampling),
            "standard_critic_mean": mean(standard_critic),
            "critic_stabilized_mean": mean(value_clip),
            "critic_stabilized_minus_standard_critic": mean(value_clip) - mean(standard_critic),
        })

    metrics_output = Path(str(plan.analysis.get("metrics_output", output_path.with_suffix(".csv"))))
    summary_output = Path(str(plan.analysis.get("summary_output", output_path.with_name("profile_summary.csv"))))
    ranking_output = Path(str(plan.analysis.get("ranking_output", output_path.with_name("profile_ranking.csv"))))
    factor_output = Path(str(plan.analysis.get("factor_output", output_path.with_name("factor_effects.csv"))))
    trajectory_output = Path(str(plan.analysis.get("trajectory_output", output_path.with_name("validation_trajectory.csv"))))
    decision_output = Path(str(plan.analysis.get("decision_output", output_path.with_name("decision.json"))))
    markdown_output = Path(str(plan.analysis.get("markdown_output", output_path.with_suffix(".md"))))

    _write_csv(metrics_output, case_rows)
    _write_csv(summary_output, summary_rows)
    _write_csv(ranking_output, ranking_rows)
    _write_csv(factor_output, factor_rows)
    _write_csv(trajectory_output, trajectory_rows)

    recommended = ranking_rows[0]
    decision = {
        "recommended_profile": recommended["profile"],
        "latest_service_feasible_seeds": int(recommended["latest_service_feasible_seeds"]),
        "ever_service_feasible_seeds": int(recommended["ever_service_feasible_seeds"]),
        "learn_then_drift_seeds": int(recommended["learn_then_drift_seeds"]),
        "latest_full_collapse_seeds": int(recommended["latest_full_collapse_seeds"]),
        "selection_rule": "robustness first: latest feasible, ever feasible, collapse, starvation/P99, then Jain",
        "scope": "Round 16A only; reward, observations, policy architecture and radio environment are fixed",
    }
    decision_output.parent.mkdir(parents=True, exist_ok=True)
    decision_output.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    rows_html = []
    for row in ranking_rows:
        rows_html.append(
            "<tr>"
            f"<td>{row['rank']}</td><td><code>{html.escape(str(row['profile']))}</code></td>"
            f"<td>{row['latest_service_feasible_seeds']}/3</td>"
            f"<td>{row['ever_service_feasible_seeds']}/3</td>"
            f"<td>{row['learn_then_drift_seeds']}/3</td>"
            f"<td>{row['latest_full_collapse_seeds']}/3</td>"
            f"<td>{row['mean_latest_goodput_bits_per_slot']:.0f}</td>"
            f"<td>{row['mean_latest_jain_fairness']:.4f}</td>"
            f"<td>{100*row['mean_latest_starvation_rate']:.2f}%</td>"
            f"<td>{row['mean_latest_p99_wait_slots']:.1f}</td>"
            f"<td>{row['mean_tail_mean_approx_kl']:.4f}</td>"
            f"<td>{row['mean_tail_critic_to_actor_grad_ratio_probe']:.2f}</td>"
            "</tr>"
        )

    title = html.escape(str(plan.analysis.get("title", "Round 16A — PPO Root-Cause Audit")))
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;line-height:1.5;color:#1f2937}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #d1d5db;padding:7px;text-align:left}}th{{background:#f3f4f6}}code{{background:#f3f4f6;padding:2px 4px;border-radius:4px}}.callout{{border-left:5px solid #2563eb;background:#eff6ff;padding:14px 16px;border-radius:7px;margin:16px 0}}</style></head><body>
<h1>{title}</h1>
<div class='callout'><b>Purpose:</b> diagnose PPO update instability without changing reward, observation features, architecture, or radio environment. The two controlled factors are sampling regime and a critic-stabilization package (value_coef 0.25 + value clipping 0.2).</div>
<h2>Profile ranking</h2>
<table><thead><tr><th>Rank</th><th>Profile</th><th>Latest feasible</th><th>Ever feasible</th><th>Learn→drift</th><th>Collapse</th><th>Goodput</th><th>JFI</th><th>Starvation</th><th>P99</th><th>Tail KL</th><th>Critic/actor grad</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table>
<h2>Interpretation guardrails</h2>
<p>This round is a 2×2 screen. A winning profile identifies a useful stabilization direction, not a final PPO configuration. If larger sampling wins, the next round can flex rollout/minibatch separately. If critic stabilization wins, value coefficient and value clipping can then be isolated. If neither wins, reward/constraint or architecture changes become stronger candidates.</p>
<p>All actor/critic gradient ratios are diagnostic probes from the first minibatch of each PPO update; they are not used to modify optimization.</p>
<h2>Decision</h2><p>Recommended within Round 16A: <code>{html.escape(str(recommended['profile']))}</code>, with {recommended['latest_service_feasible_seeds']}/3 latest service-feasible seeds.</p>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")

    markdown = [
        f"# {plan.analysis.get('title', 'Round 16A — PPO Root-Cause Audit')}",
        "",
        "Controlled 2×2 screen: standard/large sampling × standard/critic-stabilized package.",
        "Reward, 16-feature observation, feed-forward PPO architecture and radio environment are fixed.",
        "",
        f"Recommended within this screen: **{recommended['profile']}**.",
        f"Latest service-feasible seeds: **{recommended['latest_service_feasible_seeds']}/3**.",
        f"Ever service-feasible seeds: **{recommended['ever_service_feasible_seeds']}/3**.",
        f"Learn→drift seeds: **{recommended['learn_then_drift_seeds']}/3**.",
        "",
        "See CSV outputs for per-seed diagnostics, factor effects, and validation trajectory.",
    ]
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return output_path
