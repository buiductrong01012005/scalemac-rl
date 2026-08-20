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


def _profile(common: Mapping[str, Any]) -> str:
    explicit = str(common.get("sampling_profile", "")).strip()
    if explicit:
        return explicit
    rollout = int(common.get("rollout_steps", 256))
    steps = int(common.get("environment_steps", 0))
    return f"r{rollout}_e{steps}"


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


def _paired_effects(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row["profile"]), int(row["seed"])): row for row in case_rows}
    comparisons = [
        (
            "same_env_budget_large_minus_baseline",
            "r256_e100k",
            "r1024_e100k",
            "same ~98k environment interactions; isolates larger rollout/minibatch with fewer outer updates",
        ),
        (
            "same_update_count_r512_minus_baseline",
            "r256_e100k",
            "r512_e200k",
            "~384 outer updates on both sides; R512 receives 2x environment interactions",
        ),
        (
            "same_update_count_r1024_minus_baseline",
            "r256_e100k",
            "r1024_e400k",
            "~384 outer updates on both sides; R1024 receives 4x environment interactions",
        ),
        (
            "r1024_long_budget_minus_short_budget",
            "r1024_e100k",
            "r1024_e400k",
            "same R1024/MB64 sampling; isolates additional interaction budget / policy-refresh cycles",
        ),
    ]
    metrics = [
        "latest_service_feasible",
        "latest_full_collapse",
        "latest_starvation_rate",
        "latest_p99_wait_slots",
        "latest_jain_fairness",
        "latest_goodput_bits_per_slot",
        "tail_mean_approx_kl",
        "mean_ppo_early_stop_rate",
        "mean_grad_clip_fraction",
        "tail_value_explained_variance",
        "tail_critic_to_actor_grad_ratio_probe",
    ]
    rows: list[dict[str, Any]] = []
    for comparison, left, right, interpretation in comparisons:
        for metric in metrics:
            deltas = []
            matched = 0
            for seed in (1701, 2701, 3701):
                a = by_key.get((left, seed))
                b = by_key.get((right, seed))
                if a is None or b is None:
                    continue
                matched += 1
                deltas.append(float(b[metric]) - float(a[metric]))
            if not deltas:
                continue
            avg, sd = _summary(deltas)
            rows.append(
                {
                    "comparison": comparison,
                    "left_profile": left,
                    "right_profile": right,
                    "metric": metric,
                    "paired_seeds": matched,
                    "mean_right_minus_left": avg,
                    "std_right_minus_left": sd,
                    "interpretation": interpretation,
                }
            )
    return rows


def build_sampling_budget_analysis(
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
        profile = _profile(common)
        seed = int(common.get("seed", 1701))
        rollout_steps = int(common.get("rollout_steps", 256))
        minibatch_size = int(common.get("minibatch_size", 8))
        env_steps = int(common.get("environment_steps", 0))
        target_outer_updates = env_steps // rollout_steps
        tail_count = max(1, len(training) // 5)
        tail = training[-tail_count:]
        latest = validations[-1]
        feasible = [row for row in validations if _service_feasible(row)]
        ever_feasible = bool(feasible)
        latest_feasible = _service_feasible(latest)

        mean_minibatches = _avg(training, "ppo_minibatches_processed")
        estimated_grad_minibatches = len(training) * mean_minibatches
        estimated_transition_samples = estimated_grad_minibatches * minibatch_size

        row: dict[str, Any] = {
            "case_id": case.case_id,
            "profile": profile,
            "seed": seed,
            "environment_steps_requested": env_steps,
            "rollout_steps": rollout_steps,
            "minibatch_size": minibatch_size,
            "target_outer_updates": target_outer_updates,
            "training_updates": len(training),
            "validation_points": len(validations),
            "env_steps_per_outer_update": rollout_steps,
            "minibatch_fraction_of_rollout": minibatch_size / float(rollout_steps),
            "ever_service_feasible": int(ever_feasible),
            "latest_service_feasible": int(latest_feasible),
            "learn_then_drift": int(ever_feasible and not latest_feasible),
            "latest_full_collapse": int(_full_collapse(latest)),
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
            "mean_ppo_early_stop_rate": _avg(training, "ppo_early_stop"),
            "tail_ppo_early_stop_rate": _avg(tail, "ppo_early_stop"),
            "mean_minibatches_processed": mean_minibatches,
            "mean_sample_reuse": _avg(training, "ppo_sample_reuse"),
            "estimated_gradient_minibatches": estimated_grad_minibatches,
            "estimated_transition_samples_optimized": estimated_transition_samples,
            "optimized_transition_samples_per_env_transition": (
                estimated_transition_samples / float(env_steps) if env_steps else 0.0
            ),
            "tail_mean_value_loss": _avg(tail, "value_loss"),
            "tail_value_explained_variance": _avg(tail, "value_explained_variance_preupdate"),
            "tail_value_rmse": _avg(tail, "value_rmse_preupdate"),
            "tail_critic_to_actor_grad_ratio_probe": _avg(tail, "critic_to_actor_grad_ratio_probe"),
            "mean_steps_per_second": _avg(tail, "steps_per_second"),
        }
        row.update(_latest_payload(latest))
        case_rows.append(row)

        for validation in validations:
            trajectory_rows.append(
                {
                    "case_id": case.case_id,
                    "profile": profile,
                    "seed": seed,
                    "rollout_steps": rollout_steps,
                    "minibatch_size": minibatch_size,
                    "environment_steps_requested": env_steps,
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

    profile_order = ["r256_e100k", "r1024_e100k", "r512_e200k", "r1024_e400k"]
    summary_rows: list[dict[str, Any]] = []
    for profile in profile_order:
        subset = [row for row in case_rows if row["profile"] == profile]
        if not subset:
            continue
        entry: dict[str, Any] = {
            "profile": profile,
            "seeds": len(subset),
            "environment_steps_requested": subset[0]["environment_steps_requested"],
            "rollout_steps": subset[0]["rollout_steps"],
            "minibatch_size": subset[0]["minibatch_size"],
            "target_outer_updates": subset[0]["target_outer_updates"],
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
            "mean_minibatches_processed",
            "optimized_transition_samples_per_env_transition",
            "mean_steps_per_second",
        ):
            avg, sd = _summary([float(r[metric]) for r in subset])
            entry[f"mean_{metric}"] = avg
            entry[f"std_{metric}"] = sd
        summary_rows.append(entry)

    ranking_rows = sorted(
        [dict(row) for row in summary_rows],
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

    paired_rows = _paired_effects(case_rows)

    metrics_output = Path(str(plan.analysis.get("metrics_output", output_path.with_suffix(".csv"))))
    summary_output = Path(str(plan.analysis.get("summary_output", output_path.with_name("profile_summary.csv"))))
    ranking_output = Path(str(plan.analysis.get("ranking_output", output_path.with_name("profile_ranking.csv"))))
    paired_output = Path(str(plan.analysis.get("paired_output", output_path.with_name("paired_effects.csv"))))
    trajectory_output = Path(str(plan.analysis.get("trajectory_output", output_path.with_name("validation_trajectory.csv"))))
    decision_output = Path(str(plan.analysis.get("decision_output", output_path.with_name("decision.json"))))
    markdown_output = Path(str(plan.analysis.get("markdown_output", output_path.with_suffix(".md"))))

    _write_csv(metrics_output, case_rows)
    _write_csv(summary_output, summary_rows)
    _write_csv(ranking_output, ranking_rows)
    _write_csv(paired_output, paired_rows)
    _write_csv(trajectory_output, trajectory_rows)

    by_profile = {str(row["profile"]): row for row in summary_rows}
    same_env_candidates = [by_profile[p] for p in ("r256_e100k", "r1024_e100k") if p in by_profile]
    same_update_candidates = [by_profile[p] for p in ("r256_e100k", "r512_e200k", "r1024_e400k") if p in by_profile]

    def pick(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return sorted(
            rows,
            key=lambda r: (
                -int(r["latest_service_feasible_seeds"]),
                -int(r["ever_service_feasible_seeds"]),
                int(r["latest_full_collapse_seeds"]),
                float(r["mean_latest_starvation_rate"]),
                float(r["mean_latest_p99_wait_slots"]),
                -float(r["mean_latest_jain_fairness"]),
            ),
        )[0]

    overall = ranking_rows[0]
    same_env = pick(same_env_candidates)
    same_updates = pick(same_update_candidates)
    decision = {
        "recommended_robustness_profile": overall["profile"],
        "best_same_environment_budget_profile": same_env["profile"],
        "best_same_outer_update_count_profile": same_updates["profile"],
        "latest_service_feasible_seeds": int(overall["latest_service_feasible_seeds"]),
        "ever_service_feasible_seeds": int(overall["ever_service_feasible_seeds"]),
        "learn_then_drift_seeds": int(overall["learn_then_drift_seeds"]),
        "latest_full_collapse_seeds": int(overall["latest_full_collapse_seeds"]),
        "selection_rule": "robustness first; interpret winners separately for same env budget and same outer-update-count controls",
        "scope": "Round 16B only; service40 reward, observation, critic, PPO clipping/KL settings and radio environment are fixed",
    }
    decision_output.parent.mkdir(parents=True, exist_ok=True)
    decision_output.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    table_rows = []
    for row in ranking_rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['rank']}</td><td><code>{html.escape(str(row['profile']))}</code></td>"
            f"<td>{row['rollout_steps']}</td><td>{row['minibatch_size']}</td>"
            f"<td>{int(row['environment_steps_requested']):,}</td><td>{row['target_outer_updates']}</td>"
            f"<td>{row['latest_service_feasible_seeds']}/3</td><td>{row['ever_service_feasible_seeds']}/3</td>"
            f"<td>{row['latest_full_collapse_seeds']}/3</td>"
            f"<td>{row['mean_latest_goodput_bits_per_slot']:.0f}</td>"
            f"<td>{row['mean_latest_jain_fairness']:.4f}</td>"
            f"<td>{100*row['mean_latest_starvation_rate']:.2f}%</td>"
            f"<td>{row['mean_tail_mean_approx_kl']:.4f}</td>"
            "</tr>"
        )

    title = html.escape(str(plan.analysis.get("title", "Round 16B — PPO Sampling Budget Control")))
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;line-height:1.5;color:#1f2937}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #d1d5db;padding:7px;text-align:left}}th{{background:#f3f4f6}}code{{background:#f3f4f6;padding:2px 4px;border-radius:4px}}.callout{{border-left:5px solid #2563eb;background:#eff6ff;padding:14px 16px;border-radius:7px;margin:16px 0}}</style></head><body>
<h1>{title}</h1>
<div class='callout'><b>Purpose:</b> determine whether Round 16A large-rollout failure came from the rollout itself or from giving it too few policy-refresh cycles at a fixed interaction budget. Reward, observation, critic, target-KL, clipping and radio environment remain fixed.</div>
<h2>Profiles</h2>
<table><thead><tr><th>Rank</th><th>Profile</th><th>Rollout</th><th>MB</th><th>Env steps</th><th>Target updates</th><th>Latest feasible</th><th>Ever feasible</th><th>Collapse</th><th>Goodput</th><th>JFI</th><th>Starvation</th><th>Tail KL</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Controls</h2>
<p><b>Same environment budget:</b> R256/MB8 @ 98k versus R1024/MB64 @ 98k. This asks whether a larger on-policy batch helps when interaction cost is held fixed.</p>
<p><b>Same outer-update count:</b> R256/MB8 @ 98k, R512/MB32 @ 196k, and R1024/MB64 @ 393k all target 384 policy-refresh cycles. Because LR/entropy schedules are based on progress through each run, their schedules are aligned in update-space when the total update count matches.</p>
<p><b>Same sampling, more budget:</b> R1024 @ 98k versus R1024 @ 393k isolates whether the large-rollout profile in Round 16A was simply under-trained in policy-refresh count.</p>
<h2>Decision</h2>
<p>Best under the same ~98k environment budget: <code>{html.escape(str(same_env['profile']))}</code>.</p>
<p>Best among the ~384-update profiles: <code>{html.escape(str(same_updates['profile']))}</code>.</p>
<p>Overall robustness winner in this round: <code>{html.escape(str(overall['profile']))}</code>.</p>
<p>Do not interpret a 200k/400k winner as more sample-efficient; it consumed 2×/4× more environment interactions by design.</p>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")

    markdown = [
        f"# {plan.analysis.get('title', 'Round 16B — PPO Sampling Budget Control')}",
        "",
        "Controlled sampling-budget study on fixed service40 PPO.",
        "",
        f"Best same-environment-budget profile: **{same_env['profile']}**.",
        f"Best same-outer-update-count profile: **{same_updates['profile']}**.",
        f"Overall robustness profile: **{overall['profile']}**.",
        "",
        "Longer-budget profiles are not sample-efficiency winners by default; they intentionally consume more environment interactions.",
        "See CSV outputs for paired effects, per-seed diagnostics, and validation trajectories.",
    ]
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return output_path
