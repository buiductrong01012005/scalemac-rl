from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from .reward_study import RewardStudyPlan, read_csv_rows, safe_float

STARV_LIMIT = 0.0
P99_LIMIT = 50.0
MAX_WAIT_LIMIT = 60.0
TOL = 1e-12


def _feasible(row: Mapping[str, Any]) -> bool:
    return (
        safe_float(row, "max_starvation_rate") <= STARV_LIMIT + TOL
        and safe_float(row, "max_p99_wait_slots") <= P99_LIMIT + TOL
        and safe_float(row, "max_wait_slots") <= MAX_WAIT_LIMIT + TOL
    )


def _collapse(row: Mapping[str, Any]) -> bool:
    return (
        safe_float(row, "max_starvation_rate") >= 0.5
        or safe_float(row, "max_p99_wait_slots") >= 4999.0
        or safe_float(row, "max_wait_slots") >= 4999.0
    )


def _avg(rows: list[Mapping[str, Any]], key: str) -> float:
    vals = [safe_float(r, key) for r in rows if str(r.get(key, "")).strip()]
    return mean(vals) if vals else 0.0


def _mx(rows: list[Mapping[str, Any]], key: str) -> float:
    vals = [safe_float(r, key) for r in rows if str(r.get(key, "")).strip()]
    return max(vals) if vals else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def build_ppo_geometry_analysis(*, plan: RewardStudyPlan, round_dir: Path, output_path: Path) -> Path:
    cases: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for case in plan.cases:
        run = round_dir / case.case_id
        train = read_csv_rows(run / "training.csv")
        val = read_csv_rows(run / "validation.csv")
        if not train or not val:
            raise ValueError(f"missing rows for {case.case_id}")
        common = dict(plan.common); common.update(case.common_overrides)
        profile = str(common.get("geometry_profile", case.case_id.rsplit("_seed", 1)[0]))
        seed = int(common.get("seed", 0))
        latest = val[-1]
        feasible_rows = [r for r in val if _feasible(r)]
        tail = train[-max(1, len(train)//5):]
        row = {
            "case_id": case.case_id,
            "profile": profile,
            "seed": seed,
            "ppo_ratio_mode": str(common.get("ppo_ratio_mode", "joint")),
            "strict_kl_guard": int(bool(common.get("strict_kl_guard", False))),
            "strict_kl_limit": float(common.get("strict_kl_limit", common.get("target_kl", 0.02))),
            "training_updates": len(train),
            "ever_service_feasible": int(bool(feasible_rows)),
            "latest_service_feasible": int(_feasible(latest)),
            "learn_then_drift": int(bool(feasible_rows) and not _feasible(latest)),
            "latest_full_collapse": int(_collapse(latest)),
            "first_feasible_env_steps": min((int(float(r.get("global_env_steps", 0))) for r in feasible_rows), default=-1),
            "tail_mean_approx_kl": _avg(tail, "approx_kl"),
            "max_approx_kl": _mx(train, "max_approx_kl"),
            "tail_mean_post_step_kl": _avg(tail, "post_step_kl"),
            "max_post_step_kl": _mx(train, "max_post_step_kl"),
            "mean_clip_fraction": _avg(train, "clip_fraction"),
            "mean_grad_clip_fraction": _avg(train, "grad_clip_fraction"),
            "mean_early_stop_rate": _avg(train, "ppo_early_stop"),
            "mean_strict_rejections": _avg(train, "strict_kl_rejections"),
            "mean_strict_reject_fraction": _avg(train, "strict_kl_reject_fraction"),
            "max_ratio": _mx(train, "max_ratio"),
            "max_abs_log_ratio": _mx(train, "max_abs_log_ratio"),
            "latest_goodput_bits_per_slot": safe_float(latest, "mean_goodput_bits_per_slot"),
            "latest_jain_fairness": safe_float(latest, "final_jain_fairness"),
            "latest_starvation_rate": safe_float(latest, "max_starvation_rate"),
            "latest_p99_wait_slots": safe_float(latest, "max_p99_wait_slots"),
            "latest_max_wait_slots": safe_float(latest, "max_wait_slots"),
        }
        cases.append(row)
        for v in val:
            trajectories.append({
                "case_id": case.case_id, "profile": profile, "seed": seed,
                "update": int(float(v.get("update", 0))),
                "global_env_steps": int(float(v.get("global_env_steps", 0))),
                "goodput_bits_per_slot": safe_float(v, "mean_goodput_bits_per_slot"),
                "jain_fairness": safe_float(v, "final_jain_fairness"),
                "starvation_rate": safe_float(v, "max_starvation_rate"),
                "p99_wait_slots": safe_float(v, "max_p99_wait_slots"),
                "max_wait_slots": safe_float(v, "max_wait_slots"),
                "service_feasible": int(_feasible(v)),
            })

    order = ["joint", "joint_strict", "perue", "perue_strict"]
    summaries: list[dict[str, Any]] = []
    for profile in order:
        subset = [r for r in cases if r["profile"] == profile]
        if not subset: continue
        summaries.append({
            "profile": profile,
            "latest_service_feasible_seeds": sum(int(r["latest_service_feasible"]) for r in subset),
            "ever_service_feasible_seeds": sum(int(r["ever_service_feasible"]) for r in subset),
            "learn_then_drift_seeds": sum(int(r["learn_then_drift"]) for r in subset),
            "latest_full_collapse_seeds": sum(int(r["latest_full_collapse"]) for r in subset),
            "mean_latest_goodput_bits_per_slot": mean(float(r["latest_goodput_bits_per_slot"]) for r in subset),
            "mean_latest_jain_fairness": mean(float(r["latest_jain_fairness"]) for r in subset),
            "mean_latest_starvation_rate": mean(float(r["latest_starvation_rate"]) for r in subset),
            "mean_tail_approx_kl": mean(float(r["tail_mean_approx_kl"]) for r in subset),
            "max_post_step_kl": max(float(r["max_post_step_kl"]) for r in subset),
            "mean_strict_reject_fraction": mean(float(r["mean_strict_reject_fraction"]) for r in subset),
        })
    ranked = sorted(summaries, key=lambda r: (
        -int(r["latest_service_feasible_seeds"]),
        -int(r["ever_service_feasible_seeds"]),
        int(r["latest_full_collapse_seeds"]),
        float(r["mean_latest_starvation_rate"]),
        -float(r["mean_latest_jain_fairness"]),
    ))
    ranking = [{"rank": i+1, **r} for i, r in enumerate(ranked)]
    recommended = ranking[0]["profile"] if ranking else "none"
    decision = {
        "recommended_profile": recommended,
        "criterion": "robustness-first: latest feasible seeds, ever feasible seeds, collapse, starvation, Jain",
        "profiles": summaries,
    }

    analysis = plan.analysis
    metrics_path = Path(str(analysis.get("metrics_output", output_path.with_suffix(".metrics.csv"))))
    summary_path = Path(str(analysis.get("summary_output", output_path.with_suffix(".summary.csv"))))
    ranking_path = Path(str(analysis.get("ranking_output", output_path.with_suffix(".ranking.csv"))))
    trajectory_path = Path(str(analysis.get("trajectory_output", output_path.with_suffix(".trajectory.csv"))))
    decision_path = Path(str(analysis.get("decision_output", output_path.with_suffix(".decision.json"))))
    _write_csv(metrics_path, cases); _write_csv(summary_path, summaries); _write_csv(ranking_path, ranking); _write_csv(trajectory_path, trajectories)
    decision_path.parent.mkdir(parents=True, exist_ok=True); decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    rows = "".join(
        f"<tr><td>{r['rank']}</td><td>{html.escape(str(r['profile']))}</td><td>{r['latest_service_feasible_seeds']}/3</td>"
        f"<td>{r['ever_service_feasible_seeds']}/3</td><td>{r['latest_full_collapse_seeds']}/3</td>"
        f"<td>{r['mean_latest_goodput_bits_per_slot']:.0f}</td><td>{r['mean_latest_jain_fairness']:.4f}</td>"
        f"<td>{100*r['mean_latest_starvation_rate']:.2f}%</td><td>{r['mean_tail_approx_kl']:.4f}</td>"
        f"<td>{r['max_post_step_kl']:.4f}</td></tr>" for r in ranking
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Round 16C PPO Geometry</title>
<style>body{{font-family:Segoe UI,Arial;max-width:1100px;margin:30px auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#f3f4f6}}</style></head><body>
<h1>Round 16C — PPO Update Geometry</h1><p>Only PPO ratio geometry and strict KL commit protection vary. Reward, observation, rollout, minibatch, optimizer and radio environment are fixed.</p>
<table><tr><th>Rank</th><th>Profile</th><th>Latest feasible</th><th>Ever feasible</th><th>Collapse</th><th>Goodput</th><th>JFI</th><th>Starvation</th><th>Tail KL</th><th>Max post-step KL</th></tr>{rows}</table>
<p><b>Recommended:</b> {html.escape(str(recommended))}</p></body></html>""", encoding="utf-8")
    md_path = Path(str(analysis.get("markdown_output", output_path.with_suffix(".md"))))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = ["# Round 16C — PPO Update Geometry", "", f"Recommended: **{recommended}**", "", "| Rank | Profile | Latest feasible | Ever feasible | Collapse | Mean starvation |", "|---:|---|---:|---:|---:|---:|"]
    for r in ranking:
        md_lines.append(f"| {r['rank']} | {r['profile']} | {r['latest_service_feasible_seeds']}/3 | {r['ever_service_feasible_seeds']}/3 | {r['latest_full_collapse_seeds']}/3 | {r['mean_latest_starvation_rate']:.4f} |")
    md_path.write_text("\n".join(md_lines)+"\n", encoding="utf-8")
    return output_path
