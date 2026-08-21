from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .reward_study import RewardStudyPlan, read_csv_rows, safe_float


def _feasible(row: Mapping[str, Any]) -> bool:
    return (
        safe_float(row, "max_starvation_rate") <= 1e-12
        and safe_float(row, "max_p99_wait_slots") <= 50.0 + 1e-12
        and safe_float(row, "max_wait_slots") <= 60.0 + 1e-12
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


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise ValueError(f"empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def build_actor_critic_interference_analysis(*, plan: RewardStudyPlan, round_dir: Path, output_path: Path) -> Path:
    cases=[]; trajectories=[]
    for case in plan.cases:
        run=round_dir/case.case_id
        train=read_csv_rows(run/"training.csv"); val=read_csv_rows(run/"validation.csv")
        if not train or not val: raise ValueError(f"missing rows for {case.case_id}")
        common=dict(plan.common); common.update(case.common_overrides)
        profile=str(common.get("encoder_profile", "shared"))
        seed=int(common.get("seed",0)); latest=val[-1]
        feasible=[r for r in val if _feasible(r)]; tail=train[-max(1,len(train)//5):]
        cases.append({
            "case_id":case.case_id,"profile":profile,"seed":seed,
            "separate_critic_encoder":int(bool(common.get("separate_critic_encoder",False))),
            "ever_service_feasible":int(bool(feasible)),"latest_service_feasible":int(_feasible(latest)),
            "learn_then_drift":int(bool(feasible) and not _feasible(latest)),"latest_full_collapse":int(_collapse(latest)),
            "first_feasible_env_steps":min((int(float(r.get("global_env_steps",0))) for r in feasible),default=-1),
            "tail_actor_grad_norm":_avg(tail,"actor_grad_norm_probe"),
            "tail_critic_grad_norm":_avg(tail,"critic_grad_norm_probe"),
            "tail_critic_to_actor_grad_ratio":_avg(tail,"critic_to_actor_grad_ratio_probe"),
            "tail_value_explained_variance":_avg(tail,"value_explained_variance_preupdate"),
            "tail_value_loss":_avg(tail,"value_loss"),
            "tail_approx_kl":_avg(tail,"approx_kl"),"max_approx_kl":_mx(train,"max_approx_kl"),
            "mean_grad_clip_fraction":_avg(train,"grad_clip_fraction"),
            "latest_goodput_bits_per_slot":safe_float(latest,"mean_goodput_bits_per_slot"),
            "latest_jain_fairness":safe_float(latest,"final_jain_fairness"),
            "latest_starvation_rate":safe_float(latest,"max_starvation_rate"),
            "latest_p99_wait_slots":safe_float(latest,"max_p99_wait_slots"),
            "latest_max_wait_slots":safe_float(latest,"max_wait_slots"),
        })
        for v in val:
            trajectories.append({"case_id":case.case_id,"profile":profile,"seed":seed,
                "update":int(float(v.get("update",0))),"global_env_steps":int(float(v.get("global_env_steps",0))),
                "goodput_bits_per_slot":safe_float(v,"mean_goodput_bits_per_slot"),
                "jain_fairness":safe_float(v,"final_jain_fairness"),"starvation_rate":safe_float(v,"max_starvation_rate"),
                "p99_wait_slots":safe_float(v,"max_p99_wait_slots"),"service_feasible":int(_feasible(v))})
    summaries=[]
    for profile in ["shared","split"]:
        ss=[r for r in cases if r["profile"]==profile]
        if not ss: continue
        summaries.append({
            "profile":profile,
            "latest_service_feasible_seeds":sum(r["latest_service_feasible"] for r in ss),
            "ever_service_feasible_seeds":sum(r["ever_service_feasible"] for r in ss),
            "learn_then_drift_seeds":sum(r["learn_then_drift"] for r in ss),
            "latest_full_collapse_seeds":sum(r["latest_full_collapse"] for r in ss),
            "mean_latest_goodput_bits_per_slot":mean(r["latest_goodput_bits_per_slot"] for r in ss),
            "mean_latest_jain_fairness":mean(r["latest_jain_fairness"] for r in ss),
            "mean_latest_starvation_rate":mean(r["latest_starvation_rate"] for r in ss),
            "mean_tail_critic_to_actor_grad_ratio":mean(r["tail_critic_to_actor_grad_ratio"] for r in ss),
            "mean_tail_value_explained_variance":mean(r["tail_value_explained_variance"] for r in ss),
            "mean_tail_approx_kl":mean(r["tail_approx_kl"] for r in ss),
        })
    ranking=sorted(summaries,key=lambda r:(-r["latest_service_feasible_seeds"],-r["ever_service_feasible_seeds"],r["latest_full_collapse_seeds"],r["mean_latest_starvation_rate"],-r["mean_latest_jain_fairness"]))
    ranking=[{"rank":i+1,**r} for i,r in enumerate(ranking)]
    recommended=ranking[0]["profile"] if ranking else "none"
    a=plan.analysis
    _write(Path(str(a["metrics_output"])),cases); _write(Path(str(a["summary_output"])),summaries); _write(Path(str(a["ranking_output"])),ranking); _write(Path(str(a["trajectory_output"])),trajectories)
    Path(str(a["decision_output"])).parent.mkdir(parents=True,exist_ok=True)
    Path(str(a["decision_output"])).write_text(json.dumps({"recommended_profile":recommended,"profiles":summaries},indent=2),encoding="utf-8")
    rows="".join(f"<tr><td>{r['rank']}</td><td>{html.escape(r['profile'])}</td><td>{r['latest_service_feasible_seeds']}/3</td><td>{r['ever_service_feasible_seeds']}/3</td><td>{r['learn_then_drift_seeds']}/3</td><td>{r['mean_latest_jain_fairness']:.4f}</td><td>{100*r['mean_latest_starvation_rate']:.2f}%</td><td>{r['mean_tail_critic_to_actor_grad_ratio']:.2f}x</td><td>{r['mean_tail_value_explained_variance']:.3f}</td></tr>" for r in ranking)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Round 16D</title><style>body{{font-family:Segoe UI,Arial;max-width:1050px;margin:30px auto;line-height:1.5}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#f3f4f6}}</style></head><body><h1>Round 16D — Actor/Critic Encoder Interference</h1><p>Shared encoder vs independently trainable critic encoder, initialized to identical step-0 actor and critic functions.</p><table><tr><th>Rank</th><th>Profile</th><th>Latest feasible</th><th>Ever feasible</th><th>Drift</th><th>JFI</th><th>Starvation</th><th>Critic/Actor grad</th><th>Value EV</th></tr>{rows}</table><p><b>Recommended:</b> {html.escape(recommended)}</p></body></html>""",encoding="utf-8")
    md=Path(str(a["markdown_output"])); md.parent.mkdir(parents=True,exist_ok=True); md.write_text(f"# Round 16D — Actor/Critic Encoder Interference\n\nRecommended: **{recommended}**\n",encoding="utf-8")
    return output_path
