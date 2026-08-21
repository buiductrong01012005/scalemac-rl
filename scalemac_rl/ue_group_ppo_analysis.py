from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import torch

from .policy_diagnostics import DiagnosticCase, evaluate_case_mode


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _profile(case_id: str) -> str:
    if case_id.startswith("joint_"):
        return "joint"
    if case_id.startswith("uegroup_clip_"):
        return "uegroup_clip"
    if case_id.startswith("uegroup_jis_"):
        return "uegroup_jis"
    raise ValueError(case_id)


def _service_feasible(row: pd.Series) -> bool:
    return (
        float(row["max_starvation_rate"]) <= 1e-12
        and float(row["max_p99_wait_slots"]) <= 50.0 + 1e-12
        and float(row["max_wait_slots"]) <= 60.0 + 1e-12
    )


def _final_milestone(case_dir: Path, round_dir: Path) -> Path:
    manifest = pd.read_csv(case_dir / "checkpoint_manifest.csv")
    milestone = manifest[manifest["tag"].astype(str).str.startswith("milestone_")]
    if milestone.empty:
        raise FileNotFoundError(f"no milestone checkpoint recorded for {case_dir.name}")
    row = milestone.sort_values("global_env_steps").iloc[-1]
    raw = Path(str(row["checkpoint"]))
    if raw.is_absolute() and raw.is_file():
        return raw
    # Normal study outputs record a project-relative path beginning with
    # artifacts/.  Search the round directory and its ancestors instead of
    # assuming one fixed nesting depth, which also keeps analysis portable.
    for base in (round_dir, *round_dir.parents):
        candidate = base / raw
        if candidate.is_file():
            return candidate
    fallback = case_dir / "checkpoints" / raw.name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(raw)


def build_ue_group_analysis(
    *, round_dir: Path, output_dir: Path, device: str = "cpu", diagnostic_slots: int = 512
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    case_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    case_dirs = sorted(
        p for p in round_dir.iterdir() if p.is_dir() and (p / "run_config.json").is_file()
    )
    if len(case_dirs) != 9:
        raise ValueError(f"expected 9 cases, found {len(case_dirs)}")

    for case_dir in case_dirs:
        case_id = case_dir.name
        profile = _profile(case_id)
        run_config = json.loads((case_dir / "run_config.json").read_text(encoding="utf-8"))
        effective = dict(run_config.get("effective_common", run_config.get("common", {})))
        case_payload = dict(run_config.get("case", {}))
        seed = int(effective.get("environment_seed", effective.get("seed", 1701)))
        profile_seed = int(effective.get("profile_seed", seed))

        training = pd.read_csv(case_dir / "training.csv")
        validation = pd.read_csv(case_dir / "validation.csv")
        latest = validation.iloc[-1]
        tail = training.iloc[-max(1, len(training) // 5) :]
        ever = any(_service_feasible(row) for _, row in validation.iterrows())
        final_checkpoint = _final_milestone(case_dir, round_dir)

        diagnostic_case = DiagnosticCase(
            case_id=case_id,
            label=str(case_payload.get("label", case_id)),
            case_dir=case_dir,
            checkpoint_path=final_checkpoint,
            run_config=run_config,
        )
        modes: dict[str, dict[str, Any]] = {}
        for mode in ("deterministic", "stochastic"):
            summary, _, _, _ = evaluate_case_mode(
                case=diagnostic_case,
                mode=mode,
                device=dev,
                slots=diagnostic_slots,
                seed=seed,
                profile_seed=profile_seed,
                window_size=64,
                tie_epsilon=1e-6,
            )
            summary = dict(summary)
            summary.update(
                {
                    "profile": profile,
                    "evaluation_seed": seed,
                    "profile_seed": profile_seed,
                    "checkpoint_role": "final_environment_step_milestone",
                    "checkpoint": str(final_checkpoint),
                }
            )
            diagnostic_rows.append(summary)
            modes[mode] = summary

        det = modes["deterministic"]
        sto = modes["stochastic"]
        case_rows.append(
            {
                "case_id": case_id,
                "profile": profile,
                "training_seed": int(effective.get("seed", seed)),
                "environment_seed": seed,
                "profile_seed": profile_seed,
                "ever_service_feasible": int(ever),
                "latest_service_feasible": int(_service_feasible(latest)),
                "latest_goodput": float(latest["mean_goodput_bits_per_slot"]),
                "latest_jfi": float(latest["final_jain_fairness"]),
                "latest_starvation": float(latest["max_starvation_rate"]),
                "latest_p99": float(latest["max_p99_wait_slots"]),
                "latest_max_wait": float(latest["max_wait_slots"]),
                "tail_approx_kl": float(tail["approx_kl"].mean()),
                "max_approx_kl": float(training["max_approx_kl"].max()),
                "tail_clip_fraction": float(tail["clip_fraction"].mean()),
                "tail_grad_clip_fraction": float(tail["grad_clip_fraction"].mean()),
                "tail_actor_grad": float(tail["actor_grad_norm_probe"].mean()),
                "tail_critic_grad": float(tail["critic_grad_norm_probe"].mean()),
                "tail_critic_actor_ratio": float(tail["critic_to_actor_grad_ratio_probe"].mean()),
                "tail_disc_is_loss": float(tail["disc_is_loss"].mean()) if "disc_is_loss" in tail else 0.0,
                "tail_disc_alpha": float(tail["disc_is_alpha_after"].mean()) if "disc_is_alpha_after" in tail else 0.0,
                "max_disc_alpha": float(training["disc_is_alpha_after"].max()) if "disc_is_alpha_after" in training else 0.0,
                "det_jfi": float(det["final_jain_fairness"]),
                "det_starvation": float(det["max_starvation_rate"]),
                "det_unique_selected": int(det["unique_selected_ues_episode"]),
                "det_gini": float(det["selection_gini"]),
                "det_overlap": float(det["mean_selected_overlap_previous"]),
                "sto_jfi": float(sto["final_jain_fairness"]),
                "sto_starvation": float(sto["max_starvation_rate"]),
                "sto_unique_selected": int(sto["unique_selected_ues_episode"]),
                "alignment_starvation_gap": float(det["max_starvation_rate"])
                - float(sto["max_starvation_rate"]),
            }
        )

    profile_rows: list[dict[str, Any]] = []
    for profile in ("joint", "uegroup_clip", "uegroup_jis"):
        rows = [row for row in case_rows if row["profile"] == profile]
        profile_rows.append(
            {
                "profile": profile,
                "latest_service_feasible_seeds": sum(row["latest_service_feasible"] for row in rows),
                "ever_service_feasible_seeds": sum(row["ever_service_feasible"] for row in rows),
                "mean_latest_starvation": mean(row["latest_starvation"] for row in rows),
                "mean_latest_jfi": mean(row["latest_jfi"] for row in rows),
                "mean_det_starvation": mean(row["det_starvation"] for row in rows),
                "mean_det_jfi": mean(row["det_jfi"] for row in rows),
                "mean_det_unique_selected": mean(row["det_unique_selected"] for row in rows),
                "mean_det_overlap": mean(row["det_overlap"] for row in rows),
                "mean_alignment_starvation_gap": mean(row["alignment_starvation_gap"] for row in rows),
                "mean_tail_kl": mean(row["tail_approx_kl"] for row in rows),
                "mean_tail_actor_grad": mean(row["tail_actor_grad"] for row in rows),
                "mean_tail_critic_actor_ratio": mean(row["tail_critic_actor_ratio"] for row in rows),
                "mean_tail_disc_is_loss": mean(row["tail_disc_is_loss"] for row in rows),
                "mean_tail_disc_alpha": mean(row["tail_disc_alpha"] for row in rows),
                "max_disc_alpha": max(row["max_disc_alpha"] for row in rows),
            }
        )

    ranking = sorted(
        profile_rows,
        key=lambda row: (
            -row["latest_service_feasible_seeds"],
            -row["ever_service_feasible_seeds"],
            row["mean_latest_starvation"],
            -row["mean_latest_jfi"],
            -row["mean_det_unique_selected"],
        ),
    )
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank

    winner = ranking[0]
    decision = {
        "round": "19A",
        "recommended_profile": winner["profile"],
        "latest_service_feasible_seeds": winner["latest_service_feasible_seeds"],
        "ever_service_feasible_seeds": winner["ever_service_feasible_seeds"],
        "analysis_integrity": {
            "environment_seed_source": "run_config.effective_common",
            "checkpoint_role": "final environment-step milestone, not restored latest.pt",
            "deterministic_and_stochastic_diagnostics": True,
        },
        "method_note": (
            "uegroup_clip clips factorized priority/demand importance ratios per action dimension, "
            "sums the two surrogate terms within each UE, and averages over active UEs. "
            "uegroup_jis adds adaptive J_IS computed from the 2-D UE-group log-ratio, then averages J_IS over active UEs."
        ),
    }

    paths = {
        "case": output_dir / "ue_group_case_metrics.csv",
        "profile": output_dir / "ue_group_profile_summary.csv",
        "diagnostics": output_dir / "ue_group_final_diagnostics.csv",
        "decision": output_dir / "ue_group_decision.json",
        "html": output_dir / "ue_group_ppo.html",
    }
    _write_csv(paths["case"], case_rows)
    _write_csv(paths["profile"], ranking)
    _write_csv(paths["diagnostics"], diagnostic_rows)
    paths["decision"].write_text(json.dumps(decision, indent=2), encoding="utf-8")

    body = "".join(
        f"<tr><td>{row['rank']}</td><td>{row['profile']}</td>"
        f"<td>{row['latest_service_feasible_seeds']}/3</td><td>{row['ever_service_feasible_seeds']}/3</td>"
        f"<td>{row['mean_latest_starvation']*100:.2f}%</td><td>{row['mean_latest_jfi']:.3f}</td>"
        f"<td>{row['mean_det_unique_selected']:.0f}/1200</td><td>{row['mean_det_overlap']*100:.1f}%</td>"
        f"<td>{row['mean_tail_kl']:.5f}</td><td>{row['mean_tail_critic_actor_ratio']:.1f}x</td>"
        f"<td>{row['mean_tail_disc_is_loss']:.6f}</td><td>{row['max_disc_alpha']:.2g}</td></tr>"
        for row in ranking
    )
    paths["html"].write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Round19A</title>"
        "<style>body{font-family:Segoe UI,Arial;max-width:1180px;margin:32px auto}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:8px}th{background:#f3f4f6}</style></head><body>"
        "<h1>Round 19A — UE-group Dimension-wise PPO</h1>"
        f"<p>Recommended by controlled ranking: <b>{winner['profile']}</b></p>"
        "<p>Diagnostics use each case's effective environment/profile seed and the exact final environment-step milestone checkpoint.</p>"
        "<table><thead><tr><th>Rank</th><th>Profile</th><th>Latest feasible</th><th>Ever feasible</th>"
        "<th>Latest starvation</th><th>Latest JFI</th><th>Det coverage</th><th>Det overlap</th>"
        "<th>Tail KL</th><th>Critic/actor grad</th><th>UE J_IS</th><th>Max alpha</th></tr></thead><tbody>"
        + body
        + "</tbody></table><p>"
        + decision["method_note"]
        + "</p></body></html>",
        encoding="utf-8",
    )
    return paths
