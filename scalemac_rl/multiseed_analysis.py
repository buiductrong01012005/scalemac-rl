from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from scalemac_rl.reward_study import POSITIVE_COMPONENTS, RewardStudyPlan, safe_float


FINAL_METRICS: tuple[tuple[str, str, str], ...] = (
    ("mean_goodput_bits_per_slot", "Goodput", "bit/slot"),
    ("final_jain_fairness", "Jain fairness", ""),
    ("max_starvation_rate", "Max starvation", "rate"),
    ("max_p99_wait_slots", "Worst P99 wait", "slot"),
    ("max_wait_slots", "Max wait", "slot"),
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return 0.0, 0.0
    return mean(clean), stdev(clean) if len(clean) > 1 else 0.0


def _profile_for_case(plan: RewardStudyPlan, case_id: str) -> str:
    mapping = plan.analysis.get("case_profile", {})
    return str(mapping.get(case_id, case_id))


def _seed_for_case(plan: RewardStudyPlan, case_id: str) -> int:
    case = next(item for item in plan.cases if item.case_id == case_id)
    value = case.common_overrides.get("seed", plan.common.get("seed", 1701))
    return int(value)


def _formula_for_case(plan: RewardStudyPlan, case_id: str) -> str:
    case = next(item for item in plan.cases if item.case_id == case_id)
    names = {
        "throughput": "T",
        "fairness": "J",
        "service": "S",
        "deficit_service": "D",
        "pf_utility": "PF",
        "low_throughput": "L",
        "urgency_service": "U",
    }
    terms = [
        f"{weight:.4g}{names[name]}"
        for name, weight in case.positive_weights.items()
        if weight > 0
    ]
    return " + ".join(terms)


def _status(trajectory: list[dict[str, str]]) -> tuple[str, str]:
    if not trajectory:
        return "Incomplete", "Không tìm thấy validation.csv hoặc file không có dữ liệu."
    final = trajectory[-1]
    final_starvation = safe_float(final, "max_starvation_rate")
    final_p99 = safe_float(final, "max_p99_wait_slots")
    final_wait = safe_float(final, "max_wait_slots")

    def safe(row: Mapping[str, Any]) -> bool:
        return (
            safe_float(row, "max_starvation_rate") <= 1e-12
            and safe_float(row, "max_p99_wait_slots") < 64
            and safe_float(row, "max_wait_slots") < 80
        )

    had_safe = any(safe(row) for row in trajectory[:-1])
    full_collapse = (
        final_starvation > 0.10 or final_p99 >= 5000 or final_wait >= 5000
    )
    if full_collapse and had_safe:
        return "Late collapse", "Đã từng đạt vùng an toàn nhưng checkpoint cuối collapse."
    if full_collapse:
        return "Full collapse", "Checkpoint cuối mất coverage hoặc chạm trần 5000 slot."
    if final_starvation > 0 or final_p99 >= 64 or final_wait >= 80:
        return "Borderline", "Không collapse hoàn toàn nhưng còn starvation hoặc tail wait cao."
    return "Stable", "Checkpoint cuối không starvation và tail wait dưới ngưỡng cảnh báo."


def _format_metric(key: str, value: float) -> str:
    if key == "mean_goodput_bits_per_slot":
        return f"{value:,.0f}"
    if key == "final_jain_fairness":
        return f"{value:.4f}"
    if key == "max_starvation_rate":
        return f"{100 * value:.2f}%"
    return f"{value:.1f}"


def _fmt_mean_std(key: str, avg: float, sd: float) -> str:
    if key == "mean_goodput_bits_per_slot":
        return f"{avg:,.0f} ± {sd:,.0f}"
    if key == "final_jain_fairness":
        return f"{avg:.4f} ± {sd:.4f}"
    if key == "max_starvation_rate":
        return f"{100*avg:.2f}% ± {100*sd:.2f} điểm %"
    return f"{avg:.1f} ± {sd:.1f}"


def _paired_rows(
    seed_rows: list[dict[str, Any]],
    profile_order: list[str],
    baseline_profile: str,
) -> list[dict[str, Any]]:
    indexed = {(str(row["profile"]), int(row["seed"])): row for row in seed_rows}
    baseline_seeds = sorted(
        seed for profile, seed in indexed if profile == baseline_profile
    )
    output: list[dict[str, Any]] = []
    for profile in profile_order:
        if profile == baseline_profile:
            continue
        for seed in baseline_seeds:
            current = indexed.get((profile, seed))
            baseline = indexed.get((baseline_profile, seed))
            if not current or not baseline:
                continue
            row: dict[str, Any] = {"profile": profile, "seed": seed}
            for key, _, _ in FINAL_METRICS:
                row[f"delta_{key}"] = float(current[key]) - float(baseline[key])
            output.append(row)
    return output


def build_multiseed_confirmation_analysis(
    *,
    plan: RewardStudyPlan,
    round_dir: Path,
    output_path: Path,
) -> Path:
    analysis = plan.analysis
    profile_order = [str(value) for value in analysis.get("profile_order", [])]
    if not profile_order:
        profile_order = list(dict.fromkeys(_profile_for_case(plan, c.case_id) for c in plan.cases))
    baseline_profile = str(analysis.get("baseline_profile", profile_order[0]))
    profile_labels = {str(k): str(v) for k, v in analysis.get("profile_labels", {}).items()}
    profile_roles = {str(k): str(v) for k, v in analysis.get("profile_roles", {}).items()}

    seed_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    for case in plan.cases:
        profile = _profile_for_case(plan, case.case_id)
        seed = _seed_for_case(plan, case.case_id)
        validation_rows = _read_rows(round_dir / case.case_id / "validation.csv")
        status, status_reason = _status(validation_rows)
        if validation_rows:
            final = validation_rows[-1]
            row: dict[str, Any] = {
                "case_id": case.case_id,
                "profile": profile,
                "profile_label": profile_labels.get(profile, profile),
                "seed": seed,
                "formula": _formula_for_case(plan, case.case_id),
                "status": status,
                "global_env_steps": int(safe_float(final, "global_env_steps")),
                "device": final.get("device", ""),
            }
            for key, _, _ in FINAL_METRICS:
                row[key] = safe_float(final, key)
            row["mean_service_score"] = safe_float(final, "mean_service_score")
            row["mean_urgency_service_score"] = safe_float(final, "mean_urgency_service_score")
            row["mean_deficit_service_score"] = safe_float(final, "mean_deficit_service_score")
            row["mean_final_target_reward"] = safe_float(final, "mean_final_target_reward")
            seed_rows.append(row)
        stability_rows.append(
            {
                "case_id": case.case_id,
                "profile": profile,
                "seed": seed,
                "status": status,
                "reason": status_reason,
            }
        )
        for item in validation_rows:
            trajectory_rows.append(
                {
                    "case_id": case.case_id,
                    "profile": profile,
                    "seed": seed,
                    **item,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["profile"])].append(row)
    for profile in profile_order:
        rows = grouped.get(profile, [])
        summary: dict[str, Any] = {
            "profile": profile,
            "profile_label": profile_labels.get(profile, profile),
            "role": profile_roles.get(profile, ""),
            "formula": rows[0]["formula"] if rows else "",
            "completed_seeds": len(rows),
            "stable_seeds": sum(row["status"] == "Stable" for row in rows),
            "borderline_seeds": sum(row["status"] == "Borderline" for row in rows),
            "late_collapse_seeds": sum(row["status"] == "Late collapse" for row in rows),
            "full_collapse_seeds": sum(row["status"] == "Full collapse" for row in rows),
        }
        for key, _, _ in FINAL_METRICS:
            avg, sd = _mean_std(float(row[key]) for row in rows)
            summary[f"mean_{key}"] = avg
            summary[f"std_{key}"] = sd
            summary[f"min_{key}"] = min((float(row[key]) for row in rows), default=0.0)
            summary[f"max_{key}"] = max((float(row[key]) for row in rows), default=0.0)
        summary_rows.append(summary)

    paired = _paired_rows(seed_rows, profile_order, baseline_profile)

    seed_output = Path(str(analysis.get("seed_metrics_output", output_path.with_name("round_08_seed_metrics.csv"))))
    summary_output = Path(str(analysis.get("profile_summary_output", output_path.with_name("round_08_profile_summary.csv"))))
    trajectory_output = Path(str(analysis.get("trajectory_output", output_path.with_name("round_08_validation_trajectory.csv"))))
    stability_output = Path(str(analysis.get("stability_output", output_path.with_name("round_08_stability_matrix.csv"))))
    comparison_output = Path(str(analysis.get("comparison_output", output_path.with_name("round_08_paired_comparison.csv"))))
    markdown_output = Path(str(analysis.get("markdown_output", output_path.with_suffix(".md"))))

    _write_rows(seed_output, seed_rows)
    _write_rows(summary_output, summary_rows)
    _write_rows(trajectory_output, trajectory_rows)
    _write_rows(stability_output, stability_rows)
    _write_rows(comparison_output, paired)

    summary_by_profile = {str(row["profile"]): row for row in summary_rows}
    stable_all = [row for row in summary_rows if int(row["stable_seeds"]) == int(row["completed_seeds"]) and int(row["completed_seeds"]) > 0]

    def best_profile(metric: str, maximize: bool) -> str | None:
        candidates = stable_all or summary_rows
        candidates = [row for row in candidates if int(row["completed_seeds"]) > 0]
        if not candidates:
            return None
        key = f"mean_{metric}"
        chosen = max(candidates, key=lambda row: float(row[key])) if maximize else min(candidates, key=lambda row: float(row[key]))
        return str(chosen["profile"])

    winners = {
        "mean_goodput_bits_per_slot": best_profile("mean_goodput_bits_per_slot", True),
        "final_jain_fairness": best_profile("final_jain_fairness", True),
        "max_p99_wait_slots": best_profile("max_p99_wait_slots", False),
        "max_wait_slots": best_profile("max_wait_slots", False),
    }

    baseline = summary_by_profile.get(baseline_profile)
    candidate_profiles = [p for p in profile_order if p != baseline_profile]
    technical_conclusion = "Chưa đủ dữ liệu để kết luận."
    if all(int(summary_by_profile.get(p, {}).get("completed_seeds", 0)) >= 3 for p in profile_order):
        fully_stable = [p for p in candidate_profiles if int(summary_by_profile[p]["stable_seeds"]) == 3]
        if fully_stable:
            urgency_like = next((p for p in fully_stable if "urgency" in p), None)
            deficit_like = next((p for p in fully_stable if "deficit" in p), None)
            if urgency_like and baseline:
                u = summary_by_profile[urgency_like]
                if (
                    float(u["mean_final_jain_fairness"]) > float(baseline["mean_final_jain_fairness"])
                    and float(u["mean_mean_goodput_bits_per_slot"]) >= 0.98 * float(baseline["mean_mean_goodput_bits_per_slot"])
                ):
                    technical_conclusion = "Urgency vượt qua bước xác nhận nếu cải thiện fairness lặp lại mà giữ gần như toàn bộ goodput."
            if technical_conclusion.startswith("Chưa") and deficit_like:
                technical_conclusion = "Deficit ổn định đa seed nhưng cần cân nhắc mức đánh đổi goodput để đổi lấy tail delay."
            if technical_conclusion.startswith("Chưa"):
                technical_conclusion = "Có ứng viên ổn định đa seed, nhưng chưa có profile nào thắng rõ trên trade-off tổng thể."
        else:
            technical_conclusion = "Không ứng viên mới nào ổn định ở cả ba seed; nên quay lại T–J–S làm active reward."

    title = str(analysis.get("title", "Round 08 — Multi-seed confirmation"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for profile in profile_order:
        row = summary_by_profile.get(profile)
        if not row:
            continue
        cells = [
            f"<td><strong>{html.escape(profile_labels.get(profile, profile))}</strong><br><code>{html.escape(str(row['formula']))}</code></td>",
            f"<td>{int(row['stable_seeds'])}/{int(row['completed_seeds'])}</td>",
        ]
        for key, _, _ in FINAL_METRICS:
            value = _fmt_mean_std(key, float(row[f"mean_{key}"]), float(row[f"std_{key}"]))
            highlight = winners.get(key) == profile and key != "max_starvation_rate"
            cells.append(f"<td{' class=top' if highlight else ''}>{'<strong>' if highlight else ''}{html.escape(value)}{'</strong>' if highlight else ''}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    seed_table_rows = []
    for row in sorted(seed_rows, key=lambda r: (profile_order.index(str(r["profile"])), int(r["seed"]))):
        seed_table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['profile_label']))}</td>"
            f"<td>{int(row['seed'])}</td>"
            f"<td><span class='status {str(row['status']).lower().replace(' ', '-')}'>{html.escape(str(row['status']))}</span></td>"
            f"<td>{_format_metric('mean_goodput_bits_per_slot', float(row['mean_goodput_bits_per_slot']))}</td>"
            f"<td>{_format_metric('final_jain_fairness', float(row['final_jain_fairness']))}</td>"
            f"<td>{_format_metric('max_starvation_rate', float(row['max_starvation_rate']))}</td>"
            f"<td>{_format_metric('max_p99_wait_slots', float(row['max_p99_wait_slots']))}</td>"
            f"<td>{_format_metric('max_wait_slots', float(row['max_wait_slots']))}</td>"
            "</tr>"
        )

    profile_cards = []
    for profile in profile_order:
        row = summary_by_profile.get(profile)
        if not row:
            continue
        profile_cards.append(
            "<article class='card'>"
            f"<h3>{html.escape(profile_labels.get(profile, profile))}</h3>"
            f"<p><code>{html.escape(str(row['formula']))}</code></p>"
            f"<p>{html.escape(profile_roles.get(profile, ''))}</p>"
            f"<p><strong>Ổn định:</strong> {int(row['stable_seeds'])}/{int(row['completed_seeds'])} seed.</p>"
            "</article>"
        )

    html_text = f"""<!doctype html>
<html lang='vi'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f5f7fb; --card:#fff; --ink:#172033; --muted:#536178; --line:#d9e0eb; --accent:#174ea6; --top:#fff2a8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#10141c; --card:#171d28; --ink:#eef3ff; --muted:#aab5c8; --line:#303a4d; --accent:#8ab4f8; --top:#665800; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.58 system-ui,sans-serif; }}
main {{ max-width:1180px; margin:auto; padding:32px 20px 72px; }} h1,h2,h3 {{ line-height:1.2; }} h1 {{ font-size:2.1rem; }}
.summary {{ border-left:6px solid var(--accent); background:var(--card); padding:18px 22px; border-radius:10px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }} .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); font-size:.92rem; }} th,td {{ border:1px solid var(--line); padding:9px; text-align:left; vertical-align:top; }} th {{ position:sticky; top:0; background:var(--card); }}
.table-wrap {{ overflow:auto; border-radius:12px; }} .top {{ background:var(--top); }} code {{ white-space:nowrap; }} .muted {{ color:var(--muted); }}
.status {{ font-weight:700; }} .stable {{ color:#16833a; }} .borderline {{ color:#b36b00; }} .late-collapse,.full-collapse {{ color:#c43a3a; }}
</style>
</head>
<body><main>
<h1>{html.escape(title)}</h1>
<section class='summary'>
<h2>Kết luận kỹ thuật hiện tại</h2>
<p><strong>{html.escape(technical_conclusion)}</strong></p>
<p>Round này không tìm reward mới. Nó kiểm tra xem hai ứng viên Round 07 có lặp lại lợi ích trên ba seed chung hay không. Mỗi profile dùng cùng seed, cùng 100.096 bước và cùng PPO config để so sánh theo cặp.</p>
</section>
<h2>Ba profile được xác nhận</h2>
<div class='grid'>{''.join(profile_cards)}</div>
<h2>Kết quả tổng hợp mean ± std</h2>
<p>Ô vàng và chữ đậm đánh dấu profile tốt nhất trong nhóm không mất ổn định trên các seed đã hoàn thành. Goodput/Jain càng cao càng tốt; P99/max wait càng thấp càng tốt.</p>
<div class='table-wrap'><table><thead><tr><th>Profile và reward</th><th>Stable</th><th>Goodput</th><th>Jain</th><th>Starvation</th><th>P99</th><th>Max wait</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<h2>Kết quả từng seed</h2>
<div class='table-wrap'><table><thead><tr><th>Profile</th><th>Seed</th><th>Stability</th><th>Goodput</th><th>Jain</th><th>Starvation</th><th>P99</th><th>Max wait</th></tr></thead><tbody>{''.join(seed_table_rows)}</tbody></table></div>
<h2>Cách ra quyết định sau Round 08</h2>
<ul>
<li><strong>Giữ Urgency</strong> nếu nó stable ở cả ba seed, fairness tăng lặp lại và goodput không giảm đáng kể so với T–J–S.</li>
<li><strong>Giữ Deficit theo profile delay-sensitive</strong> nếu tail delay giảm lặp lại và mức mất goodput được chấp nhận.</li>
<li><strong>Quay lại T–J–S</strong> nếu hai ứng viên collapse hoặc lợi ích không lặp lại.</li>
<li>Chưa tune PPO, Beta, architecture hoặc thêm reward thứ năm trong round này.</li>
</ul>
<p class='muted'>Bằng chứng là exploratory multi-seed với ba seed. Mean ± std mô tả độ lặp lại trong đúng system model hiện tại, không phải khẳng định phổ quát.</p>
</main></body></html>"""
    output_path.write_text(html_text, encoding="utf-8")

    md_lines = [
        f"# {title}",
        "",
        "## Kết luận kỹ thuật hiện tại",
        "",
        technical_conclusion,
        "",
        "## Mean ± std theo profile",
        "",
        "| Profile | Stable | Goodput | Jain | Starvation | P99 | Max wait |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in profile_order:
        row = summary_by_profile.get(profile)
        if not row:
            continue
        md_lines.append(
            "| " + " | ".join(
                [
                    profile_labels.get(profile, profile),
                    f"{int(row['stable_seeds'])}/{int(row['completed_seeds'])}",
                    _fmt_mean_std("mean_goodput_bits_per_slot", float(row["mean_mean_goodput_bits_per_slot"]), float(row["std_mean_goodput_bits_per_slot"])),
                    _fmt_mean_std("final_jain_fairness", float(row["mean_final_jain_fairness"]), float(row["std_final_jain_fairness"])),
                    _fmt_mean_std("max_starvation_rate", float(row["mean_max_starvation_rate"]), float(row["std_max_starvation_rate"])),
                    _fmt_mean_std("max_p99_wait_slots", float(row["mean_max_p99_wait_slots"]), float(row["std_max_p99_wait_slots"])),
                    _fmt_mean_std("max_wait_slots", float(row["mean_max_wait_slots"]), float(row["std_max_wait_slots"])),
                ]
            ) + " |"
        )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    metadata = {
        "round_id": plan.round_id,
        "completed_cases": len(seed_rows),
        "expected_cases": len(plan.cases),
        "profiles": profile_order,
        "baseline_profile": baseline_profile,
        "outputs": {
            "html": str(output_path),
            "markdown": str(markdown_output),
            "seed_metrics": str(seed_output),
            "profile_summary": str(summary_output),
            "trajectory": str(trajectory_output),
            "stability": str(stability_output),
            "paired_comparison": str(comparison_output),
        },
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output_path
