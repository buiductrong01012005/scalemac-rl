from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

from scalemac_rl.reward_study import (
    DELTA_COMPONENTS,
    PENALTY_COMPONENTS,
    POSITIVE_COMPONENTS,
    pareto_front_indices,
    read_csv_rows,
    safe_float,
)


KPI_COLUMNS = (
    "mean_goodput_bits_per_slot",
    "mean_throughput_score",
    "final_jain_fairness",
    "mean_fairness_score",
    "mean_service_score",
    "mean_deficit_service_score",
    "mean_pf_utility_score",
    "mean_low_throughput_score",
    "mean_urgency_service_score",
    "mean_starvation_rate",
    "max_starvation_rate",
    "final_p99_wait_slots",
    "max_p99_wait_slots",
    "final_max_wait_slots",
    "max_wait_slots",
    "mean_final_target_reward",
    "mean_core_reward",
    "constraint_feasible",
    "global_env_steps",
    "seed",
)

CONTRIBUTION_COLUMNS = (
    "mean_reward_throughput_component",
    "mean_reward_fairness_component",
    "mean_reward_service_component",
    "mean_reward_deficit_service_component",
    "mean_reward_pf_utility_component",
    "mean_reward_low_throughput_component",
    "mean_reward_urgency_service_component",
    "mean_reward_fairness_progress_component",
    "mean_reward_pf_utility_progress_component",
    "mean_reward_starvation_penalty",
    "mean_reward_deadline_risk_penalty",
    "mean_reward_max_wait_risk_penalty",
    "mean_reward_population_wait_penalty",
)

COMPONENT_LABELS = {
    "throughput": "Throughput",
    "fairness": "Jain fairness",
    "service": "Service",
    "deficit_service": "Deficit service",
    "pf_utility": "PF utility",
    "low_throughput": "Low-throughput",
    "urgency_service": "Urgency service",
    "starvation": "Starvation penalty",
    "deadline_risk": "P99/deadline-risk penalty",
    "max_wait_risk": "Maximum-wait penalty",
    "population_wait": "Population-wait penalty",
}

COMPONENT_MEANINGS = {
    "throughput": "Thưởng khi cell truyền thành công nhiều bit trong một slot so với mức tốt nhất ước tính.",
    "fairness": "Thưởng khi throughput được phân phối đều hơn giữa 1.200 UE, kết hợp Jain fairness tích lũy và ngắn hạn.",
    "service": "Thưởng một điểm tổng hợp khi starvation, thời gian chờ trung bình và số UE gần deadline thấp.",
    "deficit_service": "Thưởng khi các UE có throughput thấp hơn mức trung bình được truyền thành công.",
    "pf_utility": "Thưởng utility log-throughput để lợi ích của việc hỗ trợ UE yếu lớn hơn việc cấp thêm cho UE vốn đã mạnh.",
    "low_throughput": "Thưởng khi throughput của nhóm UE ở đáy phân phối được nâng lên.",
    "urgency_service": "Thưởng khi UE vừa thiếu throughput vừa chờ lâu được truyền thành công.",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_by_steps(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: safe_float(row, "global_env_steps", -1.0))


def _flatten_run(
    *,
    round_id: str,
    case_dir: Path,
    config: Mapping[str, Any],
    validation_rows: list[dict[str, str]],
) -> dict[str, Any] | None:
    final = _last_by_steps(validation_rows)
    if final is None:
        return None
    case = dict(config.get("case", {}))
    actual = dict(case.get("actual_coefficients", {}))
    common = dict(config.get("common", {}))
    architecture = dict(config.get("architecture", {}))
    row: dict[str, Any] = {
        "study_id": config.get("study_id", ""),
        "round_id": round_id,
        "case_id": case.get("id", case_dir.name),
        "label": case.get("label", case_dir.name),
        "hypothesis": case.get("hypothesis", ""),
        "run_dir": str(case_dir),
        "checkpoint_path": str(case_dir / "latest.pt"),
        "evaluation_policy_mode": "deterministic_mean_action",
        "positive_scale": case.get("positive_scale", 1.0),
        "environment_steps": config.get("effective_environment_steps", ""),
        "validation_slots": config.get("effective_validation_slots", ""),
        "constraint_training": common.get("constraint_training", False),
        "starvation_threshold_slots": common.get("starvation_threshold_slots", 64),
        "gamma": common.get("gamma", ""),
        "gae_lambda": common.get("gae_lambda", ""),
        "clip_coef": common.get("clip_coef", ""),
        "learning_rate_start": common.get("learning_rate_start", ""),
        "learning_rate_end": common.get("learning_rate_end", ""),
        "entropy_coef_start": common.get("entropy_coef_start", ""),
        "entropy_coef_end": common.get("entropy_coef_end", ""),
        "hidden_dim": architecture.get("embedding_dim", ""),
        "num_ues": architecture.get("num_ues", ""),
        "top_k": architecture.get("top_k", ""),
    }
    for component in POSITIVE_COMPONENTS:
        row[f"relative_weight_{component}"] = case.get("positive_weights", {}).get(
            component, 0.0
        )
        row[f"coef_{component}"] = actual.get(f"coef_{component}", 0.0)
    for component in DELTA_COMPONENTS:
        row[f"coef_{component}_delta"] = actual.get(
            f"coef_{component}_delta", 0.0
        )
    for component in PENALTY_COMPONENTS:
        row[f"coef_{component}_penalty"] = actual.get(
            f"coef_{component}_penalty", 0.0
        )
    for key in KPI_COLUMNS + CONTRIBUTION_COLUMNS:
        row[key] = final.get(key, "")
    return row


def _trajectory_rows(
    *,
    round_id: str,
    case_dir: Path,
    config: Mapping[str, Any],
    validation_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    case = dict(config.get("case", {}))
    actual = dict(case.get("actual_coefficients", {}))
    common = dict(config.get("common", {}))
    rows: list[dict[str, Any]] = []
    for validation in validation_rows:
        row: dict[str, Any] = {
            "round_id": round_id,
            "case_id": case.get("id", case_dir.name),
            "label": case.get("label", case_dir.name),
            "hypothesis": case.get("hypothesis", ""),
            "run_dir": str(case_dir),
            "checkpoint_path": str(case_dir / "latest.pt"),
            "starvation_threshold_slots": common.get("starvation_threshold_slots", 64),
        }
        row.update(actual)
        for key in KPI_COLUMNS + CONTRIBUTION_COLUMNS:
            row[key] = validation.get(key, "")
        rows.append(row)
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _active_formula(row: Mapping[str, Any]) -> str:
    terms: list[str] = []
    for component in POSITIVE_COMPONENTS:
        coefficient = safe_float(row, f"coef_{component}")
        if coefficient > 0:
            terms.append(f"{coefficient:.3f} × {COMPONENT_LABELS[component]}")
    for component in PENALTY_COMPONENTS:
        coefficient = safe_float(row, f"coef_{component}_penalty")
        if coefficient > 0:
            terms.append(f"− {coefficient:.3f} × {COMPONENT_LABELS[component]}")
    return " + ".join(terms).replace("+ −", "−") or "Không có thành phần reward hoạt động"


def _meaning_for_row(row: Mapping[str, Any]) -> str:
    active = [
        component
        for component in POSITIVE_COMPONENTS
        if safe_float(row, f"coef_{component}") > 0
    ]
    return " ".join(COMPONENT_MEANINGS[component] for component in active)


def _kpi_glossary() -> str:
    return """
<section class="card"><h2>Cách hiểu các KPI</h2>
<table><thead><tr><th>KPI</th><th>Ý nghĩa dễ hiểu</th><th>Cách đọc</th></tr></thead><tbody>
<tr><td><strong>Goodput</strong></td><td>Số bit dữ liệu truyền thành công trong mỗi slot. Một slot là một bước ra quyết định lập lịch của mô phỏng.</td><td>Càng cao càng tốt, nhưng không nên đánh đổi bằng starvation hoặc fairness quá thấp.</td></tr>
<tr><td><strong>Jain fairness</strong></td><td>Mức độ throughput được phân phối đều giữa 1.200 UE. Giá trị gần 1 nghĩa là tương đối cân bằng; giá trị thấp nghĩa là tài nguyên tập trung vào một nhóm nhỏ.</td><td>Càng cao càng công bằng.</td></tr>
<tr><td><strong>Worst starvation</strong></td><td>Tỷ lệ UE không có lần truyền thành công trong ít nhất 64 slot liên tiếp, lấy tỷ lệ cao nhất xuất hiện trong episode đánh giá.</td><td>Mục tiêu là 0%.</td></tr>
<tr><td><strong>Worst P99 wait</strong></td><td>Tại mỗi slot, ta đo số slot kể từ lần truyền thành công gần nhất của từng UE. P99 là mức mà 99% UE không chờ lâu hơn; 1% UE tệ nhất có thể chờ lâu hơn. “Worst P99” là P99 lớn nhất trong toàn bộ 5.000 slot đánh giá.</td><td>Ví dụ P99 = 46 nghĩa là tại thời điểm xấu nhất, 99% UE đã được truyền thành công trong vòng 46 slot gần nhất. Mục tiêu hiện tại ≤ 50.</td></tr>
<tr><td><strong>Worst single-UE wait</strong></td><td>Khoảng chờ dài nhất của một UE bất kỳ tại bất kỳ thời điểm nào trong episode.</td><td>Càng thấp càng tốt; mục tiêu hiện tại ≤ 60.</td></tr>
</tbody></table>
<p class="small">Wait ở đây là khoảng cách giữa hai lần truyền thành công của UE, chưa phải độ trễ của từng packet vì môi trường hiện vẫn dùng full-buffer.</p></section>
"""


def _round_html(round_id: str, rows: list[dict[str, Any]], description: str) -> str:
    ordered = sorted(rows, key=lambda row: str(row.get("case_id", "")))
    body_rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(str(row.get('case_id', '')))}</strong><br>"
        f"<small>{html.escape(str(row.get('label', '')))}</small></td>"
        f"<td>{_fmt(row.get('mean_goodput_bits_per_slot'), 1)}</td>"
        f"<td>{_fmt(row.get('final_jain_fairness'), 4)}</td>"
        f"<td>{_fmt(100.0 * safe_float(row, 'max_starvation_rate'), 3)}%</td>"
        f"<td>{_fmt(row.get('max_p99_wait_slots'), 1)}</td>"
        f"<td>{_fmt(row.get('max_wait_slots'), 1)}</td>"
        f"<td>{'Có' if row.get('pareto_safety_filtered') else 'Không'}</td>"
        "</tr>"
        for row in ordered
    )
    configuration_rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(str(row.get('case_id', '')))}</strong><br><small>{html.escape(str(row.get('hypothesis', '')))}</small></td>"
        f"<td><code>{html.escape(_active_formula(row))}</code></td>"
        f"<td>{html.escape(_meaning_for_row(row))}</td>"
        "</tr>"
        for row in ordered
    )
    coefficient_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('case_id', '')))}</td>"
        f"<td>{_fmt(row.get('coef_throughput'), 3)}</td>"
        f"<td>{_fmt(row.get('coef_fairness'), 3)}</td>"
        f"<td>{_fmt(row.get('coef_service'), 3)}</td>"
        f"<td>{_fmt(row.get('coef_deficit_service'), 3)}</td>"
        f"<td>{_fmt(row.get('coef_pf_utility'), 3)}</td>"
        f"<td>{_fmt(row.get('coef_low_throughput'), 3)}</td>"
        f"<td>{_fmt(row.get('coef_urgency_service'), 3)}</td>"
        f"<td>{_fmt(row.get('coef_starvation_penalty'), 3)}</td>"
        f"<td>{_fmt(safe_float(row, 'coef_deadline_risk_penalty') + safe_float(row, 'coef_max_wait_risk_penalty') + safe_float(row, 'coef_population_wait_penalty'), 3)}</td>"
        "</tr>"
        for row in ordered
    )
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(round_id)} – ScaleMAC-RL reward analysis</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#172033;margin:0;line-height:1.55}}
main{{max-width:1240px;margin:auto;padding:26px}}header,.card{{background:white;border:1px solid #dfe5ee;border-radius:15px;padding:22px;margin-bottom:16px}}
header{{background:linear-gradient(135deg,#18264a,#315efb);color:white}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #e4e8ef;text-align:left;vertical-align:top}}th{{background:#f0f3f9}}small,.small{{color:#657083}}
.note{{background:#fff5d6;border-radius:10px;padding:13px}}code{{background:#eef1f7;padding:2px 5px;border-radius:4px;white-space:normal}}
</style></head><body><main>
<header><h1>{html.escape(round_id)}</h1><p>{html.escape(description)}</p></header>
<section class="card"><h2>Reward của từng case được cấu hình thế nào?</h2><table><thead><tr><th>Case và giả thuyết</th><th>Công thức reward thực tế</th><th>PPO được khuyến khích làm gì?</th></tr></thead><tbody>{configuration_rows}</tbody></table></section>
<section class="card"><h2>KPI cuối mỗi case</h2><table><thead><tr><th>Case</th><th>Goodput (bit/slot)</th><th>Jain fairness</th><th>Worst starvation</th><th>Worst P99 wait (slot)</th><th>Worst single-UE wait (slot)</th><th>Pareto sau lọc safety</th></tr></thead><tbody>{body_rows}</tbody></table></section>
{_kpi_glossary()}
<section class="card"><h2>Bảng hệ số đầy đủ</h2><table><thead><tr><th>Case</th><th>Throughput</th><th>Jain</th><th>Service</th><th>Deficit</th><th>PF</th><th>Low-throughput</th><th>Urgency</th><th>Starvation penalty</th><th>Delay penalties tổng</th></tr></thead><tbody>{coefficient_rows}</tbody></table></section>
<section class="card note"><strong>Quy tắc đọc:</strong> không chọn case chỉ vì tổng reward cao. So sánh KPI thật, đường học theo environment steps, reward decomposition và các Pareto frontier đã lọc.</section>
</main></body></html>"""


def _index_html(
    round_links: list[tuple[str, str]],
    dataset_rows: int,
    pareto_all_rows: int,
    pareto_safety_rows: int,
    pareto_strict_rows: int,
) -> str:
    links = "".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(round_id)}</a></li>'
        for round_id, filename in round_links
    ) or "<li>Chưa có round hoàn thành.</li>"
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScaleMAC-RL Reward Analysis</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#172033;margin:0;line-height:1.55}}main{{max-width:920px;margin:auto;padding:28px}}section,header{{background:white;border:1px solid #dfe5ee;border-radius:16px;padding:22px;margin-bottom:16px}}header{{background:linear-gradient(135deg,#18264a,#315efb);color:white}}a{{color:#315efb}}code{{background:#eef1f7;padding:2px 5px;border-radius:4px}}</style></head><body><main>
<header><h1>ScaleMAC-RL Reward Analysis</h1><p>Khám phá reward có kiểm soát → tune trọng số → Pareto → dataset truy vấn.</p></header>
<section><h2>Trạng thái dataset</h2><p><strong>{dataset_rows}</strong> cấu hình hoàn thành.</p><ul><li>Pareto toàn bộ: <strong>{pareto_all_rows}</strong></li><li>Pareto sau lọc safety: <strong>{pareto_safety_rows}</strong></li><li>Pareto đạt constraint nghiêm ngặt: <strong>{pareto_strict_rows}</strong></li></ul></section>
<section><h2>Các vòng đã tạo</h2><ul>{links}</ul></section>
<section><h2>Tệp dữ liệu</h2><p><code>reward_weight_dataset.csv</code><br><code>reward_weight_trajectory.csv</code><br><code>pareto_all.csv</code><br><code>pareto_safety_filtered.csv</code><br><code>pareto_strict_constraints.csv</code></p></section>
</main></body></html>"""


def _subset_front(
    rows: list[dict[str, Any]], predicate
) -> tuple[set[int], list[dict[str, Any]]]:
    selected_indices = [index for index, row in enumerate(rows) if predicate(row)]
    selected_rows = [rows[index] for index in selected_indices]
    local_front = pareto_front_indices(
        selected_rows,
        maximize=("mean_goodput_bits_per_slot", "final_jain_fairness"),
        minimize=("max_starvation_rate", "max_p99_wait_slots", "max_wait_slots"),
    )
    global_front = {selected_indices[index] for index in local_front}
    return global_front, [rows[index] for index in sorted(global_front)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reward-weight datasets, Pareto fronts, and readable HTML analyses"
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("artifacts/runs/reward_study"),
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path("docs/analysis/reward_study/generated"),
    )
    args = parser.parse_args()

    final_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    descriptions: dict[str, str] = {}
    for round_dir in sorted(path for path in args.study_root.glob("round_*") if path.is_dir()):
        snapshot_path = round_dir / "round_plan_snapshot.json"
        if snapshot_path.is_file():
            snapshot = _load_json(snapshot_path)
            descriptions[round_dir.name] = str(snapshot.get("description", ""))
        for case_dir in sorted(path for path in round_dir.iterdir() if path.is_dir()):
            config_path = case_dir / "run_config.json"
            validation_path = case_dir / "validation.csv"
            if not config_path.is_file() or not validation_path.is_file():
                continue
            config = _load_json(config_path)
            validation_rows = read_csv_rows(validation_path)
            flattened = _flatten_run(
                round_id=round_dir.name,
                case_dir=case_dir,
                config=config,
                validation_rows=validation_rows,
            )
            if flattened is not None:
                final_rows.append(flattened)
            trajectory_rows.extend(
                _trajectory_rows(
                    round_id=round_dir.name,
                    case_dir=case_dir,
                    config=config,
                    validation_rows=validation_rows,
                )
            )

    all_front = pareto_front_indices(
        final_rows,
        maximize=("mean_goodput_bits_per_slot", "final_jain_fairness"),
        minimize=("max_starvation_rate", "max_p99_wait_slots", "max_wait_slots"),
    )
    safety_front, safety_rows = _subset_front(
        final_rows,
        lambda row: safe_float(row, "max_starvation_rate") <= 1e-12
        and safe_float(row, "max_p99_wait_slots") <= 60.0
        and safe_float(row, "max_wait_slots") <= 60.0,
    )
    strict_front, strict_rows = _subset_front(
        final_rows,
        lambda row: safe_float(row, "max_starvation_rate") <= 1e-12
        and safe_float(row, "max_p99_wait_slots") <= 50.0
        and safe_float(row, "max_wait_slots") <= 60.0
        and safe_float(row, "final_jain_fairness") >= 0.60,
    )

    for index, row in enumerate(final_rows):
        row["pareto_front"] = index in all_front  # backward-compatible name
        row["pareto_all"] = index in all_front
        row["pareto_safety_filtered"] = index in safety_front
        row["pareto_strict_constraints"] = index in strict_front

    all_rows = [final_rows[index] for index in sorted(all_front)]
    _write_csv(args.study_root / "reward_weight_dataset.csv", final_rows)
    _write_csv(args.study_root / "reward_weight_trajectory.csv", trajectory_rows)
    _write_csv(args.study_root / "pareto_front.csv", all_rows)
    _write_csv(args.study_root / "pareto_all.csv", all_rows)
    _write_csv(args.study_root / "pareto_safety_filtered.csv", safety_rows)
    _write_csv(args.study_root / "pareto_strict_constraints.csv", strict_rows)

    args.docs_root.mkdir(parents=True, exist_ok=True)
    round_links: list[tuple[str, str]] = []
    for round_id in sorted({str(row.get("round_id", "")) for row in final_rows}):
        round_rows = [row for row in final_rows if row.get("round_id") == round_id]
        report = _round_html(round_id, round_rows, descriptions.get(round_id, ""))
        artifact_analysis = args.study_root / round_id / "analysis.html"
        artifact_analysis.write_text(report, encoding="utf-8")
        docs_analysis = args.docs_root / f"{round_id}.html"
        docs_analysis.write_text(report, encoding="utf-8")
        round_links.append((round_id, docs_analysis.name))

    index = _index_html(
        round_links,
        len(final_rows),
        len(all_rows),
        len(safety_rows),
        len(strict_rows),
    )
    (args.study_root / "index.html").write_text(index, encoding="utf-8")
    (args.docs_root / "index.html").write_text(index, encoding="utf-8")
    print(f"saved: {args.study_root / 'reward_weight_dataset.csv'}")
    print(f"saved: {args.study_root / 'pareto_all.csv'}")
    print(f"saved: {args.study_root / 'pareto_safety_filtered.csv'}")
    print(f"saved: {args.study_root / 'pareto_strict_constraints.csv'}")
    print(f"saved: {args.study_root / 'index.html'}")


if __name__ == "__main__":
    main()
