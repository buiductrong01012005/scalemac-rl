from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from scalemac_rl.reward_study import (
    POSITIVE_COMPONENTS,
    DELTA_COMPONENTS,
    PENALTY_COMPONENTS,
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


def _best_by_metric(
    rows: list[dict[str, str]], key: str, *, maximize: bool = True
) -> float:
    values = [safe_float(row, key, float("nan")) for row in rows]
    values = [value for value in values if value == value]
    if not values:
        return float("nan")
    return max(values) if maximize else min(values)


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
        "positive_scale": case.get("positive_scale", 1.0),
        "environment_steps": config.get("effective_environment_steps", ""),
        "validation_slots": config.get("effective_validation_slots", ""),
        "constraint_training": common.get("constraint_training", False),
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

    row["best_goodput_bits_per_slot"] = _best_by_metric(
        validation_rows, "mean_goodput_bits_per_slot"
    )
    row["best_jain_fairness"] = _best_by_metric(
        validation_rows, "final_jain_fairness"
    )
    row["lowest_max_starvation_rate"] = _best_by_metric(
        validation_rows, "max_starvation_rate", maximize=False
    )
    row["lowest_max_p99_wait_slots"] = _best_by_metric(
        validation_rows, "max_p99_wait_slots", maximize=False
    )
    row["lowest_max_wait_slots"] = _best_by_metric(
        validation_rows, "max_wait_slots", maximize=False
    )
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
    rows: list[dict[str, Any]] = []
    for validation in validation_rows:
        row: dict[str, Any] = {
            "round_id": round_id,
            "case_id": case.get("id", case_dir.name),
            "label": case.get("label", case_dir.name),
            "run_dir": str(case_dir),
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
        f"<td>{'Có' if row.get('pareto_front') else 'Không'}</td>"
        "</tr>"
        for row in ordered
    )
    coefficient_headers = "".join(
        f"<th>{html.escape(name)}</th>"
        for name in (
            "throughput",
            "fairness",
            "service",
            "deficit",
            "PF",
            "low throughput",
            "urgency",
            "starvation penalty",
            "delay penalty total",
        )
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
<title>{html.escape(round_id)} – ScaleMAC-RL reward study</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#172033;margin:0;line-height:1.5}}
main{{max-width:1200px;margin:auto;padding:26px}}header,.card{{background:white;border:1px solid #dfe5ee;border-radius:15px;padding:22px;margin-bottom:16px}}
header{{background:linear-gradient(135deg,#18264a,#315efb);color:white}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #e4e8ef;text-align:left}}th{{background:#f0f3f9}}small{{color:#657083}}
.note{{background:#fff5d6;border-radius:10px;padding:13px}}code{{background:#eef1f7;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<header><h1>{html.escape(round_id)}</h1><p>{html.escape(description)}</p></header>
<section class="card"><h2>KPI cuối mỗi case</h2><table><thead><tr><th>Case</th><th>Goodput</th><th>Jain</th><th>Starvation</th><th>Max P99</th><th>Max wait</th><th>Pareto</th></tr></thead><tbody>{body_rows}</tbody></table></section>
<section class="card"><h2>Hệ số reward thực tế</h2><table><thead><tr><th>Case</th>{coefficient_headers}</tr></thead><tbody>{coefficient_rows}</tbody></table></section>
<section class="card note"><strong>Quy tắc đọc:</strong> không chọn case chỉ vì tổng reward cao. So sánh KPI thật, độ chuyển biến theo bước học, reward decomposition và Pareto frontier.</section>
</main></body></html>"""


def _index_html(round_links: list[tuple[str, str]], dataset_rows: int, pareto_rows: int) -> str:
    links = "".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(round_id)}</a></li>'
        for round_id, filename in round_links
    ) or "<li>Chưa có round hoàn thành.</li>"
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScaleMAC-RL Reward Study</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#172033;margin:0;line-height:1.55}}main{{max-width:920px;margin:auto;padding:28px}}section,header{{background:white;border:1px solid #dfe5ee;border-radius:16px;padding:22px;margin-bottom:16px}}header{{background:linear-gradient(135deg,#18264a,#315efb);color:white}}a{{color:#315efb}}code{{background:#eef1f7;padding:2px 5px;border-radius:4px}}</style></head><body><main>
<header><h1>ScaleMAC-RL Reward Study</h1><p>Khám phá reward có kiểm soát → tune trọng số → Pareto → dataset truy vấn.</p></header>
<section><h2>Trạng thái dataset</h2><p><strong>{dataset_rows}</strong> cấu hình hoàn thành; <strong>{pareto_rows}</strong> cấu hình trên Pareto frontier.</p></section>
<section><h2>Các vòng đã tạo</h2><ul>{links}</ul></section>
<section><h2>Tệp dữ liệu</h2><p><code>artifacts/runs/reward_study/reward_weight_dataset.csv</code><br><code>artifacts/runs/reward_study/reward_weight_trajectory.csv</code><br><code>artifacts/runs/reward_study/pareto_front.csv</code></p></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the reward-weight dataset, Pareto front, and HTML reports"
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("artifacts/runs/reward_study"),
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path("docs/research/reward_study/generated"),
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

    front = pareto_front_indices(
        final_rows,
        maximize=("mean_goodput_bits_per_slot", "final_jain_fairness"),
        minimize=("max_starvation_rate", "max_p99_wait_slots", "max_wait_slots"),
    )
    for index, row in enumerate(final_rows):
        row["pareto_front"] = index in front
    pareto_rows = [row for index, row in enumerate(final_rows) if index in front]

    _write_csv(args.study_root / "reward_weight_dataset.csv", final_rows)
    _write_csv(args.study_root / "reward_weight_trajectory.csv", trajectory_rows)
    _write_csv(args.study_root / "pareto_front.csv", pareto_rows)

    args.docs_root.mkdir(parents=True, exist_ok=True)
    round_links: list[tuple[str, str]] = []
    for round_id in sorted({str(row.get("round_id", "")) for row in final_rows}):
        round_rows = [row for row in final_rows if row.get("round_id") == round_id]
        report = _round_html(round_id, round_rows, descriptions.get(round_id, ""))
        artifact_report = args.study_root / round_id / "report.html"
        artifact_report.write_text(report, encoding="utf-8")
        docs_report = args.docs_root / f"{round_id}.html"
        docs_report.write_text(report, encoding="utf-8")
        round_links.append((round_id, docs_report.name))

    index = _index_html(round_links, len(final_rows), len(pareto_rows))
    (args.study_root / "index.html").write_text(index, encoding="utf-8")
    (args.docs_root / "index.html").write_text(index, encoding="utf-8")
    print(f"saved: {args.study_root / 'reward_weight_dataset.csv'}")
    print(f"saved: {args.study_root / 'pareto_front.csv'}")
    print(f"saved: {args.study_root / 'index.html'}")


if __name__ == "__main__":
    main()
