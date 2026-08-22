from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any, Mapping

from scalemac_rl.reward_study import POSITIVE_COMPONENTS, RewardStudyPlan, safe_float


COMPONENT_GUIDE: dict[str, dict[str, str]] = {
    "throughput": {
        "name": "Throughput",
        "meaning": "Thưởng khi cell truyền thành công nhiều bit trong mỗi slot.",
        "expected": "Khuyến khích chọn UE có khả năng sử dụng PRB hiệu quả.",
    },
    "fairness": {
        "name": "Jain fairness",
        "meaning": "Thưởng khi delivered throughput được phân phối đồng đều hơn giữa 1.200 UE.",
        "expected": "Hạn chế chênh lệch throughput dài hạn giữa các UE.",
    },
    "schedule_fairness": {
        "name": "Schedule-frequency fairness",
        "meaning": "Thưởng khi số cơ hội được scheduler chọn được phân phối đều giữa các UE, tách biệt với CQI và số bit thực sự truyền thành công.",
        "expected": "Giảm việc một nhóm UE được chọn lặp lại quá thường xuyên trong khi UE khác hiếm khi được schedule.",
    },
    "service": {
        "name": "Service",
        "meaning": (
            "Thưởng cho trạng thái toàn cell có ít UE bị bỏ phục vụ lâu, thời gian chờ "
            "trung bình thấp và ít UE tiến gần ngưỡng chờ nguy hiểm."
        ),
        "expected": "Kiểm tra liệu tín hiệu tổng hợp về phục vụ có cải thiện delay/starvation hay không.",
    },
    "deficit_service": {
        "name": "Throughput-deficit service",
        "meaning": "Thưởng khi phục vụ UE đang có throughput thấp hơn mức trung bình của cell.",
        "expected": "Bù tài nguyên cho UE bị thiếu throughput.",
    },
    "pf_utility": {
        "name": "PF utility",
        "meaning": "Thưởng theo log-throughput để cân bằng hiệu suất tổng và lợi ích của UE yếu.",
        "expected": "Tạo trade-off kiểu Proportional Fair.",
    },
    "low_throughput": {
        "name": "Low-throughput",
        "meaning": "Thưởng khi nhóm UE ở đáy phân phối throughput được cải thiện.",
        "expected": "Kéo nhóm UE yếu nhất lên.",
    },
    "urgency_service": {
        "name": "Urgency service",
        "meaning": "Thưởng khi phục vụ UE vừa thiếu throughput vừa chờ lâu.",
        "expected": "Ưu tiên UE có nguy cơ bị bỏ quên.",
    },
}

KPI_COLUMNS: tuple[tuple[str, str], ...] = (
    ("mean_goodput_bits_per_slot", "Goodput (bit/slot)"),
    ("final_jain_fairness", "Jain fairness"),
    ("max_starvation_rate", "Starvation rate"),
    ("max_p99_wait_slots", "Worst P99 wait (slot)"),
    ("max_wait_slots", "Max wait (slot)"),
)


def _read_last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _fmt_metric(key: str, value: float) -> str:
    if key == "mean_goodput_bits_per_slot":
        return f"{value:,.1f}"
    if key == "max_starvation_rate":
        return f"{100.0 * value:.2f}%"
    if key == "final_jain_fairness":
        return f"{value:.4f}"
    return f"{value:.1f}"


def _formula(coefficients: Mapping[str, float]) -> str:
    terms: list[str] = []
    for component in POSITIVE_COMPONENTS:
        coefficient = float(coefficients.get(f"coef_{component}", 0.0))
        if coefficient > 0.0:
            name = COMPONENT_GUIDE[component]["name"]
            terms.append(f"{coefficient:.4g} × {name}")
    return " + ".join(terms) or "0"


def _metric_delta_text(key: str, current: float, reference: float) -> str:
    delta = current - reference
    if key == "mean_goodput_bits_per_slot":
        relative = 100.0 * delta / reference if abs(reference) > 1e-12 else 0.0
        return f"{delta:+,.1f} ({relative:+.2f}%)"
    if key == "max_starvation_rate":
        return f"{100.0 * delta:+.2f} điểm %"
    if key == "final_jain_fairness":
        return f"{delta:+.4f}"
    return f"{delta:+.1f} slot"


def _dominant_component(coefficients: Mapping[str, float]) -> str:
    active = {
        component: float(coefficients.get(f"coef_{component}", 0.0))
        for component in POSITIVE_COMPONENTS
    }
    return max(active, key=active.get)


def _plain_language_observations(
    current: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    focus_component: str,
) -> list[str]:
    if reference is None:
        return [
            "Chưa tìm thấy run tham chiếu trong artifacts, nên báo cáo hiện chỉ mô tả case mới.",
            "Sau khi đặt run tham chiếu đúng đường dẫn, chạy lại lệnh tạo analysis để có bảng chênh lệch.",
        ]

    current_values = {key: safe_float(current, key) for key, _ in KPI_COLUMNS}
    reference_values = {key: safe_float(reference, key) for key, _ in KPI_COLUMNS}
    observations: list[str] = []

    goodput_delta = current_values["mean_goodput_bits_per_slot"] - reference_values["mean_goodput_bits_per_slot"]
    fairness_delta = current_values["final_jain_fairness"] - reference_values["final_jain_fairness"]
    starvation_delta = current_values["max_starvation_rate"] - reference_values["max_starvation_rate"]
    p99_delta = current_values["max_p99_wait_slots"] - reference_values["max_p99_wait_slots"]
    max_wait_delta = current_values["max_wait_slots"] - reference_values["max_wait_slots"]

    component_name = COMPONENT_GUIDE[focus_component]["name"]
    if focus_component == "throughput":
        if goodput_delta > 0:
            observations.append("Tăng trọng số Throughput đã làm goodput tăng so với mốc ba thành phần bằng nhau.")
        else:
            observations.append("Tăng trọng số Throughput chưa làm goodput tăng ở checkpoint cuối; cần xem trajectory trước khi kết luận.")
    elif focus_component == "fairness":
        if fairness_delta > 0:
            observations.append("Tăng trọng số Jain fairness đã làm fairness deterministic tăng so với mốc bằng nhau.")
        else:
            observations.append("Tăng trọng số Jain fairness chưa làm fairness deterministic tăng; có thể policy đang học score phẳng hoặc phụ thuộc exploration.")
    elif focus_component == "service":
        if starvation_delta < -1e-6 or p99_delta < -1e-6 or max_wait_delta < -1e-6:
            observations.append("Tăng trọng số Service đã cải thiện ít nhất một KPI chờ/starvation.")
        else:
            observations.append("Tăng trọng số Service chưa cải thiện KPI chờ/starvation ở checkpoint cuối.")
    else:
        observations.append(f"Case này kiểm tra tác dụng khi tăng {component_name} so với mốc bằng nhau.")

    if goodput_delta > 0:
        observations.append("Goodput tăng so với run tham chiếu.")
    elif goodput_delta < 0:
        observations.append("Goodput giảm so với run tham chiếu.")

    if fairness_delta > 0:
        observations.append("Jain fairness tăng so với run tham chiếu.")
    elif fairness_delta < 0:
        observations.append("Jain fairness giảm so với run tham chiếu.")

    if starvation_delta > 1e-6 or p99_delta > 1e-6 or max_wait_delta > 1e-6:
        observations.append("Ít nhất một KPI coverage/thời gian chờ xấu đi so với mốc bằng nhau.")
    elif starvation_delta < -1e-6 or p99_delta < -1e-6 or max_wait_delta < -1e-6:
        observations.append("Ít nhất một KPI coverage/thời gian chờ tốt lên so với mốc bằng nhau.")

    if current_values["max_starvation_rate"] > 0.10:
        observations.append(
            "Policy đang collapse về coverage: hơn 10% UE từng vượt quá ngưỡng 64 slot không có lần truyền thành công."
        )
    elif current_values["max_starvation_rate"] == 0.0:
        observations.append("Không quan sát thấy starvation trong episode validation cuối.")

    return observations


def _reward_contribution_table(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return ""
    entries: list[tuple[str, float]] = []
    for component in POSITIVE_COMPONENTS:
        key = f"mean_reward_{component}_component"
        if key in row and str(row[key]).strip() != "":
            entries.append((component, safe_float(row, key)))
    if not entries:
        return ""
    total = sum(max(value, 0.0) for _, value in entries)
    rows = []
    for component, value in entries:
        share = 100.0 * max(value, 0.0) / total if total > 1e-12 else 0.0
        rows.append(
            "<tr>"
            f"<td>{html.escape(COMPONENT_GUIDE[component]['name'])}</td>"
            f"<td>{value:.6f}</td><td>{share:.1f}%</td>"
            "</tr>"
        )
    return (
        "<h3>Đóng góp reward thực tế ở checkpoint cuối</h3>"
        "<table><thead><tr><th>Thành phần</th><th>Mean contribution</th><th>Tỷ trọng trong reward dương</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p><small>Trọng số trong config và tỷ trọng đóng góp thực tế có thể khác nhau vì raw score của từng thành phần không cùng độ lớn.</small></p>"
    )


def _integrated_status(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "Chưa có kết quả"
    starvation = safe_float(row, "max_starvation_rate")
    p99_wait = safe_float(row, "max_p99_wait_slots")
    max_wait = safe_float(row, "max_wait_slots")
    if starvation > 0.10 or p99_wait >= 5000 or max_wait >= 5000:
        return "Collapse coverage"
    if starvation > 0.0 or p99_wait > 64 or max_wait > 80:
        return "Không ổn định"
    return "Ổn định"


def _regime_order(analysis_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    configured = analysis_cfg.get("regime_order")
    if isinstance(configured, (list, tuple)):
        values = tuple(str(value) for value in configured if str(value).strip())
        if values:
            return values
    return ("equal_quarter", "new_component_heavy", "anchor_preserving")


def _integrated_interpretation(
    *,
    component: str,
    rows: Mapping[str, Mapping[str, Any] | None],
    analysis_cfg: Mapping[str, Any],
) -> str:
    statuses = {regime: _integrated_status(row) for regime, row in rows.items()}
    name = COMPONENT_GUIDE[component]["name"]
    completed = {key: value for key, value in statuses.items() if value != "Chưa có kết quả"}
    if not completed:
        return f"Chưa có kết quả để kết luận vai trò của {name}."

    stable = [key for key, value in completed.items() if value == "Ổn định"]
    unstable = [key for key, value in completed.items() if value != "Ổn định"]
    labels = dict(analysis_cfg.get("regime_labels", {}))
    families = dict(analysis_cfg.get("regime_family", {}))
    sentences = [f"{name} ổn định ở {len(stable)}/{len(completed)} regime đã hoàn thành."]

    equal_status = statuses.get("equal_quarter")
    heavy_status = statuses.get("new_component_heavy")
    if equal_status == "Ổn định":
        sentences.append("Equal-quarter cho thấy component có thể được thêm vào nền T–J–S mà chưa phá coverage.")
    elif equal_status and equal_status != "Chưa có kết quả":
        sentences.append("Equal-quarter không ổn định, nên chỉ việc thêm component đã làm policy rời vùng tốt.")
    if heavy_status == "Ổn định":
        sentences.append("New-component-heavy vẫn ổn định, vì vậy component có tiềm năng làm objective mạnh hơn.")
    elif heavy_status and heavy_status != "Chưa có kết quả":
        sentences.append("New-component-heavy thất bại; component không nên được xem là objective chi phối ở công thức hiện tại.")

    single_stable = [
        key for key in stable if families.get(key) == "single_hold"
    ]
    pair_stable = [
        key for key in stable if families.get(key) == "pair_hold"
    ]
    if single_stable:
        names = ", ".join(labels.get(key, key).split(":", 1)[0] for key in single_stable)
        sentences.append(f"Các phép giữ một anchor còn ổn định: {names}; đây là bằng chứng component phụ thuộc vào anchor cụ thể.")
    if pair_stable:
        names = ", ".join(labels.get(key, key).split(":", 1)[0] for key in pair_stable)
        sentences.append(f"Các nhóm giữ được policy ổn định: {names}; dùng chúng để xác định X có thể thay reward nào.")
    if unstable and not stable:
        sentences.append("Mọi regime đã chạy đều không ổn định; nên xem lại công thức score trước khi tune hệ số sâu hơn.")
    return " ".join(sentences)


def _integrated_component_comparison(
    *,
    plan: RewardStudyPlan,
    round_path: Path,
    analysis_cfg: Mapping[str, Any],
) -> str:
    component_case_map = dict(analysis_cfg.get("component_case_map", {}))
    regime_labels = dict(analysis_cfg.get("regime_labels", {}))
    order = _regime_order(analysis_cfg)
    sections: list[str] = []
    for component, mapping_raw in component_case_map.items():
        mapping = dict(mapping_raw)
        rows_by_regime: dict[str, Mapping[str, Any] | None] = {}
        table_rows: list[str] = []
        for regime in order:
            case_id = str(mapping.get(regime, ""))
            row = _read_last_csv_row(round_path / case_id / "validation.csv") if case_id else None
            rows_by_regime[regime] = row
            if row is None:
                table_rows.append(
                    "<tr>"
                    f"<td>{html.escape(regime_labels.get(regime, regime))}</td>"
                    "<td colspan='6'>Chưa có kết quả</td></tr>"
                )
                continue
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(regime_labels.get(regime, regime))}</td>"
                f"<td>{safe_float(row, 'mean_goodput_bits_per_slot'):,.1f}</td>"
                f"<td>{safe_float(row, 'final_jain_fairness'):.4f}</td>"
                f"<td>{100.0 * safe_float(row, 'max_starvation_rate'):.2f}%</td>"
                f"<td>{safe_float(row, 'max_p99_wait_slots'):.1f}</td>"
                f"<td>{safe_float(row, 'max_wait_slots'):.1f}</td>"
                f"<td>{html.escape(_integrated_status(row))}</td>"
                "</tr>"
            )
        interpretation = _integrated_interpretation(
            component=component,
            rows=rows_by_regime,
            analysis_cfg=analysis_cfg,
        )
        sections.append(
            "<section class='card'>"
            f"<h2>Tám regime của {html.escape(COMPONENT_GUIDE[component]['name'])}</h2>"
            f"<p>{html.escape(COMPONENT_GUIDE[component]['meaning'])}</p>"
            "<table><thead><tr><th>Regime</th><th>Goodput</th><th>Jain</th><th>Starvation</th>"
            "<th>Worst P99</th><th>Max wait</th><th>Trạng thái</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table>"
            f"<div class='callout info'><strong>Cách đọc:</strong> {html.escape(interpretation)}</div>"
            "</section>"
        )

    # Cross-component reading: one table per regime so all four reward candidates
    # can be compared under exactly the same weight geometry.
    for regime in order:
        table_rows = []
        for component, mapping_raw in component_case_map.items():
            case_id = str(dict(mapping_raw).get(regime, ""))
            row = _read_last_csv_row(round_path / case_id / "validation.csv") if case_id else None
            if row is None:
                table_rows.append(
                    "<tr>"
                    f"<td>{html.escape(COMPONENT_GUIDE[component]['name'])}</td>"
                    "<td colspan='6'>Chưa có kết quả</td></tr>"
                )
                continue
            table_rows.append(
                "<tr>"
                f"<td>{html.escape(COMPONENT_GUIDE[component]['name'])}</td>"
                f"<td>{safe_float(row, 'mean_goodput_bits_per_slot'):,.1f}</td>"
                f"<td>{safe_float(row, 'final_jain_fairness'):.4f}</td>"
                f"<td>{100.0 * safe_float(row, 'max_starvation_rate'):.2f}%</td>"
                f"<td>{safe_float(row, 'max_p99_wait_slots'):.1f}</td>"
                f"<td>{safe_float(row, 'max_wait_slots'):.1f}</td>"
                f"<td>{html.escape(_integrated_status(row))}</td>"
                "</tr>"
            )
        sections.append(
            "<section class='card'>"
            f"<h2>So sánh bốn reward tại regime: {html.escape(regime_labels.get(regime, regime))}</h2>"
            "<table><thead><tr><th>Reward thứ tư</th><th>Goodput</th><th>Jain</th><th>Starvation</th>"
            "<th>Worst P99</th><th>Max wait</th><th>Trạng thái</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody></table>"
            "</section>"
        )
    return "".join(sections)

def _export_integrated_round_tables(
    *,
    plan: RewardStudyPlan,
    round_path: Path,
    analysis_cfg: Mapping[str, Any],
    reference_row: Mapping[str, Any] | None,
) -> None:
    case_focus = dict(analysis_cfg.get("case_focus", {}))
    case_regime = dict(analysis_cfg.get("case_regime", {}))
    final_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for case in plan.cases:
        coefficients = case.actual_coefficients()
        metadata = {
            "case_id": case.case_id,
            "label": case.label,
            "focus_component": case_focus.get(case.case_id, ""),
            "regime": case_regime.get(case.case_id, ""),
            **coefficients,
        }
        validation_rows = _read_csv_rows(round_path / case.case_id / "validation.csv")
        for row in validation_rows:
            trajectory_rows.append({**metadata, **row})
        if not validation_rows:
            continue
        final = {**metadata, **validation_rows[-1]}
        final_rows.append(final)
        comparison = dict(final)
        comparison["status"] = _integrated_status(final)
        if reference_row is not None:
            for key, _ in KPI_COLUMNS:
                comparison[f"delta_{key}"] = safe_float(final, key) - safe_float(reference_row, key)
        comparison_rows.append(comparison)

    stability_rows: list[dict[str, Any]] = []
    component_case_map = dict(analysis_cfg.get("component_case_map", {}))
    regime_order = _regime_order(analysis_cfg)
    for component, mapping_raw in component_case_map.items():
        mapping = dict(mapping_raw)
        row: dict[str, Any] = {
            "focus_component": component,
            "component_label": COMPONENT_GUIDE.get(component, {}).get("name", component),
        }
        for regime in regime_order:
            case_id = str(mapping.get(regime, ""))
            final = _read_last_csv_row(round_path / case_id / "validation.csv") if case_id else None
            row[f"{regime}_case_id"] = case_id
            row[f"{regime}_status"] = _integrated_status(final)
        stability_rows.append(row)

    regime_summary_rows: list[dict[str, Any]] = []
    for regime in regime_order:
        matching = [row for row in comparison_rows if row.get("regime") == regime]
        completed = [row for row in matching if row.get("status") != "Chưa có kết quả"]
        stable = [row for row in completed if row.get("status") == "Ổn định"]
        regime_summary_rows.append({
            "regime": regime,
            "regime_label": dict(analysis_cfg.get("regime_labels", {})).get(regime, regime),
            "completed_cases": len(completed),
            "stable_cases": len(stable),
            "collapse_or_unstable_cases": len(completed) - len(stable),
            "best_goodput_bits_per_slot": max((safe_float(row, "mean_goodput_bits_per_slot") for row in stable), default=0.0),
            "best_jain_fairness": max((safe_float(row, "final_jain_fairness") for row in stable), default=0.0),
            "lowest_p99_wait_slots": min((safe_float(row, "max_p99_wait_slots") for row in stable), default=0.0),
        })

    outputs = [
        ("final_metrics_output", final_rows),
        ("trajectory_output", trajectory_rows),
        ("comparison_output", comparison_rows),
        ("stability_output", stability_rows),
        ("regime_summary_output", regime_summary_rows),
    ]
    for config_key, rows in outputs:
        raw = str(analysis_cfg.get(config_key, "")).strip()
        if raw:
            _write_csv_rows(Path(raw), rows)


def build_incremental_reward_analysis(
    *,
    plan: RewardStudyPlan,
    round_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a plain-language HTML analysis for an incremental reward round."""
    round_path = Path(round_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    analysis_cfg = dict(plan.analysis)
    design = str(analysis_cfg.get("design", "single_component_increment"))
    focus_component = str(analysis_cfg.get("focus_component", "service"))
    reference_run_raw = str(analysis_cfg.get("reference_run", "")).strip()
    reference_path = Path(reference_run_raw) if reference_run_raw else None
    reference_row = (
        _read_last_csv_row(reference_path / "validation.csv")
        if reference_path is not None
        else None
    )
    if reference_row is None and isinstance(analysis_cfg.get("reference_metrics"), Mapping):
        reference_row = {
            str(key): str(value)
            for key, value in dict(analysis_cfg["reference_metrics"]).items()
        }
    reference_label = str(analysis_cfg.get("reference_label", "Run tham chiếu"))

    case_sections: list[str] = []
    summary_rows: list[str] = []
    for case in plan.cases:
        case_dir = round_path / case.case_id
        row = _read_last_csv_row(case_dir / "validation.csv")
        coefficients = case.actual_coefficients()
        if design in {"three_component_coordinate_perturbation", "fourth_component_equal_screen", "fourth_component_integrated_screen", "fourth_component_comprehensive_screen"}:
            focus_map = dict(analysis_cfg.get("case_focus", {}))
            case_focus = str(focus_map.get(case.case_id, _dominant_component(coefficients)))
        else:
            case_focus = (
                _dominant_component(coefficients)
                if design in {
                    "directional_three_component",
                    "hold_service_trade_throughput_jain",
                }
                else focus_component
            )

        coefficient_rows = []
        for component in POSITIVE_COMPONENTS:
            coefficient = coefficients[f"coef_{component}"]
            coefficient_rows.append(
                "<tr>"
                f"<td>{html.escape(COMPONENT_GUIDE[component]['name'])}</td>"
                f"<td>{coefficient:.6g}</td>"
                f"<td>{html.escape(COMPONENT_GUIDE[component]['meaning'])}</td>"
                "</tr>"
            )

        if row is None:
            result_block = (
                "<div class='callout warn'><strong>Chưa có kết quả.</strong> "
                "Case chưa train xong hoặc thiếu validation.csv.</div>"
            )
            summary_rows.append(
                "<tr>"
                f"<td>{html.escape(case.label)}</td><td colspan='6'>Chưa có kết quả</td>"
                "</tr>"
            )
        else:
            metric_rows = []
            for key, label in KPI_COLUMNS:
                current_value = safe_float(row, key)
                reference_cell = "—"
                if reference_row is not None:
                    reference_value = safe_float(reference_row, key)
                    reference_cell = (
                        f"{_fmt_metric(key, reference_value)} "
                        f"<small>({_metric_delta_text(key, current_value, reference_value)})</small>"
                    )
                metric_rows.append(
                    "<tr>"
                    f"<td>{html.escape(label)}</td>"
                    f"<td>{_fmt_metric(key, current_value)}</td>"
                    f"<td>{reference_cell}</td>"
                    "</tr>"
                )

            observations = _plain_language_observations(row, reference_row, case_focus)
            observations_html = "".join(
                f"<li>{html.escape(item)}</li>" for item in observations
            )
            result_block = (
                "<h3>Kết quả cuối run</h3>"
                "<table><thead><tr><th>KPI</th><th>Case mới</th>"
                f"<th>{html.escape(reference_label)} và chênh lệch</th></tr></thead>"
                f"<tbody>{''.join(metric_rows)}</tbody></table>"
                f"{_reward_contribution_table(row)}"
                "<h3>Diễn giải ban đầu</h3>"
                f"<ul>{observations_html}</ul>"
            )
            summary_rows.append(
                "<tr>"
                f"<td>{html.escape(case.label)}</td>"
                f"<td>{safe_float(row, 'mean_goodput_bits_per_slot'):,.1f}</td>"
                f"<td>{safe_float(row, 'final_jain_fairness'):.4f}</td>"
                f"<td>{100.0 * safe_float(row, 'max_starvation_rate'):.2f}%</td>"
                f"<td>{safe_float(row, 'max_p99_wait_slots'):.1f}</td>"
                f"<td>{safe_float(row, 'max_wait_slots'):.1f}</td>"
                f"<td>{html.escape(COMPONENT_GUIDE[case_focus]['name'])}"
                + (
                    " · " + html.escape(str(dict(analysis_cfg.get('regime_labels', {})).get(
                        dict(analysis_cfg.get('case_regime', {})).get(case.case_id, ''),
                        dict(analysis_cfg.get('case_regime', {})).get(case.case_id, ''),
                    )))
                    if design in {"fourth_component_integrated_screen", "fourth_component_comprehensive_screen"}
                    else ""
                )
                + "</td></tr>"
            )

        case_body = (
            f"<p><strong>Câu hỏi:</strong> {html.escape(case.hypothesis)}</p>"
            f"<pre>Reward = {html.escape(_formula(coefficients))}</pre>"
            "<table><thead><tr><th>Thành phần</th><th>Hệ số thực tế</th>"
            f"<th>Thành phần này đo gì?</th></tr></thead><tbody>{''.join(coefficient_rows)}</tbody></table>"
            f"{result_block}"
        )
        if design == "fourth_component_comprehensive_screen":
            case_sections.append(
                "<details class='card'>"
                f"<summary><strong>{html.escape(case.label)}</strong></summary>"
                f"{case_body}</details>"
            )
        else:
            case_sections.append(
                "<section class='card'>"
                f"<h2>{html.escape(case.label)}</h2>"
                f"{case_body}</section>"
            )

    reference_note = (
        f"Run tham chiếu được đọc từ <code>{html.escape(str(reference_path))}</code>."
        if reference_path is not None
        else "Plan chưa khai báo run tham chiếu."
    )

    if design == "directional_three_component":
        objective_text = (
            "Bắt đầu từ mốc ba thành phần bằng nhau. Mỗi case chỉ tăng một thành phần lên 0,50 "
            "và giảm đều hai thành phần còn lại xuống 0,25. Mục tiêu là đo hướng tác động của "
            "Throughput, Jain fairness và Service trước khi tune cục bộ."
        )
        decision_text = (
            "Sau khi xem xu hướng của ba case, chọn một thành phần đáng giữ ở mức cao hoặc một "
            "mức cố định hợp lý. Vòng kế tiếp mới giữ thành phần đó và tăng/giảm hai thành phần "
            "còn lại với một số ít config. Không sweep dày và không tối ưu riêng một case thất bại."
        )
    elif design == "hold_service_trade_throughput_jain":
        objective_text = (
            "Giữ Service ở mức 1/3 đã cho policy ổn định, sau đó chỉ chuyển một lượng nhỏ "
            "trọng số giữa Throughput và Jain fairness. Mục tiêu là kiểm tra hai hướng cải thiện "
            "cục bộ mà không rời khỏi vùng coverage ổn định."
        )
        decision_text = (
            "So sánh hai case với mốc 1/3–1/3–1/3. Chỉ giữ một hướng nếu KPI mục tiêu cải thiện "
            "mà starvation và wait vẫn ổn định. Nếu cả hai xấu, giữ mốc bằng nhau và chuyển sang "
            "khám phá thành phần reward tiếp theo thay vì sweep thêm."
        )
    elif design == "three_component_coordinate_perturbation":
        objective_text = (
            "Dùng mốc 1/3–1/3–1/3 làm tâm. Trong mỗi cặp thí nghiệm, giữ một thành phần ở 1/3, "
            "tăng một thành phần lên 0,40 và giảm thành phần còn lại xuống 0,2667. Sáu case bao phủ "
            "toàn bộ ba mặt cắt cục bộ của reward, nhưng vẫn chỉ thay đổi nhẹ để tránh lặp lại các "
            "collapse ở mức 0,50 của Round 05."
        )
        decision_text = (
            "Đọc từng cặp theo thành phần được giữ cố định. Một hướng chỉ được xem là hữu ích khi KPI "
            "đúng mục tiêu cải thiện, starvation/wait không collapse và trajectory không chỉ tốt ở một "
            "checkpoint ngắn. Sau vòng này mới chọn 1–2 hướng đáng tinh chỉnh; không sweep dày."
        )
    elif design == "fourth_component_comprehensive_screen":
        objective_text = (
            "Với mỗi reward thứ tư, Round 07 dùng tám regime trong cùng một plan: equal-quarter; component-heavy; "
            "ba phép giữ riêng Throughput, Jain hoặc Service; và ba phép giữ nhóm T+J, T+S hoặc J+S. "
            "Thiết kế này kiểm tra đồng thời lợi ích biên, khả năng chi phối, phụ thuộc vào từng anchor và khả năng thay thế từng reward nền."
        )
        decision_text = (
            "Đọc kết quả theo hai chiều. Theo hàng component để biết reward mới cần anchor nào; theo cột regime để so sánh bốn reward dưới cùng hình học trọng số. "
            "Chỉ giữ component nếu nó ổn định ở nhiều hơn một regime liên quan và cải thiện KPI mục tiêu. Một case đơn lẻ tốt chưa đủ; phải kiểm tra trajectory và late-collapse."
        )
    elif design in {"fourth_component_integrated_screen", "fourth_component_comprehensive_screen"}:
        objective_text = (
            "Với mỗi reward thứ tư, Round 07 chạy ba regime trong cùng một plan: thêm bằng nhau; tăng component "
            "mới lên 0,40 và giảm đều ba thành phần nền; sau đó giữ Throughput và Service ở 0,30 rồi thay phần lớn "
            "Jain bằng component mới. Thiết kế này phân biệt ba nguyên nhân: component có lợi ích biên hay không; "
            "component có chịu được vai trò chi phối hay không; và collapse có phải do làm yếu hai neo Throughput–Service hay không."
        )
        decision_text = (
            "Đọc ba case của cùng một component như một ablation thống nhất. Equal-quarter đo lợi ích biên; heavy đo khả năng làm objective chính; "
            "anchor-preserving đo khả năng thay một phần Jain khi hai neo còn mạnh. Chỉ giữ component nếu KPI mục tiêu cải thiện ở ít nhất một regime ổn định "
            "và trajectory không late-collapse. Không mở Round 08 chỉ để tune một component không có bằng chứng."
        )
    elif design == "fourth_component_equal_screen":
        objective_text = (
            "Giữ cùng nền Throughput–Jain–Service và thêm đúng một reward thứ tư trong mỗi case. "
            "Bốn thành phần đang hoạt động đều có hệ số 0,25. Đây là vòng sàng lọc tác dụng biên: "
            "component mới tạo KPI mới, trùng lặp tín hiệu hay phá vùng ổn định."
        )
        decision_text = (
            "Không tune đồng thời cả bốn case. Chỉ component nào cải thiện đúng KPI mục tiêu, không "
            "collapse deterministic và có trajectory ổn định mới được giữ để tăng/giảm hệ số ở vòng sau. "
            "Các component còn lại được ghi nhận như kết quả âm hoặc cần thiết kế lại công thức."
        )
    else:
        focus = COMPONENT_GUIDE.get(focus_component, COMPONENT_GUIDE["service"])
        objective_text = (
            f"Ta chỉ thêm hoặc thay đổi <strong>{html.escape(focus['name'])}</strong> trong reward "
            "để trả lời một câu hỏi cụ thể, chưa tune nhiều hệ số cùng lúc. "
            f"{html.escape(focus['meaning'])}"
        )
        decision_text = (
            "Nếu component mới cải thiện KPI đúng mục tiêu mà không làm goodput/fairness collapse, "
            "ta giữ và chỉ tune nhẹ hệ số. Nếu không tạo chuyển biến hoặc gây collapse, ta giảm hệ số "
            "hoặc ghi nhận rằng công thức hiện tại chưa phù hợp; không sweep dày."
        )

    summary_block = ""
    if design in {"directional_three_component", "three_component_coordinate_perturbation", "fourth_component_equal_screen", "fourth_component_integrated_screen", "fourth_component_comprehensive_screen"}:
        reference_summary = ""
        if reference_row is not None:
            reference_summary = (
                "<tr><td><strong>Mốc bằng nhau 1/3–1/3–1/3</strong></td>"
                f"<td>{safe_float(reference_row, 'mean_goodput_bits_per_slot'):,.1f}</td>"
                f"<td>{safe_float(reference_row, 'final_jain_fairness'):.4f}</td>"
                f"<td>{100.0 * safe_float(reference_row, 'max_starvation_rate'):.2f}%</td>"
                f"<td>{safe_float(reference_row, 'max_p99_wait_slots'):.1f}</td>"
                f"<td>{safe_float(reference_row, 'max_wait_slots'):.1f}</td>"
                "<td>Mốc trung tâm</td></tr>"
            )
        summary_block = (
            "<section class='card'><h2>Bảng so sánh tổng quát</h2>"
            "<table><thead><tr><th>Case</th><th>Goodput</th><th>Jain</th><th>Starvation</th>"
            "<th>Worst P99 wait</th><th>Max wait</th><th>Hướng thay đổi chính</th></tr></thead>"
            f"<tbody>{reference_summary}{''.join(summary_rows)}</tbody></table></section>"
        )

    integrated_sections = ""
    if design in {"fourth_component_integrated_screen", "fourth_component_comprehensive_screen"}:
        _export_integrated_round_tables(
            plan=plan,
            round_path=round_path,
            analysis_cfg=analysis_cfg,
            reference_row=reference_row,
        )
        integrated_sections = _integrated_component_comparison(
            plan=plan,
            round_path=round_path,
            analysis_cfg=analysis_cfg,
        )

    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(plan.round_id)} – Phân tích reward</title>
<style>
body{{margin:0;background:#f4f7fb;color:#172033;font-family:Segoe UI,Arial,sans-serif;line-height:1.6}}
main{{max-width:1120px;margin:auto;padding:28px 18px 60px}}header,.card{{background:#fff;border:1px solid #dfe5ee;border-radius:16px;padding:22px;margin-bottom:16px}}
header{{background:linear-gradient(135deg,#18264a,#315efb);color:#fff}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dfe5ee;text-align:left;vertical-align:top}}th{{background:#f0f3f8}}
pre{{background:#111827;color:#edf2f7;padding:15px;border-radius:10px;overflow:auto}}code{{background:#eef1f7;color:#172033;padding:2px 5px;border-radius:5px}}.callout{{padding:14px 16px;border-radius:11px}}.warn{{background:#fff5d8}}.info{{background:#eaf1ff}}small{{color:#5f6e82}}.nav{{margin:0 0 14px;padding:10px 14px;background:#fff;border:1px solid #dfe5ee;border-radius:12px}}.nav a{{color:#315efb;text-decoration:none}}details summary{{cursor:pointer;font-size:18px;padding:4px 0}}details[open] summary{{margin-bottom:14px}}
</style></head><body><main>
<nav class="nav"><a href="../index.html">← Mục lục reward study</a> · <a href="../../index.html">Kho phân tích</a></nav>
<header><h1>{html.escape(plan.round_id)}</h1><p>{html.escape(plan.description)}</p></header>
<section class="card"><h2>Mục tiêu của vòng này</h2><p>{objective_text}</p><p>{reference_note}</p></section>
{summary_block}
{integrated_sections}
{''.join(case_sections)}
<section class="card"><h2>Giải nghĩa KPI dễ nhầm</h2>
<p><strong>Starvation rate:</strong> tỷ lệ UE từng có hơn 64 slot liên tiếp không có lần truyền thành công. Một slot tương ứng một action scheduling.</p>
<p><strong>Worst P99 wait:</strong> tại mỗi slot, sắp xếp thời gian chờ của 1.200 UE. P99 = 49 nghĩa là 99% UE đã được phục vụ trong 49 slot gần nhất; 1% UE chờ lâu hơn. “Worst” là giá trị P99 lớn nhất trong toàn episode đánh giá.</p>
<p><strong>Max wait:</strong> thời gian chờ dài nhất của UE tệ nhất trong episode.</p></section>
<section class="card"><h2>Quy tắc quyết định vòng sau</h2><p>{decision_text}</p></section>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output
