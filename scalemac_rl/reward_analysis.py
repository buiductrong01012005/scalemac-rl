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
        "meaning": "Thưởng khi throughput được phân phối đồng đều hơn giữa 1.200 UE.",
        "expected": "Hạn chế việc chỉ phục vụ một nhóm UE nhỏ.",
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
        case_focus = (
            _dominant_component(coefficients)
            if design == "directional_three_component"
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
                f"<td>{html.escape(COMPONENT_GUIDE[case_focus]['name'])}</td>"
                "</tr>"
            )

        case_sections.append(
            "<section class='card'>"
            f"<h2>{html.escape(case.label)}</h2>"
            f"<p><strong>Câu hỏi:</strong> {html.escape(case.hypothesis)}</p>"
            f"<pre>Reward = {html.escape(_formula(coefficients))}</pre>"
            "<table><thead><tr><th>Thành phần</th><th>Hệ số thực tế</th>"
            f"<th>Thành phần này đo gì?</th></tr></thead><tbody>{''.join(coefficient_rows)}</tbody></table>"
            f"{result_block}"
            "</section>"
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
    if design == "directional_three_component":
        reference_summary = ""
        if reference_row is not None:
            reference_summary = (
                "<tr><td><strong>Mốc bằng nhau 1/3–1/3–1/3</strong></td>"
                f"<td>{safe_float(reference_row, 'mean_goodput_bits_per_slot'):,.1f}</td>"
                f"<td>{safe_float(reference_row, 'final_jain_fairness'):.4f}</td>"
                f"<td>{100.0 * safe_float(reference_row, 'max_starvation_rate'):.2f}%</td>"
                f"<td>{safe_float(reference_row, 'max_p99_wait_slots'):.1f}</td>"
                f"<td>{safe_float(reference_row, 'max_wait_slots'):.1f}</td>"
                "<td>Không tăng riêng thành phần nào</td></tr>"
            )
        summary_block = (
            "<section class='card'><h2>Bảng so sánh tổng quát</h2>"
            "<table><thead><tr><th>Case</th><th>Goodput</th><th>Jain</th><th>Starvation</th>"
            "<th>Worst P99 wait</th><th>Max wait</th><th>Thành phần được tăng</th></tr></thead>"
            f"<tbody>{reference_summary}{''.join(summary_rows)}</tbody></table></section>"
        )

    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(plan.round_id)} – Phân tích reward</title>
<style>
body{{margin:0;background:#f4f7fb;color:#172033;font-family:Segoe UI,Arial,sans-serif;line-height:1.6}}
main{{max-width:1120px;margin:auto;padding:28px 18px 60px}}header,.card{{background:#fff;border:1px solid #dfe5ee;border-radius:16px;padding:22px;margin-bottom:16px}}
header{{background:linear-gradient(135deg,#18264a,#315efb);color:#fff}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dfe5ee;text-align:left;vertical-align:top}}th{{background:#f0f3f8}}
pre{{background:#111827;color:#edf2f7;padding:15px;border-radius:10px;overflow:auto}}code{{background:#eef1f7;color:#172033;padding:2px 5px;border-radius:5px}}.callout{{padding:14px 16px;border-radius:11px}}.warn{{background:#fff5d8}}small{{color:#5f6e82}}
</style></head><body><main>
<header><h1>{html.escape(plan.round_id)}</h1><p>{html.escape(plan.description)}</p></header>
<section class="card"><h2>Mục tiêu của vòng này</h2><p>{objective_text}</p><p>{reference_note}</p></section>
{summary_block}
{''.join(case_sections)}
<section class="card"><h2>Giải nghĩa KPI dễ nhầm</h2>
<p><strong>Starvation rate:</strong> tỷ lệ UE từng có hơn 64 slot liên tiếp không có lần truyền thành công. Một slot tương ứng một action scheduling.</p>
<p><strong>Worst P99 wait:</strong> tại mỗi slot, sắp xếp thời gian chờ của 1.200 UE. P99 = 49 nghĩa là 99% UE đã được phục vụ trong 49 slot gần nhất; 1% UE chờ lâu hơn. “Worst” là giá trị P99 lớn nhất trong toàn episode đánh giá.</p>
<p><strong>Max wait:</strong> thời gian chờ dài nhất của UE tệ nhất trong episode.</p></section>
<section class="card"><h2>Quy tắc quyết định vòng sau</h2><p>{decision_text}</p></section>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output
