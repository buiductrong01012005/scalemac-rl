from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnalysisPage:
    title: str
    href: str
    question: str
    finding: str


PAGES: tuple[AnalysisPage, ...] = (
    AnalysisPage(
        "Baseline v0.8.3",
        "baselines/v083_full_control_baseline.html",
        "PPO full-control có học được scheduling không rule hay không?",
        "PPO học được safety nhưng reward phức tạp chưa tạo fairness đủ cao.",
    ),
    AnalysisPage(
        "Round 01 — từng reward riêng lẻ",
        "round_01/round_01_component_screen_analysis.html",
        "Mỗi reward riêng lẻ khiến policy học hành vi gì?",
        "Throughput và Jain học được; Service, urgency và low-throughput có thể bị khai thác hoặc mất tín hiệu.",
    ),
    AnalysisPage(
        "Round 02 — Throughput + Jain",
        "round_02/round_02_throughput_jain_analysis.html",
        "Khi ghép hai KPI chính, tỷ lệ nào tạo policy ổn định?",
        "37,5/62,5 là điểm ổn định; các tỷ lệ khác cho thấy trọng số config không đồng nghĩa ảnh hưởng thực tế.",
    ),
    AnalysisPage(
        "Round 03 — chẩn đoán policy",
        "round_03/policy_diagnostics_analysis.html",
        "Vì sao 25/75 vẫn starvation dù Jain có trọng số lớn?",
        "Stochastic noise tạo rotation lúc train; deterministic mean-policy giữ lặp một nhóm UE và quay vòng quá chậm.",
    ),
    AnalysisPage(
        "Round 04 — thêm Service bằng nhau",
        "round_04/add_service_equal_analysis.html",
        "Service thêm giá trị gì khi ghép với Throughput và Jain?",
        "Service chủ yếu là shaping giúp thoát starvation sớm; cuối run nó phần lớn trùng hành vi coverage đã được Jain tạo ra.",
    ),
    AnalysisPage(
        "Round 05 — tăng riêng từng thành phần",
        "round_05/three_component_directional_analysis.html",
        "Nếu một reward tăng lên 0,50 thì policy dịch theo hướng nào?",
        "Throughput-heavy tái hiện opportunistic/greedy behavior; Jain-heavy không chuyển thành deterministic fairness; Service-heavy collapse.",
    ),
    AnalysisPage(
        "Round 06 — sáu hướng cục bộ",
        "round_06/experiment_plan.html",
        "Giữ một thành phần và dịch nhẹ trọng số giữa hai thành phần còn lại thì KPI đổi thế nào?",
        "Đang chờ kết quả; đây là vòng tiếp theo cần chạy.",
    ),
)


def build_reward_analysis_index(root: str | Path = "docs/analysis/reward_study") -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        "<tr>"
        f"<td><a href='{page.href}'>{page.title}</a></td>"
        f"<td>{page.question}</td>"
        f"<td>{page.finding}</td>"
        "</tr>"
        for page in PAGES
    )
    html = f"""<!doctype html>
<html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>ScaleMAC-RL Reward Study</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f7fb;color:#172033;margin:0;line-height:1.58}}
main{{max-width:1180px;margin:auto;padding:28px}}header,section{{background:#fff;border:1px solid #dfe5ee;border-radius:16px;padding:22px;margin-bottom:16px}}
header{{background:linear-gradient(135deg,#18264a,#315efb);color:#fff}}table{{width:100%;border-collapse:collapse}}th,td{{padding:11px;border-bottom:1px solid #dfe5ee;text-align:left;vertical-align:top}}th{{background:#f0f3f8}}a{{color:#315efb}}code{{background:#eef1f7;padding:2px 5px;border-radius:5px}}
</style></head><body><main>
<header><h1>ScaleMAC-RL — Mục lục phân tích reward</h1><p>Chuỗi bằng chứng phục vụ nghiên cứu và tổng hợp khóa luận. Các trang được sắp theo thứ tự thí nghiệm.</p></header>
<section><p><a href='../index.html'>← Kho phân tích</a> · <a href='methodology.html'>Phương pháp nghiên cứu reward</a> · <a href='synthesis_round_01_to_05.html'>Tổng hợp Round 01–05</a> · <a href='literature_context.html'>Liên hệ tài liệu nền</a></p></section>
<section><table><thead><tr><th>Trang</th><th>Câu hỏi nghiên cứu</th><th>Kết luận hiện tại</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Phạm vi hiện tại</h2><p>Đây là giai đoạn khám phá môi trường và reward. Chưa sinh dataset trọng số, chưa làm Pareto chính thức và chưa tối ưu surrogate model.</p></section>
</main></body></html>"""
    output = root_path / "index.html"
    output.write_text(html, encoding="utf-8")
    return output
