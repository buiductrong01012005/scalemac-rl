from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value in {None, ""} else float(value)
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    return int(round(_float(row, key, float(default))))


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _best_validation(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            _float(row, "final_target_total_constraint_excess", float("inf")),
            _float(row, "total_constraint_excess", float("inf")),
            -_float(row, "mean_final_target_reward", float("-inf")),
        ),
    )


def _manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    preferred = {
        "best_tradeoff",
        "best_feasible",
        "best_lowest_violation",
        "best_reward",
        "latest",
    }
    selected = [row for row in rows if row.get("tag") in preferred]
    return selected or rows[-5:]


def build_report(
    *,
    training_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
    title: str,
) -> str:
    lines: list[str] = [f"# {title}", ""]
    lines.extend(
        [
            "## 1. Câu hỏi nghiên cứu",
            "",
            "Policy PPO-only có thể tự lập lịch toàn bộ UE và tạo trade-off tốt hơn Rule-only mà không dùng candidate filter hay safety rule hay không?",
            "",
        ]
    )

    if training_rows:
        first = training_rows[0]
        last = training_rows[-1]
        total_steps = _int(last, "global_env_steps")
        lines.extend(
            [
                "## 2. Tóm tắt training",
                "",
                f"- Updates: **{len(training_rows):,}**",
                f"- Environment steps: **{total_steps:,}**",
                f"- UE ở stage cuối: **{_int(last, 'num_ues'):,}**",
                f"- Learning rate cuối: **{_float(last, 'learning_rate'):.6g}**",
                f"- Entropy coefficient cuối: **{_float(last, 'entropy_coef'):.6g}**",
                "",
                "| Điểm | Goodput | Jain fairness | Starvation | P99 | Max wait | Final-target reward |",
                "|---|---:|---:|---:|---:|---:|---:|",
                (
                    f"| Đầu | {_float(first, 'mean_goodput_bits_per_slot'):,.1f} | "
                    f"{_fmt(_float(first, 'mean_jain_fairness'))} | "
                    f"{_fmt(_float(first, 'mean_starvation_rate'))} | "
                    f"{_float(first, 'mean_p99_wait_slots'):.1f} | "
                    f"{_float(first, 'mean_max_wait_slots'):.1f} | "
                    f"{_fmt(_float(first, 'mean_final_target_reward'))} |"
                ),
                (
                    f"| Cuối | {_float(last, 'mean_goodput_bits_per_slot'):,.1f} | "
                    f"{_fmt(_float(last, 'mean_jain_fairness'))} | "
                    f"{_fmt(_float(last, 'mean_starvation_rate'))} | "
                    f"{_float(last, 'mean_p99_wait_slots'):.1f} | "
                    f"{_float(last, 'mean_max_wait_slots'):.1f} | "
                    f"{_fmt(_float(last, 'mean_final_target_reward'))} |"
                ),
                "",
            ]
        )

    best = _best_validation(validation_rows)
    if best is not None:
        lines.extend(
            [
                "## 3. Validation tốt nhất theo fixed target",
                "",
                f"- Update: **{_int(best, 'update')}**",
                f"- Steps: **{_int(best, 'global_env_steps'):,}**",
                f"- Goodput: **{_float(best, 'mean_goodput_bits_per_slot'):,.1f} bit/slot**",
                f"- Jain fairness: **{_fmt(_float(best, 'mean_jain_fairness'))}**",
                f"- Worst starvation: **{_fmt(_float(best, 'worst_starvation_rate'))}**",
                f"- Worst P99: **{_float(best, 'worst_p99_wait_slots'):.1f} slot**",
                f"- Worst max wait: **{_float(best, 'worst_max_wait_slots'):.1f} slot**",
                f"- Feasible: **{best.get('constraint_feasible', '')}**",
                "",
            ]
        )

    selected_manifest = _manifest_rows(manifest_rows)
    if selected_manifest:
        lines.extend(
            [
                "## 4. Checkpoint provenance",
                "",
                "| Tag | Update | Steps | Lý do | Final excess | Final reward |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for row in selected_manifest:
            lines.append(
                f"| {row.get('tag', '')} | {_int(row, 'update')} | "
                f"{_int(row, 'global_env_steps'):,} | {row.get('selection_reason', '')} | "
                f"{_fmt(sum(_float(row, key) for key in ('starvation_excess', 'max_wait_excess', 'p99_wait_excess', 'fairness_excess')))} | "
                f"{_fmt(_float(row, 'final_target_reward'))} |"
            )
        lines.append("")

    if evaluation_rows:
        lines.extend(
            [
                "## 5. So sánh scheduler",
                "",
                "| Method | Goodput | Jain fairness | Starvation | Max P99 | Max wait | Balanced | Worst gap | Pareto dominated |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in evaluation_rows:
            lines.append(
                f"| {row.get('method', '')} | {_float(row, 'mean_goodput_bits_per_slot'):,.1f} | "
                f"{_fmt(_float(row, 'final_jain_fairness'))} | "
                f"{_fmt(_float(row, 'mean_starvation_rate'))} | "
                f"{_float(row, 'max_p99_wait_slots'):.1f} | "
                f"{_float(row, 'max_wait_slots'):.1f} | "
                f"{_fmt(_float(row, 'balanced_score'))} | "
                f"{_fmt(_float(row, 'worst_kpi_gap'))} | "
                f"{row.get('pareto_dominated', '')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. Reward decomposition cần kiểm tra",
            "",
            "- throughput component có lấn át PF/P10 không?",
            "- low-throughput score có tăng cùng Jain fairness không?",
            "- population wait penalty có giảm trước khi P99 vượt target không?",
            "- dual multiplier tăng có thực sự đổi hành vi policy không?",
            "- entropy có collapse trước khi policy học được service cycle không?",
            "",
            "## 7. Kết luận nghiên cứu",
            "",
            "Điền sau khi đối chiếu learning curve, validation, unified evaluation và baseline. Không kết luận chỉ từ reward.",
            "",
            "## 8. Thay đổi duy nhất cho vòng tiếp theo",
            "",
            "Chọn đúng một nhóm: input, reward, PPO hyperparameter hoặc architecture. Không đổi đồng thời nhiều nhóm.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a research Markdown report from full-control PPO CSV outputs"
    )
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, default=None)
    parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    parser.add_argument("--evaluation", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="ScaleMAC-RL Full-Control PPO Research Report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/full_control_v2_research_report.md"),
    )
    args = parser.parse_args()

    training_rows = _read_csv(args.training)
    if not training_rows:
        parser.error(f"training CSV is missing or empty: {args.training}")
    report = build_report(
        training_rows=training_rows,
        validation_rows=_read_csv(args.validation_summary),
        manifest_rows=_read_csv(args.checkpoint_manifest),
        evaluation_rows=_read_csv(args.evaluation),
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
