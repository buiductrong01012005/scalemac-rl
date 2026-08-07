from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence


Row = dict[str, Any]


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1000.0:
            return f"{value:,.2f}"
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV report")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Row], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    selected = list(columns or dict.fromkeys(key for row in rows for key in row.keys()))
    header = "| " + " | ".join(selected) + " |"
    separator = "| " + " | ".join("---" for _ in selected) + " |"
    body = [
        "| "
        + " | ".join(_format_markdown_value(row.get(column, "")) for column in selected)
        + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_markdown(
    path: Path,
    *,
    title: str,
    rows: Sequence[Row],
    description: str = "",
    columns: Sequence[str] | None = None,
    notes: Iterable[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [f"# {title}"]
    if description:
        chunks.extend(["", description])
    chunks.extend(["", markdown_table(rows, columns)])
    notes = list(notes)
    if notes:
        chunks.extend(["", "## Notes", ""])
        chunks.extend(f"- {note}" for note in notes)
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def summarize_by_group(
    rows: Sequence[Row],
    *,
    group_key: str,
    numeric_fields: Sequence[str],
) -> list[Row]:
    groups: dict[Any, list[Row]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)

    summaries: list[Row] = []
    for group, group_rows in groups.items():
        summary: Row = {group_key: group, "runs": len(group_rows)}
        for field in numeric_fields:
            values = [float(row[field]) for row in group_rows]
            summary[f"{field}_mean"] = mean(values)
            summary[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def sibling_with_stem(path: Path, suffix: str, extension: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{extension}")


def markdown_report_path(
    data_path: Path,
    *,
    suffix: str = "",
    docs_dir: Path = Path("docs/reports"),
) -> Path:
    """Return the repository documentation path for a generated Markdown report."""
    return docs_dir / f"{data_path.stem}{suffix}.md"
