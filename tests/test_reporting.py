from pathlib import Path

from scalemac_rl.reporting import summarize_by_group, write_csv, write_markdown


def test_reports_are_created(tmp_path: Path) -> None:
    rows = [
        {"scheduler": "rr", "score": 0.5},
        {"scheduler": "rr", "score": 0.7},
    ]
    csv_path = tmp_path / "report.csv"
    md_path = tmp_path / "report.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, title="Report", rows=rows)
    summary = summarize_by_group(rows, group_key="scheduler", numeric_fields=["score"])

    assert csv_path.exists()
    assert md_path.exists()
    assert summary[0]["score_mean"] == 0.6
    assert summary[0]["score_std"] > 0.0
