from __future__ import annotations

from pathlib import Path


def test_internal_analysis_outputs_are_gitignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "docs/analysis/" in gitignore
    assert "*.log" in gitignore


def test_public_test_suite_does_not_require_internal_analysis_archive() -> None:
    for path in Path("tests").glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        content = path.read_text(encoding="utf-8")
        assert 'Path("docs/analysis/' not in content
        assert "Path('docs/analysis/" not in content
