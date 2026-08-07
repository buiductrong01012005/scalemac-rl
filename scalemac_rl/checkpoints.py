from __future__ import annotations

from pathlib import Path


def require_checkpoint(path: Path) -> Path:
    """Return an existing checkpoint path or raise a clear, actionable error."""
    path = Path(path)
    if path.is_file():
        return path

    alternatives = [path.with_name("best_reward.pt"), path.with_name("latest.pt")]
    existing = [candidate for candidate in alternatives if candidate.is_file()]
    if path.name == "best_feasible.pt":
        reason = (
            "No best_feasible.pt was created because no final-stage validation "
            "checkpoint satisfied all configured service constraints."
        )
    else:
        reason = f"Checkpoint does not exist: {path}"

    if existing:
        options = ", ".join(str(candidate) for candidate in existing)
        reason += (
            f" Available diagnostic checkpoint(s): {options}. "
            "These are not certified feasible unless their evaluation says so."
        )
    raise FileNotFoundError(reason)
