from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POSITIVE_COMPONENTS: tuple[str, ...] = (
    "throughput",
    "fairness",
    "service",
    "deficit_service",
    "pf_utility",
    "low_throughput",
    "urgency_service",
)
DELTA_COMPONENTS: tuple[str, ...] = ("fairness", "pf_utility")
PENALTY_COMPONENTS: tuple[str, ...] = (
    "starvation",
    "deadline_risk",
    "max_wait_risk",
    "population_wait",
)

POSITIVE_CLI = {
    "throughput": "--reward-throughput-weight",
    "fairness": "--reward-fairness-weight",
    "service": "--reward-service-weight",
    "deficit_service": "--reward-deficit-service-weight",
    "pf_utility": "--reward-pf-utility-weight",
    "low_throughput": "--reward-low-throughput-weight",
    "urgency_service": "--reward-urgency-service-weight",
}
DELTA_CLI = {
    "fairness": "--reward-fairness-delta-weight",
    "pf_utility": "--reward-pf-utility-delta-weight",
}
PENALTY_CLI = {
    "starvation": "--reward-starvation-penalty-weight",
    "deadline_risk": "--deadline-risk-penalty-weight",
    "max_wait_risk": "--max-wait-risk-penalty-weight",
    "population_wait": "--population-wait-penalty-weight",
}


def _complete_weights(
    raw: Mapping[str, Any] | None,
    names: Sequence[str],
    *,
    field_name: str,
) -> dict[str, float]:
    raw = raw or {}
    unknown = sorted(set(raw) - set(names))
    if unknown:
        raise ValueError(f"unknown {field_name}: {', '.join(unknown)}")
    values = {name: float(raw.get(name, 0.0)) for name in names}
    if any(value < 0.0 for value in values.values()):
        raise ValueError(f"{field_name} must be non-negative")
    return values


@dataclass(frozen=True, slots=True)
class RewardCase:
    case_id: str
    label: str
    hypothesis: str
    positive_scale: float
    positive_weights: dict[str, float]
    delta_weights: dict[str, float]
    penalty_weights: dict[str, float]
    common_overrides: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RewardCase":
        case_id = str(payload.get("id", "")).strip()
        if not case_id:
            raise ValueError("reward case id is required")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in case_id):
            raise ValueError(
                f"reward case id must use lowercase letters, numbers, '_' or '-': {case_id}"
            )
        positive_scale = float(payload.get("positive_scale", 1.0))
        if positive_scale < 0.0:
            raise ValueError("positive_scale must be non-negative")
        positive = _complete_weights(
            payload.get("positive_weights"),
            POSITIVE_COMPONENTS,
            field_name="positive reward components",
        )
        positive_sum = sum(positive.values())
        if abs(positive_sum - 1.0) > 1e-6:
            raise ValueError(
                f"positive_weights for {case_id} must sum to 1; got {positive_sum:.9f}"
            )
        return cls(
            case_id=case_id,
            label=str(payload.get("label", case_id)),
            hypothesis=str(payload.get("hypothesis", "")),
            positive_scale=positive_scale,
            positive_weights=positive,
            delta_weights=_complete_weights(
                payload.get("delta_weights"),
                DELTA_COMPONENTS,
                field_name="delta reward components",
            ),
            penalty_weights=_complete_weights(
                payload.get("penalty_weights"),
                PENALTY_COMPONENTS,
                field_name="penalty reward components",
            ),
            common_overrides=dict(payload.get("common_overrides", {})),
        )

    def actual_coefficients(self) -> dict[str, float]:
        values = {
            f"coef_{name}": self.positive_scale * weight
            for name, weight in self.positive_weights.items()
        }
        values.update(
            {f"coef_{name}_delta": weight for name, weight in self.delta_weights.items()}
        )
        values.update(
            {f"coef_{name}_penalty": weight for name, weight in self.penalty_weights.items()}
        )
        return values

    def cli_args(self) -> list[str]:
        args = ["--reward-positive-scale", str(self.positive_scale)]
        for name in POSITIVE_COMPONENTS:
            args.extend([POSITIVE_CLI[name], str(self.positive_weights[name])])
        for name in DELTA_COMPONENTS:
            args.extend([DELTA_CLI[name], str(self.delta_weights[name])])
        for name in PENALTY_COMPONENTS:
            args.extend([PENALTY_CLI[name], str(self.penalty_weights[name])])
        return args

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "label": self.label,
            "hypothesis": self.hypothesis,
            "positive_scale": self.positive_scale,
            "positive_weights": self.positive_weights,
            "delta_weights": self.delta_weights,
            "penalty_weights": self.penalty_weights,
            "common_overrides": self.common_overrides,
            "actual_coefficients": self.actual_coefficients(),
        }


@dataclass(frozen=True, slots=True)
class RewardStudyPlan:
    study_id: str
    round_id: str
    description: str
    common: dict[str, Any]
    analysis: dict[str, Any]
    cases: tuple[RewardCase, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "RewardStudyPlan":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        cases = tuple(RewardCase.from_dict(case) for case in payload.get("cases", []))
        if not cases:
            raise ValueError("reward study plan must contain at least one case")
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("reward study case ids must be unique")
        round_id = str(payload.get("round_id", "")).strip()
        if not round_id:
            raise ValueError("round_id is required")
        return cls(
            study_id=str(payload.get("study_id", "reward_study")),
            round_id=round_id,
            description=str(payload.get("description", "")),
            common=dict(payload.get("common", {})),
            analysis=dict(payload.get("analysis", {})),
            cases=cases,
        )

    def selected_cases(self, requested: Iterable[str] | None = None) -> tuple[RewardCase, ...]:
        if requested is None:
            return self.cases
        requested_set = {item.strip() for item in requested if item.strip()}
        unknown = requested_set - {case.case_id for case in self.cases}
        if unknown:
            raise ValueError(f"unknown reward cases: {', '.join(sorted(unknown))}")
        return tuple(case for case in self.cases if case.case_id in requested_set)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        raw = row.get(key, default)
        if raw in {None, ""}:
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def pareto_front_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
    tolerance: float = 1e-12,
) -> set[int]:
    """Return non-dominated row indices for mixed maximize/minimize objectives."""
    front: set[int] = set()
    for i, candidate in enumerate(rows):
        dominated = False
        for j, challenger in enumerate(rows):
            if i == j:
                continue
            no_worse = all(
                safe_float(challenger, key) >= safe_float(candidate, key) - tolerance
                for key in maximize
            ) and all(
                safe_float(challenger, key) <= safe_float(candidate, key) + tolerance
                for key in minimize
            )
            strictly_better = any(
                safe_float(challenger, key) > safe_float(candidate, key) + tolerance
                for key in maximize
            ) or any(
                safe_float(challenger, key) < safe_float(candidate, key) - tolerance
                for key in minimize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front
