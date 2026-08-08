from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ServiceConstraints:
    """Validation limits for starvation, tail delay, and fairness.

    ``starvation_rate`` is based on time since the last successful delivery.
    ``max_wait_slots`` is the single worst UE service gap in an episode.
    """

    max_starvation_rate: float = 0.0
    max_p99_wait_slots: float = 50.0
    min_jain_fairness: float = 0.60
    max_wait_slots: float = 60.0

    def validate(self) -> None:
        if not 0.0 <= self.max_starvation_rate < 1.0:
            raise ValueError("max_starvation_rate must be in [0, 1)")
        if self.max_p99_wait_slots <= 0.0:
            raise ValueError("max_p99_wait_slots must be positive")
        if not 0.0 <= self.min_jain_fairness <= 1.0:
            raise ValueError("min_jain_fairness must be in [0, 1]")
        if self.max_wait_slots <= 0.0:
            raise ValueError("max_wait_slots must be positive")

    def excesses(
        self,
        *,
        starvation_rate: float,
        p99_wait_slots: float,
    ) -> tuple[float, float]:
        """Backward-compatible starvation and P99 excesses."""
        self.validate()
        starvation_excess = max(
            0.0, float(starvation_rate) - self.max_starvation_rate
        )
        wait_excess = max(
            0.0,
            (float(p99_wait_slots) - self.max_p99_wait_slots)
            / self.max_p99_wait_slots,
        )
        return starvation_excess, wait_excess

    def all_excesses(
        self,
        *,
        starvation_rate: float,
        p99_wait_slots: float,
        jain_fairness: float,
        max_wait_slots: float,
    ) -> tuple[float, float, float, float]:
        starvation_excess, p99_excess = self.excesses(
            starvation_rate=starvation_rate,
            p99_wait_slots=p99_wait_slots,
        )
        fairness_excess = max(
            0.0,
            (self.min_jain_fairness - float(jain_fairness))
            / max(self.min_jain_fairness, 1e-9),
        )
        max_wait_excess = max(
            0.0,
            (float(max_wait_slots) - self.max_wait_slots)
            / self.max_wait_slots,
        )
        return starvation_excess, p99_excess, fairness_excess, max_wait_excess

    def feasible(
        self,
        *,
        starvation_rate: float,
        p99_wait_slots: float,
        jain_fairness: float | None = None,
        max_wait_slots: float | None = None,
    ) -> bool:
        self.validate()
        if jain_fairness is None or max_wait_slots is None:
            starvation_excess, wait_excess = self.excesses(
                starvation_rate=starvation_rate,
                p99_wait_slots=p99_wait_slots,
            )
            return starvation_excess <= 1e-12 and wait_excess <= 1e-12
        return all(
            excess <= 1e-12
            for excess in self.all_excesses(
                starvation_rate=starvation_rate,
                p99_wait_slots=p99_wait_slots,
                jain_fairness=jain_fairness,
                max_wait_slots=max_wait_slots,
            )
        )


@dataclass(slots=True)
class LagrangeController:
    starvation_multiplier: float = 5.0
    wait_multiplier: float = 1.0
    fairness_multiplier: float = 1.0
    max_wait_multiplier: float = 1.0
    learning_rate: float = 0.10
    max_multiplier: float = 50.0

    def validate(self) -> None:
        multipliers = (
            self.starvation_multiplier,
            self.wait_multiplier,
            self.fairness_multiplier,
            self.max_wait_multiplier,
        )
        if any(value < 0.0 for value in multipliers):
            raise ValueError("Lagrange multipliers must be non-negative")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if self.max_multiplier <= 0.0:
            raise ValueError("max_multiplier must be positive")

    def adjusted_reward(
        self,
        base_reward: float,
        *,
        starvation_excess: float,
        wait_excess: float,
        fairness_excess: float = 0.0,
        max_wait_excess: float = 0.0,
    ) -> tuple[float, float]:
        self.validate()
        penalty = (
            self.starvation_multiplier * max(0.0, float(starvation_excess))
            + self.wait_multiplier * max(0.0, float(wait_excess))
            + self.fairness_multiplier * max(0.0, float(fairness_excess))
            + self.max_wait_multiplier * max(0.0, float(max_wait_excess))
        )
        return float(base_reward) - penalty, penalty

    def update(
        self,
        *,
        mean_starvation_excess: float,
        mean_wait_excess: float,
        mean_fairness_excess: float = 0.0,
        mean_max_wait_excess: float = 0.0,
    ) -> None:
        self.validate()
        updates = (
            ("starvation_multiplier", mean_starvation_excess),
            ("wait_multiplier", mean_wait_excess),
            ("fairness_multiplier", mean_fairness_excess),
            ("max_wait_multiplier", mean_max_wait_excess),
        )
        for attribute, excess in updates:
            current = float(getattr(self, attribute))
            setattr(
                self,
                attribute,
                min(
                    self.max_multiplier,
                    max(0.0, current + self.learning_rate * max(0.0, float(excess))),
                ),
            )


def validation_feasible(
    rows: Iterable[Mapping[str, object]],
    constraints: ServiceConstraints,
) -> bool:
    materialized = list(rows)
    if not materialized:
        return False
    return all(
        constraints.feasible(
            starvation_rate=float(row["max_starvation_rate"]),
            p99_wait_slots=float(row["max_p99_wait_slots"]),
            jain_fairness=float(row.get("final_jain_fairness", 1.0)),
            max_wait_slots=float(
                row.get("max_wait_slots", row["max_p99_wait_slots"])
            ),
        )
        for row in materialized
    )
