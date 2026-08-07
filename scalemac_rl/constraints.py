from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ServiceConstraints:
    max_starvation_rate: float = 0.0
    max_p99_wait_slots: float = 50.0

    def validate(self) -> None:
        if not 0.0 <= self.max_starvation_rate < 1.0:
            raise ValueError("max_starvation_rate must be in [0, 1)")
        if self.max_p99_wait_slots <= 0.0:
            raise ValueError("max_p99_wait_slots must be positive")

    def excesses(self, *, starvation_rate: float, p99_wait_slots: float) -> tuple[float, float]:
        self.validate()
        starvation_excess = max(0.0, float(starvation_rate) - self.max_starvation_rate)
        wait_excess = max(
            0.0,
            (float(p99_wait_slots) - self.max_p99_wait_slots) / self.max_p99_wait_slots,
        )
        return starvation_excess, wait_excess

    def feasible(self, *, starvation_rate: float, p99_wait_slots: float) -> bool:
        starvation_excess, wait_excess = self.excesses(
            starvation_rate=starvation_rate,
            p99_wait_slots=p99_wait_slots,
        )
        return starvation_excess <= 1e-12 and wait_excess <= 1e-12


@dataclass(slots=True)
class LagrangeController:
    starvation_multiplier: float = 5.0
    wait_multiplier: float = 1.0
    learning_rate: float = 0.10
    max_multiplier: float = 50.0

    def validate(self) -> None:
        if self.starvation_multiplier < 0.0 or self.wait_multiplier < 0.0:
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
    ) -> tuple[float, float]:
        self.validate()
        penalty = (
            self.starvation_multiplier * max(0.0, float(starvation_excess))
            + self.wait_multiplier * max(0.0, float(wait_excess))
        )
        return float(base_reward) - penalty, penalty

    def update(self, *, mean_starvation_excess: float, mean_wait_excess: float) -> None:
        self.validate()
        self.starvation_multiplier = min(
            self.max_multiplier,
            max(
                0.0,
                self.starvation_multiplier
                + self.learning_rate * max(0.0, float(mean_starvation_excess)),
            ),
        )
        self.wait_multiplier = min(
            self.max_multiplier,
            max(
                0.0,
                self.wait_multiplier + self.learning_rate * max(0.0, float(mean_wait_excess)),
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
        )
        for row in materialized
    )
