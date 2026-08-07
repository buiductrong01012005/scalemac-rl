from __future__ import annotations

import numpy as np

from .base import Scheduler


class RoundRobinScheduler(Scheduler):
    def __init__(self, max_selected_ues: int):
        self.max_selected_ues = max_selected_ues
        self.pointer = 0

    def reset(self) -> None:
        self.pointer = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        num_ues = observation.shape[0]
        action = np.zeros((num_ues, 2), dtype=np.float32)
        order = (self.pointer + np.arange(num_ues)) % num_ues
        selected = order[: min(self.max_selected_ues, num_ues)]
        # Strict ordering prevents ties in the projector.
        action[selected, 0] = np.linspace(1.0, 0.5, selected.size, dtype=np.float32)
        action[:, 1] = 1.0
        self.pointer = int((self.pointer + selected.size) % num_ues)
        return action
