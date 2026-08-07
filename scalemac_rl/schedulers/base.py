from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class Scheduler(ABC):
    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return a [num_ues, 2] array of priority and demand scores."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset scheduler state between episodes."""
