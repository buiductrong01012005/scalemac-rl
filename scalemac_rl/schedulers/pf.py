from __future__ import annotations

import numpy as np

from scalemac_rl.env import CQI, DEMAND, EWMA_THROUGHPUT
from .base import Scheduler


class ProportionalFairScheduler(Scheduler):
    def __init__(self, epsilon: float = 1e-3):
        self.epsilon = epsilon

    def act(self, observation: np.ndarray) -> np.ndarray:
        instant_rate = np.maximum(observation[:, CQI], self.epsilon)
        historical_rate = observation[:, EWMA_THROUGHPUT]
        metric = instant_rate / (historical_rate + self.epsilon)
        metric /= max(float(metric.max(initial=1.0)), self.epsilon)

        action = np.empty((observation.shape[0], 2), dtype=np.float32)
        action[:, 0] = metric.astype(np.float32)
        action[:, 1] = np.clip(0.5 * metric + 0.5 * observation[:, DEMAND], 0.0, 1.0)
        return action
