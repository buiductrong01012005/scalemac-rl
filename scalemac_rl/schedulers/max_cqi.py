from __future__ import annotations

import numpy as np

from scalemac_rl.env import CQI, DEMAND
from .base import Scheduler


class MaxCqiScheduler(Scheduler):
    def act(self, observation: np.ndarray) -> np.ndarray:
        action = np.empty((observation.shape[0], 2), dtype=np.float32)
        action[:, 0] = observation[:, CQI]
        action[:, 1] = np.clip(0.25 + observation[:, DEMAND], 0.0, 1.0)
        return action
