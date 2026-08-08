from __future__ import annotations

import numpy as np

from .base import Scheduler


class RuleOnlyScheduler(Scheduler):
    """Neutral action source for the rule-only projector mode.

    UE selection is performed entirely by the environment's rule-only projector
    (HARQ first, then oldest successful-delivery wait). Equal PRB-demand scores
    produce an approximately equal allocation among selected UEs.
    """

    def act(self, observation: np.ndarray) -> np.ndarray:
        action = np.zeros((observation.shape[0], 2), dtype=np.float32)
        action[:, 1] = 1.0
        return action
