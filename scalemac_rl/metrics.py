from __future__ import annotations

import numpy as np


def jain_fairness(values: np.ndarray) -> float:
    """Return Jain's fairness index; zero when no UE has received service."""
    x = np.asarray(values, dtype=np.float64)
    numerator = float(np.square(x.sum()))
    denominator = float(x.size * np.square(x).sum())
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, percentile))
