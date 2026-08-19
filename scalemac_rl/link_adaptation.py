from __future__ import annotations

import math
import numpy as np

# 3GPP TS 38.214, CQI Table 5.2.2.1-2 (4-bit CQI Table 1).
CQI_TABLE1_EFFICIENCY = np.asarray(
    [
        0.1523,
        0.2344,
        0.3770,
        0.6016,
        0.8770,
        1.1758,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,
    ],
    dtype=np.float64,
)

# 3GPP TS 38.214, PDSCH MCS Table 5.1.3.1-1. Reserved entries are excluded.
MCS_TABLE1_INDEX = np.arange(29, dtype=np.int16)
MCS_TABLE1_MOD_ORDER = np.asarray(
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    dtype=np.int16,
)
MCS_TABLE1_CODE_RATE_X1024 = np.asarray(
    [120, 157, 193, 251, 308, 379, 449, 526, 602, 679, 340, 378, 434, 490, 553, 616, 658, 438, 466, 517, 567, 616, 666, 719, 772, 822, 873, 910, 948],
    dtype=np.float64,
)
MCS_TABLE1_EFFICIENCY = np.asarray(
    [
        0.2344, 0.3066, 0.3770, 0.4902, 0.6016, 0.7402, 0.8770, 1.0273, 1.1758,
        1.3262, 1.3281, 1.4766, 1.6953, 1.9141, 2.1602, 2.4063, 2.5703, 2.5664,
        2.7305, 3.0293, 3.3223, 3.6094, 3.9023, 4.2129, 4.5234, 4.8164, 5.1152,
        5.3320, 5.5547,
    ],
    dtype=np.float64,
)

# CQI and MCS alphabets are tiny and fixed, so precompute the two mappings used
# in every environment step instead of scanning the tables per UE.
_CQI_TO_MCS_TABLE1 = np.asarray(
    [
        (np.flatnonzero(MCS_TABLE1_EFFICIENCY <= eff + 1e-12)[-1]
         if np.any(MCS_TABLE1_EFFICIENCY <= eff + 1e-12) else 0)
        for eff in CQI_TABLE1_EFFICIENCY
    ],
    dtype=np.int16,
)
_MCS_TO_REQUIRED_CQI = np.asarray(
    [
        (np.flatnonzero(CQI_TABLE1_EFFICIENCY >= eff - 1e-12)[0] + 1
         if np.any(CQI_TABLE1_EFFICIENCY >= eff - 1e-12) else 15)
        for eff in MCS_TABLE1_EFFICIENCY
    ],
    dtype=np.int16,
)


def select_mcs_from_reported_cqi(
    reported_cqi: np.ndarray | int,
    *,
    cqi_backoff: int = 0,
) -> np.ndarray:
    """Map reported CQI to MCS Table-1 index using a spectral-efficiency ceiling.

    The selected MCS is the highest non-reserved MCS whose tabulated spectral
    efficiency does not exceed the (optionally backed-off) CQI efficiency.
    CQI=1 has no Table-1 MCS below its efficiency, so MCS 0 is used and the BLER
    model naturally marks it as more fragile.
    """
    cqi = np.asarray(reported_cqi, dtype=np.int16)
    effective_cqi = np.clip(cqi - int(cqi_backoff), 1, 15)
    return _CQI_TO_MCS_TABLE1[effective_cqi - 1]


def required_cqi_for_mcs(mcs_index: np.ndarray | int) -> np.ndarray:
    """Return the minimum CQI index whose Table-1 efficiency supports an MCS."""
    indices = np.asarray(mcs_index, dtype=np.int16)
    return _MCS_TO_REQUIRED_CQI[indices]


def bler_probability_from_cqi_mismatch(
    *,
    true_cqi: np.ndarray | int,
    mcs_index: np.ndarray | int,
    target_bler: float = 0.10,
    mismatch_slope: float = 1.5,
) -> np.ndarray:
    """NR-inspired smooth BLER abstraction from CQI/MCS mismatch.

    This is intentionally *not* a 3GPP link-level BLER curve. 38.214 provides
    CQI/MCS tables, while exact BLER depends on SINR, coding, TBS, receiver and
    channel realization. We anchor BLER to ``target_bler`` when the true CQI is
    exactly the minimum CQI supporting the selected MCS, then change the log-odds
    by ``mismatch_slope`` per CQI-index mismatch.
    """
    if not 0.0 < target_bler < 1.0:
        raise ValueError("target_bler must be in (0, 1) for CQI/MCS BLER mode")
    if mismatch_slope <= 0.0:
        raise ValueError("mismatch_slope must be positive")
    true = np.asarray(true_cqi, dtype=np.float64)
    required = required_cqi_for_mcs(mcs_index).astype(np.float64)
    mismatch = required - true
    logit_target = math.log(target_bler / (1.0 - target_bler))
    logits = np.clip(logit_target + float(mismatch_slope) * mismatch, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-logits))


def mcs_efficiency(mcs_index: np.ndarray | int) -> np.ndarray:
    return MCS_TABLE1_EFFICIENCY[np.asarray(mcs_index, dtype=np.int16)]


def mcs_modulation_order(mcs_index: np.ndarray | int) -> np.ndarray:
    return MCS_TABLE1_MOD_ORDER[np.asarray(mcs_index, dtype=np.int16)]
