# ScaleMAC-RL v0.13.1

## Purpose

Restore exact legacy-PHY arithmetic after the Round 12 regression check exposed a precision drift.

## Fix

- `legacy_fixed_bler` again keeps the pre-v0.13 float32 CQI-efficiency arithmetic path.
- The v0.13.0 implementation had cast legacy CQI efficiency to float64 before computing attempted bits.
- That changed reward values only at tiny numerical scale, but PPO amplified the difference over training and the legacy reference trajectory collapsed.
- `cqi_mcs_bler` behavior is unchanged.
- Added a regression test that pins the pre-v0.13 legacy attempted-bit calculation.

## Experiment implication

Only `legacy_delayed_csi_reference` needs to be rerun. The three completed MCS/BLER cases from Round 12 remain valid because the fix is isolated to `legacy_fixed_bler`.
