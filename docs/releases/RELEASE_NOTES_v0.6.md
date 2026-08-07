# ScaleMAC-RL v0.6

## Goal

Measure the current scheduler's in-scenario performance ceiling by training much
longer on one frozen 1,200-UE CQI/demand profile. This is intentionally an
upper-bound/overfitting experiment, not a generalization claim.

## Changes

- Added `python -m scalemac_rl.scripts.train_single_seed`.
- Default upper-bound budget: 524,288 environment steps, one worker, one 1,200-UE stage.
- Reuses one static CQI/demand profile while retaining stochastic HARQ outcomes.
- Supports resuming full PPO checkpoints, including optimizer and Lagrange state.
- Fills all 16 safety grants using mandatory HARQ and oldest-waiting UEs.
- Keeps 48 learned grants from a 128-UE candidate pool.
- Adds dense tail-delay risk shaping before the P99 deadline is violated.
- Tightens the final P99 target through 80, 65, 55, and 50 slots.
- Saves `best_lowest_violation.pt`, `best_reward.pt`, and `best_feasible.pt` separately.
- Evaluation and paired-evaluation scripts reproduce frozen-profile settings from checkpoint metadata.
- Source release remains source-only and contains no `artifacts/` directory.

## Interpretation boundary

Results from the training seed show optimization capacity on a known scenario.
A separate multi-seed and OOD phase is still required before claiming robustness.
