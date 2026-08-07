# ScaleMAC-RL v0.4

## Purpose

Close the main v0.3 failure mode: PPO increased throughput while accepting non-zero starvation and excessive P99 waiting time.

## Changes

- Added explicit starvation and P99-wait service constraints.
- Added projected Lagrange multipliers to the PPO training objective.
- Added held-out validation during curriculum training.
- Added feasibility-first checkpoint selection:
  - `latest.pt`;
  - `best_reward.pt`;
  - `best_feasible.pt`;
  - periodic update checkpoints.
- Changed the learned scheduler to process compact candidate tensors rather than all UEs.
- Prioritized pending HARQ and near-starvation UEs in candidate filtering.
- Added candidate coverage and retention diagnostics.
- Added candidate-count performance ablation.
- Added component-level latency profiling for filtering, gathering, tensor conversion, encoder, actor, and projection.
- Added tests for constraints, candidate compaction, retention, and deterministic inference.

## Important limitation

Constraint satisfaction is evaluated in the fast surrogate and does not imply 5G-LENA or real-time deployment compliance.
