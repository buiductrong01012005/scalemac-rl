# ScaleMAC-RL v0.5

## Purpose

Train for substantially longer while separating deterministic service safety from learned UE selection.

## Main changes

- Increase the default curriculum budget to 32,768 environment steps per UE stage.
- Use a 128-UE candidate pool and reserve 16 of 64 grants for HARQ or long-waiting UEs.
- Let PPO rank the remaining grants and predict PRB demand.
- Report safety-selected versus learned-selected grants.
- Use repeated held-out validation episodes.
- Update Lagrange multipliers from held-out validation violations as well as rollout violations.
- Roll back to the best feasible stage checkpoint after repeated unsafe validations.
- Save `best_stage_<UE>.pt` and `best_feasible_stage_<UE>.pt` for each curriculum scale.
- Continue each new curriculum stage from the safest available checkpoint.
- Use progressive P99-wait limits of 80, 80, 80, and 50 slots across the default curriculum.
- Keep the official final target at zero starvation and P99 wait no greater than 50 slots.

## Important interpretation

The actor now has a genuine UE-selection role because the default candidate pool (128) is larger than Top-K (64), while only part of Top-K is reserved by deterministic safety logic.

The implementation remains a fast surrogate and is not yet a 5G-LENA result.
