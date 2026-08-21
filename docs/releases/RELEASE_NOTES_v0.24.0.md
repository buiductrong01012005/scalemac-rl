# ScaleMAC-RL v0.24.0

## Round 19A — UE-group dimension-wise PPO

This release adds a scale-aware PPO objective that follows the natural action structure of ScaleMAC: each UE owns two continuous decisions, priority and PRB demand.

- `ue_group` ratio mode keeps factorized per-dimension importance ratios.
- PPO clipping is applied per action dimension.
- The two clipped surrogate terms are summed inside each UE, then averaged over active UEs, so actor-loss scale does not grow with the 1,200-UE population.
- Optional adaptive `J_IS` is computed from each 2-D UE-group log-ratio and averaged across UEs, rather than summing log-ratios over the full ~2,400-D joint action.
- Round 19A compares joint PPO, UE-group clipping without J_IS, and UE-group clipping with per-UE adaptive J_IS across seeds 1701/2701/3701.
- Round 19A diagnostics use `effective_common` seeds and the exact final environment-step milestone checkpoint, fixing the analysis ambiguity discovered after Round 18.
