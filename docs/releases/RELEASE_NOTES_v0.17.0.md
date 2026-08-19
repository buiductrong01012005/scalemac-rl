# ScaleMAC-RL v0.17.0

## Round 14C — controlled feature initialization + oracle sanity

- Adds baseline-compatible initialization for feed-forward observation ablations.
- The original 16 input columns and all downstream weights exactly match the paired baseline; added feature columns start at zero.
- Restores the post-baseline-initialization Torch RNG state so step-0 policy and subsequent sampling start from the same paired state.
- Screens baseline, CSI age, and reported-CQI trend across seeds 1701/2701/3701.
- Adds a current-state privileged service-aware oracle and PF/PPO sanity evaluation to determine whether difficult seeds remain service-feasible under the same Dynamic-CQI + delayed-CSI + MCS/BLER environment.
- The oracle has true current CQI/service-wait access but no future knowledge; it is a diagnostic upper-reference, not a deployable scheduler.
- Reward, PPO recipe, radio environment, CSI, MCS, BLER and HARQ behavior are otherwise unchanged.
