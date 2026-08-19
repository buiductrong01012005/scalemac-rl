# ScaleMAC-RL v0.16.0

## Round 14B — PPO observation feature ablation

This release keeps the selected feed-forward PPO recipe, T–J–S reward and realistic radio environment fixed, and changes only the scheduler observation.

Four paired profiles are evaluated across seeds 1701, 2701 and 3701:

1. Baseline 16 features.
2. Baseline + CSI report age.
3. Baseline + reported-CQI trend between the two latest delivered CSI reports.
4. Baseline + both CSI age and reported-CQI trend.

The two new features are opt-in and therefore preserve all earlier observation/checkpoint behavior by default. The study asks whether explicit temporal context is enough for feed-forward PPO before recurrent memory is required.

Internal experiment analysis remains under `docs/analysis/` and is Git-ignored.
