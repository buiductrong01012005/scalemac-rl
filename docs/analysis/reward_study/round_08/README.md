# Round 08 — Multi-seed confirmation

Round 08 does not add another reward component. It confirms whether the two retained
Round 07 candidates repeat their advantages across three common seeds.

## Profiles

1. `tjs_equal`: baseline `1/3 Throughput + 1/3 Jain + 1/3 Service`.
2. `urgency_hold_throughput`: balanced candidate `0.25 T + 0.20 J + 0.20 S + 0.35 U`.
3. `deficit_group_ts`: delay candidate `0.30 T + 0.10 J + 0.30 S + 0.30 D`.

Each profile is run with seeds `1701`, `2701`, and `3701`, giving nine cases. For
each case the training seed, static-profile seed, and validation seed are the same.
The environment, PPO architecture, hyperparameters, step budget and deterministic
validation protocol are otherwise unchanged.

After training, open `multiseed_confirmation_analysis.html` first. The report also
exports per-seed metrics, mean/std by profile, paired deltas, trajectory data, and a
stability matrix.
