# ScaleMAC-RL v0.25.0

## Round 20 — schedule-frequency fairness reward study

This release separates two different notions of fairness in the scheduler reward:

- **Throughput Jain fairness**: fairness of successfully delivered throughput.
- **Schedule-frequency fairness**: fairness of UE selection opportunities, independent of how many bits are successfully delivered after CQI/MCS/BLER effects.

The new scheduling fairness signal maintains cumulative scheduling counts and an EWMA scheduling-rate vector for every UE. Its reward score uses the same 60% cumulative / 40% short-term blend as the existing throughput fairness score.

Two controlled 12-run plans are included:

- Round 20A: schedule-fairness dose / equal-four-way study.
- Round 20B: pairwise interactions between schedule fairness and throughput, throughput fairness, or service.

All Round 20 runs keep the current joint PPO, 1200-UE environment, Service40-era radio realism, 16-feature observation, R256/MB8 sampling, and 98,304-step training budget fixed.
