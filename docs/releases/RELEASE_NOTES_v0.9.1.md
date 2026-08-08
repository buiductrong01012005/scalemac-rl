# ScaleMAC-RL v0.9.1

## Purpose

Continue controlled reward exploration after the three-component directional experiment.

## Findings archived

- The equal-third Throughput/Jain/Service baseline was the only stable policy in Round 05.
- Throughput-heavy briefly approached a safe state, then collapsed back to concentrated scheduling.
- Jain-heavy did not improve deterministic fairness and remained unsafe.
- Service-heavy collapsed, confirming that the current Service score is useful as shaping but not as a dominant objective.
- The complete plain-language analysis is stored under `docs/analysis/reward_study/round_05/`.

## New experiment

Round 06 fixes Service at one third and moves only a small amount of weight between Throughput and Jain:

- 0.40 Throughput / 0.2667 Jain / 0.3333 Service
- 0.2667 Throughput / 0.40 Jain / 0.3333 Service

All other rewards, penalties, environment settings and PPO hyperparameters remain unchanged.
