# ScaleMAC-RL v0.8.5

## Purpose

Start Round 02 of the controlled reward study without changing the 16-feature observation, shared Set Encoder, PPO architecture, environment, or hyperparameters.

## Changes

- adds a four-case throughput/Jain-fairness weight sweep;
- reuses the throughput-only and Jain-only endpoints from Round 01;
- moves experiment analyses to `docs/analysis/reward_study/`;
- explains goodput, Jain fairness, starvation, P99 wait, and maximum UE wait in plain language;
- makes generated HTML show the exact reward formula and actual coefficient of every case;
- writes `pareto_all.csv`, `pareto_safety_filtered.csv`, and `pareto_strict_constraints.csv`;
- keeps `pareto_front.csv` as a backward-compatible alias of the full Pareto frontier;
- removes mixed-checkpoint “best metric” columns from the main reward-weight dataset.
