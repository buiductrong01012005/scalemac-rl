# ScaleMAC-RL v0.8.4

## Reward-study framework

- stores the v0.8.3 full-control analysis under `docs/research/reward_study/baselines/`;
- adds a controlled component-screen round and a cumulative equal-family round;
- keeps 16 features/UE, the shared Set Encoder, 64-dimensional embeddings, and PPO full control unchanged;
- disables Lagrangian training penalties during reward attribution runs while continuing to measure official constraints;
- exposes the starvation penalty and a positive-reward scale as explicit experiment parameters;
- stores every case in `artifacts/runs/reward_study/<round>/<case>/`;
- builds a reusable reward-weight dataset, learning trajectory dataset, and Pareto front;
- generates HTML reports under both the run folder and `docs/research/reward_study/generated/`;
- adds a query command for selecting past weights by KPI requirements;
- adds a simplex sweep-plan generator for later weight tuning.

## Scientific intent

The release separates three questions that were previously mixed together:

1. Can PPO learn from an individual reward signal?
2. What changes when objective families are added with comparable coefficients?
3. Which weight combinations are non-dominated under real KPIs?

No new policy architecture is introduced in this release.
