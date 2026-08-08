# ScaleMAC-RL v0.8.6

## Purpose

Diagnose why a Jain-heavy reward can appear fair during PPO rollouts but collapse
under deterministic Top-K inference, before changing the reward or model.

## Added

- deterministic-versus-stochastic checkpoint evaluation;
- per-slot priority spread and Top-64/65 margin logging;
- near-tie counts around the Top-K cutoff;
- rolling 64-slot selection and successful-delivery coverage;
- per-UE selection/success frequency and concentration metrics;
- an HTML explanation under `docs/analysis/reward_study/round_03`;
- a single diagnostic output folder under `artifacts/runs/reward_study`.

## Workflow correction

Reward study runs no longer build weight datasets or Pareto files automatically.
That future workflow is opt-in through `--build-dataset`; the current phase stays
focused on environment and reward exploration.
