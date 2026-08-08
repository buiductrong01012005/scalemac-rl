# ScaleMAC-RL v0.8.7

## Purpose

Round 03 confirmed that the 25% throughput / 75% Jain policy still allocates all
64 scheduling positions, but reuses too many of the same UEs. Stochastic Beta
actions rotate UEs broadly, while the deterministic mean-policy remains too flat
and concentrated.

## Changes

- Archive the complete Round 03 analysis under `docs/analysis/reward_study/round_03/`.
- Add configurable Beta action concentration to PPO training.
- Support a linearly scheduled concentration from an exploratory value to a
  lower-noise final value.
- Log priority and PRB-demand concentration in every training row and checkpoint.
- Add Round 04 with one fixed reward and three exploration settings:
  - learned concentration near 20;
  - scheduled concentration 20 → 80;
  - scheduled concentration 20 → 200.
- Automatically run deterministic and stochastic policy diagnostics after all
  Round 04 cases finish.
- Keep environment, 16-feature input, Set Encoder, PPO action scope, reward,
  optimizer, seed, and training budget unchanged.
- Do not generate reward datasets or Pareto files.

## Research question

Can reducing stochastic action noise over training force the deterministic PPO
mean-policy to learn UE rotation, instead of receiving Jain fairness mainly from
exploration noise?
