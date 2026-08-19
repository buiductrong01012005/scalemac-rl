# ScaleMAC-RL v0.15.0

## Purpose

Start the PPO optimization phase with a controlled training-stability screen before changing observations, reward design, or recurrent architecture.

## Round 14A — PPO training stability

The round is a self-contained 2×2 factorial experiment across three paired seeds (1701, 2701, 3701):

- standard LR schedule (`1e-4 -> 2.5e-5`) + 4 PPO epochs;
- low LR schedule (`5e-5 -> 1.25e-5`) + 4 PPO epochs;
- standard LR schedule + 2 PPO epochs;
- low LR schedule + 2 PPO epochs.

This produces 12 feed-forward PPO cases. Re-running the baseline in the same round avoids cross-session/runtime ambiguity and makes the factor comparison self-contained.

## Controlled variables

The following remain fixed:

- T-J-S reward at equal thirds;
- the existing 16 UE observation features;
- feed-forward shared Set PPO architecture;
- Slow Dynamic CQI;
- CSI reporting every 4 slots with 2-slot delivery delay;
- CQI-to-MCS mapping and true-CQI-dependent BLER;
- HARQ model, action space, Top-K and full-control scheduling;
- clip coefficient 0.1 and target KL 0.02.

## Analysis

The round exports per-seed metrics, per-recipe mean/std, zero-starvation/full-collapse counts, tail KL/clip-fraction diagnostics and descriptive 2×2 factor/interaction effects.

This is a stability screen, not a statistical significance test. Clip coefficient and target-KL tuning are intentionally deferred until the LR/epoch effects are known.

## Repository hygiene

Internal `docs/analysis/`, `docs/research/`, runtime logs and artifacts remain local-only and ignored by Git.
