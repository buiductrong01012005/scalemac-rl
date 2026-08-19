# ScaleMAC-RL v0.14.0

## Purpose

Add the first controlled feed-forward PPO versus recurrent PPO checkpoint under the selected stale-CSI Link Adaptation environment.

## What changes

- Add `RecurrentSharedSetActorCritic` with one shared GRUCell applied independently to every UE.
- Preserve permutation equivariance by sharing encoder, GRU, actor and critic parameters across UEs.
- Add truncated BPTT training for recurrent PPO with configurable sequence length and recurrent minibatch size.
- Add recurrent deterministic evaluation and recurrent checkpoint loading.
- Add Round 13 with paired PPO/RPPO cases on seeds 1701, 2701 and 3701.
- Export per-seed and mean/std architecture comparisons after the round finishes.

## Controlled variables

Round 13 keeps the following fixed across PPO and RPPO:

- 1 gNB / 1 cell / 1200 UEs / 273 PRBs.
- Full-control PPO-only scheduling, Top-K 64, no forced HARQ scheduling rule.
- T-J-S reward with equal thirds.
- Slow Dynamic CQI (`rho=0.97`, innovation std `0.35`).
- Periodic CSI every 4 slots with a 2-slot delivery delay and no measurement noise.
- CQI-to-MCS Link Adaptation and true-CQI-dependent BLER.
- PPO objective coefficients, learning-rate schedule, entropy schedule and training budget.

The only research variable is temporal policy memory. The recurrent model uses hidden size 64 and truncated-BPTT sequences of 16 slots.

## Scope

The recurrent implementation currently requires `candidate_mode=all`, which preserves a stable UE-to-hidden-state identity. This matches the current 1200-UE full-control experiments. Dynamic candidate filtering should not be combined with recurrent memory until hidden states are explicitly indexed/scattered by UE id.

## Repository hygiene

`docs/analysis/`, `docs/research/`, runtime logs and artifacts are local-only and ignored by Git. Public release notes remain tracked.
