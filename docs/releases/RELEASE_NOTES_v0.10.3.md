# ScaleMAC-RL v0.10.3

## Purpose

Documentation-only snapshot before Dynamic CQI work begins.

## Changes

- Replaced the outdated v0.8 environment note with a source-verified v0.10.2 environment/MDP reference.
- Documented topology, PRB/Top-K contract, CQI generation and spectral-efficiency table, full-buffer queue behavior, simplified HARQ, all 16 observation features, PPO actions, projector behavior, active T–J–S reward and delay definitions.
- Added explicit `LOCKED`, `FLEX-PLAN`, `FLEX-CONFIG`, `DYNAMIC`, `META-ONLY` and `NOT-MODELED` labels.
- Added `NETWORK_VARIABLE_REFERENCE.csv` for quick parameter lookup.
- Explicitly recorded current realism gaps before Dynamic CQI: static CQI, constant BLER independent of CQI, speed metadata only, no physical slot duration, no packet-arrival process, no QoS/5QI and simplified HARQ.
- Added `*.log` to `.gitignore` so local diagnostic logs are not committed.

## Environment behavior

No environment, PPO, reward, architecture, scheduler, state transition or action logic is changed in v0.10.3.
