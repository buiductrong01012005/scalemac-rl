# Superseded experiment plan

The service-only local tuning plan in this historical release was deferred before execution. v0.9.0 replaces it with a symmetric three-component directional experiment.

# ScaleMAC-RL v0.8.9

## Purpose

Round 04 showed that adding `service` with an equal one-third coefficient improved
waiting-time KPIs by only about two slots while the realized service contribution
accounted for more than half of the positive reward.

This release keeps the environment, 16-feature Set Encoder and PPO full-control
architecture unchanged. It adds a small local coefficient study instead of a broad
sweep.

## New reward cases

The non-service portion preserves the proven throughput:Jain ratio `37.5:62.5`.

| Case | Throughput | Jain fairness | Service |
|---|---:|---:|---:|
| `service_005` | 0.35625 | 0.59375 | 0.05 |
| `service_010` | 0.33750 | 0.56250 | 0.10 |
| `service_015` | 0.31875 | 0.53125 | 0.15 |

All other reward components, standalone penalties, reward deltas and Lagrangian
multipliers remain disabled.

## Analysis archive

- Adds the real Round 04 result analysis under
  `docs/analysis/reward_study/round_04/add_service_equal_analysis.html`.
- Adds the Round 05 plan under
  `docs/analysis/reward_study/round_05/experiment_plan.html`.
- Stores only per-round analysis files; no reward dataset or Pareto workflow is run.

## Run

```powershell
pytest -q

python -m scalemac_rl.scripts.run_reward_study `
  --plan .\configs\reward_study\round_05_service_weight_tuning.json
```
