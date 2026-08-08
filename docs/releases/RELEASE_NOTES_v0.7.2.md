# ScaleMAC-RL v0.7.2

- Add one fixed `unified-v1` evaluation protocol for RR, PF, Max-CQI,
  rule-only, hybrid PPO, candidate PPO-only, and full PPO.
- Make `evaluate_ppo` and scheduler attribution build the environment through
  the same protocol and policy-runtime resolver.
- Keep checkpoint training rewards as provenance only; they no longer change
  evaluation reward weights or KPI constraints.
- Freeze one static CQI/demand profile and reuse the same rollout/HARQ seed for
  every scheduler in each comparison row.
- Export protocol, scenario, scheduler-runtime, and checkpoint SHA-256 hashes.
- Record checkpoint observation width and whether the legacy 8-to-10 feature
  compatibility adapter was applied.
- Add `run_unified_evaluation` as the preferred command while retaining
  `run_scheduler_attribution` for backward compatibility.
- Add an automated `verify_evaluation_consistency` command and regression tests
  proving standalone and cross-scheduler evaluation return identical KPI values
  under the same checkpoint and seeds.

This release changes evaluation reproducibility and reporting. It does not
retrain or modify existing checkpoints.
