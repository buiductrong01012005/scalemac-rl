# ScaleMAC-RL v0.7.1

- Fix scheduler-attribution loading of legacy v0.6.x hybrid checkpoints.
- Preserve legacy reward weights when `reward_deficit_service_weight` is absent.
- Add regression tests for old and current checkpoint metadata.

This patch does not change trained weights, the simulator, or evaluation KPI definitions.
