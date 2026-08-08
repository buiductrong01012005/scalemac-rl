# ScaleMAC-RL v0.7.0

This release separates scheduler attribution from safety rules and adds a clean PPO-from-scratch research path.

- expands the per-UE observation from 8 to 10 features with demand-normalized throughput deficit and service-cycle deficit;
- adds dense fairness credit through deficit service, change in Jain fairness, and change in proportional-fair utility;
- keeps throughput as the largest positive reward term while increasing action-level information for fairness;
- adds explicit scheduler modes: `hybrid`, `ppo_only`, and `rule_only`;
- limits the projector to Top-K and exact-PRB feasibility in `ppo_only` mode;
- allows PPO to select all 64 grants from 128 candidates or directly from all 1,200 UEs;
- adds random-initialization training through `train_ppo_from_scratch` with a default budget of 200,192 environment steps and tqdm timing;
- adds progressive fairness targets `0.50 -> 0.55 -> 0.60`;
- adds honest attribution metrics for safety-selected, scheduler-selected, PPO-selected, and rule-selected grants;
- adds `run_scheduler_attribution` to compare RR, PF, Max-CQI, rule-only, hybrid PPO, candidate PPO-only, and full PPO;
- keeps compatibility with legacy 8-feature imitation/PPO checkpoints by zero-initializing the two new input columns;
- source archives remain source-only and never include `artifacts/`.
