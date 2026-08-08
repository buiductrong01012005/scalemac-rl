# ScaleMAC-RL v0.7.6

## PPO-only actor with a small safety guard

- Generalized the fixed-weight rule/PPO split tool to accept any actor checkpoint.
- Added `run_ppo_guard_ablation` for reserves 0, 4, 8, 12, and 16 using one frozen
  PPO-only actor.
- Added `train_ppo_guarded` to fine-tune a selected small reserve from the PPO-only
  checkpoint rather than from the rule-dependent hybrid checkpoint.
- Added an architecture experiment plan for recurrent-set PPO and sorted 1-D CNN PPO.

The source archive remains source-only and does not include `artifacts/`.
