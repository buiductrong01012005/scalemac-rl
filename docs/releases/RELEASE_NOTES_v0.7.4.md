# ScaleMAC-RL v0.7.4

## 300k diagnostic PPO runs

This release reduces the long-run budget from about one million to **300,032**
environment steps for both learned schedulers:

- `python -m scalemac_rl.scripts.train_hybrid_300k`
- `python -m scalemac_rl.scripts.train_ppo_only_300k`
- `python -m scalemac_rl.scripts.train_both_ppo_300k`

Each run saves milestones near **100k, 200k, and 300k** environment steps. This
makes it possible to check whether goodput, Jain fairness, starvation, P99 wait,
and maximum wait are still improving before spending time on a longer run.

The trade-off checkpoint selector and unified evaluation protocol are unchanged.
All generated files use `hybrid_300k`, `ppo_only_300k`, or `unified_300k`
prefixes. The source archive remains source-only and excludes `artifacts/`.
