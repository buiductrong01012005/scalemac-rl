# ScaleMAC-RL v0.9.8

v0.9.8 adds the pre-registered Round 08 multi-seed confirmation workflow. It does
not change the environment, observation, action, PPO architecture, reward formulas,
or the Round 07 conclusions.

## Round 08 plan

- adds `configs/reward_study/round_08_multiseed_confirmation.json`;
- runs three reward profiles on seeds 1701, 2701 and 3701 (nine cases total);
- compares equal-third T–J–S, Urgency hold-Throughput, and Deficit group T+S;
- keeps 100,096 steps and the full-control PPO protocol fixed;
- pre-registers keep/reject rules before execution.

## Runner and analysis

- reward cases can now provide `common_overrides`, used for seed, profile-seed and
  validation-seed overrides while retaining one shared plan;
- the runner records the effective merged common config for every case;
- adds a multi-seed analysis builder with CSV outputs for per-seed metrics,
  profile mean/std, paired deltas, validation trajectories and stability;
- adds reader-facing HTML and Markdown summaries under
  `docs/analysis/reward_study/round_08/`;
- keeps resume/skip-completed behavior and source-only packaging.

## Scope

No new reward component, PPO hyperparameter tuning, Beta tuning, architecture
change, dataset generation, or Pareto optimization is introduced in this release.
