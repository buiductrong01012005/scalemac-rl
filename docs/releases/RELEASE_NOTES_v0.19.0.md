# ScaleMAC-RL v0.19.0

## Round 16A — PPO Root-Cause Audit

- Adds a controlled 2x2 audit of sampling regime and PPO value-function clipping on the fixed service40 reward.
- Adds PPO diagnostics for KL/ratio behavior, target-KL early stopping, gradient clipping, actor/critic gradient probes, return/advantage statistics, value RMSE and explained variance.
- Adds optional PPO value clipping through `--value-clip-coef`; disabled by default for backward compatibility.
- Keeps the 16-feature observation, feed-forward PPO architecture, reward definition and radio environment unchanged.
- Adds multi-seed root-cause analysis outputs for 12 paired cases.
