# ScaleMAC-RL v0.21.0

- Add Round 16C PPO update-geometry study.
- Add per-UE grouped PPO probability-ratio/clipping mode.
- Add strict post-step KL guard that rolls back model and Adam state when a minibatch exceeds the hard KL limit.
- Keep joint-ratio behavior as the default for backward compatibility.
