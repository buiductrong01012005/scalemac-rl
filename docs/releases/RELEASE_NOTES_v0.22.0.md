# ScaleMAC-RL v0.22.0

- Adds an explicit environment RNG seed independent of PPO/model RNG while preserving legacy defaults.
- Adds Round 17A, a full 3×3 training-RNG × environment-seed decoupling study.
- Keeps joint/shared PPO, rollout 256, minibatch 8, service40 reward, 16 observations and the radio model unchanged.
