# ScaleMAC-RL v0.20.0

## Round 16B — PPO Sampling Budget Control

- Adds a controlled sampling-budget study on the fixed `service40` reward and 16-feature feed-forward PPO.
- Compares R256/MB8 @ 98k and R1024/MB64 @ 98k under the same environment-interaction budget.
- Adds R512/MB32 @ 196k and R1024/MB64 @ 393k so all three R256/R512/R1024 profiles target about 384 outer PPO updates.
- Keeps PPO clipping, target-KL, critic, reward, observations, and radio environment unchanged from the standard-critic Round 16A setup.
- Preserves PPO audit diagnostics and exports paired same-budget, same-update-count, and long-vs-short R1024 comparisons.
- Keeps validation/checkpoint cadence approximately aligned in environment steps across rollout sizes.
