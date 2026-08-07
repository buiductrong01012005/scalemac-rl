# ScaleMAC-RL v0.6.1

This patch makes the single-seed upper-bound run easier to interpret and monitor.

- reduces the default run to 200,192 environment steps (approximately 200k);
- adds a tqdm progress bar with elapsed time, ETA, step rate, reward, P99 wait, and goodput;
- exports elapsed seconds, steps/second, and ETA in `ppo_training.csv`;
- separates core reward, active-curriculum reward, constrained training reward, and fixed-final-target reward;
- replaces clipped tail-delay risk with a smooth non-saturating logarithmic score;
- ranks `best_reward.pt` and `best_lowest_violation.pt` against the fixed final P99 target of 50 slots;
- keeps source releases free of `artifacts/`.
