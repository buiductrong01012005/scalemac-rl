# ScaleMAC-RL v0.22.1

- Adds validation-driven service-only rollback that protects starvation/P99/max-wait feasibility without requiring the Jain target.
- Adds optional learning-rate decay after each rollback with a configurable floor.
- Records pre-rollback and protected post-rollback validation state, rollback count and LR multiplier.
- Adds Round 17B: baseline vs service rollback vs service rollback + LR×0.5 across three paired seeds.
