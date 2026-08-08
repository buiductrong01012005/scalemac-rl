# ScaleMAC-RL v0.9.2

## Reward exploration

- Replaces the narrow Round 06 plan with one six-case coordinate experiment around the stable equal-third reward.
- Holds one of Throughput, Jain fairness, or Service fixed at one third while shifting a small amount between the other two.
- Keeps the environment, PPO architecture, seed, training budget and all undeclared reward terms unchanged.
- Does not create reward datasets or Pareto outputs.

## Analysis archive

- Adds a linked HTML index for the complete reward-study sequence.
- Adds a synthesis page connecting Rounds 01–05.
- Adds a literature-context page that distinguishes qualitative opportunistic/greedy behavior from formal theorem reproduction.
- Clarifies that Service-heavy still selects 64 UEs per slot; it produces ineffective concentrated scheduling rather than literally allocating to nobody.
