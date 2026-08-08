# ScaleMAC-RL v0.9.5

v0.9.5 expands Round 07 into one comprehensive fourth-component experiment so
all weight-geometry questions can be analyzed together instead of being split
across several later rounds.

## Thirty-two controlled cases

For each of deficit service, PF utility, low-throughput and urgency service, the
plan runs eight regimes:

1. equal-quarter addition;
2. new-component-heavy;
3. hold Throughput;
4. hold Jain fairness;
5. hold Service;
6. preserve the Throughput + Jain group;
7. preserve the Throughput + Service group;
8. preserve the Jain + Service group.

This yields 4 × 8 = 32 cases. Every case keeps the same environment, PPO,
seed, validation protocol and 100,096 training steps. Only positive reward
weights change.

## Integrated evidence output

The generated HTML reads the experiment in both directions: all regimes of one
new component, and all four new components under one regime. Individual case
sections are collapsible to keep the report readable. The run also exports a
stability matrix and a regime-level summary in addition to final metrics,
validation trajectories and reference deltas.

No reward dataset, Pareto optimization, architecture change, heuristic rule or
constraint penalty is introduced. Source archives remain source-only and exclude
`artifacts/`.
