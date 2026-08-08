# ScaleMAC-RL v0.9.4

v0.9.4 turns Round 07 into one integrated fourth-component ablation so the
screening and first directional checks can be interpreted together.

## Twelve controlled cases

For each of deficit service, PF utility, low-throughput and urgency service:

1. equal-quarter: Throughput/Jain/Service/new component = 0.25 each;
2. new-component-heavy: 0.20/0.20/0.20/0.40;
3. anchor-preserving: Throughput 0.30, Jain 0.10, Service 0.30, new component 0.30.

The third regime preserves the empirically important Throughput and Service
signals while testing whether the new component can replace part of Jain. This
distinguishes a weak component from a case that merely collapses because both
anchors were diluted.

## Integrated evidence output

After training, the same command generates:

- `fourth_component_integrated_analysis.html`;
- `round_07_final_metrics.csv`;
- `round_07_validation_trajectory.csv`;
- `round_07_regime_comparison.csv`.

The HTML groups the three regimes by component and gives explicit decision rules
for secondary shaping, dominant-objective failure, anchor dependence and score
redesign.

No reward dataset, Pareto optimization, architecture change, heuristic rule or
constraint penalty is introduced. Source archives remain source-only and exclude
`artifacts/`.
