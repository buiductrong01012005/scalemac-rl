# ScaleMAC-RL v0.9.0

## Purpose

Continue controlled reward exploration from the equal-third Throughput–Jain–Service baseline without prematurely tuning only Service.

## New experiment

`configs/reward_study/round_05_three_component_directional.json`

| Case | Throughput | Jain fairness | Service |
|---|---:|---:|---:|
| `throughput_heavy` | 0.50 | 0.25 | 0.25 |
| `jain_heavy` | 0.25 | 0.50 | 0.25 |
| `service_heavy` | 0.25 | 0.25 | 0.50 |

Each case raises one component and reduces the other two equally. All other reward components, penalties, reward deltas, Lagrangian multipliers and rule overrides remain disabled.

## Analysis changes

- Reframed Round 04 as the central `1/3–1/3–1/3` evidence point.
- Removed the premature conclusion that Service must immediately be reduced to 5–15%.
- Added an automatic three-case comparison against the equal baseline.
- Added realized reward-contribution tables and plain-language KPI explanations.
- Added fallback Round 04 reference metrics so the HTML comparison can still be generated when old artifacts are not present locally.

## Output

```text
artifacts/runs/reward_study/round_05_three_component_directional/
├── throughput_heavy/
├── jain_heavy/
├── service_heavy/
└── round_plan_snapshot.json
```

The generated analysis is written to:

```text
docs/analysis/reward_study/round_05/three_component_directional_analysis.html
```
