# Round 10 — Dynamic CQI screen

Reward/PPO/state/action remain fixed. Only CQI temporal dynamics change.

| Case | CQI | Goodput | Jain | Starvation | P99 | Max wait | Mean |ΔCQI| | Changed UE fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Static CQI baseline | static | 97411 | 0.2676 | 0.00% | 47 | 48 | 0.000 | 0.0% |
| Slow correlated Dynamic CQI | correlated | 110637 | 0.1813 | 3.75% | 101 | 113 | 0.275 | 27.5% |
| Faster correlated Dynamic CQI | correlated | 15113 | 0.0437 | 94.67% | 5000 | 5000 | 0.717 | 58.9% |

## Interpretation rule
Dynamic CQI is considered manageable only if the scheduler remains non-collapsed while preserving useful goodput/fairness/delay relative to the static baseline.
BLER is still fixed and CQI-independent in this round; Link Adaptation is intentionally not changed yet.
