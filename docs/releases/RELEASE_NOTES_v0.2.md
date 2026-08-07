# ScaleMAC-RL v0.2

## What changed

- Reward inputs are explicitly normalized before weighting.
- Starvation is now a separate constraint penalty, so a scheduler cannot obtain a high reward only by serving the best-channel UEs.
- Every experiment records the reward decomposition.
- Baseline, imitation-training, and policy-evaluation scripts automatically create CSV and Markdown reports.
- Policy evaluation reports mean goodput over the complete episode rather than labeling the last slot as a mean.
- Policy evaluation records mean, P95, P99, and maximum inference latency.
- Baseline and policy reports include summaries across seeds.

## Validation result

For 1,200 UEs, 273 PRBs, Top-K 64, 1,000 slots, and three seeds:

- Max-CQI still achieves the highest goodput, as expected.
- Its reward is now much lower because it starves about 85% of UEs.
- RR and PF receive higher total rewards because they maintain service without starvation.

This is a reward sanity check, not a claim that the final reward weights are optimal.
