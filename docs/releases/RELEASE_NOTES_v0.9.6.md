# ScaleMAC-RL v0.9.6

v0.9.6 archives the completed Round 07 analysis. Scheduler, environment, PPO,
state and action definitions are unchanged.

## Round 07 complete

- verified 32/32 completed cases;
- all cases use 100,096 environment steps, seed/profile/validation seed 1701,
  and actual CPU execution;
- exported final metrics, validation trajectories, regime comparison, stability
  matrix, regime summary and claim-evidence CSV;
- replaced the provisional report with a comprehensive HTML analysis containing
  final KPI tables, trajectory figures, reward-mechanism interpretation and all
  32 case details.

## Main research decision

`urgency_service_hold_throughput` is the leading balanced candidate. It reaches
97,877 bit/slot, Jain 0.3248, zero starvation, P99 wait 46 and max wait 48 in the
current one-seed setting. `deficit_service_anchor_preserving` is retained as a
fairness/tail-delay alternative.

PF utility is removed from the active objective in its current normalization.
Low-throughput percentile is also removed until its P10=0 dead-zone is fixed.
The next experiment should be a small common-seed confirmation of the equal-third
reference and the two retained candidates, not another broad weight sweep.

Source archives remain source-only and exclude `artifacts/`.
