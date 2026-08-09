# ScaleMAC-RL v0.9.7

v0.9.7 is a documentation-focused release. Scheduler, environment, PPO,
state/action definitions, reward formulas and Round 07 numerical results are
unchanged.

## Reader-friendly Round 07 report

- replaced the main Round 07 entry page with a Vietnamese reader-friendly report;
- explains the baseline T–J–S reward and the exact formulas for Deficit, PF,
  Low-throughput and Urgency;
- decodes all eight weight regimes in plain language;
- adds a case-by-case table describing what changed, final KPI, raw X score,
  stability and a plain-language verdict;
- highlights the top three Stable cases for goodput, Jain, P99 wait and max wait;
- separates the overall winner, the second candidate, stable-but-low-value cases,
  and rewards that should be parked or redesigned;
- adds an explicit roadmap for multi-seed confirmation and future redundancy
  ablations (Service versus Urgency; Jain versus Deficit);
- preserves the previous technical HTML as
  `fourth_component_technical_appendix.html`.

## New audit files

- `round_07_case_guide.csv`;
- `round_07_reward_decision.csv`;
- `round_07_top3_metrics.csv`;
- `round_07/README.md` explaining which report to open first.

The source archive remains source-only and excludes `artifacts/`.
