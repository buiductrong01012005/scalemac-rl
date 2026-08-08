# ScaleMAC-RL v0.7.5

v0.7.5 tests the hypothesis that rule-based selection is carrying the hybrid PPO
scheduler instead of assuming that the combined result belongs to PPO alone.

## Added

- fixed-weight rule/PPO split ablation at 0, 8, 16, 24, 32, 48, and 64 rule grants;
- a pure same-weights PPO reference with projector safety rules disabled;
- a HARQ-only point separating mandatory retransmission support from oldest-UE reserve;
- optional all-1,200-UE evaluation to expose dependence on the heuristic candidate filter;
- per-KPI `rule_lift_*` values where positive values always mean improvement;
- seed-level, summary, dependency-curve, and provenance reports;
- generic 120,064-step fine-tuning for a selected fixed rule/PPO split.

## Interpretation contract

The main dependency curve keeps actor weights and the heuristic 128-candidate filter
fixed. It therefore measures the contribution of the projector safety rule. Removing
the candidate filter is a separate out-of-distribution ablation. Final claims about a
split still require training each promising split from the same starting checkpoint.

## Source hygiene

The release archive contains source, tests, configuration, and documentation only.
Generated `artifacts/` and experiment reports are excluded.
