# ScaleMAC-RL v0.18.0

## Round 15 — TJS reward × checkpoint stability

This release adds a controlled four-profile T/J/S reward re-check over seeds 1701, 2701, and 3701 while keeping the feed-forward PPO recipe, 16-feature observation, Dynamic CQI, delayed CSI, CQI→MCS link adaptation, BLER and HARQ unchanged.

Profiles:

- equal: T/J/S = 1/3, 1/3, 1/3;
- throughput40: 0.40, 0.30, 0.30;
- jain40: 0.30, 0.40, 0.30;
- service40: 0.30, 0.30, 0.40.

The new checkpoint-stability analysis reads the full validation trajectory and separates three outcomes: never service-feasible, service-feasible and retained, and learn-then-drift. It also compares the saved `best_tradeoff` checkpoint against `latest` and exports case, profile, trajectory, ranking and decision artifacts.

Service-feasible is reported separately from the stricter full target: zero maximum starvation, P99 wait no greater than 50 slots, and maximum wait no greater than 60 slots.
