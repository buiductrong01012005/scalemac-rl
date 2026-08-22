# ScaleMAC-RL v0.26.0

## Round 21 — teacher-free PPO retention and schedule-state observability

- adds Round 21A: Equal4 baseline vs service-aware rollback vs rollback + moderate
  Beta concentration annealing;
- adds Round 21B: controlled observation study for scheduler-owned per-UE history;
- new optional observation features:
  - time since the UE was last scheduled;
  - recent EWMA schedule-rate deficit relative to the expected Top-K share;
  - relative schedule-rate deficit rank;
- expanded-input actors use baseline-compatible initialization: the original 16
  feature weights, downstream actor/critic parameters, and post-init RNG state are
  paired exactly, while added feature weights start at zero;
- Round 21 post-hoc diagnostics evaluate both deterministic and stochastic policies
  on each case's effective environment seed using the exact final milestone;
- no PF/oracle checkpoint or expert action label is used by either Round 21 study;
- records hierarchical MAPPO (12x100 / 16x75 + global coordinator + shared local
  actors + centralized critic) as the next architecture family after pure-PPO work.
