# Round 10 — Dynamic CQI Screen

- 3 cases: static / slow-correlated / faster-correlated CQI.
- 1 controlled seed: 1701.
- 100,096 environment steps mỗi case.
- Reward khóa: T/J/S = 1/3 / 1/3 / 1/3.
- Không đổi PPO, architecture, traffic, HARQ abstraction hoặc scheduler control.
- Output: per-case CSV/checkpoint + `dynamic_cqi_metrics.csv` + Markdown/HTML analysis.

Mục tiêu round này là đo tác động của channel non-stationarity, không phải tuning reward.
