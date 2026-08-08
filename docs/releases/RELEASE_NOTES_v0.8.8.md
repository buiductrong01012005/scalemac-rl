# ScaleMAC-RL v0.8.8

## Purpose

Return the project to incremental reward exploration rather than optimizing the failed 25/75 case.

## Changes

- Added `round_04_add_service_equal.json` with one controlled three-component case.
- Uses equal coefficients for throughput, Jain fairness and service before tuning.
- Added automatic plain-language HTML analysis with exact formulas, coefficients, KPI definitions and comparison to the 37.5/62.5 reference run.
- Moved the unused exploration-alignment plan into `configs/reward_study/archive/`.
- Kept dataset and Pareto generation opt-in and outside the current research phase.
- Updated the reward methodology to add one component at a time and tune only when its effect is understood.
