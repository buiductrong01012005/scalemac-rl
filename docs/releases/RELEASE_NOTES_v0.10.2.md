# ScaleMAC-RL v0.10.2

## Round 09 result

- 3/3 identical T–J–S seed-1701 repeats completed.
- All 391 training rows are numerically identical.
- RNG, initial-model, final-model and latest-checkpoint hashes match across repeats.
- Final deterministic KPI is identical: goodput 97,411.019 bit/slot, Jain 0.267640, starvation 0, P99 47, max-wait 48.
- Local CPU pipeline is therefore reproducible under the recorded runtime.
- Cross-runtime/session differences remain a separate concern and should not be confused with seed robustness.

## Research decision

- Freeze the active reward to Throughput + Jain + Service.
- Do not continue fourth-component reward tuning.
- Move next to Dynamic CQI while keeping PPO/reward/state/action architecture fixed.

## Usability

- Removed the PowerShell wrapper script; run commands are provided directly instead.
- Added reader-friendly Round 09 analysis and audit CSVs.
