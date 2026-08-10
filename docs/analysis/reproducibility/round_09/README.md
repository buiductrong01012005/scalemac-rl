# Round 09 — T–J–S reproducibility diagnostic

This experiment deliberately does **not** change the reward or system model.

It runs the selected reward three times with the same configuration:

`R = (1/3) Throughput + (1/3) Jain + (1/3) Service`

All repeats use training seed, static-profile seed and validation seed `1701`.

The goal is to answer one question before Dynamic CQI is introduced:

> If ScaleMAC-RL is launched repeatedly inside the same runtime with identical inputs, does it learn the same trajectory and final deterministic policy?

Generated after the run:

- `reproducibility_analysis.html`
- `reproducibility_summary.md`
- `repeat_metrics.csv`
- `pairwise_repeatability.csv`
- `repeatability_summary.csv`
- `runtime_fingerprints.csv`

If repeatability passes, the next development phase is Dynamic CQI. If it fails, the next diagnostic locks CPU threading and deterministic algorithms before any system-model change.
