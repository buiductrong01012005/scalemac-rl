# ScaleMAC-RL v0.10.0

## Purpose

Freeze the active reward at Throughput + Jain + Service and add a controlled reproducibility diagnostic before making the radio/network model more realistic.

## Experiment

Round 09 is not a reward sweep. It runs the exact same T–J–S configuration three times with seed/profile seed/validation seed 1701.

## New diagnostics

- Per-run Python/NumPy/PyTorch/CPU/thread/determinism fingerprint.
- Seeded NumPy and Torch RNG fingerprints.
- Initial model parameter hash.
- Final model parameter hash from checkpoints.
- Pairwise training-trajectory comparison excluding timing columns.
- First divergent training row at exact, 1e-12 and 1e-9 tolerances.
- HTML, Markdown and CSV reproducibility summaries.

## Research constraint

No reward formula, PPO hyperparameter, architecture, state/action design, environment model, traffic model or CQI model is changed in this release. Dynamic CQI starts only after this diagnostic is interpreted.
