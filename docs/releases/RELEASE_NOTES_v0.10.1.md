# ScaleMAC-RL v0.10.1

## Purpose

Make the Round 09 reproducibility diagnostic easy to run on a local Windows machine without a Kaggle notebook.

## Added

- `run_round09_local.ps1` at the repository root.
- One-command local workflow that:
  - prints the Python/NumPy/PyTorch runtime;
  - runs the full test suite;
  - dry-runs the three-repeat Round 09 plan;
  - runs/resumes all three CPU repeats;
  - rebuilds the reproducibility report;
  - checks that all required result files exist;
  - creates `scalemac_results_v0101_round09_reproducibility.zip` for analysis.

## Research behavior

No PPO, reward, environment, state, action, architecture, seed, or Round 09 experiment setting changed from v0.10.0.
