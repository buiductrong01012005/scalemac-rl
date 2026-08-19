# ScaleMAC-RL v0.13.3

## Purpose

Fix the internal-docs policy test on Windows.

## Fix

- Use `Path(__file__).name` when excluding `test_internal_docs_policy.py` from its own scan.
- Prevent the test from detecting its own `docs/analysis/` guard strings and failing on Windows path separators.
- Keep `docs/analysis/` and `*.log` ignored.
- No PPO, reward, environment, CQI, CSI, MCS, BLER, HARQ, or training behavior changes.
