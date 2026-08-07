# ScaleMAC-RL v0.4.1

- Excludes the generated `artifacts/` directory from source archives.
- Fixes the Python 3.11 syntax error in `evaluate_ppo.py`.
- Adds clear errors when `best_feasible.pt` was not created.
- Lists `best_reward.pt` and `latest.pt` as diagnostic alternatives without treating them as feasible.
- Updates PowerShell evaluation commands to select an available checkpoint safely.
