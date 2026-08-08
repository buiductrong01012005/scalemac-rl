# ScaleMAC-RL v0.7.3

## Purpose

Run the hybrid PPO and candidate-128 PPO-only experiments for approximately one
million environment steps without losing early or intermediate operating points.
The release also adds a multi-KPI checkpoint and scheduler ranking aligned with the
project goal: remain close to the best deployable scheduler across goodput, fairness,
starvation, P99 wait, and maximum UE wait.

## New training entry points

- `python -m scalemac_rl.scripts.train_hybrid_1m`
- `python -m scalemac_rl.scripts.train_ppo_only_1m`
- `python -m scalemac_rl.scripts.train_both_ppo_1m`

Both long runs use 999,936 environment steps, one frozen 1,200-UE profile, one worker,
256-step rollouts, 5,000-slot validation, and tqdm timing. Hybrid fine-tuning resumes
from the strongest existing hybrid checkpoint when available. PPO-only always starts
from random actor-critic weights and receives no safety-selected grants.

## Trade-off checkpoint

Training now exports `best_tradeoff.pt`. Validation computes fixed-target attainment:

- throughput score relative to the soft target 0.45;
- Jain fairness relative to 0.60;
- P99 wait relative to 50 slots;
- maximum successful-delivery wait relative to 60 slots;
- starvation relative to zero.

The selector first rejects starvation failures, then minimizes the largest KPI gap,
then maximizes the geometric balanced score, and finally prefers higher goodput.
`best_lowest_violation`, `best_reward`, and `best_feasible` remain available and are
not replaced.

## Milestones

Requested checkpoints are saved after crossing approximately 250k, 500k, 750k, and
1M environment steps. This makes it possible to determine whether longer training
improves the policy, plateaus, or collapses after an earlier optimum.

## Unified trade-off report

`run_unified_evaluation` now exports an additional `*_tradeoff.csv` and Markdown report
containing:

- goodput, fairness, P99, max-wait, and starvation proximity;
- geometric `balanced_score`;
- `worst_kpi_gap`;
- starvation eligibility;
- Pareto dominance;
- per-seed trade-off rank.

The ideal point is formed only from starvation-feasible methods. Max-CQI remains in the
report but cannot set the deployable goodput reference when it starves UEs.

## Compatibility

No retraining is required to evaluate old checkpoints. Legacy 8-feature hybrid
checkpoints continue to use the existing compatibility adapter. Source archives do not
contain `artifacts/`.
