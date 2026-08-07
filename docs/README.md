# ScaleMAC-RL

> Repository documentation lives under `docs/`; generated Markdown experiment reports are written to `docs/reports/`.

ScaleMAC-RL is a fast DRL training surrogate for **single-cell 5G NR downlink MAC scheduling** with:

- 1 gNB / 1 cell;
- 1,200 active, full-buffer UEs;
- heterogeneous static CQI and demand profiles;
- 273 schedulable PRBs per slot;
- at most 64 selected UEs per slot;
- one downlink queue per UE and no QoS flow;
- simplified HARQ;
- required output: selected Top-K UEs and PRBs per selected UE;
- KPIs: throughput, fairness/service, delay/starvation, candidate coverage, and inference latency.

## What v0.5 adds

v0.5 addresses the v0.4 validation mismatch and the weak Top-K learning claim:

- a 128-UE candidate pool feeds a 64-UE scheduler output;
- up to 16 grants are reserved for HARQ and long-waiting UEs;
- PPO ranks the remaining grants and predicts PRB demand;
- training defaults to 32,768 environment steps per curriculum stage;
- repeated held-out validation updates the Lagrange controller;
- unsafe validation streaks trigger rollback to the best feasible stage checkpoint;
- every curriculum stage saves its lowest-violation and best-feasible checkpoints;
- grant-attribution reports distinguish deterministic safety selection from learned selection;
- the final official constraint remains zero starvation and P99 waiting time no greater than 50 slots.

## Current roadmap status

| Roadmap block | Status in v0.5 |
|---|---|
| Fix DL scenario, active UEs, PRBs, traffic | Done in fast surrogate |
| RR, PF, Max-CQI baselines | Done in fast surrogate |
| KPI/reward sanity checks over multiple seeds | Done |
| Fast surrogate with CQI, queue, HARQ, traffic abstraction | Done |
| Collect PF demonstrations | Done online during imitation |
| Train imitation policy | Done |
| UE curriculum | Implemented for PPO |
| Candidate filtering and action masking | Implemented with compact candidate tensors |
| Actor priority and RB-demand scores | Done |
| Hybrid safety-reserve action projector | Done |
| RL training | Long-run constrained PPO with validation rollback implemented |
| Feasibility-first checkpoint selection | Implemented with held-out validation seeds |
| Distributed RL | Partial: vectorized workers in one Python process |
| OOD evaluation | Not yet automated |
| Transfer to 5G-LENA AI scheduler | Not yet implemented |
| Full 5G-LENA 1,200-UE evaluation | Not yet implemented |
| Shadow mode / hardware-in-the-loop | Not yet implemented |

## Objective and constraints

The base reward remains normalized and interpretable:

```text
0.55 * throughput_score
+ 0.30 * fairness_score
+ 0.15 * service_score
- 0.50 * starvation_violation
```

PPO additionally optimizes a constrained reward:

```text
constrained_reward
= base_reward
- lambda_starvation * starvation_excess
- lambda_wait * normalized_P99_wait_excess
```

Default validation constraints:

```text
maximum starvation rate = 0
maximum P99 waiting time = 50 slots
```

A checkpoint is marked feasible only when **every held-out validation seed** satisfies both limits. These defaults are starting hypotheses and must later be justified through baseline results and 5G-LENA timing.

## Commands

Run each PowerShell command on one line.

```powershell
# Test
pytest -q

# Baselines
python -m scalemac_rl.scripts.run_baselines --num-ues 1200 --slots 1000 --seeds 3

# PF imitation pretraining
python -m scalemac_rl.scripts.train_imitation --num-ues 1200 --steps 2000

# Constrained PPO training with held-out validation
python -m scalemac_rl.scripts.train_ppo --init-checkpoint .\artifacts\pf_imitation.pt --curriculum 128,256,600,1200 --steps-per-stage 32768 --workers 4 --rollout-steps 64 --episode-slots 500 --max-candidates 128 --safety-reserve-ues 16 --stage-p99-wait-limits 80,80,80,50 --validation-seeds 9001,9002,9003 --validation-repeats 2 --validation-slots 1000 --validate-every 16 --rollback-patience 2 --max-starvation-rate 0 --max-p99-wait-slots 50

# Evaluate the feasibility-first checkpoint
python -m scalemac_rl.scripts.evaluate_ppo .\artifacts\best_feasible.pt --num-ues 1200 --slots 1000 --seeds 3 --max-starvation-rate 0 --max-p99-wait-slots 50

# Paired comparison with RR, Max-CQI, and PF
python -m scalemac_rl.scripts.run_paired_evaluation .\artifacts\best_feasible.pt --num-ues 1200 --slots 1000 --seeds 5 --max-starvation-rate 0 --max-p99-wait-slots 50

# Candidate-count performance ablation
python -m scalemac_rl.scripts.run_candidate_ablation .\artifacts\best_feasible.pt --candidate-counts 64,128,256 --num-ues 1200 --slots 1000 --seeds 5

# Component latency and candidate-count benchmark
python -m scalemac_rl.scripts.benchmark_inference .\artifacts\best_feasible.pt --num-ues 1200 --candidate-counts 64,128,256 --warmup 100 --repeats 1000 --deadlines-us 500,1000 --torch-threads 1
```

For a quick CPU smoke run:

```powershell
python -m scalemac_rl.scripts.train_ppo --init-checkpoint .\artifacts\pf_imitation.pt --curriculum 64,128 --stage-p99-wait-limits 100,100 --steps-per-stage 256 --workers 1 --rollout-steps 16 --episode-slots 80 --max-candidates 128 --safety-reserve-ues 8 --validation-seeds 9101 --validation-repeats 1 --validation-slots 80 --validate-every 1 --checkpoint-every 2 --output .\artifacts\smoke_latest.pt --best-feasible-output .\artifacts\smoke_best_feasible.pt --best-reward-output .\artifacts\smoke_best_reward.pt --log-output .\artifacts\smoke_ppo_training.csv --validation-output .\artifacts\smoke_ppo_validation.csv
```

## Files to send back after v0.5

Include both numeric artifacts and generated readable reports:

```powershell
Compress-Archive -Path .\artifacts\*,.\docs\reports\* -DestinationPath .\scalemac_results_v05.zip -Force
```

Most important files:

```text
artifacts/
├── latest.pt
├── best_reward.pt
├── best_feasible.pt                 # only created when final-stage validation is feasible
├── checkpoints/
├── ppo_training.csv
├── ppo_validation.csv
├── ppo_validation_summary.csv
├── ppo_evaluation.csv
├── ppo_evaluation_summary.csv
├── paired_evaluation.csv
├── paired_evaluation_summary.csv
├── candidate_ablation.csv
├── candidate_ablation_summary.csv
└── inference_benchmark.csv

docs/reports/
├── ppo_training.md
├── ppo_validation.md
├── ppo_validation_summary.md
├── ppo_evaluation.md
├── paired_evaluation.md
├── candidate_ablation.md
└── inference_benchmark.md
```

When `best_feasible.pt` is absent, send `latest.pt`, `best_reward.pt`, and the validation reports. The absence itself is an important result: the selected constraints were not met.

## Scope boundary

This remains a **fast surrogate**, not a 3GPP-complete or 5G-LENA scheduler. Static CQI, aggregated HARQ, full PDSCH use of all 273 PRBs, abstract CQI-to-rate mapping, and simplified control feasibility remain deliberate MVP assumptions.

## Repository layout

```text
scalemac-rl/
├── artifacts/          # generated at runtime; never included in source archives
├── docs/
│   ├── README.md
│   ├── releases/       # Version notes
│   └── reports/        # Generated Markdown experiment reports
├── scalemac_rl/
└── tests/
```

ZIP archives are ignored through both `*.zip` and the requested `*.*zip` pattern.

## Source archive policy

Release ZIP files contain source code, tests, configs, and documentation only. They do not contain an `artifacts/` directory, so extracting a new source release cannot overwrite existing experiment results or checkpoints. Scripts create output directories automatically when needed.
