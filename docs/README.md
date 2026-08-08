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


## What v0.7.2 adds

v0.7.2 makes scheduler comparison reproducible through one `unified-v1`
evaluation protocol. Standalone PPO evaluation and cross-scheduler evaluation
now share the same environment builder, candidate/runtime resolver, normalized
evaluation reward, KPI constraints, static profile, and rollout seed. Every row
records protocol, scenario, scheduler-runtime, and checkpoint hashes.

## What v0.6.2 adds

v0.6.2 focuses on a more realistic fairness/service objective for the 1,200-UE
single-seed experiment:

- the default run remains 200,192 environment steps with tqdm timing and ETA;
- starvation now means **no successful delivery for at least 64 consecutive slots**;
- a separate scheduling-wait counter shows whether a UE was selected but failed HARQ;
- positive reward weights are throughput `0.50`, fairness `0.35`, and service `0.15`;
- fairness training uses 60% cumulative Jain fairness and 40% EWMA Jain fairness;
- the fixed validation targets are Jain fairness >= 0.60, P99 wait <= 50 slots,
  maximum successful-delivery wait <= 60 slots, and starvation rate = 0;
- the reward includes an additional non-saturating risk term for the single worst UE wait;
- constrained PPO has separate dual multipliers for starvation, P99 wait, fairness,
  and maximum wait;
- checkpoint selection saves the exact validated weights before rollback and exports
  `checkpoint_manifest.csv` with update, step count, and selection reason;
- source releases never contain `artifacts/`.

The single-seed mode remains an in-scenario upper-bound experiment, not evidence of
multi-seed robustness or OOD generalization.

## Current roadmap status

| Roadmap block | Status in v0.6 |
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
| RL training | Long-run constrained PPO plus a single-seed upper-bound mode |
| Feasibility-first checkpoint selection | Implemented with held-out validation seeds |
| Distributed RL | Partial: vectorized workers in one Python process |
| OOD evaluation | Not yet automated |
| Transfer to 5G-LENA AI scheduler | Not yet implemented |
| Full 5G-LENA 1,200-UE evaluation | Not yet implemented |
| Shadow mode / hardware-in-the-loop | Not yet implemented |

## Objective and constraints

The primary waiting counter is the number of consecutive slots since a UE last
received **successfully delivered bits**. A UE selected for transmission but failing
HARQ is therefore still waiting. Scheduling-only waiting time is logged separately.

Default starvation definition:

```text
starved UE = successful_delivery_wait >= 64 slots
starvation_rate = number of starved UEs / number of active UEs
```

The positive reward is:

```text
core_reward
= 0.50 * throughput_score
+ 0.35 * fairness_score
+ 0.15 * service_score
- starvation_penalty
```

where:

```text
fairness_score
= 0.60 * cumulative_Jain_fairness
+ 0.40 * EWMA_Jain_fairness
```

Tail-delay penalties are then applied:

```text
active_reward
= core_reward
- P99_deadline_risk_penalty
- maximum_wait_risk_penalty
```

PPO additionally uses Lagrange penalties for four validation constraints:

```text
starvation rate = 0
P99 successful-delivery wait <= 50 slots
minimum Jain fairness >= 0.60
single worst UE successful-delivery wait <= 60 slots
```

`best_lowest_violation.pt` is ranked lexicographically: starvation first, then
maximum wait, P99 wait, fairness deficit, and finally final-target reward. This
prevents a small throughput gain from hiding a severe service failure.

## Commands

Run each PowerShell command on one line.

```powershell
# Test
pytest -q

# Baselines
python -m scalemac_rl.scripts.run_baselines --num-ues 1200 --slots 1000 --seeds 3

# PF imitation pretraining
python -m scalemac_rl.scripts.train_imitation --num-ues 1200 --steps 2000

# Primary v0.6.2 experiment: one frozen 1,200-UE profile, about 200k steps
python -m scalemac_rl.scripts.train_single_seed

# Override the budget when needed, for example 300k aligned steps
python -m scalemac_rl.scripts.train_single_seed --steps-per-stage 300032

# General multi-seed curriculum remains available separately
python -m scalemac_rl.scripts.train_ppo --init-checkpoint .\artifacts\pf_imitation.pt --curriculum 128,256,600,1200 --steps-per-stage 32768 --workers 4 --rollout-steps 64 --episode-slots 500 --max-candidates 128 --safety-reserve-ues 16 --stage-p99-wait-limits 80,80,80,50 --final-stage-p99-schedule 80,65,55,50 --validation-seeds 9001,9002,9003 --validation-repeats 2 --validation-slots 1000 --validate-every 16 --rollback-patience 2 --max-starvation-rate 0 --max-p99-wait-slots 50

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

## Files to send back after v0.6.2

Include both numeric artifacts and generated readable reports:

```powershell
Compress-Archive `
  -Path .\artifacts\ppo_training.csv,`
        .\artifacts\ppo_validation.csv,`
        .\artifacts\ppo_validation_summary.csv,`
        .\artifacts\ppo_evaluation*.csv,`
        .\artifacts\paired_evaluation*.csv,`
        .\artifacts\checkpoint_manifest.csv,`
        .\docs\reports\*.md `
  -DestinationPath .\scalemac_results_v062_single_seed.zip `
  -Force
```

Most important files:

```text
artifacts/
├── latest.pt
├── best_reward.pt
├── best_lowest_violation.pt
├── best_feasible.pt                 # only created when final-stage validation is feasible
├── checkpoints/
├── ppo_training.csv
├── ppo_validation.csv
├── ppo_validation_summary.csv
├── checkpoint_manifest.csv
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

When `best_feasible.pt` is absent, send the CSV/Markdown reports first. Send `best_lowest_violation.pt` only when direct checkpoint inspection is needed. The absence itself is an important result: the selected constraints were not met.

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

## v0.7 pure-PPO and attribution experiments

The environment now exposes 10 per-UE features. The two new features are:

- `throughput_deficit`: demand-normalized deficit relative to the current cell mean;
- `service_deficit`: successful-delivery wait relative to the expected round-robin service cycle.

The scheduler can run in three modes:

- `hybrid`: rule-selected safety grants plus PPO-selected grants;
- `ppo_only`: PPO selects every grant and the projector only enforces feasibility;
- `rule_only`: HARQ/oldest-wait rules select every grant.

Train PPO from random weights with 128 candidates:

```powershell
python -m scalemac_rl.scripts.train_ppo_from_scratch
```

Expose all 1,200 UEs directly to PPO after the candidate experiment is stable:

```powershell
python -m scalemac_rl.scripts.train_ppo_from_scratch --full-ues
```

Run scheduler attribution after checkpoints are available:

```powershell
python -m scalemac_rl.scripts.run_scheduler_attribution `
  --hybrid-checkpoint .\artifacts\best_lowest_violation.pt `
  --ppo-candidate-checkpoint .\artifacts\ppo_scratch_candidate128_best_lowest_violation.pt `
  --ppo-full-checkpoint .\artifacts\ppo_scratch_full1200_best_lowest_violation.pt
```

All learned modes keep the projector, but in `ppo_only` mode it may only enforce Top-64 selection, non-negative integer PRBs, at least one PRB per selected UE, and an exact total of 273 PRBs.

## v0.7.2 unified evaluation protocol

All scheduler comparisons now use one fixed evaluation contract:

- the same frozen CQI/demand profile;
- the same rollout seed and HARQ randomness;
- the same 10-feature observation schema;
- the same normalized evaluation reward;
- the same starvation, P99, Jain-fairness, and maximum-wait constraints;
- the same Top-64, exact-273-PRB projector contract.

A checkpoint's training reward is retained only as provenance. It cannot change
its evaluation reward or KPI definitions.

Evaluate one checkpoint:

```powershell
python -m scalemac_rl.scripts.evaluate_ppo `
  .\artifacts\best_lowest_violation.pt `
  --num-ues 1200 --slots 5000 --seed 1701 --seeds 1
```

Compare all available schedulers under the same protocol:

```powershell
python -m scalemac_rl.scripts.run_unified_evaluation `
  --hybrid-checkpoint .\artifacts\best_lowest_violation.pt `
  --ppo-candidate-checkpoint .\artifacts\ppo_scratch_candidate128_best_lowest_violation.pt `
  --num-ues 1200 --slots 5000 --seed 1701 --seeds 1
```

The legacy `run_scheduler_attribution` command invokes the same implementation.
Both commands export a manifest containing:

```text
evaluation_protocol_hash
scenario_hash
scheduler_runtime_hash
checkpoint_sha256
checkpoint_training_reward_version
checkpoint_observation_features
compatibility_adapter_applied
```

For a given checkpoint and identical CLI arguments, the KPI row produced by
`evaluate_ppo` must match the corresponding row produced by
`run_unified_evaluation`.

Verify that standalone and unified evaluation rows are identical:

```powershell
python -m scalemac_rl.scripts.verify_evaluation_consistency `
  .\artifacts\ppo_evaluation.csv `
  .\artifacts\scheduler_attribution.csv
```

The check matches rows by checkpoint SHA-256, protocol hash, scenario hash, and
scheduler-runtime hash before comparing KPI values. Inference timing is excluded
because wall-clock measurements naturally vary between runs.
