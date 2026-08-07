# ScaleMAC-RL

ScaleMAC-RL is a fast DRL training surrogate for **single-cell 5G NR downlink MAC scheduling** with:

- 1 gNB / 1 cell;
- 1,200 active, full-buffer UEs;
- heterogeneous static CQI and demand profiles;
- 273 schedulable PRBs per slot;
- at most 64 selected UEs per slot;
- one downlink queue per UE and no QoS flow;
- simplified HARQ;
- required output: selected Top-K UEs and PRBs per selected UE;
- KPIs: throughput, fairness/service, delay/starvation, and inference latency.

## What v0.3 adds

v0.2 trained only **behavioral cloning from PF**. That is real neural-network training, but it is not reinforcement learning. v0.3 adds the first actual RL stage:

- stochastic permutation-equivariant actor and pooled set critic;
- PPO fine-tuning initialized from `pf_imitation.pt`;
- UE-count curriculum, default `128 -> 256 -> 600 -> 1200`;
- deterministic safety-aware candidate filtering and action masking;
- hard action projection that preserves Top-K and the exact PRB budget;
- vectorized rollout workers in one Python process;
- paired PPO/PF/RR/Max-CQI evaluation under identical seeds;
- warm inference benchmark with deadline-miss rates;
- automatic CSV and Markdown reports.

## Roadmap status

| Roadmap block | Status in v0.3 |
|---|---|
| Fix DL scenario, active UEs, PRBs, traffic | Done in fast surrogate |
| RR, PF, Max-CQI baselines | Done in fast surrogate |
| KPI/reward sanity checks over multiple seeds | Done |
| Fast surrogate with CQI, queue, HARQ, traffic abstraction | Done |
| Collect PF demonstrations | Done online during imitation |
| Train imitation policy | Done |
| UE curriculum | Implemented for PPO |
| Candidate filtering and action masking | Implemented; actor outputs are masked to candidates |
| Actor priority and RB-demand scores | Done |
| Constraint-aware action projector | Done |
| RL training | PPO implemented in v0.3 |
| Distributed RL | Partial: multiple vectorized workers, not multiprocessing/distributed yet |
| OOD evaluation | Not yet automated |
| Transfer to 5G-LENA AI scheduler | Not yet implemented |
| Full 5G-LENA 1,200-UE evaluation | Not yet implemented |
| Shadow mode / hardware-in-the-loop | Not yet implemented |

## Reward contract

All positive components are normalized to `[0, 1]`:

```text
0.55 * throughput_score
+ 0.30 * fairness_score
+ 0.15 * service_score
- 0.50 * starvation_violation
```

Each component is logged separately so PPO cannot appear successful merely by maximizing throughput while starving UEs.

## Commands

Run each PowerShell command on one line.

```powershell
# Test
pytest -q

# Baselines
python -m scalemac_rl.scripts.run_baselines --num-ues 1200 --slots 1000 --seeds 3

# PF imitation pretraining
python -m scalemac_rl.scripts.train_imitation --num-ues 1200 --steps 2000

# Actual PPO training with UE curriculum
python -m scalemac_rl.scripts.train_ppo --init-checkpoint .\artifacts\pf_imitation.pt --curriculum 128,256,600,1200 --steps-per-stage 2048 --workers 4 --rollout-steps 64 --max-candidates 256

# Evaluate PPO
python -m scalemac_rl.scripts.evaluate_ppo .\artifacts\scalemac_ppo.pt --num-ues 1200 --slots 1000 --seeds 3 --max-candidates 256

# Paired comparison with RR, Max-CQI, and PF
python -m scalemac_rl.scripts.run_paired_evaluation .\artifacts\scalemac_ppo.pt --num-ues 1200 --slots 1000 --seeds 5 --max-candidates 256

# Warm inference benchmark
python -m scalemac_rl.scripts.benchmark_inference .\artifacts\scalemac_ppo.pt --num-ues 1200 --max-candidates 256 --warmup 100 --repeats 1000 --deadlines-us 500,1000
```

For a quick CPU smoke run before the full curriculum:

```powershell
python -m scalemac_rl.scripts.train_ppo --init-checkpoint .\artifacts\pf_imitation.pt --curriculum 128,256 --steps-per-stage 256 --workers 2 --rollout-steps 32 --max-candidates 128 --output .\artifacts\smoke_ppo.pt --log-output .\artifacts\smoke_ppo_training.csv
```

## Files to send back after v0.3

Zip the complete `artifacts` folder:

```powershell
Compress-Archive -Path .\artifacts\* -DestinationPath .\scalemac_artifacts_v03.zip -Force
```

The most important files are:

```text
artifacts/
├── ppo_training.csv
├── ppo_training.md
├── scalemac_ppo.pt
├── ppo_evaluation.csv
├── ppo_evaluation_summary.csv
├── paired_evaluation.csv
├── paired_evaluation_summary.csv
├── inference_benchmark.csv
└── inference_benchmark.md
```

## Scope boundary

This is still a **fast surrogate**, not a 3GPP-complete or 5G-LENA scheduler. Static CQI, aggregated HARQ, full PDSCH use of all 273 PRBs, abstract CQI-to-rate mapping, and simplified control feasibility remain deliberate MVP assumptions.
