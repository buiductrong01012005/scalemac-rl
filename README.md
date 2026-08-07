# ScaleMAC-RL

Fast DRL-ready surrogate for **single-cell 5G NR downlink MAC scheduling** with the agreed MVP contract:

- 1 gNB / 1 cell;
- 1,200 active, full-buffer UEs;
- heterogeneous static CQI and heterogeneous demand profiles;
- 273 schedulable PRBs per slot;
- maximum Top-K = 64 scheduled UEs per slot;
- one downlink queue per UE, no QoS flow;
- simplified HARQ enabled;
- mandatory output: selected Top-K UEs and PRBs per selected UE;
- primary KPIs: cell throughput, UE fairness/service, delay/starvation.

## Milestone v0.2

This version adds the experiment contract needed before PPO training:

- normalized throughput, fairness, and service scores in `[0, 1]`;
- a separate starvation constraint penalty so Max-CQI cannot obtain a high reward while starving most UEs;
- per-component reward logging;
- automatic CSV and Markdown reports for baselines, imitation training, and policy evaluation;
- full-episode mean goodput instead of the old last-slot label;
- policy inference latency statistics;
- baseline and evaluation summaries across seeds.

## Reward contract

The positive objective is:

```text
0.55 * throughput_score
+ 0.30 * fairness_score
+ 0.15 * service_score
```

A separate penalty is then subtracted:

```text
0.50 * starvation_violation
```

Definitions:

- `throughput_score`: current goodput divided by a per-slot oracle upper bound that obeys Top-K and PRB constraints;
- `fairness_score`: Jain fairness over cumulative delivered bits;
- `service_score`: one minus the normalized wait/starvation penalty;
- `starvation_violation`: normalized fraction of UEs beyond the starvation threshold.

Every component is reported independently. PPO should not be added until RR, PF, and Max-CQI sanity checks confirm that the reward reflects the intended trade-off.

## Environment contract

Observation per UE has eight values:

1. normalized CQI;
2. normalized queue backlog;
3. normalized demand factor;
4. normalized EWMA throughput;
5. normalized time since service;
6. HARQ-pending flag;
7. normalized HARQ retransmission count;
8. eligibility mask.

Raw learned action has shape `[num_ues, 2]`:

1. UE priority score;
2. PRB-demand score.

The projector turns this into:

- `selected_ues` with at most 64 UEs;
- `prbs_per_selected_ue` whose sum is exactly 273;
- pending HARQ UEs forced before new transmissions when capacity permits.

## Scope boundary

This is a **fast training surrogate**, not yet a full 5G-LENA or 3GPP-compliant scheduler. HARQ is aggregated to one simplified pending process per UE; CQI is static within each episode; all 273 PRBs are treated as PDSCH-schedulable; control-channel feasibility, exact MCS/TBS tables, frequency-selective CQI, and PHY propagation are deferred to the 5G-LENA milestone.

## Commands

Run each PowerShell command on one line:

```powershell
pytest -q
python -m scalemac_rl.scripts.run_baselines --num-ues 1200 --slots 1000 --seeds 3
python -m scalemac_rl.scripts.train_imitation --num-ues 1200 --steps 2000
python -m scalemac_rl.scripts.evaluate_policy .\artifacts\pf_imitation.pt --num-ues 1200 --slots 1000 --seeds 3
```

Generated files:

```text
artifacts/
├── baselines.csv
├── baselines.md
├── baselines_summary.csv
├── baselines_summary.md
├── imitation_training.csv
├── imitation_training.md
├── pf_imitation.pt
├── evaluation.csv
├── evaluation.md
├── evaluation_summary.csv
└── evaluation_summary.md
```
