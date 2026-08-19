from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from scalemac_rl.config import ScaleMacConfig
from scalemac_rl.oracle_sanity import evaluate_policy, load_feedforward_policy, service_aware_oracle_action, write_csv
from scalemac_rl.schedulers import ProportionalFairScheduler


def _config(seed: int, slots: int) -> ScaleMacConfig:
    return ScaleMacConfig(
        num_ues=1200, num_prbs=273, max_selected_ues=64, episode_slots=slots,
        scheduler_mode="ppo_only", force_harq_retransmissions=False,
        freeze_static_profiles=True, static_profile_seed=seed, seed=seed,
        cqi_mode="correlated", cqi_temporal_correlation=0.97,
        cqi_innovation_std=0.35, cqi_update_interval_slots=1,
        cqi_max_delta_per_update=1, csi_report_mode="periodic",
        csi_report_period_slots=4, csi_report_delay_slots=2,
        csi_report_error_std=0.0, link_adaptation_mode="cqi_mcs_bler",
        link_adaptation_cqi_backoff=0, bler_mismatch_slope=1.5,
        starvation_threshold_slots=64,
        reward_throughput_weight=1/3, reward_fairness_weight=1/3,
        reward_service_weight=1/3, reward_deficit_service_weight=0.0,
        reward_pf_utility_weight=0.0, reward_low_throughput_weight=0.0,
        reward_urgency_service_weight=0.0, reward_fairness_delta_weight=0.0,
        reward_pf_utility_delta_weight=0.0, reward_starvation_penalty_weight=0.0,
        reward_deadline_risk_penalty_weight=0.0,
        reward_max_wait_risk_penalty_weight=0.0,
        reward_population_wait_penalty_weight=0.0,
    )


def _summary(rows):
    out=[]
    for policy in ("oracle", "pf", "ppo"):
        group=[r for r in rows if r["policy"]==policy]
        rec={"policy":policy,"seeds":len(group),"zero_starvation_seeds":sum(int(r["zero_starvation"]) for r in group),"service_feasible_seeds":sum(int(r["service_feasible_under_64"]) for r in group)}
        for key in ("mean_goodput_bits_per_slot","mean_spectral_efficiency_bps_hz","final_jain_fairness","max_starvation_rate","max_p99_wait_slots","max_wait_slots","mean_observed_bler","mean_harq_retransmission_fraction"):
            vals=[float(r[key]) for r in group]
            rec[key+"_mean"]=mean(vals); rec[key+"_std"]=stdev(vals) if len(vals)>1 else 0.0
        out.append(rec)
    return out


def main():
    ap=argparse.ArgumentParser(description="Oracle/PF/PPO seed-feasibility sanity check")
    ap.add_argument("--training-root",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--seeds",default="1701,2701,3701")
    ap.add_argument("--slots",type=int,default=5000)
    ap.add_argument("--device",choices=["cpu","cuda","auto"],default="cpu")
    args=ap.parse_args()
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    if args.device=="cuda" or (args.device=="auto" and torch.cuda.is_available()): device=torch.device("cuda")
    else: device=torch.device("cpu")
    rows=[]
    for seed in seeds:
        cfg=_config(seed,args.slots)
        rows.append(evaluate_policy(name="oracle",config=cfg,seed=seed,action_fn=lambda env,obs: service_aware_oracle_action(env)))
        pf=ProportionalFairScheduler(); pf.reset()
        rows.append(evaluate_policy(name="pf",config=cfg,seed=seed,action_fn=lambda env,obs,pf=pf: pf.act(obs)))
        ckpt=args.training_root/f"baseline_seed{seed}"/"latest.pt"
        model=load_feedforward_policy(ckpt,device)
        def ppo_action(env,obs,model=model):
            with torch.no_grad():
                t=torch.as_tensor(obs,dtype=torch.float32,device=device)
                return model.deterministic_action(t).action.detach().cpu().numpy()
        rows.append(evaluate_policy(name="ppo",config=cfg,seed=seed,action_fn=ppo_action))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/"oracle_sanity_metrics.csv",rows)
    summary=_summary(rows); write_csv(args.output_dir/"oracle_sanity_summary.csv",summary)
    payload={"seeds":seeds,"slots":args.slots,"oracle_definition":"current-state privileged service-aware oracle using true CQI and exact service wait; no future knowledge","all_oracle_seeds_service_feasible":all(int(r["service_feasible_under_64"]) for r in rows if r["policy"]=="oracle")}
    (args.output_dir/"oracle_sanity_decision.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2)); print(f"saved: {args.output_dir}")

if __name__=="__main__": main()
