from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import torch

from .policy_diagnostics import DiagnosticCase, evaluate_case_mode


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _profile(case_id: str) -> str:
    if case_id.startswith('joint_'): return 'joint'
    if case_id.startswith('disc_exact_'): return 'disc_exact'
    if case_id.startswith('disc_norm_'): return 'disc_norm'
    raise ValueError(case_id)


def _service_feasible(row: pd.Series) -> bool:
    return float(row['max_starvation_rate']) <= 1e-12 and float(row['max_p99_wait_slots']) <= 50.0 + 1e-12 and float(row['max_wait_slots']) <= 60.0 + 1e-12


def build_dimensionwise_analysis(*, round_dir: Path, output_dir: Path, device: str='cpu', diagnostic_slots: int=512) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev=torch.device(device)
    case_rows=[]; diagnostic_rows=[]
    case_dirs=sorted(p for p in round_dir.iterdir() if p.is_dir() and (p/'run_config.json').is_file())
    if len(case_dirs)!=9: raise ValueError(f'expected 9 cases, found {len(case_dirs)}')
    for case_dir in case_dirs:
        case_id=case_dir.name; profile=_profile(case_id)
        rc=json.loads((case_dir/'run_config.json').read_text(encoding='utf-8'))
        common=dict(rc.get('common',{})); cp=dict(rc.get('case',{}))
        seed=int(common.get('environment_seed',common.get('seed',1701))); profile_seed=int(common.get('profile_seed',seed))
        tr=pd.read_csv(case_dir/'training.csv'); va=pd.read_csv(case_dir/'validation.csv'); latest=va.iloc[-1]
        tail=tr.iloc[-max(1,len(tr)//5):]
        ever=any(_service_feasible(r) for _,r in va.iterrows())
        dcase=DiagnosticCase(case_id=case_id,label=str(cp.get('label',case_id)),case_dir=case_dir,checkpoint_path=case_dir/'latest.pt',run_config=rc)
        modes={}
        for mode in ('deterministic','stochastic'):
            sm,_,_,_=evaluate_case_mode(case=dcase,mode=mode,device=dev,slots=diagnostic_slots,seed=seed,profile_seed=profile_seed,window_size=64,tie_epsilon=1e-6)
            sm=dict(sm); sm['profile']=profile; diagnostic_rows.append(sm); modes[mode]=sm
        det=modes['deterministic']; sto=modes['stochastic']
        case_rows.append({
            'case_id':case_id,'profile':profile,'seed':seed,
            'ever_service_feasible':int(ever),'latest_service_feasible':int(_service_feasible(latest)),
            'latest_goodput':float(latest['mean_goodput_bits_per_slot']),'latest_jfi':float(latest['final_jain_fairness']),
            'latest_starvation':float(latest['max_starvation_rate']),'latest_p99':float(latest['max_p99_wait_slots']),
            'tail_approx_kl':float(tail['approx_kl'].mean()),'max_approx_kl':float(tr['max_approx_kl'].max()),
            'tail_clip_fraction':float(tail['clip_fraction'].mean()),'tail_grad_clip_fraction':float(tail['grad_clip_fraction'].mean()),
            'tail_disc_is_loss':float(tail['disc_is_loss'].mean()) if 'disc_is_loss' in tail else 0.0,
            'tail_disc_alpha':float(tail['disc_is_alpha_after'].mean()) if 'disc_is_alpha_after' in tail else 0.0,
            'det_jfi':float(det['final_jain_fairness']),'det_starvation':float(det['max_starvation_rate']),
            'det_unique_selected':int(det['unique_selected_ues_episode']),'det_gini':float(det['selection_gini']),'det_overlap':float(det['mean_selected_overlap_previous']),
            'sto_jfi':float(sto['final_jain_fairness']),'sto_starvation':float(sto['max_starvation_rate']),
            'sto_unique_selected':int(sto['unique_selected_ues_episode']),
            'alignment_starvation_gap':float(det['max_starvation_rate'])-float(sto['max_starvation_rate']),
        })
    profile_rows=[]
    for profile in ('joint','disc_exact','disc_norm'):
        rows=[r for r in case_rows if r['profile']==profile]
        profile_rows.append({
            'profile':profile,
            'latest_service_feasible_seeds':sum(r['latest_service_feasible'] for r in rows),
            'ever_service_feasible_seeds':sum(r['ever_service_feasible'] for r in rows),
            'mean_det_starvation':mean(r['det_starvation'] for r in rows),
            'mean_det_jfi':mean(r['det_jfi'] for r in rows),
            'mean_det_unique_selected':mean(r['det_unique_selected'] for r in rows),
            'mean_alignment_starvation_gap':mean(r['alignment_starvation_gap'] for r in rows),
            'mean_tail_kl':mean(r['tail_approx_kl'] for r in rows),
            'mean_tail_clip_fraction':mean(r['tail_clip_fraction'] for r in rows),
            'mean_tail_disc_is_loss':mean(r['tail_disc_is_loss'] for r in rows),
            'mean_tail_disc_alpha':mean(r['tail_disc_alpha'] for r in rows),
        })
    ranking=sorted(profile_rows,key=lambda r:(-r['latest_service_feasible_seeds'],-r['ever_service_feasible_seeds'],r['mean_det_starvation'],-r['mean_det_unique_selected'],r['mean_alignment_starvation_gap']))
    for i,r in enumerate(ranking,1): r['rank']=i
    decision={'round':'18B','recommended_profile':ranking[0]['profile'],'latest_service_feasible_seeds':ranking[0]['latest_service_feasible_seeds'],'ever_service_feasible_seeds':ranking[0]['ever_service_feasible_seeds'],'method_note':'disc_exact implements the on-policy dimension-wise clipping + adaptive J_IS core of Han & Sung (2019), without replay-buffer reuse or GAE-V; disc_norm is a ScaleMAC loss-scale-normalized ablation.'}
    paths={'case':output_dir/'dimensionwise_case_metrics.csv','profile':output_dir/'dimensionwise_profile_summary.csv','diagnostics':output_dir/'dimensionwise_latest_diagnostics.csv','decision':output_dir/'dimensionwise_decision.json','html':output_dir/'dimensionwise_ppo.html'}
    _write_csv(paths['case'],case_rows); _write_csv(paths['profile'],ranking); _write_csv(paths['diagnostics'],diagnostic_rows); paths['decision'].write_text(json.dumps(decision,indent=2),encoding='utf-8')
    rows_html=''.join(f"<tr><td>{r['rank']}</td><td>{r['profile']}</td><td>{r['latest_service_feasible_seeds']}/3</td><td>{r['ever_service_feasible_seeds']}/3</td><td>{r['mean_det_starvation']*100:.2f}%</td><td>{r['mean_det_jfi']:.3f}</td><td>{r['mean_det_unique_selected']:.0f}/1200</td><td>{r['mean_tail_kl']:.5f}</td><td>{r['mean_tail_disc_is_loss']:.5f}</td></tr>" for r in ranking)
    paths['html'].write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Round18B</title><style>body{{font-family:Segoe UI,Arial;max-width:1100px;margin:32px auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}th{{background:#f3f4f6}}</style></head><body><h1>Round 18B — Dimension-wise PPO</h1><p>Recommended: <b>{ranking[0]['profile']}</b></p><table><thead><tr><th>Rank</th><th>Profile</th><th>Latest feasible</th><th>Ever feasible</th><th>Det starvation</th><th>Det JFI</th><th>UE coverage</th><th>Tail KL</th><th>J_IS</th></tr></thead><tbody>{rows_html}</tbody></table><p>{decision['method_note']}</p></body></html>""",encoding='utf-8')
    return paths
