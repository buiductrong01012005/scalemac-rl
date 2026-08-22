from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .policy_diagnostics import DiagnosticCase, evaluate_case_mode, resolve_device
from .reward_study import RewardStudyPlan


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        return default if value in {None, ''} else float(value)
    except (TypeError, ValueError):
        return default


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
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _profile(case_id: str) -> str:
    return case_id.rsplit('_seed', 1)[0] if '_seed' in case_id else case_id


def _service_feasible(row: dict[str, Any]) -> bool:
    return (
        _f(row, 'max_starvation_rate') <= 1e-12
        and _f(row, 'max_p99_wait_slots', 1e9) <= 50.0
        and _f(row, 'max_wait_slots', 1e9) <= 60.0
    )


def _final_milestone(case_dir: Path) -> Path:
    manifest = _read_csv(case_dir / 'checkpoint_manifest.csv')
    milestones = [r for r in manifest if str(r.get('tag', '')).startswith('milestone_')]
    if milestones:
        row = max(milestones, key=lambda r: int(float(r.get('global_env_steps', 0) or 0)))
        path = Path(str(row['checkpoint']))
        if path.is_file():
            return path
        candidate = case_dir.parents[3] / path if not path.is_absolute() else path
        if candidate.is_file():
            return candidate
        # Most manifests store paths relative to project root. Resolve from the
        # nearest project-style ancestor containing artifacts/.
        for parent in [case_dir, *case_dir.parents]:
            candidate = parent / path
            if candidate.is_file():
                return candidate
    direct = case_dir / 'checkpoints' / 'milestone_0098304.pt'
    if direct.is_file():
        return direct
    raise FileNotFoundError(f'final milestone checkpoint not found for {case_dir}')


def _rollback_count(training: list[dict[str, str]]) -> int:
    if not training:
        return 0
    values = [int(_f(row, 'rollback_count', 0.0)) for row in training]
    return max(values, default=0)


def build_round21_analysis(
    *,
    plan_path: str | Path,
    round_dir: str | Path,
    output_dir: str | Path,
    device_name: str = 'cpu',
    diagnostic_slots: int = 512,
) -> dict[str, Path]:
    plan = RewardStudyPlan.from_json(plan_path)
    root = Path(round_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in plan.cases:
        case_dir = root / case.case_id
        validation = _read_csv(case_dir / 'validation.csv')
        training = _read_csv(case_dir / 'training.csv')
        cfg_path = case_dir / 'run_config.json'
        if not validation or not cfg_path.is_file():
            raise FileNotFoundError(f'incomplete case: {case.case_id}')
        run_config = json.loads(cfg_path.read_text(encoding='utf-8'))
        common = run_config.get('effective_common', run_config.get('common', {}))
        seed = int(common.get('seed', case.case_id.rsplit('_seed', 1)[-1]))
        profile_seed = int(common.get('profile_seed', seed))
        latest = validation[-1]
        feasible = [r for r in validation if _service_feasible(r)]
        tail_count = max(1, len(training) // 5) if training else 0
        tail = training[-tail_count:] if tail_count else []

        final_ckpt = _final_milestone(case_dir)
        diag_case = DiagnosticCase(
            case_id=case.case_id,
            label=case.label,
            case_dir=case_dir,
            checkpoint_path=final_ckpt,
            run_config=run_config,
        )
        per_mode: dict[str, dict[str, Any]] = {}
        for mode in ('deterministic', 'stochastic'):
            summary, _, _, _ = evaluate_case_mode(
                case=diag_case,
                mode=mode,
                device=device,
                slots=diagnostic_slots,
                seed=seed,
                profile_seed=profile_seed,
                window_size=64,
                tie_epsilon=1e-4,
            )
            diag_rows.append(summary)
            per_mode[mode] = summary

        det = per_mode['deterministic']
        sto = per_mode['stochastic']
        weights = case.positive_weights
        row = {
            'case_id': case.case_id,
            'profile': _profile(case.case_id),
            'seed': seed,
            'weight_throughput': weights.get('throughput', 0.0),
            'weight_bandwidth_fairness': weights.get('fairness', 0.0),
            'weight_schedule_fairness': weights.get('schedule_fairness', 0.0),
            'weight_service': weights.get('service', 0.0),
            'input_features': int(run_config.get('architecture', {}).get('observation_features_per_ue', 16)),
            'include_time_since_schedule': int(bool(common.get('observation_include_time_since_schedule', False))),
            'include_schedule_rate_deficit': int(bool(common.get('observation_include_schedule_rate_deficit', False))),
            'include_schedule_rate_rank': int(bool(common.get('observation_include_schedule_rate_rank', False))),
            'baseline_compatible_feature_init': int(bool(common.get('baseline_compatible_feature_init', False))),
            'rollback_patience': int(common.get('rollback_patience', 1000000)),
            'rollback_lr_factor': float(common.get('rollback_lr_factor', 1.0)),
            'beta_concentration_end': float(common.get('beta_concentration_end', common.get('beta_concentration_start', 20.0))),
            'rollback_count': _rollback_count(training),
            'latest_service_feasible': int(_service_feasible(latest)),
            'ever_service_feasible': int(bool(feasible)),
            'learn_then_drift': int(bool(feasible) and not _service_feasible(latest)),
            'first_feasible_env_steps': int(min((_f(r, 'global_env_steps', 0.0) for r in feasible), default=-1)),
            'latest_goodput_bits_per_slot': _f(latest, 'mean_goodput_bits_per_slot'),
            'latest_jain_fairness': _f(latest, 'final_jain_fairness'),
            'latest_schedule_fairness': _f(latest, 'final_schedule_fairness'),
            'latest_starvation_rate': _f(latest, 'max_starvation_rate'),
            'latest_scheduling_starvation_rate': _f(latest, 'max_scheduling_starvation_rate'),
            'latest_p99_wait_slots': _f(latest, 'max_p99_wait_slots'),
            'latest_max_wait_slots': _f(latest, 'max_wait_slots'),
            'tail_approx_kl': mean(_f(r, 'approx_kl') for r in tail) if tail else 0.0,
            'tail_actor_grad': mean(_f(r, 'actor_grad_norm_probe') for r in tail) if tail else 0.0,
            'det_service_feasible_512': int(
                det['max_starvation_rate'] <= 1e-12
                and det['max_p99_wait_slots'] <= 50.0
                and det['max_wait_slots'] <= 60.0
            ),
            'det_goodput_512': det['mean_goodput_bits_per_slot'],
            'det_jfi_512': det['final_jain_fairness'],
            'det_schedule_fairness_512': det.get('final_schedule_fairness', 0.0),
            'det_starvation_512': det['max_starvation_rate'],
            'det_p99_512': det['max_p99_wait_slots'],
            'det_unique_selected_512': det['unique_selected_ues_episode'],
            'det_selection_gini_512': det['selection_gini'],
            'det_overlap_512': det['mean_selected_overlap_previous'],
            'stoch_starvation_512': sto['max_starvation_rate'],
            'stoch_schedule_fairness_512': sto.get('final_schedule_fairness', 0.0),
            'stoch_unique_selected_512': sto['unique_selected_ues_episode'],
            'stoch_selection_gini_512': sto['selection_gini'],
            'train_deploy_starvation_gap_512': det['max_starvation_rate'] - sto['max_starvation_rate'],
        }
        rows.append(row)
        grouped[row['profile']].append(row)

    summaries: list[dict[str, Any]] = []
    for profile, rs in grouped.items():
        first = rs[0]
        summaries.append({
            'profile': profile,
            'input_features': first['input_features'],
            'include_time_since_schedule': first['include_time_since_schedule'],
            'include_schedule_rate_deficit': first['include_schedule_rate_deficit'],
            'include_schedule_rate_rank': first['include_schedule_rate_rank'],
            'rollback_patience': first['rollback_patience'],
            'rollback_lr_factor': first['rollback_lr_factor'],
            'beta_concentration_end': first['beta_concentration_end'],
            'latest_service_feasible_seeds': sum(r['latest_service_feasible'] for r in rs),
            'ever_service_feasible_seeds': sum(r['ever_service_feasible'] for r in rs),
            'learn_then_drift_seeds': sum(r['learn_then_drift'] for r in rs),
            'det_service_feasible_512_seeds': sum(r['det_service_feasible_512'] for r in rs),
            'total_rollbacks': sum(r['rollback_count'] for r in rs),
            'mean_latest_goodput_bits_per_slot': mean(r['latest_goodput_bits_per_slot'] for r in rs),
            'mean_latest_jain_fairness': mean(r['latest_jain_fairness'] for r in rs),
            'mean_latest_schedule_fairness': mean(r['latest_schedule_fairness'] for r in rs),
            'mean_latest_starvation_rate': mean(r['latest_starvation_rate'] for r in rs),
            'mean_det_selection_gini_512': mean(r['det_selection_gini_512'] for r in rs),
            'mean_det_overlap_512': mean(r['det_overlap_512'] for r in rs),
            'mean_train_deploy_starvation_gap_512': mean(r['train_deploy_starvation_gap_512'] for r in rs),
            'mean_tail_approx_kl': mean(r['tail_approx_kl'] for r in rs),
        })

    ranking = sorted(
        summaries,
        key=lambda r: (
            -r['latest_service_feasible_seeds'],
            -r['ever_service_feasible_seeds'],
            r['learn_then_drift_seeds'],
            r['mean_latest_starvation_rate'],
            r['mean_det_selection_gini_512'],
            -r['mean_latest_jain_fairness'],
        ),
    )
    ranking_rows = [{'rank': i + 1, **r} for i, r in enumerate(ranking)]
    best = ranking_rows[0] if ranking_rows else None
    decision = {
        'study_id': plan.study_id,
        'round_id': plan.round_id,
        'cases': len(rows),
        'profiles': len(summaries),
        'best_profile': best['profile'] if best else None,
        'best_latest_service_feasible_seeds': best['latest_service_feasible_seeds'] if best else None,
        'best_ever_service_feasible_seeds': best['ever_service_feasible_seeds'] if best else None,
        'principle': 'All policies are trained from scratch with PPO. No PF/oracle demonstrations or expert action labels are used.',
        'diagnostic_note': 'Post-hoc deterministic and stochastic diagnostics use each case effective environment/profile seed and the exact final environment-step milestone checkpoint.',
    }

    case_csv = out / 'round21_case_metrics.csv'
    summary_csv = out / 'round21_profile_summary.csv'
    ranking_csv = out / 'round21_ranking.csv'
    diag_csv = out / 'round21_final_diagnostics.csv'
    decision_json = out / 'round21_decision.json'
    html_path = out / 'round21_analysis.html'
    _write_csv(case_csv, rows)
    _write_csv(summary_csv, summaries)
    _write_csv(ranking_csv, ranking_rows)
    _write_csv(diag_csv, diag_rows)
    decision_json.write_text(json.dumps(decision, indent=2), encoding='utf-8')

    body = ''.join(
        '<tr>'
        f"<td>{r['rank']}</td><td>{html.escape(str(r['profile']))}</td>"
        f"<td>{r['latest_service_feasible_seeds']}/3</td>"
        f"<td>{r['ever_service_feasible_seeds']}/3</td>"
        f"<td>{r['learn_then_drift_seeds']}/3</td>"
        f"<td>{r['det_service_feasible_512_seeds']}/3</td>"
        f"<td>{r['mean_latest_jain_fairness']:.4f}</td>"
        f"<td>{r['mean_latest_schedule_fairness']:.4f}</td>"
        f"<td>{100*r['mean_latest_starvation_rate']:.2f}%</td>"
        f"<td>{r['mean_det_selection_gini_512']:.4f}</td>"
        '</tr>'
        for r in ranking_rows
    )
    html_path.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Round 21 analysis</title>'
        '<style>body{font-family:Segoe UI,Arial;max-width:1200px;margin:30px auto;padding:0 18px;line-height:1.5}'
        'table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:7px}th{background:#f3f4f6}</style>'
        '</head><body>'
        f'<h1>{html.escape(plan.round_id)}</h1>'
        '<p>Pure PPO study: no teacher imitation, no expert action supervision.</p>'
        '<table><thead><tr><th>Rank</th><th>Profile</th><th>Latest feasible</th><th>Ever feasible</th><th>Drift</th><th>Final diag feasible</th><th>Jain</th><th>Schedule fair</th><th>Starvation</th><th>Selection Gini</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
        f'<pre>{html.escape(json.dumps(decision, indent=2))}</pre>'
        '</body></html>',
        encoding='utf-8',
    )
    return {
        'case_metrics': case_csv,
        'profile_summary': summary_csv,
        'ranking': ranking_csv,
        'final_diagnostics': diag_csv,
        'decision': decision_json,
        'html': html_path,
    }
