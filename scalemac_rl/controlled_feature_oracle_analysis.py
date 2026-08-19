from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_controlled_feature_oracle_report(*, docs_dir: Path, output: Path) -> Path:
    feature_path = docs_dir / "controlled_feature_metrics.csv"
    oracle_path = docs_dir / "oracle_sanity_metrics.csv"
    feature = _read(feature_path)
    oracle = _read(oracle_path)
    oracle_by_seed = {int(r["seed"]): r for r in oracle if r["policy"] == "oracle"}
    rows=[]
    for r in feature:
        seed=int(r["seed"])
        o=oracle_by_seed.get(seed,{})
        rows.append(
            f"<tr><td>{html.escape(r['profile'])}</td><td>{seed}</td>"
            f"<td>{float(r['goodput_bits_per_slot']):,.0f}</td>"
            f"<td>{float(r['jain_fairness']):.4f}</td>"
            f"<td>{100*float(r['starvation_rate']):.2f}%</td>"
            f"<td>{float(r['p99_wait_slots']):.0f}</td>"
            f"<td>{'YES' if int(o.get('service_feasible_under_64','0')) else 'NO'}</td></tr>"
        )
    oracle_rows=[]
    for r in oracle:
        oracle_rows.append(
            f"<tr><td>{r['policy'].upper()}</td><td>{r['seed']}</td>"
            f"<td>{float(r['mean_goodput_bits_per_slot']):,.0f}</td>"
            f"<td>{float(r['final_jain_fairness']):.4f}</td>"
            f"<td>{100*float(r['max_starvation_rate']):.2f}%</td>"
            f"<td>{float(r['max_p99_wait_slots']):.0f}</td>"
            f"<td>{float(r['max_wait_slots']):.0f}</td></tr>"
        )
    all_feasible=all(int(r["service_feasible_under_64"]) for r in oracle if r["policy"]=="oracle")
    text=f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Round 14C</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1120px;margin:32px auto;padding:0 18px;line-height:1.55;color:#1f2937}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d1d5db;padding:8px;text-align:left}}th{{background:#f3f4f6}}.callout{{border-left:5px solid #2563eb;background:#eff6ff;padding:14px 16px;margin:18px 0}}</style></head><body>
<h1>Round 14C — Controlled Features + Oracle Sanity</h1>
<div class='callout'><b>Oracle sanity:</b> {'PASS — all three seeds remain service-feasible under the privileged current-state oracle.' if all_feasible else 'NOT FULLY PASSED — at least one seed is not service-feasible under the diagnostic oracle.'}</div>
<p>Feature cases use baseline-compatible initialization: the original 16-feature policy and post-initialization RNG are paired exactly; the added feature column starts with zero weight.</p>
<h2>Controlled feature cases</h2><table><thead><tr><th>Profile</th><th>Seed</th><th>Goodput</th><th>JFI</th><th>Starvation</th><th>P99</th><th>Oracle feasible seed?</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Environment sanity policies</h2><table><thead><tr><th>Policy</th><th>Seed</th><th>Goodput</th><th>JFI</th><th>Max starvation</th><th>Max P99</th><th>Max wait</th></tr></thead><tbody>{''.join(oracle_rows)}</tbody></table>
<p><b>Interpretation rule:</b> if PPO collapses on a seed where the oracle keeps zero starvation and P99 below 64 slots, the seed is not dismissed as an intrinsically impossible environment; the failure remains attributable to policy/training/observation limitations under this abstraction.</p>
</body></html>"""
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(text,encoding="utf-8")
    return output
