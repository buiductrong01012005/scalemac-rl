from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np
import torch

from scalemac_rl.reproducibility import checkpoint_model_sha256, sha256_file
from scalemac_rl.reward_study import RewardStudyPlan, safe_float

_METRICS = (
    ("mean_goodput_bits_per_slot", "Goodput", True),
    ("final_jain_fairness", "Jain", True),
    ("max_starvation_rate", "Starvation", False),
    ("max_p99_wait_slots", "P99 wait", False),
    ("max_wait_slots", "Max wait", False),
)
_TIMING_COLUMNS = {"elapsed_seconds", "steps_per_second", "eta_seconds"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    items = [float(v) for v in values if math.isfinite(float(v))]
    if not items:
        return 0.0, 0.0
    return mean(items), stdev(items) if len(items) > 1 else 0.0


def _status(validation_rows: list[dict[str, str]]) -> str:
    if not validation_rows:
        return "Incomplete"
    final = validation_rows[-1]
    starvation = safe_float(final, "max_starvation_rate")
    p99 = safe_float(final, "max_p99_wait_slots")
    wait = safe_float(final, "max_wait_slots")
    if starvation > 0.10 or p99 >= 5000 or wait >= 5000:
        return "Full collapse"
    if starvation > 0 or p99 >= 64 or wait >= 80:
        return "Borderline"
    return "Stable"


def _training_matrix(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray]:
    if not rows:
        return [], np.empty((0, 0), dtype=np.float64)
    columns: list[str] = []
    for key in rows[0]:
        if key in _TIMING_COLUMNS or key == "device":
            continue
        try:
            float(rows[0][key])
        except (TypeError, ValueError):
            continue
        columns.append(key)
    matrix = np.array([[float(row[key]) for key in columns] for row in rows], dtype=np.float64)
    return columns, matrix


def _first_divergence(reference: np.ndarray, other: np.ndarray, tolerance: float) -> tuple[int | None, float]:
    if reference.shape != other.shape:
        return 0, float("inf")
    if reference.size == 0:
        return None, 0.0
    diff = np.abs(reference - other)
    row_max = np.nanmax(diff, axis=1)
    indices = np.flatnonzero(row_max > tolerance)
    return (int(indices[0] + 1) if indices.size else None), float(np.nanmax(diff))


def _checkpoint_initial_hash(run_dir: Path) -> str:
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt")) if (run_dir / "checkpoints").is_dir() else []
    if not checkpoints:
        return ""
    # The first milestone/checkpoint is not the true initialized model, so runtime metadata is preferred.
    return ""


def build_reproducibility_analysis(*, plan: RewardStudyPlan, round_dir: Path, output_path: Path) -> Path:
    repeat_rows: list[dict[str, Any]] = []
    training_by_case: dict[str, tuple[list[str], np.ndarray]] = {}
    metadata_by_case: dict[str, dict[str, Any]] = {}

    for case in plan.cases:
        run_dir = round_dir / case.case_id
        validation = _read_csv(run_dir / "validation.csv")
        training = _read_csv(run_dir / "training.csv")
        training_by_case[case.case_id] = _training_matrix(training)
        metadata_path = run_dir / "runtime_fingerprint.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        metadata_by_case[case.case_id] = metadata
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "repeat": case.common_overrides.get("repeat_index", ""),
            "seed": int(case.common_overrides.get("seed", plan.common.get("seed", 1701))),
            "status": _status(validation),
            "training_rows": len(training),
            "validation_rows": len(validation),
        }
        if validation:
            final = validation[-1]
            for key, _, _ in _METRICS:
                row[key] = safe_float(final, key)
            row["global_env_steps"] = int(safe_float(final, "global_env_steps"))
            row["device"] = final.get("device", "")
        else:
            for key, _, _ in _METRICS:
                row[key] = float("nan")
            row["global_env_steps"] = 0
            row["device"] = ""
        if (run_dir / "latest.pt").is_file():
            row["final_model_parameter_sha256"] = checkpoint_model_sha256(run_dir / "latest.pt")
            row["latest_checkpoint_file_sha256"] = sha256_file(run_dir / "latest.pt")
        else:
            row["final_model_parameter_sha256"] = ""
            row["latest_checkpoint_file_sha256"] = ""
        row["initial_model_parameter_sha256"] = metadata.get("initial_model_parameter_sha256", "")
        row["numpy_rng_after_seed_sha256"] = metadata.get("rng_after_seed", {}).get("numpy", "")
        row["torch_rng_after_seed_sha256"] = metadata.get("rng_after_seed", {}).get("torch_cpu", "")
        repeat_rows.append(row)

    completed = [row for row in repeat_rows if row["status"] != "Incomplete"]
    reference_case = completed[0]["case_id"] if completed else ""
    pair_rows: list[dict[str, Any]] = []
    if reference_case:
        ref_columns, ref_matrix = training_by_case[reference_case]
        for row in completed[1:]:
            case_id = str(row["case_id"])
            columns, matrix = training_by_case[case_id]
            same_columns = columns == ref_columns
            divergence_0, max_diff = _first_divergence(ref_matrix, matrix, 0.0) if same_columns else (0, float("inf"))
            divergence_1e12, _ = _first_divergence(ref_matrix, matrix, 1e-12) if same_columns else (0, float("inf"))
            divergence_1e9, _ = _first_divergence(ref_matrix, matrix, 1e-9) if same_columns else (0, float("inf"))
            pair_rows.append({
                "reference_case": reference_case,
                "comparison_case": case_id,
                "same_numeric_columns": same_columns,
                "first_divergent_training_row_exact": divergence_0 if divergence_0 is not None else "none",
                "first_divergent_training_row_tol_1e-12": divergence_1e12 if divergence_1e12 is not None else "none",
                "first_divergent_training_row_tol_1e-9": divergence_1e9 if divergence_1e9 is not None else "none",
                "max_abs_training_metric_diff": max_diff,
                "same_initial_model_hash": row.get("initial_model_parameter_sha256") == completed[0].get("initial_model_parameter_sha256"),
                "same_final_model_hash": row.get("final_model_parameter_sha256") == completed[0].get("final_model_parameter_sha256"),
                "same_numpy_rng_after_seed": row.get("numpy_rng_after_seed_sha256") == completed[0].get("numpy_rng_after_seed_sha256"),
                "same_torch_rng_after_seed": row.get("torch_rng_after_seed_sha256") == completed[0].get("torch_rng_after_seed_sha256"),
            })

    summary_rows: list[dict[str, Any]] = []
    for key, label, _ in _METRICS:
        avg, sd = _mean_std(row[key] for row in completed if math.isfinite(float(row[key])))
        values = [float(row[key]) for row in completed if math.isfinite(float(row[key]))]
        summary_rows.append({
            "metric": key,
            "label": label,
            "mean": avg,
            "std": sd,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
            "range": max(values) - min(values) if values else float("nan"),
        })

    analysis = plan.analysis
    repeat_output = Path(str(analysis.get("repeat_metrics_output", output_path.with_name("repeat_metrics.csv"))))
    pair_output = Path(str(analysis.get("pairwise_output", output_path.with_name("pairwise_repeatability.csv"))))
    summary_output = Path(str(analysis.get("summary_output", output_path.with_name("repeatability_summary.csv"))))
    markdown_output = Path(str(analysis.get("markdown_output", output_path.with_suffix(".md"))))
    _write_csv(repeat_output, repeat_rows)
    _write_csv(pair_output, pair_rows)
    _write_csv(summary_output, summary_rows)

    identical_initial = bool(completed) and len({row.get("initial_model_parameter_sha256", "") for row in completed}) == 1
    identical_final = bool(completed) and len({row.get("final_model_parameter_sha256", "") for row in completed}) == 1
    all_stable = bool(completed) and all(row["status"] == "Stable" for row in completed)
    pair_exact = bool(pair_rows) and all(row["first_divergent_training_row_exact"] == "none" for row in pair_rows)
    reproducible = len(completed) == len(plan.cases) and identical_initial and identical_final and pair_exact

    if reproducible:
        conclusion = "Ba lần chạy cùng seed/config cho trajectory và model cuối giống hệt nhau trong cùng runtime: within-session reproducibility đạt yêu cầu."
        next_step = "Có thể chuyển sang Dynamic CQI; vẫn nên lưu runtime fingerprint cho các round tiếp theo."
    else:
        conclusion = "Các repeat cùng seed/config chưa trùng hoàn toàn; pipeline hiện tại chưa đạt within-session reproducibility đủ mạnh để tiếp tục reward/model comparison một cách tự tin."
        next_step = "Chạy pha deterministic-lock (1 CPU thread + deterministic algorithms), sau đó so sánh lại trước khi mở Dynamic CQI."

    runtime_rows = []
    for case_id, metadata in metadata_by_case.items():
        runtime = metadata.get("runtime", metadata)
        torch_runtime = runtime.get("torch_runtime", {}) if isinstance(runtime, dict) else {}
        packages = runtime.get("packages", {}) if isinstance(runtime, dict) else {}
        thread_env = runtime.get("thread_environment", {}) if isinstance(runtime, dict) else {}
        runtime_rows.append({
            "case_id": case_id,
            "python": runtime.get("python", {}).get("version", "") if isinstance(runtime, dict) else "",
            "numpy": packages.get("numpy", ""),
            "torch": packages.get("torch", ""),
            "cpu_model": runtime.get("platform", {}).get("cpu_model", "") if isinstance(runtime, dict) else "",
            "torch_num_threads": torch_runtime.get("num_threads", ""),
            "torch_num_interop_threads": torch_runtime.get("num_interop_threads", ""),
            "deterministic_algorithms": torch_runtime.get("deterministic_algorithms_enabled", ""),
            "OMP_NUM_THREADS": thread_env.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": thread_env.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": thread_env.get("OPENBLAS_NUM_THREADS", ""),
        })
    runtime_output = Path(str(analysis.get("runtime_output", output_path.with_name("runtime_fingerprints.csv"))))
    _write_csv(runtime_output, runtime_rows)

    title = str(analysis.get("title", "Reproducibility diagnostic"))
    rows_html = "".join(
        "<tr>" +
        f"<td><code>{html.escape(str(r['case_id']))}</code></td>" +
        f"<td>{html.escape(str(r['status']))}</td>" +
        f"<td>{float(r['mean_goodput_bits_per_slot']):,.0f}</td>" +
        f"<td>{float(r['final_jain_fairness']):.4f}</td>" +
        f"<td>{100*float(r['max_starvation_rate']):.2f}%</td>" +
        f"<td>{float(r['max_p99_wait_slots']):.0f}</td>" +
        f"<td>{float(r['max_wait_slots']):.0f}</td>" +
        f"<td><code>{html.escape(str(r['final_model_parameter_sha256']))[:12]}…</code></td>" +
        "</tr>"
        for r in completed
    )
    pair_html = "".join(
        "<tr>" +
        f"<td><code>{html.escape(str(r['comparison_case']))}</code></td>" +
        f"<td>{html.escape(str(r['first_divergent_training_row_exact']))}</td>" +
        f"<td>{html.escape(str(r['first_divergent_training_row_tol_1e-12']))}</td>" +
        f"<td>{html.escape(str(r['first_divergent_training_row_tol_1e-9']))}</td>" +
        f"<td>{'Có' if r['same_initial_model_hash'] else 'Không'}</td>" +
        f"<td>{'Có' if r['same_final_model_hash'] else 'Không'}</td>" +
        "</tr>"
        for r in pair_rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title><style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:auto;padding:28px;color:#172033}}h1,h2{{line-height:1.2}}.box{{padding:16px 20px;border:1px solid #d8dee9;border-radius:12px;margin:16px 0}}.good{{border-left:6px solid #16833a}}.bad{{border-left:6px solid #c43a3a}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d8dee9;padding:8px;text-align:left}}th{{background:#f3f6fa}}code{{background:#f2f4f7;padding:2px 5px;border-radius:4px}}</style></head><body>
<h1>{html.escape(title)}</h1>
<div class='box {'good' if reproducible else 'bad'}'><h2>Kết luận</h2><p><b>{html.escape(conclusion)}</b></p><p>{html.escape(next_step)}</p></div>
<h2>Ta đang kiểm tra cái gì?</h2><p>Ba case dùng đúng cùng reward <code>R = 1/3 T + 1/3 J + 1/3 S</code>, seed/profile seed/validation seed = 1701 và cùng PPO config. Khác biệt duy nhất là thư mục output/repeat id. Mục tiêu là đo xem cùng một thí nghiệm có cho cùng trajectory hay không.</p>
<h2>Kết quả từng repeat</h2><table><thead><tr><th>Repeat</th><th>Status</th><th>Goodput</th><th>Jain</th><th>Starvation</th><th>P99</th><th>Max wait</th><th>Model hash</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>Trajectory bắt đầu lệch ở đâu?</h2><p>So với repeat đầu. Các cột timing bị loại khỏi phép so sánh vì tốc độ chạy không phải tín hiệu học.</p><table><thead><tr><th>Repeat</th><th>Lệch exact ở row</th><th>Lệch &gt;1e-12</th><th>Lệch &gt;1e-9</th><th>Initial model giống?</th><th>Final model giống?</th></tr></thead><tbody>{pair_html}</tbody></table>
<h2>Cách ra quyết định</h2><ul><li>Nếu cả ba repeat giống trajectory/model cuối: within-session reproducibility đạt, chuyển sang Dynamic CQI.</li><li>Nếu initial model/RNG giống nhưng trajectory tách: nghi ngờ numerical/thread nondeterminism; chạy deterministic-lock.</li><li>Nếu initial model/RNG đã khác: sửa seeding/RNG path trước.</li><li>Không tune reward thêm trong diagnostic này.</li></ul>
<p><b>All stable:</b> {all_stable}. <b>Initial hash giống:</b> {identical_initial}. <b>Final hash giống:</b> {identical_final}. <b>Within-session reproducible:</b> {reproducible}.</p>
</body></html>""", encoding="utf-8")

    markdown_output.write_text(
        f"# {title}\n\n"
        f"- Completed: {len(completed)}/{len(plan.cases)} repeats.\n"
        f"- Conclusion: {conclusion}\n"
        f"- Next: {next_step}\n"
        f"- Identical initial model hash: {identical_initial}.\n"
        f"- Identical final model hash: {identical_final}.\n"
        f"- Within-session reproducible: {reproducible}.\n",
        encoding="utf-8",
    )
    return output_path
