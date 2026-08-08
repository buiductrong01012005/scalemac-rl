from __future__ import annotations

import json
from pathlib import Path

import torch

from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.policy_diagnostics import (
    DiagnosticCase,
    build_html_report,
    discover_cases,
    parse_modes,
    run_diagnostics,
)


def _write_case(root: Path, case_id: str = "test_case") -> DiagnosticCase:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    run_config = {
        "architecture": {
            "num_ues": 16,
            "num_prbs": 16,
            "top_k": 8,
            "embedding_dim": 16,
        },
        "common": {
            "starvation_threshold_slots": 8,
            "profile_seed": 11,
        },
        "case": {
            "id": case_id,
            "label": "50% throughput + 50% Jain fairness",
            "positive_scale": 1.0,
            "positive_weights": {
                "throughput": 0.5,
                "fairness": 0.5,
                "service": 0.0,
                "deficit_service": 0.0,
                "pf_utility": 0.0,
                "low_throughput": 0.0,
                "urgency_service": 0.0,
            },
            "delta_weights": {"fairness": 0.0, "pf_utility": 0.0},
            "penalty_weights": {
                "starvation": 0.0,
                "deadline_risk": 0.0,
                "max_wait_risk": 0.0,
                "population_wait": 0.0,
            },
            "actual_coefficients": {
                "coef_throughput": 0.5,
                "coef_fairness": 0.5,
            },
        },
    }
    (case_dir / "run_config.json").write_text(
        json.dumps(run_config), encoding="utf-8"
    )
    model = SharedSetActorCritic(input_dim=16, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": 16,
            "checkpoint_tag": "test",
        },
        case_dir / "latest.pt",
    )
    return DiagnosticCase(
        case_id=case_id,
        label=run_config["case"]["label"],
        case_dir=case_dir,
        checkpoint_path=case_dir / "latest.pt",
        run_config=run_config,
    )


def test_parse_modes_rejects_unknown_mode() -> None:
    assert parse_modes("deterministic,stochastic") == (
        "deterministic",
        "stochastic",
    )
    try:
        parse_modes("invalid")
    except ValueError as exc:
        assert "unknown evaluation modes" in str(exc)
    else:
        raise AssertionError("unknown mode should fail")


def test_discover_cases_reads_checkpoint_and_config(tmp_path: Path) -> None:
    expected = _write_case(tmp_path)
    found = discover_cases(
        study_root=tmp_path,
        case_ids=[expected.case_id],
        checkpoint_name="latest.pt",
    )
    assert len(found) == 1
    assert found[0].label == expected.label


def test_small_policy_diagnostic_run_writes_single_folder(tmp_path: Path) -> None:
    case = _write_case(tmp_path / "study")
    output_root = tmp_path / "artifacts" / "round_03"
    docs_output = tmp_path / "docs" / "analysis" / "diagnostic.html"
    paths = run_diagnostics(
        cases=[case],
        modes=("deterministic", "stochastic"),
        device=torch.device("cpu"),
        slots=16,
        first_seed=11,
        seeds=1,
        profile_seed=11,
        window_size=8,
        tie_epsilon=1e-6,
        output_root=output_root,
        docs_output=docs_output,
    )
    assert paths["summary"].is_file()
    assert paths["slot"].is_file()
    assert paths["window"].is_file()
    assert paths["ue"].is_file()
    assert paths["html"].is_file()
    assert docs_output.is_file()
    content = docs_output.read_text(encoding="utf-8")
    assert "deterministic" in content
    assert "stochastic" in content
    assert "Top-64" in content


def test_html_report_explains_starvation_window(tmp_path: Path) -> None:
    case = _write_case(tmp_path / "study")
    output = tmp_path / "analysis.html"
    rows = [
        {
            "case_id": case.case_id,
            "case_label": case.label,
            "mode": "deterministic",
            "mean_goodput_bits_per_slot": 1.0,
            "final_jain_fairness": 0.5,
            "max_starvation_rate": 0.0,
            "max_p99_wait_slots": 4.0,
            "max_wait_slots": 5.0,
            "unique_selected_ues_episode": 16,
            "never_selected_ues_episode": 0,
            "selection_top64_share": 0.5,
            "mean_mean_top_k_margin": 0.01,
        }
    ]
    build_html_report(
        output=output,
        summaries=rows,
        cases=[case],
        window_size=64,
        tie_epsilon=1e-6,
    )
    text = output.read_text(encoding="utf-8")
    assert "64 action" in text
    assert "UE từng được chọn" in text
