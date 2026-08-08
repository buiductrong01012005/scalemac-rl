from scalemac_rl.scripts.verify_evaluation_consistency import compare_rows


def _row(method: str, reward: float = 0.5) -> dict[str, str]:
    row = {
        "method": method,
        "checkpoint_sha256": "abc",
        "evaluation_protocol_hash": "protocol",
        "scenario_hash": "scenario",
        "scheduler_runtime_hash": "runtime",
    }
    for field in (
        "mean_reward",
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "mean_starvation_rate",
        "max_starvation_rate",
        "final_p99_wait_slots",
        "max_p99_wait_slots",
        "final_max_wait_slots",
        "max_wait_slots",
    ):
        row[field] = str(reward)
    return row


def test_consistency_matches_different_method_labels_by_hashes() -> None:
    results = compare_rows([_row("checkpoint_stem")], [_row("hybrid_ppo")], tolerance=0.0)
    assert results[0]["matched"] is True
    assert results[0]["consistent"] is True


def test_consistency_detects_kpi_difference() -> None:
    standalone = _row("checkpoint_stem")
    unified = _row("hybrid_ppo")
    unified["final_jain_fairness"] = "0.4"
    results = compare_rows([standalone], [unified], tolerance=1e-9)
    assert results[0]["consistent"] is False
