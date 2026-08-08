from scalemac_rl.tradeoff import annotate_tradeoff_scores, validation_tradeoff_metrics


def _row(method: str, goodput: float, fairness: float, p99: float, max_wait: float, starvation: float):
    return {
        "method": method,
        "seed": 1,
        "mean_goodput_bits_per_slot": goodput,
        "final_jain_fairness": fairness,
        "max_p99_wait_slots": p99,
        "max_wait_slots": max_wait,
        "mean_starvation_rate": starvation,
    }


def test_tradeoff_ideal_uses_starvation_feasible_methods() -> None:
    rows = annotate_tradeoff_scores(
        [
            _row("unsafe", 200.0, 0.05, 5000.0, 5000.0, 0.9),
            _row("balanced", 100.0, 0.60, 30.0, 35.0, 0.0),
            _row("fair", 70.0, 0.75, 20.0, 25.0, 0.0),
        ],
        max_starvation_rate=0.0,
    )
    by_name = {row["method"]: row for row in rows}
    assert by_name["balanced"]["goodput_proximity"] == 1.0
    assert by_name["unsafe"]["tradeoff_eligible"] is False
    assert by_name["unsafe"]["balanced_score"] < by_name["balanced"]["balanced_score"]


def test_tradeoff_marks_dominated_policy() -> None:
    rows = annotate_tradeoff_scores(
        [
            _row("better", 100.0, 0.60, 30.0, 35.0, 0.0),
            _row("worse", 90.0, 0.55, 35.0, 40.0, 0.0),
        ]
    )
    by_name = {row["method"]: row for row in rows}
    assert by_name["better"]["pareto_dominated"] is False
    assert by_name["worse"]["pareto_dominated"] is True
    assert by_name["better"]["tradeoff_rank"] == 1


def test_validation_tradeoff_penalizes_one_bad_kpi() -> None:
    good = validation_tradeoff_metrics(
        mean_throughput_score=0.5,
        minimum_jain_fairness=0.60,
        worst_starvation_rate=0.0,
        worst_p99_wait_slots=50.0,
        worst_max_wait_slots=60.0,
        target_throughput_score=0.50,
        target_jain_fairness=0.60,
        target_starvation_rate=0.0,
        target_p99_wait_slots=50.0,
        target_max_wait_slots=60.0,
    )
    bad = validation_tradeoff_metrics(
        mean_throughput_score=0.5,
        minimum_jain_fairness=0.30,
        worst_starvation_rate=0.0,
        worst_p99_wait_slots=50.0,
        worst_max_wait_slots=60.0,
        target_throughput_score=0.50,
        target_jain_fairness=0.60,
        target_starvation_rate=0.0,
        target_p99_wait_slots=50.0,
        target_max_wait_slots=60.0,
    )
    assert bad["target_balanced_score"] < good["target_balanced_score"]
    assert bad["target_worst_kpi_gap"] > good["target_worst_kpi_gap"]
