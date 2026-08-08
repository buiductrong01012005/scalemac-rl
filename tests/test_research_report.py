from scalemac_rl.scripts.build_full_control_report import build_report


def test_research_report_contains_training_and_evaluation_sections() -> None:
    report = build_report(
        training_rows=[
            {
                "global_env_steps": "256",
                "num_ues": "1200",
                "learning_rate": "0.0001",
                "entropy_coef": "0.005",
                "mean_goodput_bits_per_slot": "90000",
                "mean_jain_fairness": "0.6",
                "mean_starvation_rate": "0",
                "mean_p99_wait_slots": "45",
                "mean_max_wait_slots": "50",
                "mean_final_target_reward": "0.5",
            }
        ],
        validation_rows=[],
        manifest_rows=[],
        evaluation_rows=[
            {
                "method": "ppo_full",
                "mean_goodput_bits_per_slot": "90000",
                "final_jain_fairness": "0.6",
                "mean_starvation_rate": "0",
                "max_p99_wait_slots": "45",
                "max_wait_slots": "50",
                "balanced_score": "0.8",
                "worst_kpi_gap": "0.2",
                "pareto_dominated": "False",
            }
        ],
        title="Test report",
    )
    assert "# Test report" in report
    assert "## 2. Tóm tắt training" in report
    assert "ppo_full" in report
