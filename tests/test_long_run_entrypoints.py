from scalemac_rl.scripts.train_hybrid_300k import (
    BUDGET_STEPS as HYBRID_BUDGET,
    MILESTONES as HYBRID_MILESTONES,
)
from scalemac_rl.scripts.train_ppo_only_300k import (
    BUDGET_STEPS as PPO_ONLY_BUDGET,
    MILESTONES as PPO_ONLY_MILESTONES,
)


def test_diagnostic_run_budgets_are_rollout_aligned() -> None:
    assert HYBRID_BUDGET == PPO_ONLY_BUDGET == 300_032
    assert HYBRID_BUDGET % 256 == 0


def test_diagnostic_milestones_are_rollout_aligned_and_include_final_step() -> None:
    assert HYBRID_MILESTONES == PPO_ONLY_MILESTONES == "100096,200192,300032"
    milestones = [int(value) for value in HYBRID_MILESTONES.split(",")]
    assert milestones[-1] == HYBRID_BUDGET
    assert all(value % 256 == 0 for value in milestones)
