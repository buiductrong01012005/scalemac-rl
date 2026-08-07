from scalemac_rl.constraints import LagrangeController, ServiceConstraints, validation_feasible


def test_service_constraints_report_normalized_excess() -> None:
    constraints = ServiceConstraints(max_starvation_rate=0.0, max_p99_wait_slots=50.0)
    starvation_excess, wait_excess = constraints.excesses(
        starvation_rate=0.02,
        p99_wait_slots=75.0,
    )
    assert starvation_excess == 0.02
    assert wait_excess == 0.5
    assert not constraints.feasible(starvation_rate=0.02, p99_wait_slots=75.0)


def test_lagrange_controller_penalizes_and_increases_multipliers() -> None:
    controller = LagrangeController(
        starvation_multiplier=5.0,
        wait_multiplier=1.0,
        learning_rate=0.5,
        max_multiplier=10.0,
    )
    adjusted, penalty = controller.adjusted_reward(
        0.6,
        starvation_excess=0.02,
        wait_excess=0.5,
    )
    assert penalty == 0.6
    assert adjusted == 0.0
    controller.update(mean_starvation_excess=0.02, mean_wait_excess=0.5)
    assert controller.starvation_multiplier > 5.0
    assert controller.wait_multiplier > 1.0


def test_validation_feasibility_is_worst_seed_based() -> None:
    constraints = ServiceConstraints(max_starvation_rate=0.0, max_p99_wait_slots=50.0)
    rows = [
        {"max_starvation_rate": 0.0, "max_p99_wait_slots": 40.0},
        {"max_starvation_rate": 0.0, "max_p99_wait_slots": 55.0},
    ]
    assert not validation_feasible(rows, constraints)


def test_progressive_p99_schedule_selects_expected_segments() -> None:
    from scalemac_rl.scripts.train_ppo import _active_p99_limit

    schedule = [80.0, 65.0, 55.0, 50.0]
    assert _active_p99_limit(
        stage_index=1, stage_count=1, stage_env_steps=0, steps_per_stage=400,
        default_limit=50.0, final_stage_schedule=schedule,
    ) == 80.0
    assert _active_p99_limit(
        stage_index=1, stage_count=1, stage_env_steps=200, steps_per_stage=400,
        default_limit=50.0, final_stage_schedule=schedule,
    ) == 55.0
    assert _active_p99_limit(
        stage_index=1, stage_count=1, stage_env_steps=399, steps_per_stage=400,
        default_limit=50.0, final_stage_schedule=schedule,
    ) == 50.0
