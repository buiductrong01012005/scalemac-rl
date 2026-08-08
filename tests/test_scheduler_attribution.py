from scalemac_rl.scripts.run_scheduler_attribution import _learned_config


def test_legacy_hybrid_checkpoint_preserves_convex_reward_weights() -> None:
    checkpoint = {
        "training": {
            "reward_throughput_weight": 0.50,
            "reward_fairness_weight": 0.35,
            "reward_service_weight": 0.15,
            # v0.6.x has no reward_deficit_service_weight field.
        }
    }
    config, max_candidates, _ = _learned_config(
        checkpoint=checkpoint,
        num_ues=1200,
        slots=5000,
        fixed_profile_seed=1701,
        scheduler_mode="hybrid",
        candidate_mode="heuristic",
    )
    assert config.reward_deficit_service_weight == 0.0
    assert abs(
        config.reward_throughput_weight
        + config.reward_fairness_weight
        + config.reward_service_weight
        + config.reward_deficit_service_weight
        - 1.0
    ) < 1e-9
    assert max_candidates == 128


def test_current_checkpoint_keeps_explicit_deficit_weight() -> None:
    checkpoint = {
        "training": {
            "reward_throughput_weight": 0.45,
            "reward_fairness_weight": 0.35,
            "reward_service_weight": 0.15,
            "reward_deficit_service_weight": 0.05,
        }
    }
    config, _, _ = _learned_config(
        checkpoint=checkpoint,
        num_ues=1200,
        slots=5000,
        fixed_profile_seed=1701,
        scheduler_mode="ppo_only",
        candidate_mode="heuristic",
    )
    assert config.reward_deficit_service_weight == 0.05
