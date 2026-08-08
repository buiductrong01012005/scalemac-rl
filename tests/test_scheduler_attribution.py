from pathlib import Path

import torch

from scalemac_rl.evaluation_protocol import (
    UnifiedEvaluationProtocol,
    checkpoint_input_features,
    learned_provenance,
    load_policy_checkpoint,
    resolve_policy_runtime,
)
from scalemac_rl.models import SharedSetActorCritic
from scalemac_rl.rl_evaluation import evaluate_actor_critic


def _checkpoint(*, input_dim: int = 10, scheduler_mode: str = "hybrid") -> dict:
    model = SharedSetActorCritic(input_dim=input_dim, hidden_dim=16)
    return {
        "checkpoint_type": f"{scheduler_mode}_constrained_ppo_actor_critic",
        "checkpoint_tag": "test",
        "input_dim": input_dim,
        "hidden_dim": 16,
        "model_state_dict": model.state_dict(),
        "training": {
            "scheduler_mode": scheduler_mode,
            "candidate_mode": "heuristic",
            "max_candidates": 64,
            "safety_reserve_ues": 8,
            "force_harq_retransmissions": True,
            "long_wait_threshold": 0.8,
            "reward_throughput_weight": 0.50,
            "reward_fairness_weight": 0.35,
            "reward_service_weight": 0.15,
        },
    }


def test_protocol_reward_is_independent_of_checkpoint_training_reward() -> None:
    protocol = UnifiedEvaluationProtocol(num_ues=64, slots=10, profile_seed=11)
    config = protocol.build_config(
        scheduler_mode="hybrid",
        safety_reserve_ues=8,
        force_harq_retransmissions=True,
    )
    assert config.reward_throughput_weight == 0.45
    assert config.reward_fairness_weight == 0.35
    assert config.reward_service_weight == 0.15
    assert config.reward_deficit_service_weight == 0.05
    assert abs(
        config.reward_throughput_weight
        + config.reward_fairness_weight
        + config.reward_service_weight
        + config.reward_deficit_service_weight
        - 1.0
    ) < 1e-9


def test_runtime_recovers_execution_choices_not_reward_weights() -> None:
    runtime = resolve_policy_runtime(_checkpoint(), num_ues=1200)
    assert runtime.scheduler_mode == "hybrid"
    assert runtime.candidate_mode == "heuristic"
    assert runtime.max_candidates == 64
    assert runtime.safety_reserve_ues == 8
    assert runtime.force_harq_retransmissions is True


def test_ppo_only_runtime_disables_rule_override() -> None:
    runtime = resolve_policy_runtime(
        _checkpoint(scheduler_mode="hybrid"),
        num_ues=1200,
        scheduler_mode="ppo_only",
    )
    assert runtime.safety_reserve_ues == 0
    assert runtime.force_harq_retransmissions is False


def test_legacy_input_dimension_is_recorded_and_adapted(tmp_path: Path) -> None:
    checkpoint = _checkpoint(input_dim=8)
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    model, loaded = load_policy_checkpoint(path, torch.device("cpu"))
    assert model.input_dim == 10
    assert checkpoint_input_features(loaded) == 8

    protocol = UnifiedEvaluationProtocol(num_ues=64, slots=10, profile_seed=11)
    runtime = resolve_policy_runtime(loaded, num_ues=64)
    provenance = learned_provenance(
        checkpoint_path=path,
        checkpoint=loaded,
        protocol=protocol,
        runtime=runtime,
        rollout_seed=11,
    )
    assert provenance["compatibility_adapter_applied"] is True
    assert len(str(provenance["checkpoint_sha256"])) == 64


def test_standalone_and_attribution_paths_have_identical_kpis(tmp_path: Path) -> None:
    checkpoint = _checkpoint(input_dim=10)
    path = tmp_path / "policy.pt"
    torch.save(checkpoint, path)
    device = torch.device("cpu")
    model, loaded = load_policy_checkpoint(path, device)
    protocol = UnifiedEvaluationProtocol(
        num_ues=64,
        slots=25,
        num_prbs=64,
        max_selected_ues=16,
        profile_seed=77,
        starvation_threshold_slots=20,
        p99_wait_target_slots=15.0,
        max_wait_target_slots=20.0,
        min_jain_fairness=0.0,
    )
    runtime = resolve_policy_runtime(
        loaded,
        num_ues=64,
        max_candidates=32,
        safety_reserve_ues=4,
    )
    config_a = protocol.build_config(
        scheduler_mode=runtime.scheduler_mode,
        safety_reserve_ues=runtime.safety_reserve_ues,
        force_harq_retransmissions=runtime.force_harq_retransmissions,
        safety_wait_threshold_ratio=runtime.long_wait_threshold,
    )
    config_b = protocol.build_config(
        scheduler_mode=runtime.scheduler_mode,
        safety_reserve_ues=runtime.safety_reserve_ues,
        force_harq_retransmissions=runtime.force_harq_retransmissions,
        safety_wait_threshold_ratio=runtime.long_wait_threshold,
    )
    row_a = evaluate_actor_critic(
        model=model,
        device=device,
        config=config_a,
        seed=91,
        name="standalone",
        max_candidates=runtime.max_candidates,
        candidate_mode=runtime.candidate_mode,
        long_wait_threshold=runtime.long_wait_threshold,
        constraints=protocol.constraints(),
    )
    row_b = evaluate_actor_critic(
        model=model,
        device=device,
        config=config_b,
        seed=91,
        name="attribution",
        max_candidates=runtime.max_candidates,
        candidate_mode=runtime.candidate_mode,
        long_wait_threshold=runtime.long_wait_threshold,
        constraints=protocol.constraints(),
    )
    comparable = (
        "mean_reward",
        "mean_goodput_bits_per_slot",
        "final_jain_fairness",
        "mean_starvation_rate",
        "final_p99_wait_slots",
        "max_p99_wait_slots",
        "max_wait_slots",
    )
    for key in comparable:
        assert row_a[key] == row_b[key]
    assert protocol.scenario_hash(91) == protocol.scenario_hash(91)
