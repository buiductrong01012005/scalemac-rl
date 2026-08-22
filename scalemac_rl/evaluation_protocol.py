from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .config import ScaleMacConfig
from .constraints import ServiceConstraints
from .env import OBSERVATION_FEATURES
from .models import RecurrentSharedSetActorCritic, SharedSetActorCritic, SplitEncoderActorCritic


UNIFIED_EVALUATION_VERSION = "unified-v1"
EVALUATION_REWARD_VERSION = "balanced-kpi-v1"
OBSERVATION_SCHEMA_VERSION = "ue-features-16-v2"
PROJECTOR_CONTRACT_VERSION = "top64-exact273-v1"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_input_features(checkpoint: dict[str, Any]) -> int:
    if "input_dim" in checkpoint:
        return int(checkpoint["input_dim"])
    state_dict = checkpoint.get("model_state_dict", {})
    weight = state_dict.get("encoder.0.weight")
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[1])
    return 8


def checkpoint_reward_version(checkpoint: dict[str, Any]) -> str:
    training = checkpoint.get("training", {})
    if "reward_pf_utility_weight" in training or "reward_low_throughput_weight" in training:
        return "full-control-tradeoff-v0.8"
    if "reward_deficit_service_weight" in training:
        return "dense-deficit-v0.7"
    if "reward_max_wait_risk_penalty_weight" in training or "max_wait_target_slots" in training:
        return "successful-delivery-v0.6.2"
    return "legacy-v0.6"


def checkpoint_reward_signature(checkpoint: dict[str, Any]) -> str:
    training = checkpoint.get("training", {})
    keys = (
        "reward_throughput_weight",
        "reward_fairness_weight",
        "reward_schedule_fairness_weight",
        "reward_service_weight",
        "reward_deficit_service_weight",
        "reward_pf_utility_weight",
        "reward_low_throughput_weight",
        "reward_urgency_service_weight",
        "reward_fairness_delta_weight",
        "reward_pf_utility_delta_weight",
        "deadline_risk_penalty_weight",
        "max_wait_risk_penalty_weight",
        "population_wait_penalty_weight",
        "low_throughput_percentile",
        "starvation_threshold_slots",
        "reference_deadline_target_slots",
        "max_wait_target_slots",
    )
    payload = {key: training.get(key) for key in keys if key in training}
    return _canonical_json(payload)


@dataclass(frozen=True, slots=True)
class UnifiedEvaluationProtocol:
    """One fixed environment, KPI, reward, and constraint contract for all schedulers.

    Checkpoint training rewards are recorded as provenance but never used to build
    the evaluation environment. Only scheduler-execution choices such as hybrid
    reserve size and candidate mode are recovered from a checkpoint.
    """

    num_ues: int = 1200
    slots: int = 5000
    num_prbs: int = 273
    max_selected_ues: int = 64
    profile_seed: int = 1701

    starvation_threshold_slots: int = 64
    p99_wait_target_slots: float = 50.0
    max_wait_target_slots: float = 60.0
    min_jain_fairness: float = 0.60
    max_starvation_rate: float = 0.0

    target_bler: float = 0.10
    max_harq_retransmissions: int = 3
    deadline_risk_start_ratio: float = 0.60

    reward_throughput_weight: float = 0.45
    reward_fairness_weight: float = 0.35
    reward_service_weight: float = 0.15
    reward_deficit_service_weight: float = 0.05
    reward_fairness_delta_weight: float = 0.03
    reward_pf_utility_delta_weight: float = 0.02
    reward_deadline_risk_penalty_weight: float = 0.15
    reward_max_wait_risk_penalty_weight: float = 0.10

    def validate(self) -> None:
        if self.num_ues <= 0:
            raise ValueError("num_ues must be positive")
        if self.slots <= 0:
            raise ValueError("slots must be positive")
        if self.num_prbs < min(self.max_selected_ues, self.num_ues):
            raise ValueError("num_prbs must support one PRB per selected UE")
        if self.profile_seed < 0:
            raise ValueError("profile_seed must be non-negative")
        self.constraints().validate()
        self.build_config(
            scheduler_mode="ppo_only",
            safety_reserve_ues=0,
            force_harq_retransmissions=False,
        ).validate()

    def constraints(self) -> ServiceConstraints:
        return ServiceConstraints(
            max_starvation_rate=self.max_starvation_rate,
            max_p99_wait_slots=self.p99_wait_target_slots,
            min_jain_fairness=self.min_jain_fairness,
            max_wait_slots=self.max_wait_target_slots,
        )

    def shared_payload(self) -> dict[str, Any]:
        return {
            "evaluation_version": UNIFIED_EVALUATION_VERSION,
            "evaluation_reward_version": EVALUATION_REWARD_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "projector_contract_version": PROJECTOR_CONTRACT_VERSION,
            **asdict(self),
        }

    @property
    def protocol_hash(self) -> str:
        return _sha256_text(_canonical_json(self.shared_payload()))

    def scenario_hash(self, rollout_seed: int) -> str:
        return _sha256_text(
            _canonical_json(
                {
                    "protocol_hash": self.protocol_hash,
                    "static_profile_seed": self.profile_seed,
                    "rollout_seed": int(rollout_seed),
                }
            )
        )

    def build_config(
        self,
        *,
        scheduler_mode: str,
        safety_reserve_ues: int,
        force_harq_retransmissions: bool,
        safety_wait_threshold_ratio: float = 0.80,
    ) -> ScaleMacConfig:
        max_selected = min(self.max_selected_ues, self.num_ues)
        reserve = 0 if scheduler_mode == "ppo_only" else min(
            safety_reserve_ues, max_selected
        )
        config = ScaleMacConfig(
            num_ues=self.num_ues,
            num_prbs=self.num_prbs,
            max_selected_ues=max_selected,
            episode_slots=self.slots,
            scheduler_mode=scheduler_mode,
            force_harq_retransmissions=force_harq_retransmissions,
            safety_reserve_ues=reserve,
            safety_wait_threshold_ratio=safety_wait_threshold_ratio,
            freeze_static_profiles=True,
            static_profile_seed=self.profile_seed,
            starvation_threshold_slots=self.starvation_threshold_slots,
            target_bler=self.target_bler,
            max_harq_retransmissions=self.max_harq_retransmissions,
            deadline_target_slots=self.p99_wait_target_slots,
            reference_deadline_target_slots=self.p99_wait_target_slots,
            deadline_risk_start_ratio=self.deadline_risk_start_ratio,
            max_wait_target_slots=self.max_wait_target_slots,
            reward_throughput_weight=self.reward_throughput_weight,
            reward_fairness_weight=self.reward_fairness_weight,
            reward_service_weight=self.reward_service_weight,
            reward_deficit_service_weight=self.reward_deficit_service_weight,
            reward_fairness_delta_weight=self.reward_fairness_delta_weight,
            reward_pf_utility_delta_weight=self.reward_pf_utility_delta_weight,
            reward_deadline_risk_penalty_weight=(
                self.reward_deadline_risk_penalty_weight
            ),
            reward_max_wait_risk_penalty_weight=(
                self.reward_max_wait_risk_penalty_weight
            ),
        )
        config.validate()
        return config


@dataclass(frozen=True, slots=True)
class PolicyRuntime:
    scheduler_mode: str
    candidate_mode: str
    max_candidates: int
    safety_reserve_ues: int
    force_harq_retransmissions: bool
    long_wait_threshold: float

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def runtime_hash(self) -> str:
        return _sha256_text(_canonical_json(self.payload()))


def resolve_policy_runtime(
    checkpoint: dict[str, Any],
    *,
    num_ues: int,
    scheduler_mode: str | None = None,
    candidate_mode: str | None = None,
    max_candidates: int | None = None,
    safety_reserve_ues: int | None = None,
    force_harq_retransmissions: bool | None = None,
    long_wait_threshold: float | None = None,
) -> PolicyRuntime:
    training = checkpoint.get("training", {})
    resolved_scheduler_mode = str(
        scheduler_mode or training.get("scheduler_mode", "hybrid")
    )
    resolved_candidate_mode = str(
        candidate_mode or training.get("candidate_mode", "heuristic")
    )
    if resolved_candidate_mode not in {"heuristic", "all"}:
        raise ValueError("candidate_mode must be heuristic or all")
    if resolved_scheduler_mode not in {"hybrid", "ppo_only", "rule_only"}:
        raise ValueError("scheduler_mode must be hybrid, ppo_only, or rule_only")

    resolved_max_candidates = int(
        num_ues
        if resolved_candidate_mode == "all"
        else max_candidates or training.get("max_candidates", 128)
    )
    resolved_max_candidates = min(max(resolved_max_candidates, 64), num_ues)

    default_force_harq = resolved_scheduler_mode in {"hybrid", "rule_only"}
    resolved_force_harq = bool(
        training.get("force_harq_retransmissions", default_force_harq)
        if force_harq_retransmissions is None
        else force_harq_retransmissions
    )
    resolved_reserve = int(
        training.get("safety_reserve_ues", 16)
        if safety_reserve_ues is None
        else safety_reserve_ues
    )
    if resolved_scheduler_mode == "ppo_only":
        resolved_reserve = 0
        resolved_force_harq = False
    elif resolved_scheduler_mode == "rule_only":
        resolved_reserve = min(64, num_ues)
        resolved_force_harq = True
    else:
        resolved_reserve = min(max(resolved_reserve, 0), min(64, num_ues))

    resolved_long_wait = float(
        training.get("long_wait_threshold", 0.80)
        if long_wait_threshold is None
        else long_wait_threshold
    )
    if resolved_long_wait < 0.0:
        raise ValueError("long_wait_threshold must be non-negative")

    return PolicyRuntime(
        scheduler_mode=resolved_scheduler_mode,
        candidate_mode=resolved_candidate_mode,
        max_candidates=resolved_max_candidates,
        safety_reserve_ues=resolved_reserve,
        force_harq_retransmissions=resolved_force_harq,
        long_wait_threshold=resolved_long_wait,
    )


def load_policy_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[SharedSetActorCritic | SplitEncoderActorCritic | RecurrentSharedSetActorCritic, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    architecture = str(checkpoint.get("policy_architecture", "feedforward"))
    model: SharedSetActorCritic | SplitEncoderActorCritic | RecurrentSharedSetActorCritic
    if architecture == "recurrent":
        model = RecurrentSharedSetActorCritic(
            input_dim=OBSERVATION_FEATURES,
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(device)
    elif bool(checkpoint.get("separate_critic_encoder", False)):
        model = SplitEncoderActorCritic(
            input_dim=OBSERVATION_FEATURES,
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(device)
    else:
        model = SharedSetActorCritic(
            input_dim=OBSERVATION_FEATURES,
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(device)
    model.load_compatible_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def learned_provenance(
    *,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    protocol: UnifiedEvaluationProtocol,
    runtime: PolicyRuntime,
    rollout_seed: int,
) -> dict[str, Any]:
    source_input_features = checkpoint_input_features(checkpoint)
    return {
        "evaluation_protocol_version": UNIFIED_EVALUATION_VERSION,
        "evaluation_protocol_hash": protocol.protocol_hash,
        "evaluation_reward_version": EVALUATION_REWARD_VERSION,
        "evaluation_observation_schema": OBSERVATION_SCHEMA_VERSION,
        "evaluation_observation_features": OBSERVATION_FEATURES,
        "projector_contract_version": PROJECTOR_CONTRACT_VERSION,
        "scenario_hash": protocol.scenario_hash(rollout_seed),
        "rollout_seed": int(rollout_seed),
        "static_profile_seed": protocol.profile_seed,
        "scheduler_runtime_hash": runtime.runtime_hash,
        "scheduler_runtime_json": _canonical_json(runtime.payload()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_type": str(checkpoint.get("checkpoint_type", "unknown")),
        "checkpoint_tag": str(checkpoint.get("checkpoint_tag", "unknown")),
        "checkpoint_training_reward_version": checkpoint_reward_version(checkpoint),
        "checkpoint_training_reward_signature": checkpoint_reward_signature(checkpoint),
        "checkpoint_observation_features": source_input_features,
        "compatibility_adapter_applied": source_input_features != OBSERVATION_FEATURES,
    }


def classical_provenance(
    *,
    protocol: UnifiedEvaluationProtocol,
    scheduler_name: str,
    scheduler_mode: str,
    rollout_seed: int,
) -> dict[str, Any]:
    runtime_payload = {
        "scheduler_name": scheduler_name,
        "scheduler_mode": scheduler_mode,
        "candidate_mode": "not_applicable",
        "projector": PROJECTOR_CONTRACT_VERSION,
    }
    return {
        "evaluation_protocol_version": UNIFIED_EVALUATION_VERSION,
        "evaluation_protocol_hash": protocol.protocol_hash,
        "evaluation_reward_version": EVALUATION_REWARD_VERSION,
        "evaluation_observation_schema": OBSERVATION_SCHEMA_VERSION,
        "evaluation_observation_features": OBSERVATION_FEATURES,
        "projector_contract_version": PROJECTOR_CONTRACT_VERSION,
        "scenario_hash": protocol.scenario_hash(rollout_seed),
        "rollout_seed": int(rollout_seed),
        "static_profile_seed": protocol.profile_seed,
        "scheduler_runtime_hash": _sha256_text(_canonical_json(runtime_payload)),
        "scheduler_runtime_json": _canonical_json(runtime_payload),
        "checkpoint_path": "",
        "checkpoint_sha256": "",
        "checkpoint_type": "classical_scheduler",
        "checkpoint_tag": "not_applicable",
        "checkpoint_training_reward_version": "not_applicable",
        "checkpoint_training_reward_signature": "{}",
        "checkpoint_observation_features": "",
        "compatibility_adapter_applied": False,
    }
