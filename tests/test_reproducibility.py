from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scalemac_rl.reproducibility import (
    collect_runtime_fingerprint,
    numpy_global_rng_sha256,
    tensor_mapping_sha256,
    torch_cpu_rng_sha256,
)
from scalemac_rl.reward_study import RewardStudyPlan


def test_seeded_rng_fingerprints_repeat() -> None:
    np.random.seed(1701)
    torch.manual_seed(1701)
    first = (numpy_global_rng_sha256(), torch_cpu_rng_sha256())
    np.random.seed(1701)
    torch.manual_seed(1701)
    second = (numpy_global_rng_sha256(), torch_cpu_rng_sha256())
    assert first == second


def test_tensor_mapping_hash_depends_on_values_not_dict_order() -> None:
    a = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    b = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    c = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    assert tensor_mapping_sha256(a) == tensor_mapping_sha256(b)
    assert tensor_mapping_sha256(a) != tensor_mapping_sha256(c)


def test_runtime_fingerprint_has_required_fields() -> None:
    payload = collect_runtime_fingerprint()
    assert payload["packages"]["torch"]
    assert payload["packages"]["numpy"]
    assert "num_threads" in payload["torch_runtime"]
    assert "cpu_model" in payload["platform"]


def test_round09_plan_is_three_identical_seeded_tjs_repeats() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = RewardStudyPlan.from_json(root / "configs/reproducibility/round_09_tjs_repeatability.json")
    assert len(plan.cases) == 3
    assert plan.analysis["design"] == "reproducibility_repeat"
    for index, case in enumerate(plan.cases, start=1):
        assert case.common_overrides["seed"] == 1701
        assert case.common_overrides["profile_seed"] == 1701
        assert case.common_overrides["validation_seeds"] == [1701]
        assert case.common_overrides["repeat_index"] == index
        assert case.positive_weights["throughput"] == case.positive_weights["fairness"]
        assert case.positive_weights["fairness"] == case.positive_weights["service"]
        assert sum(v > 0 for v in case.positive_weights.values()) == 3
