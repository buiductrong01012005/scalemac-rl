from __future__ import annotations

import json
from pathlib import Path

from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command


def test_round17a_full_3x3_seed_grid() -> None:
    plan = RewardStudyPlan.from_json("configs/optimization/round_17a_seed_decoupling.json")
    assert len(plan.cases) == 9
    pairs = {
        (int(case.common_overrides["seed"]), int(case.common_overrides["environment_seed"]))
        for case in plan.cases
    }
    assert pairs == {(t, e) for t in (1701, 2701, 3701) for e in (1701, 2701, 3701)}
    for case in plan.cases:
        env_seed = int(case.common_overrides["environment_seed"])
        assert int(case.common_overrides["profile_seed"]) == env_seed
        assert case.common_overrides["validation_seeds"] == [env_seed]
        assert case.positive_weights["throughput"] == 0.3
        assert case.positive_weights["fairness"] == 0.3
        assert case.positive_weights["service"] == 0.4


def test_reward_study_command_decouples_policy_and_environment_seed(tmp_path: Path) -> None:
    common = {
        "seed": 1701,
        "environment_seed": 2701,
        "profile_seed": 2701,
        "validation_seeds": [2701],
    }
    command = _common_command(
        common=common,
        run_dir=tmp_path,
        steps_override=1024,
        validation_slots_override=128,
        progress=False,
        device="cpu",
    )
    joined = " ".join(command)
    assert "--seed 1701" in joined
    assert "--environment-seed 2701" in joined
    assert "--fixed-profile-seed 2701" in joined
    assert "--validation-seeds 2701" in joined
