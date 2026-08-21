from __future__ import annotations

from pathlib import Path

from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command


def test_round17b_has_three_profiles_across_three_seeds() -> None:
    plan = RewardStudyPlan.from_json("configs/optimization/round_17b_service_rollback.json")
    assert len(plan.cases) == 9
    ids = {case.case_id for case in plan.cases}
    for prefix in ("baseline", "rollback", "rollback_lr50"):
        for seed in (1701, 2701, 3701):
            assert f"{prefix}_seed{seed}" in ids


def test_round17b_rollback_command_flags(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json("configs/optimization/round_17b_service_rollback.json")
    case = next(case for case in plan.cases if case.case_id == "rollback_lr50_seed1701")
    common = dict(plan.common)
    common.update(case.common_overrides)
    command = _common_command(
        common=common,
        run_dir=tmp_path,
        steps_override=1024,
        validation_slots_override=128,
        progress=False,
        device="cpu",
    )
    joined = " ".join(command)
    assert "--rollback-mode service" in joined
    assert "--rollback-patience 1" in joined
    assert "--rollback-lr-factor 0.5" in joined
    assert "--rollback-min-lr-multiplier 0.125" in joined
