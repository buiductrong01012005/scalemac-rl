from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from scalemac_rl.models import RecurrentSharedSetActorCritic
from scalemac_rl.policy_architecture_analysis import build_policy_architecture_analysis
from scalemac_rl.reward_study import RewardStudyPlan
from scalemac_rl.scripts.run_reward_study import _common_command


PLAN = Path("configs/channel_study/round_13_ppo_vs_rppo.json")


def test_recurrent_shared_set_policy_is_permutation_equivariant() -> None:
    torch.manual_seed(3)
    model = RecurrentSharedSetActorCritic(input_dim=16, hidden_dim=8)
    observation = torch.randn(6, 16)
    hidden = torch.randn(6, 8)
    order = torch.tensor([4, 1, 5, 0, 3, 2])

    original = model.deterministic_action(observation, hidden)
    permuted = model.deterministic_action(observation[order], hidden[order])

    assert torch.allclose(permuted.action, original.action[order], atol=1e-6)
    assert torch.allclose(permuted.hidden_state, original.hidden_state[order], atol=1e-6)


def test_recurrent_policy_carries_temporal_memory() -> None:
    torch.manual_seed(7)
    model = RecurrentSharedSetActorCritic(input_dim=16, hidden_dim=8)
    observation = torch.randn(5, 16)
    hidden = model.initial_state(1, 5).squeeze(0)
    first = model.deterministic_action(observation, hidden)
    second = model.deterministic_action(observation, first.hidden_state)
    assert not torch.allclose(first.hidden_state, second.hidden_state)
    assert not torch.allclose(first.action, second.action)


def test_recurrent_sequence_evaluation_shapes() -> None:
    torch.manual_seed(11)
    model = RecurrentSharedSetActorCritic(input_dim=16, hidden_dim=8)
    batch, steps, ues = 2, 4, 5
    observations = torch.randn(batch, steps, ues, 16)
    masks = torch.ones(batch, steps, ues, dtype=torch.bool)
    actions = torch.full((batch, steps, ues, 2), 0.5)
    dones = torch.zeros(batch, steps)
    hidden = model.initial_state(batch, ues)
    log_prob, entropy, value, mean_action = model.evaluate_sequence(
        observations, masks, actions, hidden, dones
    )
    assert log_prob.shape == (batch, steps)
    assert entropy.shape == (batch, steps)
    assert value.shape == (batch, steps)
    assert mean_action.shape == (batch, steps, ues, 2)


def test_round13_plan_pairs_ppo_and_rppo_on_three_common_seeds(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    assert plan.analysis["design"] == "policy_architecture_screen"
    assert len(plan.cases) == 6
    expected = {1701, 2701, 3701}
    for architecture in ("feedforward", "recurrent"):
        cases = [c for c in plan.cases if c.common_overrides["policy_architecture"] == architecture]
        assert {int(c.common_overrides["seed"]) for c in cases} == expected
        for case in cases:
            effective = dict(plan.common)
            effective.update(case.common_overrides)
            command = _common_command(
                common=effective,
                run_dir=tmp_path / case.case_id,
                steps_override=256,
                validation_slots_override=64,
                progress=False,
                device="cpu",
            )
            assert command[command.index("--policy-architecture") + 1] == architecture
            seed = str(case.common_overrides["seed"])
            assert command[command.index("--seed") + 1] == seed
            assert command[command.index("--fixed-profile-seed") + 1] == seed
            assert command[command.index("--validation-seeds") + 1] == seed
            assert command[command.index("--link-adaptation-mode") + 1] == "cqi_mcs_bler"
            assert command[command.index("--csi-report-delay-slots") + 1] == "2"


def test_policy_architecture_analysis_exports_case_and_mean_std_tables(tmp_path: Path) -> None:
    plan = RewardStudyPlan.from_json(PLAN)
    round_dir = tmp_path / plan.round_id
    fields = [
        "mean_goodput_bits_per_slot",
        "mean_spectral_efficiency_bps_hz",
        "final_jain_fairness",
        "max_starvation_rate",
        "max_p99_wait_slots",
        "max_wait_slots",
        "mean_observed_bler",
        "mean_harq_retransmission_fraction",
        "mean_inference_us",
        "p99_inference_us",
    ]
    for idx, case in enumerate(plan.cases):
        case_dir = round_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "validation.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "mean_goodput_bits_per_slot": 90000 + idx * 100,
                    "mean_spectral_efficiency_bps_hz": 1.9 + idx * 0.01,
                    "final_jain_fairness": 0.35 + idx * 0.01,
                    "max_starvation_rate": 0.0,
                    "max_p99_wait_slots": 45 + idx,
                    "max_wait_slots": 48 + idx,
                    "mean_observed_bler": 0.16,
                    "mean_harq_retransmission_fraction": 0.14,
                    "mean_inference_us": 100 + idx,
                    "p99_inference_us": 150 + idx,
                }
            )
    plan.analysis.update(
        {
            "output": str(tmp_path / "arch.html"),
            "markdown_output": str(tmp_path / "arch.md"),
            "metrics_output": str(tmp_path / "arch.csv"),
            "summary_output": str(tmp_path / "summary.csv"),
        }
    )
    result = build_policy_architecture_analysis(
        plan=plan, round_dir=round_dir, output_path=tmp_path / "arch.html"
    )
    assert result.is_file()
    assert (tmp_path / "arch.csv").is_file()
    assert (tmp_path / "summary.csv").is_file()
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {row["architecture"] for row in rows} == {"feedforward", "recurrent"}
    assert all(int(row["seeds"]) == 3 for row in rows)
