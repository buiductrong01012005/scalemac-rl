from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import time
from typing import Any

from scalemac_rl.reward_study import RewardStudyPlan, write_json


def _csv(values: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _common_command(
    *,
    common: dict[str, Any],
    run_dir: Path,
    steps_override: int | None,
    validation_slots_override: int | None,
    progress: bool,
    device: str,
) -> list[str]:
    steps = int(steps_override or common.get("environment_steps", 65_536))
    rollout_steps = int(common.get("rollout_steps", 256))
    validation_slots = int(
        validation_slots_override or common.get("validation_slots", 5_000)
    )
    validation_seeds = [int(value) for value in common.get("validation_seeds", [1701])]
    validation_repeats = int(common.get("validation_repeats", 1))
    seed = int(common.get("seed", 1701))
    profile_seed = int(common.get("profile_seed", seed))
    constraint_training = bool(common.get("constraint_training", False))

    command = [
        sys.executable,
        "-m",
        "scalemac_rl.scripts.train_ppo",
        "--single-seed-upper-bound",
        "--freeze-static-profiles",
        "--curriculum",
        "1200",
        "--stage-p99-wait-limits",
        "50",
        "--steps-per-stage",
        str(steps),
        "--workers",
        "1",
        "--rollout-steps",
        str(rollout_steps),
        "--episode-slots",
        str(int(common.get("episode_slots", 5_000))),
        "--candidate-mode",
        "all",
        "--max-candidates",
        "1200",
        "--scheduler-mode",
        "ppo_only",
        "--safety-reserve-ues",
        "0",
        "--no-force-harq-retransmissions",
        "--final-stage-p99-schedule",
        "50",
        "--fairness-target-schedule",
        "0.60",
        "--validation-seeds",
        _csv(validation_seeds),
        "--validation-repeats",
        str(validation_repeats),
        "--validation-slots",
        str(validation_slots),
        "--validate-every",
        str(int(common.get("validate_every", 64))),
        "--rollback-patience",
        "1000000",
        "--checkpoint-every",
        str(int(common.get("checkpoint_every", 128))),
        "--milestone-env-steps",
        str(steps),
        "--seed",
        str(seed),
        "--fixed-profile-seed",
        str(profile_seed),
        "--hidden-dim",
        str(int(common.get("hidden_dim", 64))),
        "--lr",
        str(float(common.get("learning_rate_start", 1.0e-4))),
        "--lr-end",
        str(float(common.get("learning_rate_end", 2.5e-5))),
        "--gamma",
        str(float(common.get("gamma", 0.999))),
        "--gae-lambda",
        str(float(common.get("gae_lambda", 0.97))),
        "--clip-coef",
        str(float(common.get("clip_coef", 0.10))),
        "--entropy-coef",
        str(float(common.get("entropy_coef_start", 5.0e-3))),
        "--entropy-coef-end",
        str(float(common.get("entropy_coef_end", 5.0e-4))),
        "--update-epochs",
        str(int(common.get("update_epochs", 4))),
        "--minibatch-size",
        str(int(common.get("minibatch_size", 8))),
        "--target-kl",
        str(float(common.get("target_kl", 0.02))),
        "--deadline-risk-start-ratio",
        str(float(common.get("deadline_risk_start_ratio", 0.60))),
        "--low-throughput-percentile",
        str(float(common.get("low_throughput_percentile", 10.0))),
        "--starvation-threshold-slots",
        str(int(common.get("starvation_threshold_slots", 64))),
        "--max-starvation-rate",
        "0",
        "--max-p99-wait-slots",
        "50",
        "--min-jain-fairness",
        "0.60",
        "--min-throughput-score",
        "0.43",
        "--max-wait-slots",
        "60",
        "--init-checkpoint",
        "artifacts/__random_initialization__.pt",
        "--device",
        device,
        "--output",
        str(run_dir / "latest.pt"),
        "--best-feasible-output",
        str(run_dir / "best_feasible.pt"),
        "--best-reward-output",
        str(run_dir / "best_reward.pt"),
        "--best-tradeoff-output",
        str(run_dir / "best_tradeoff.pt"),
        "--best-lowest-violation-output",
        str(run_dir / "best_lowest_violation.pt"),
        "--checkpoint-dir",
        str(run_dir / "checkpoints"),
        "--log-output",
        str(run_dir / "training.csv"),
        "--validation-output",
        str(run_dir / "validation.csv"),
        "--checkpoint-manifest-output",
        str(run_dir / "checkpoint_manifest.csv"),
        "--report-dir",
        str(run_dir / "reports"),
        "--progress" if progress else "--no-progress",
    ]
    if constraint_training:
        command.extend(
            [
                "--starvation-multiplier",
                str(float(common.get("starvation_multiplier", 8.0))),
                "--wait-multiplier",
                str(float(common.get("wait_multiplier", 1.5))),
                "--fairness-multiplier",
                str(float(common.get("fairness_multiplier", 2.0))),
                "--max-wait-multiplier",
                str(float(common.get("max_wait_multiplier", 1.5))),
                "--lagrangian-lr",
                str(float(common.get("lagrangian_lr", 0.05))),
            ]
        )
    else:
        command.extend(
            [
                "--starvation-multiplier",
                "0",
                "--wait-multiplier",
                "0",
                "--fairness-multiplier",
                "0",
                "--max-wait-multiplier",
                "0",
                "--lagrangian-lr",
                "0",
                "--validation-lagrangian-scale",
                "0",
            ]
        )
    return command


def _completed(run_dir: Path) -> bool:
    required = (
        run_dir / "training.csv",
        run_dir / "validation.csv",
        run_dir / "validation_summary.csv",
        run_dir / "latest.pt",
    )
    return all(path.is_file() for path in required)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled full-control PPO reward experiments from a JSON plan"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/reward_study/round_01_component_screen.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/runs/reward_study"),
    )
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case ids; empty runs every case in the plan",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--validation-slots", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-build-dataset",
        action="store_true",
        help="do not rebuild CSV/Pareto/HTML summaries after the round",
    )
    args = parser.parse_args()

    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.validation_slots is not None and args.validation_slots <= 0:
        parser.error("--validation-slots must be positive")

    try:
        plan = RewardStudyPlan.from_json(args.plan)
        requested = [value.strip() for value in args.cases.split(",") if value.strip()]
        cases = plan.selected_cases(requested or None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    round_dir = args.output_root / plan.round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        round_dir / "round_plan_snapshot.json",
        {
            "study_id": plan.study_id,
            "round_id": plan.round_id,
            "description": plan.description,
            "common": plan.common,
            "cases": [case.to_dict() for case in cases],
            "source_plan": str(args.plan),
        },
    )

    failed: list[str] = []
    for index, case in enumerate(cases, start=1):
        run_dir = round_dir / case.case_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if _completed(run_dir) and not args.force:
            print(f"[{index}/{len(cases)}] skip completed: {case.case_id}")
            continue

        command = _common_command(
            common=plan.common,
            run_dir=run_dir,
            steps_override=args.steps,
            validation_slots_override=args.validation_slots,
            progress=args.progress,
            device=args.device,
        )
        command.extend(case.cli_args())
        command_text = shlex.join(command)
        (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
        write_json(
            run_dir / "run_config.json",
            {
                "study_id": plan.study_id,
                "round_id": plan.round_id,
                "case": case.to_dict(),
                "common": plan.common,
                "effective_environment_steps": int(
                    args.steps or plan.common.get("environment_steps", 65_536)
                ),
                "effective_validation_slots": int(
                    args.validation_slots or plan.common.get("validation_slots", 5_000)
                ),
                "architecture": {
                    "observation_features_per_ue": 16,
                    "encoder": "shared_set_encoder",
                    "embedding_dim": int(plan.common.get("hidden_dim", 64)),
                    "candidate_mode": "all",
                    "scheduler_mode": "ppo_only",
                    "num_ues": 1200,
                    "top_k": 64,
                    "num_prbs": 273,
                },
                "command": command,
            },
        )
        print(f"[{index}/{len(cases)}] {case.case_id}: {case.label}")
        if args.dry_run:
            print(command_text)
            continue

        write_json(
            run_dir / "status.json",
            {"status": "running", "started_unix": time()},
        )
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failed.append(case.case_id)
            write_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "return_code": result.returncode,
                    "finished_unix": time(),
                },
            )
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
        else:
            write_json(
                run_dir / "status.json",
                {"status": "completed", "return_code": 0, "finished_unix": time()},
            )

    if not args.dry_run and not args.skip_build_dataset:
        command = [
            sys.executable,
            "-m",
            "scalemac_rl.scripts.build_reward_study_dataset",
            "--study-root",
            str(args.output_root),
            "--docs-root",
            "docs/research/reward_study/generated",
        ]
        subprocess.run(command, check=False)

    if failed:
        print(f"failed cases: {', '.join(failed)}")
        raise SystemExit(1)
    print(f"round output: {round_dir}")


if __name__ == "__main__":
    main()
