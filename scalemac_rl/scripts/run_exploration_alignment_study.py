from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import time
from typing import Any

from scalemac_rl.scripts.run_reward_study import _common_command
from scalemac_rl.reward_study import write_json


REWARD_ARGS = [
    "--reward-positive-scale", "1.0",
    "--reward-throughput-weight", "0.25",
    "--reward-fairness-weight", "0.75",
    "--reward-service-weight", "0.0",
    "--reward-deficit-service-weight", "0.0",
    "--reward-pf-utility-weight", "0.0",
    "--reward-low-throughput-weight", "0.0",
    "--reward-urgency-service-weight", "0.0",
    "--reward-fairness-delta-weight", "0.0",
    "--reward-pf-utility-delta-weight", "0.0",
    "--reward-starvation-penalty-weight", "0.0",
    "--deadline-risk-penalty-weight", "0.0",
    "--max-wait-risk-penalty-weight", "0.0",
    "--population-wait-penalty-weight", "0.0",
]


def _completed(run_dir: Path) -> bool:
    return all(
        (run_dir / name).is_file()
        for name in ("latest.pt", "training.csv", "validation.csv", "run_config.json")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether annealing Beta exploration noise makes the deterministic "
            "PPO mean-policy learn UE rotation under the fixed 25/75 reward."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/reward_study/archive/exploration_alignment_25_75.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/runs/reward_study"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.plan.read_text(encoding="utf-8"))
    round_id = str(payload["round_id"])
    common: dict[str, Any] = dict(payload["common"])
    cases: list[dict[str, Any]] = list(payload["cases"])
    round_dir = args.output_root / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    write_json(round_dir / "round_plan_snapshot.json", payload)

    failed: list[str] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        run_dir = round_dir / case_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if _completed(run_dir) and not args.force:
            print(f"[{index}/{len(cases)}] skip completed: {case_id}")
            continue

        command = _common_command(
            common=common,
            run_dir=run_dir,
            steps_override=None,
            validation_slots_override=None,
            progress=args.progress,
            device=args.device,
        )
        # Remove common concentration switches and replace with case-specific values.
        filtered: list[str] = []
        skip_next = False
        for token in command:
            if skip_next:
                skip_next = False
                continue
            if token in {"--beta-concentration-start", "--beta-concentration-end"}:
                skip_next = True
                continue
            if token in {"--freeze-beta-concentration", "--no-freeze-beta-concentration"}:
                continue
            filtered.append(token)
        command = filtered + REWARD_ARGS
        command.extend(
            [
                "--beta-concentration-start",
                str(float(case["beta_concentration_start"])),
                "--beta-concentration-end",
                str(float(case["beta_concentration_end"])),
                "--freeze-beta-concentration"
                if bool(case["freeze_beta_concentration"])
                else "--no-freeze-beta-concentration",
            ]
        )

        positive_weights = {
            "throughput": 0.25,
            "fairness": 0.75,
            "service": 0.0,
            "deficit_service": 0.0,
            "pf_utility": 0.0,
            "low_throughput": 0.0,
            "urgency_service": 0.0,
        }
        run_config = {
            "study_id": "reward_exploration",
            "round_id": round_id,
            "case": {
                "id": case_id,
                "label": str(case["label"]),
                "hypothesis": str(case.get("hypothesis", "")),
                "positive_scale": 1.0,
                "positive_weights": positive_weights,
                "delta_weights": {"fairness": 0.0, "pf_utility": 0.0},
                "penalty_weights": {
                    "starvation": 0.0,
                    "deadline_risk": 0.0,
                    "max_wait_risk": 0.0,
                    "population_wait": 0.0,
                },
                "beta_concentration_start": float(case["beta_concentration_start"]),
                "beta_concentration_end": float(case["beta_concentration_end"]),
                "freeze_beta_concentration": bool(case["freeze_beta_concentration"]),
            },
            "common": common,
            "architecture": {
                "observation_features_per_ue": 16,
                "encoder": "shared_set_encoder",
                "embedding_dim": int(common.get("hidden_dim", 64)),
                "candidate_mode": "all",
                "scheduler_mode": "ppo_only",
                "num_ues": 1200,
                "top_k": 64,
                "num_prbs": 273,
            },
            "command": command,
        }
        write_json(run_dir / "run_config.json", run_config)
        (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        print(f"[{index}/{len(cases)}] {case_id}: {case['label']}")
        write_json(run_dir / "status.json", {"status": "running", "started_unix": time()})
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            failed.append(case_id)
            write_json(
                run_dir / "status.json",
                {"status": "failed", "return_code": result.returncode, "finished_unix": time()},
            )
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
        else:
            write_json(
                run_dir / "status.json",
                {"status": "completed", "return_code": 0, "finished_unix": time()},
            )

    if failed:
        print(f"failed cases: {', '.join(failed)}")
        raise SystemExit(1)

    if not args.skip_diagnostics:
        case_ids = ",".join(str(case["id"]) for case in cases)
        diagnostics_dir = round_dir / "diagnostics"
        docs_output = Path(
            "docs/analysis/reward_study/round_04/exploration_alignment_diagnostics.html"
        )
        diag_command = [
            sys.executable,
            "-m",
            "scalemac_rl.scripts.run_policy_diagnostics",
            "--study-root",
            str(round_dir),
            "--cases",
            case_ids,
            "--checkpoint",
            "latest.pt",
            "--modes",
            "deterministic,stochastic",
            "--slots",
            str(int(common.get("validation_slots", 5000))),
            "--seed",
            str(int(common.get("seed", 1701))),
            "--seeds",
            "1",
            "--profile-seed",
            str(int(common.get("profile_seed", common.get("seed", 1701)))),
            "--window-size",
            str(int(common.get("starvation_threshold_slots", 64))),
            "--device",
            args.device,
            "--output-root",
            str(diagnostics_dir),
            "--docs-output",
            str(docs_output),
        ]
        result = subprocess.run(diag_command, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print(f"round output: {round_dir}")


if __name__ == "__main__":
    main()
