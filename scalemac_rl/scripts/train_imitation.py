from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch import nn

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.models import SharedSetPolicy
from scalemac_rl.reporting import markdown_report_path, write_csv, write_markdown
from scalemac_rl.schedulers import ProportionalFairScheduler


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ues", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episode-slots", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pf_imitation.pt"))
    parser.add_argument(
        "--log-output",
        type=Path,
        default=Path("artifacts/imitation_training.csv"),
    )
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")

    device = _resolve_device(args.device)

    config = ScaleMacConfig(num_ues=args.num_ues, episode_slots=args.episode_slots)
    if config.max_selected_ues > config.num_ues:
        config.max_selected_ues = config.num_ues
    if config.num_prbs < config.max_selected_ues:
        config.max_selected_ues = config.num_prbs
    config.validate()

    env = ScaleMacDownlinkEnv(config)
    teacher = ProportionalFairScheduler()
    policy = SharedSetPolicy(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    observation, _ = env.reset(seed=config.seed)
    teacher.reset()
    window: dict[str, list[float]] = {
        "loss": [],
        "teacher_reward": [],
        "teacher_throughput_score": [],
        "teacher_fairness_score": [],
        "teacher_service_score": [],
        "teacher_starvation_rate": [],
    }
    log_rows: list[dict[str, Any]] = []

    def flush_log(step: int) -> None:
        if not window["loss"]:
            return
        row = {
            "step": step,
            "window_steps": len(window["loss"]),
            "mean_loss": mean(window["loss"]),
            "mean_teacher_reward": mean(window["teacher_reward"]),
            "mean_teacher_throughput_score": mean(window["teacher_throughput_score"]),
            "mean_teacher_fairness_score": mean(window["teacher_fairness_score"]),
            "mean_teacher_service_score": mean(window["teacher_service_score"]),
            "mean_teacher_starvation_rate": mean(window["teacher_starvation_rate"]),
            "device": str(device),
        }
        log_rows.append(row)
        print(
            f"step={step:6d} loss={row['mean_loss']:.6f} "
            f"teacher_reward={row['mean_teacher_reward']:.4f} device={device}"
        )
        for values in window.values():
            values.clear()

    for step in range(1, args.steps + 1):
        target = teacher.act(observation)
        x = torch.from_numpy(observation).to(device)
        y = torch.from_numpy(target).to(device)

        prediction = policy(x)
        loss = loss_fn(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        observation, _, terminated, truncated, info = env.step(target)
        window["loss"].append(float(loss.item()))
        window["teacher_reward"].append(float(info["reward_total"]))
        window["teacher_throughput_score"].append(float(info["throughput_score"]))
        window["teacher_fairness_score"].append(float(info["fairness_score"]))
        window["teacher_service_score"].append(float(info["service_score"]))
        window["teacher_starvation_rate"].append(float(info["starvation_rate"]))

        if terminated or truncated:
            observation, _ = env.reset()
            teacher.reset()

        if step % args.log_every == 0:
            flush_log(step)

    flush_log(args.steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "hidden_dim": args.hidden_dim,
            "input_dim": 8,
            "config": config.to_dict(),
            "training": {
                "steps": args.steps,
                "learning_rate": args.lr,
                "device": str(device),
                "teacher": "proportional_fair",
                "final_logged_mean_loss": log_rows[-1]["mean_loss"],
            },
        },
        args.output,
    )

    write_csv(args.log_output, log_rows)
    write_markdown(
        markdown_report_path(args.log_output),
        title="ScaleMAC-RL PF imitation training",
        description=(
            "Behavioral cloning of the proportional-fair teacher. The report includes "
            "training loss and the normalized reward profile generated by the teacher trajectories."
        ),
        rows=log_rows,
        notes=(
            f"Checkpoint: `{args.output}`",
            "The teacher reward is diagnostic only; imitation optimizes MSE against the teacher action.",
        ),
    )

    print(f"saved: {args.output}")
    print(f"saved: {args.log_output}")
    print(f"saved: {markdown_report_path(args.log_output)}")


if __name__ == "__main__":
    main()
