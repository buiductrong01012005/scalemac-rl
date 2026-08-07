from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.models import SharedSetPolicy
from scalemac_rl.schedulers import ProportionalFairScheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ues", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--episode-slots", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pf_imitation.pt"))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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
    running_loss = 0.0

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

        running_loss += float(loss.item())
        observation, _, terminated, truncated, _ = env.step(target)
        if terminated or truncated:
            observation, _ = env.reset()
            teacher.reset()

        if step % 100 == 0:
            print(f"step={step:6d} loss={running_loss / 100:.6f} device={device}")
            running_loss = 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "hidden_dim": args.hidden_dim,
            "input_dim": 8,
            "config": config.to_dict(),
        },
        args.output,
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
