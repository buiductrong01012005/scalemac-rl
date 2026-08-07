from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.models import SharedSetPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = SharedSetPolicy(
        input_dim=checkpoint.get("input_dim", 8),
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    config = ScaleMacConfig(num_ues=args.num_ues, episode_slots=args.slots)
    config.validate()
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=args.seed)

    total_reward = 0.0
    final_info = {}
    with torch.inference_mode():
        while True:
            action = policy(torch.from_numpy(observation).to(device)).cpu().numpy()
            observation, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

    print(f"mean_reward={total_reward / args.slots:.6f}")
    print(f"final_fairness={final_info['jain_fairness']:.6f}")
    print(f"final_p99_wait_slots={final_info['p99_wait_slots']:.1f}")
    print(f"mean_last_slot_goodput_bits={final_info['cell_goodput_bits']:.1f}")


if __name__ == "__main__":
    main()
