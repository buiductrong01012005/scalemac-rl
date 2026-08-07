from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

from scalemac_rl import ScaleMacConfig, ScaleMacDownlinkEnv
from scalemac_rl.schedulers import (
    MaxCqiScheduler,
    ProportionalFairScheduler,
    RoundRobinScheduler,
)


def evaluate(name: str, scheduler, config: ScaleMacConfig, seed: int) -> dict[str, float | int | str]:
    env = ScaleMacDownlinkEnv(config)
    observation, _ = env.reset(seed=seed)
    scheduler.reset()

    rewards: list[float] = []
    goodput: list[float] = []
    fairness: list[float] = []
    starvation: list[float] = []
    p99_wait: list[float] = []

    while True:
        action = scheduler.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        goodput.append(info["cell_goodput_bits"])
        fairness.append(info["jain_fairness"])
        starvation.append(info["starvation_rate"])
        p99_wait.append(info["p99_wait_slots"])
        if terminated or truncated:
            break

    return {
        "scheduler": name,
        "seed": seed,
        "num_ues": config.num_ues,
        "slots": config.episode_slots,
        "mean_reward": mean(rewards),
        "mean_goodput_bits_per_slot": mean(goodput),
        "final_jain_fairness": fairness[-1],
        "mean_starvation_rate": mean(starvation),
        "final_p99_wait_slots": p99_wait[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ues", type=int, default=1200)
    parser.add_argument("--slots", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baselines.csv"))
    args = parser.parse_args()

    config = ScaleMacConfig(num_ues=args.num_ues, episode_slots=args.slots)
    config.validate()

    factories = {
        "rr": lambda: RoundRobinScheduler(config.max_selected_ues),
        "max_cqi": MaxCqiScheduler,
        "pf": ProportionalFairScheduler,
    }

    rows = []
    for seed in range(args.seeds):
        for name, factory in factories.items():
            row = evaluate(name, factory(), config, seed)
            rows.append(row)
            print(
                f"{name:8s} seed={seed} reward={row['mean_reward']:.4f} "
                f"goodput={row['mean_goodput_bits_per_slot']:.1f} "
                f"fairness={row['final_jain_fairness']:.4f} "
                f"starvation={row['mean_starvation_rate']:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
