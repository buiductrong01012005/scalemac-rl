# ScaleMAC-RL v0.3

## Main correction

The previous milestone trained a PF imitation network but did not yet optimize a policy through reinforcement learning. v0.3 adds actual PPO interaction and policy updates in the fast scheduler environment.

## Added

- `SharedSetActorCritic` with a bounded stochastic actor and set-pooled critic.
- PPO fine-tuning initialized from the PF imitation checkpoint.
- UE curriculum from small populations to 1,200 UEs.
- Safety-aware candidate filtering that retains HARQ and long-waiting UEs.
- Candidate action masking while preserving the full `[num_ues, 2]` output contract.
- Vectorized multi-environment rollout collection.
- Paired baseline/PPO evaluation using the same seeds.
- Warm inference benchmark and deadline-miss reports.
- CSV/Markdown output for PPO training and evaluation.

## Not yet included

- Multiprocessing or cluster-distributed rollout workers.
- Automated OOD suites across channel/traffic models.
- 5G-LENA integration and full 3GPP validation.
- Shadow mode or hardware-in-the-loop.
