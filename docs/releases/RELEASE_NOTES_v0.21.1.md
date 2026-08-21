# ScaleMAC-RL v0.21.1

- Add Round 16D shared-vs-split actor/critic encoder interference study.
- Add a split critic encoder initialized as an exact copy of the legacy shared encoder without consuming extra RNG.
- Persist and reload the split-encoder architecture in PPO checkpoints.
