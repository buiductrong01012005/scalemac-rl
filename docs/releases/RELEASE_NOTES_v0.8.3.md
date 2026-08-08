# ScaleMAC-RL v0.8.3

- Keeps the 16-feature per-UE observation and unchanged shared set encoder.
- Stores every full-control experiment inside one run directory under `artifacts/runs/`.
- Stores checkpoints, CSV logs, manifests, and Markdown reports together.
- Adds `--run-dir` to the full-control training wrapper.
