# ScaleMAC-RL v0.8.1

- Restores the previous 10-feature per-UE observation.
- Restores the previous shared set encoder with hidden dimension 64.
- Keeps full PPO control over all 1,200 UEs.
- Keeps the projector limited to Top-64 and exact 273-PRB feasibility.
- Keeps rule, candidate filtering, and forced HARQ selection disabled.
- Records reward and hyperparameter changes separately from representation changes.
