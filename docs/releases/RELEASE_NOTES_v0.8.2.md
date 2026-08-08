# ScaleMAC-RL v0.8.2

- Restores the expanded 16-feature per-UE observation from v0.8.0.
- Keeps the existing shared set-encoder architecture unchanged: one shared UE MLP, hidden dimension 64, and global mean/max pooling.
- Keeps PPO in full control of UE selection and PRB-demand scores over all 1,200 UEs.
- Keeps the action projector limited to Top-64 and exact 273-PRB feasibility.
- Keeps candidate filtering, oldest-UE rules, safety reserves, and forced HARQ selection disabled.
- Clarifies that feature expansion and encoder architecture are separate design choices.
