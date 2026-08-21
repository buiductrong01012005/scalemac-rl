# ScaleMAC-RL v0.22.2

Round 17C adds an evaluation-only environment stress audit.

- evaluates the privileged service-aware Oracle and PF over 18 deterministic environment seeds;
- includes the three main study seeds 1701, 2701, and 3701;
- uses the current Dynamic-CQI + delayed-CSI + CQI/MCS/BLER/HARQ environment;
- exports per-seed metrics, population summary, and per-metric hardness percentiles for the three key seeds;
- performs no PPO training and changes no scheduler learning behavior.
