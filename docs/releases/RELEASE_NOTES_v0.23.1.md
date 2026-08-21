# ScaleMAC-RL v0.23.1

Round 18B adds a high-dimensional PPO objective study.

- Keeps the shared entity actor/critic architecture, service40 reward, R256/MB8, learned Beta exploration, 16 features and radio model fixed.
- Adds factorized per-action-dimension Beta log probabilities.
- Implements the on-policy core of Han & Sung (ICML 2019) dimension-wise IS clipping with adaptive J_IS coefficient control.
- Compares the paper-form dimension-sum objective with a ScaleMAC loss-scale-normalized ablation.
- This is not the full DISC algorithm: replay-buffer reuse and GAE-V remain intentionally excluded so the study isolates the clipping/objective intervention.
- The normalized ablation scales the joint log-ratio control statistic by sqrt(active action dimensions) so the J_IS target is comparable when ScaleMAC has ~2400 active action dimensions.
