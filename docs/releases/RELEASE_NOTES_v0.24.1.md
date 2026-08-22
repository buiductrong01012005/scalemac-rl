# ScaleMAC-RL v0.24.1

Round 19B corrects the UE-group actor scale discovered in Round 19A.

- add `ue_group_sum` PPO ratio mode;
- clip priority/demand ratios per dimension and sum UE surrogate terms before minibatch averaging;
- keep adaptive J_IS controlled by mean 2-D per-UE J_IS while scaling its actor penalty by the same UE sum;
- add Round 19B 3×3 controlled study and final deterministic/stochastic diagnostics;
- retain Service40 reward, 16-feature observation, learned Beta exploration, radio environment and PPO sampling budget.
