# Round 03: deterministic vs stochastic policy diagnostics

This round does not train a new policy and does not change reward, input features,
architecture, or environment. It compares the same checkpoint under:

- deterministic mean action, used for deployment-style validation;
- stochastic Beta action sampling, used during PPO rollout collection.

The generated HTML is written to `policy_diagnostics.html` after the diagnostic
command completes. Detailed CSV files remain under the matching artifact run
folder.
