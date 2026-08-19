# ScaleMAC-RL v0.13.2

Test-suite maintenance release after moving experiment-analysis documents to local-only storage.

## Fix

- Remove tests that required ignored `docs/analysis/` HTML/CSV artifacts to exist in a clean source checkout.
- Keep behavioral tests for reward-study plans and report generators that operate on temporary data.
- Add explicit tests that `docs/analysis/` and runtime `*.log` files remain ignored.
- Preserve the v0.13.1 legacy float32 PHY regression fix.
- No PPO, reward, channel, CSI, MCS, BLER, HARQ, state, or action behavior is changed.
