# ScaleMAC-RL v0.11.0

## Dynamic CQI

- Add a backward-compatible `static` CQI mode and a temporally correlated Dynamic CQI mode.
- Preserve each UE's heterogeneous CQI anchor while allowing controlled mean-reverting variation over slots.
- Add configurable temporal correlation, innovation standard deviation, update interval, and per-update CQI rate limit.
- Use a separate channel RNG stream so CQI innovations do not perturb the HARQ random sequence.
- Log CQI dynamics into training and validation outputs.
- Add a controlled three-case Round 10 screen: static, slow dynamic, faster dynamic.
- Keep T–J–S reward, PPO, state/action architecture, full-buffer traffic, fixed BLER and full-control scheduling unchanged.

## Scope boundary

This release is not yet Link Adaptation: BLER remains CQI-independent and the agent does not choose MCS or transmission layers.
