# ScaleMAC-RL v0.12.1

Documentation-only release for completed Round 11 CSI reporting study.

## Evidence
- 4/4 CSI cases completed.
- Perfect CSI maximizes goodput but ends with weaker Jain fairness and non-zero starvation.
- Periodic reporting removes starvation and improves service interval.
- Periodic reporting with 2-slot delivery delay gives the strongest balanced outcome in this screen.
- Adding measurement noise of 0.75 CQI-index units produces only a small Jain change while slightly worsening tail wait.

## Decision
- Perfect CSI remains an information-reference case.
- Periodic CSI (period 4, delay 2) becomes the primary realism profile for the next Link Adaptation round.
- Delayed + noisy CSI remains a robustness/stress case.
- Reward, PPO, environment transition logic, traffic, and HARQ are unchanged in v0.12.1.
