# ScaleMAC-RL v0.13.0

## Link Adaptation foundation

This release adds the first joint PHY/MAC realism layer without changing the PPO action space or T–J–S reward.

- Keep Slow Dynamic CQI and the CSI-reporting abstraction.
- Add an NR-inspired CQI Table-1 / PDSCH MCS Table-1 mapping.
- Select MCS from scheduler-visible **reported CQI**.
- Compute transmission BLER from the mismatch between selected MCS support and **true CQI**.
- Preserve the legacy fixed-BLER path for regression comparison.
- Export effective spectral efficiency, MCS, predicted/observed BLER and HARQ retransmission diagnostics.
- Add Round 12 with legacy, perfect-CSI, delayed-CSI and delayed+noisy-CSI Link Adaptation cases.

## Scope

The CQI and MCS table values follow the 3GPP NR table structure. The smooth BLER-vs-CQI-mismatch function is intentionally a simulator abstraction, not a link-level 3GPP BLER curve. Exact link-level BLER requires SINR, coding/TBS, channel and receiver modeling.

## Repository hygiene

`docs/analysis/` and generated HTML/CSV research reports are ignored so experiment-analysis documents remain local unless explicitly promoted to public documentation.
