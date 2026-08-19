# ScaleMAC-RL v0.12.0

## Purpose

Add an explicit CSI-reporting observation layer on top of the selected Slow Dynamic CQI environment before introducing Link Adaptation.

## Added

- `true CQI` and `reported CQI` are separated.
- configurable CSI report mode, period, delay and measurement error;
- independent CSI RNG stream;
- scheduler CQI/CQI-rank observations now use reported CQI;
- PHY capacity and throughput oracle continue to use true CQI;
- CSI age/error/staleness metrics in training and validation CSV;
- Round 11 four-case controlled CSI screen and HTML/Markdown/CSV analysis;
- tests for perfect CSI compatibility, delayed-report semantics and RNG isolation.

## Scientific controls

- Slow Dynamic CQI is fixed in all Round 11 cases.
- Reward remains equal-third Throughput + Jain + Service.
- PPO architecture and 16-feature observation width remain unchanged.
- MCS, CQI-dependent BLER, RPPO and transmission layers are intentionally deferred.

## Next checkpoint

After CSI reporting, add the first Link Adaptation abstraction: reported CQI drives MCS choice while true CQI + chosen MCS determines BLER. Then reassess whether feed-forward PPO remains adequate or whether RPPO/memory and objective changes are justified.
