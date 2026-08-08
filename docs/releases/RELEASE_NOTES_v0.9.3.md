# ScaleMAC-RL v0.9.3

v0.9.3 archives the completed Round 06 coordinate experiment and starts the
next controlled reward-screening stage.

## Round 06 evidence

- archives the six-case deterministic validation trajectory under
  `docs/analysis/reward_study/round_06/`;
- identifies a local stable region in which Throughput and Service act as
  complementary anchors while Jain controls distribution;
- records late-training collapse when Service is weakened;
- compares stochastic training metrics with deterministic validation;
- keeps the equal-third case as the balanced reference and the mild
  Throughput tilt as a throughput-oriented alternative.

## Round 07

`configs/reward_study/round_07_fourth_component_screen.json` adds exactly one
new reward component per case:

- throughput-deficit service;
- PF utility;
- low-throughput percentile score;
- urgency service.

Each case uses four equal positive coefficients of `0.25`. All undeclared
rewards, deltas, penalties, constraints, rules and forced HARQ selections stay
disabled. Coefficients are not tuned until the marginal effect of each new
component is understood.

## Documentation navigation

- adds `synthesis_round_01_to_06.html`;
- adds `ablation_map.html`;
- links Round 06 results and the Round 07 plan from both analysis indexes;
- updates the literature context to distinguish current multi-objective
  behavior from Maximum QoS Satisfaction frameworks with traffic classes.

Source archives remain source-only and do not include `artifacts/`.
