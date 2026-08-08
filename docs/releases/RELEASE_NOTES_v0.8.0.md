# ScaleMAC-RL v0.8.0

## Research focus

v0.8.0 starts the rule-free, end-to-end PPO research track. PPO now has an entry point that observes all 1,200 UEs, selects the Top-64, and predicts PRB demand without heuristic candidate filtering, oldest-UE reserves, forced HARQ selection, or PF imitation.

## Observation v2

The per-UE observation grows from 10 to 16 features by adding:

- CQI percentile rank;
- demand-normalized throughput-deficit rank;
- successful-delivery wait rank;
- throughput relative to the cell mean;
- wait relative to the active deadline;
- previous-slot PRB share.

Legacy 8-feature and 10-feature checkpoints remain loadable through zero-padded input adapters.

## Reward v2

Optional positive reward terms support:

- proportional-fair utility level;
- low-percentile UE throughput;
- successful service of urgent/underserved UEs.

A population-wide wait-risk penalty provides dense pressure before P99 or maximum wait crosses a hard target. Legacy reward defaults are preserved; the new full-control entry point enables the v2 profile explicitly.

## PPO optimization

- gamma 0.999 and GAE lambda 0.97 in the main profile;
- linear learning-rate decay;
- linear entropy-coefficient decay;
- smaller PPO clipping for stable full-UE updates;
- 300,032-step direct full-control experiment;
- optional 128→256→600→1200 curriculum;
- short balanced/fairness/stable profile tuning runner.

## Documentation

A research notebook under `docs/research/` records:

- confirmed findings from rule-dependency and PPO-only experiments;
- environment/MDP definitions;
- the full-control PPO plan;
- reproducibility protocol;
- architecture decisions for RPPO and CNN;
- a report template and decision register.

`build_full_control_report` generates a Markdown research report from CSV outputs.

## Source archive policy

The source ZIP contains code, tests, configs, and docs only. It does not include `artifacts/`.
