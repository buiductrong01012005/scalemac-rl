# Decision Register

| ID | Quyết định | Lý do | Trạng thái |
|---|---|---|---|
| D01 | Giữ projector | Cần action hợp lệ; projector không được tự scheduling UE | Giữ |
| D02 | Bỏ oldest-UE rule khỏi nhánh chính | Rule che năng lực PPO và gánh safety cho Hybrid cũ | Đã áp dụng v0.8 |
| D03 | Bỏ candidate filter khỏi nhánh full-control | Cần kiểm tra PPO thực sự xử lý 1.200 UE | Đã áp dụng v0.8 |
| D04 | Không dùng PF imitation | Muốn quan sát PPO học từ đầu | Đã áp dụng v0.8 |
| D05 | Thêm rank/ratio features | Giá trị tương đối trong cell hữu ích hơn UE ID hoặc giá trị tuyệt đối | Đã áp dụng v0.8 |
| D06 | Thêm PF/P10/urgency reward | Jain toàn cell có credit assignment yếu | Đã áp dụng v0.8 |
| D07 | Gamma 0.999 và LR/entropy decay | Hậu quả starvation dài hạn, cần update ổn định | Đang thử |
| D08 | RPPO sau feed-forward baseline | Tránh đổi input, reward và architecture cùng lúc | Chờ kết quả v0.8 |
| D09 | CNN không chạy theo UE ID | Quan hệ lân cận UE ID là giả | Giữ |

## D-008 — Expand the observation without changing the encoder architecture

**Decision:** Use the 16-feature per-UE observation while preserving the existing shared set-encoder architecture with hidden dimension 64 and global mean/max pooling. PPO controls all UE selection and PRB-demand decisions.

**Reason:** The research question concerns whether richer state information and reward/hyperparameter tuning let PPO learn the full scheduling trade-off. Changing to GNN, CNN, attention, or recurrent layers in the same experiment would confound representation architecture with observation design.

## DR-008 — Treat reward design as a staged empirical study

**Decision:** Freeze environment, 16-feature input, Set Encoder, PPO architecture, and PPO hyperparameters during initial reward discovery. Screen reward components independently, add objective families gradually with comparable coefficients, then tune retained weights and analyze the Pareto frontier.

**Reason:** The previous reward combined seven positive components, dense deltas, explicit penalties, and Lagrangian penalties. This made causal attribution weak even when KPI outcomes were measurable.

**Operational consequence:** Reward-study runs disable undeclared shaping and Lagrangian training penalties. Each run records exact coefficients, decomposition, KPIs, seed, hyperparameters, and checkpoint provenance in a reusable dataset.

## Decision: diagnose policy action mode before adding more reward terms

- **Decision:** compare deterministic and stochastic execution of the same Round 02 checkpoints.
- **Reason:** a larger Jain coefficient produced worse deterministic fairness, suggesting flat priorities or Top-K tie behaviour rather than a simple reward-weight problem.
- **Controlled variables:** 16 features/UE, shared Set Encoder, PPO, full-control 1200 UE environment, reward weights, and checkpoint remain unchanged.
- **Deferred:** new rewards, hyperparameter tuning, architecture changes, dataset generation, and Pareto analysis.


## v0.8.8 — Incremental reward exploration

- Current goal: understand the environment and the effect of each reward component.
- Add one component at a time, begin with equal coefficients, then tune only a small number of weights after explaining the observed KPI changes.
- Do not optimize the failed 25/75 case in isolation.
- Do not generate a reward-weight dataset or Pareto study in the current phase.
- Round 04 adds `service` to the throughput–Jain base using equal one-third coefficients.

## Decision 12 — Add one reward component at a time

**Status:** active in v0.8.8.

**Decision:** After screening individual components and establishing the throughput–Jain base, add only one new reward component per round. Start with equal coefficients among active components, explain the KPI effect, then tune only a small number of coefficients if the component is useful.

**Current case:** throughput + Jain fairness + service, each with coefficient `1/3`.

**Deferred:** dense weight grids, optimization of the failed 25/75 case, dataset generation, Pareto analysis, and architecture changes.


## Decision 13 — Retain two Round 07 candidates for multi-seed confirmation

**Status:** active after v0.9.6.

**Decision:** retain `urgency_service_hold_throughput` as the balanced candidate
and `deficit_service_anchor_preserving` as the fairness/tail-delay candidate.
Compare both against the equal-third T–J–S reference using at least three common
seeds.

**Reason:** urgency hold-T is the only Round 07 case that improves deterministic
Jain and P99 without reducing goodput or coverage at the final checkpoint. Deficit
group T+S provides the best tail delay and improved Jain but trades about 4.9%
goodput. These represent two distinct trade-off directions worth confirming.

**Deferred:** coefficient micro-sweeps, PPO hyperparameter tuning, Beta schedule
changes and architecture changes. PF utility and low-throughput percentile are
removed from the active objective until their score definitions are revised.


## Decision 14 — Separate reader-facing conclusions from the technical audit

**Status:** active after v0.9.7.

**Decision:** use the reader-friendly Round 07 HTML as the main entry point, while
preserving the original technical report as an appendix. Every future reward
round should state the reward formula, decode each case geometry, report KPI and
stability, distinguish meaningful marginal gain from merely non-collapsing
behaviour, and record whether a component is retained, parked or redesigned.

**Reason:** a technically complete report is not reusable if a future reader
cannot reconstruct what each case added or why a reward was kept or removed.

## Decision 15 — Confirm retained reward candidates on common seeds before tuning

**Status:** active in v0.9.8.

**Decision:** run exactly nine cases: three fixed reward profiles on seeds 1701,
2701 and 3701. The profiles are equal-third T–J–S, Urgency hold-Throughput, and
Deficit group Throughput+Service. Use the same seed as training, static-profile and
validation seed within each paired comparison.

**Reason:** Round 07 used one seed and therefore identified candidates but did not
establish repeatability. A common-seed design isolates profile differences more
cleanly than another broad coefficient sweep.

**Pre-registered decision rule:** Urgency must be stable on all seeds, improve Jain
repeatedly, and retain at least 98% of baseline mean goodput. Deficit may be retained
as a delay-sensitive profile only if tail-delay gains repeat and its goodput cost is
acceptable. If neither condition is met, equal-third T–J–S remains the active reward.

**Deferred:** new reward components, coefficient micro-sweeps, PPO tuning, Beta
concentration tuning, architecture changes, and dataset/Pareto work.
