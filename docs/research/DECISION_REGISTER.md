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
