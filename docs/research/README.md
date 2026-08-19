# ScaleMAC-RL Research Notebook

Thư mục này ghi lại **quá trình khám phá môi trường**, các giả thuyết, quyết định thiết kế và kết quả thí nghiệm. Mục tiêu không chỉ là tìm một checkpoint có reward cao, mà còn giải thích được vì sao policy hoạt động hoặc thất bại.

## Tài liệu chính

- [`RESEARCH_LOG.md`](RESEARCH_LOG.md): nhật ký theo phiên bản và các phát hiện đã xác nhận.
- [`ENVIRONMENT_AND_MDP.md`](ENVIRONMENT_AND_MDP.md): snapshot environment/MDP hiện tại, gồm Dynamic CQI và CSI reporting.
- [`DYNAMIC_CQI_DESIGN.md`](DYNAMIC_CQI_DESIGN.md): thiết kế Dynamic CQI v0.11.0, tham số FLEX, metric và phạm vi Round 10.
- [`CSI_REPORTING_DESIGN.md`](CSI_REPORTING_DESIGN.md): thiết kế true CQI → reported CSI, period/delay/error và phạm vi Round 11.
- [`DYNAMIC_CQI_VARIABLE_REFERENCE.csv`](DYNAMIC_CQI_VARIABLE_REFERENCE.csv): bảng tra nhanh các biến Dynamic CQI và scope Link Adaptation tương lai.
- [`NETWORK_VARIABLE_REFERENCE.csv`](NETWORK_VARIABLE_REFERENCE.csv): bảng tra cứu nhanh các biến mạng và khả năng cấu hình.
- [`FULL_CONTROL_PPO_PLAN.md`](FULL_CONTROL_PPO_PLAN.md): kế hoạch PPO điều khiển toàn bộ scheduler, không rule và không candidate filter.
- [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md): cách train, chọn checkpoint và đánh giá tái lập.
- [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md): khung báo cáo nghiên cứu sau mỗi vòng chạy.
- [`DECISION_REGISTER.md`](DECISION_REGISTER.md): các quyết định quan trọng và lý do.

## Nguyên tắc nghiên cứu

1. **Tách KPI khỏi reward.** Reward là tín hiệu học; kết luận dựa trên goodput, Jain fairness, P99/max wait, starvation và inference latency.
2. **Một thay đổi mỗi vòng thí nghiệm.** Không đổi đồng thời input, reward, hyperparameter và kiến trúc nếu chưa có ablation.
3. **So sánh cùng protocol.** Cùng CQI/demand profile, HARQ seed, số PRB, Top-K và thời lượng rollout.
4. **Không để rule che năng lực PPO.** Nhánh nghiên cứu chính là PPO-only full control; rule chỉ còn là baseline hoặc ablation.
5. **Ghi lại kết quả âm.** Plateau, collapse và checkpoint không feasible đều là bằng chứng về môi trường và thiết kế học.

## Reward study

Các bản phân tích HTML được lưu riêng tại [`../analysis/reward_study/index.html`](../analysis/reward_study/index.html). `docs/research/` giữ kế hoạch và nhật ký; `docs/analysis/` giữ kết quả phân tích để sau này tổng hợp thành khóa luận; `docs/reports/` dành cho báo cáo chính thức.
