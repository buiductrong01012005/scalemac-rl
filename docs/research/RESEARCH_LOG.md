# Nhật ký nghiên cứu ScaleMAC-RL

## Câu hỏi trung tâm

Liệu một PPO scheduler có thể tự học trade-off giữa throughput, fairness và delay cho 1.200 UE mà không dựa vào oldest-UE rule hay candidate filter hay không?

## Bối cảnh cố định

- Một gNB, một cell, downlink.
- 1.200 UE active và full-buffer.
- 273 PRB/slot, tối đa 64 UE được cấp lịch.
- Static heterogeneous CQI trong episode.
- HARQ abstraction.
- Mục tiêu triển khai: starvation bằng 0, fairness cao, P99/max wait thấp và giữ goodput cạnh tranh.

## Các phát hiện đã xác nhận

### 1. Hybrid PPO tăng goodput nhưng rule gánh safety

Ablation giữ nguyên actor Hybrid và tắt oldest-UE reserve cho thấy starvation và tail delay sụp mạnh. Khi tăng số rule grant, P99/max wait giảm rõ và goodput cũng tăng, nhưng fairness giảm. Kết luận đúng cho actor Hybrid cũ là:

```text
oldest-UE rule  -> chống starvation và kiểm soát delay
PPO demand head -> tăng throughput bằng phân bổ PRB không đều
```

Do đó không thể dùng Hybrid cũ để claim PPO đã tự học toàn bộ scheduler.

### 2. PPO-only học từ đầu có thể tự đạt safety

PPO-only candidate-128 sau khoảng 300k bước đạt gần:

```text
Jain fairness ≈ 0.60
starvation = 0
P99 ≈ 50 slot
max wait ≈ 51 slot
goodput > RR
```

Điều này chứng minh PPO có khả năng học scheduling thật, nhưng candidate filter vẫn giới hạn bài toán từ 1.200 UE xuống 128 candidates.

### 3. Guard nhỏ không giải quyết câu hỏi nghiên cứu

Thêm 8–16 rule grant vào PPO-only cải thiện goodput và delay nhưng giảm fairness. Guard hữu ích cho triển khai an toàn, nhưng không trả lời liệu PPO có thể tự học full control. Vì vậy guard không còn là hướng chính.

### 4. Train lâu hơn không tự sửa reward/credit assignment

Hybrid plateau sớm; tăng số bước chỉ tăng nhẹ goodput và có thể làm fairness giảm. Vấn đề chính là input tương đối, reward credit assignment và long-horizon hyperparameters, không phải chỉ thiếu step.

## Quyết định cho v0.8

Nhánh chính chuyển sang:

```text
1.200 UE
→ PPO quan sát toàn bộ UE
→ PPO chấm priority và PRB demand cho toàn bộ UE
→ PPO chọn Top-64
→ projector chỉ đảm bảo action hợp lệ và tổng đúng 273 PRB
```

Không dùng:

- heuristic candidate filter;
- oldest-UE reserve;
- forced HARQ selection;
- PF imitation checkpoint.

## Giả thuyết v0.8

### H1 — Input tương đối giúp policy tổng quát hơn

Các feature rank và ratio giúp PPO biết một UE đang đứng ở đâu trong cell, thay vì chỉ nhìn giá trị tuyệt đối:

- CQI rank;
- throughput-deficit rank;
- wait rank;
- throughput/cell mean;
- wait/deadline;
- PRB share ở slot trước.

### H2 — Reward PF và low-percentile tạo credit fairness tốt hơn Jain tổng

Jain fairness toàn cell thay đổi chậm và khó quy trách nhiệm cho một action. Reward v2 thêm:

- proportional-fair utility level;
- P10 throughput so với cell mean;
- thưởng phục vụ UE vừa thiếu throughput vừa gần deadline;
- population-wide wait pressure.

### H3 — Long-horizon PPO cần gamma cao và update nhỏ

Starvation xuất hiện sau hàng chục slot, nên thử:

- gamma 0.999;
- GAE lambda 0.97;
- rollout 256;
- clip 0.08–0.10;
- learning-rate decay;
- entropy decay.

## Điều kiện để coi PPO thắng Rule-only

PPO không cần đứng đầu từng KPI. PPO được xem là tốt hơn về trade-off khi:

- starvation bằng 0;
- không bị Rule-only Pareto-dominate;
- goodput cao hơn Rule-only đáng kể;
- fairness và tail delay không cách quá xa Rule-only;
- balanced score cao hơn hoặc worst KPI gap thấp hơn;
- kết quả giữ được qua nhiều seed.

## Các bước sau v0.8

1. Train `balanced` profile 300k bước.
2. Nếu policy collapse: chạy short tuning `balanced,fairness,stable` khoảng 100k/profile.
3. Phân tích input/reward component và checkpoint trajectory.
4. Khi Set PPO baseline ổn định mới thử Recurrent Set PPO.
5. Sorted-CNN chỉ thử sau khi định nghĩa thứ tự UE có ý nghĩa và ổn định.

## v0.8.2 correction — 16 features, unchanged set-encoder architecture

The intended experiment keeps the expanded 16-feature observation but does not replace the existing encoder architecture. Every UE is still processed by the same shared MLP with a 64-dimensional hidden representation, followed by global set pooling. PPO then scores all 1,200 UEs, selects the Top-64, and predicts PRB demand. No candidate rule or safety rule selects UEs for the policy.

## v0.8.4 — Controlled reward discovery

The v0.8.3 rule-free full-control run learned zero starvation and acceptable tail delay, but Jain fairness remained far below the official target. The next phase therefore freezes the 16-feature observation, shared Set Encoder, PPO architecture, full 1,200-UE action scope, and optimizer settings while varying only reward structure.

Two initial rounds are registered:

- `round_01_component_screen`: each normalized positive reward component is trained alone;
- `round_02_throughput_jain_sweep`: keep the environment and PPO fixed, then sweep four interior throughput/Jain coefficient pairs; reuse the two single-component endpoints from Round 01.

Lagrangian training penalties are disabled in these attribution rounds so the observed behaviour can be traced to the declared reward case. Constraint KPIs remain active for validation and reporting.

Every completed run is appended to a reward-weight dataset and checked for Pareto dominance. Later weight sweeps and multi-seed confirmations will use the same dataset schema.


## v0.8.5 — Round 02 reward-weight sweep

- Move experiment analyses to `docs/analysis/`; reserve `docs/reports/` for later formal reports.
- Define P99 wait in plain language in generated HTML.
- Sweep only throughput and Jain fairness before introducing any safety penalty.
- Store all, safety-filtered, and strict-constraint Pareto fronts separately.

## v0.8.6 — Diagnose train/inference fairness gap before further reward tuning

Round 02 showed that the 25/75 throughput–Jain configuration could have high
fairness during stochastic PPO rollouts but severe starvation under deterministic
validation. The next experiment therefore keeps reward, environment, observations,
and architecture fixed and measures whether Beta sampling is breaking near-ties
that deterministic Top-K resolves by repeatedly selecting a small UE subset.

No dataset or Pareto workflow is part of the current phase. Those remain a possible
future study after the scheduler and reward have been optimized and understood.

## v0.8.7 — Exploration alignment after Round 03

Round 03 showed that the deterministic 25/75 policy selected only about 374 unique
UEs in a typical 64-slot window and retained about 52 of the previous slot's 64
UEs. The 37.5/62.5 policy covered almost all 1,200 UEs in the same window. Exact
Top-K ties were not the main failure; the 25/75 mean priority distribution was too
flat, while stochastic Beta noise was roughly sixty times larger than its mean
priority variation.

Before changing reward again, Round 04 fixes the reward at 25% throughput and 75%
Jain fairness and changes only the Beta concentration schedule. The goal is to
measure whether gradually reducing exploration makes deterministic and stochastic
performance converge. Dataset and Pareto work remain explicitly out of scope.


## v0.8.8 — Incremental reward exploration

- Current goal: understand the environment and the effect of each reward component.
- Add one component at a time, begin with equal coefficients, then tune only a small number of weights after explaining the observed KPI changes.
- Do not optimize the failed 25/75 case in isolation.
- Do not generate a reward-weight dataset or Pareto study in the current phase.
- Round 04 adds `service` to the throughput–Jain base using equal one-third coefficients.
