# Môi trường và MDP của ScaleMAC-RL v0.8

## 1. State của simulator

State đầy đủ ở mỗi slot gồm, với từng UE:

- CQI và demand profile;
- queue/backlog;
- EWMA throughput và cumulative delivered bits;
- số slot từ lần truyền thành công gần nhất;
- số slot từ lần được scheduling gần nhất;
- HARQ pending và số retransmission;
- eligibility;
- grant ở slot trước.

Policy không nhất thiết nhìn thấy toàn bộ state thô. Observation là biểu diễn chuẩn hóa từ state.

## 2. Observation v2 — 16 feature/UE

| # | Feature | Ý nghĩa |
|---:|---|---|
| 0 | `cqi` | CQI chuẩn hóa 0–1 |
| 1 | `queue` | Queue chuẩn hóa; trong full-buffer ít thông tin |
| 2 | `demand` | Demand class chuẩn hóa |
| 3 | `ewma_throughput` | Throughput EWMA tương đối |
| 4 | `time_since_service` | Slot từ lần truyền thành công gần nhất |
| 5 | `harq_pending` | Có HARQ pending hay không |
| 6 | `harq_retx_count` | Số retransmission chuẩn hóa |
| 7 | `eligible` | UE có hợp lệ để cấp lịch không |
| 8 | `throughput_deficit` | Thiếu throughput so với cell mean sau chuẩn hóa demand |
| 9 | `service_deficit` | Wait so với chu kỳ phục vụ kỳ vọng |
| 10 | `cqi_rank` | Percentile CQI trong cell |
| 11 | `throughput_deficit_rank` | Rank UE bị thiếu throughput; cao là bị thiếu nhiều |
| 12 | `wait_rank` | Percentile wait; cao là chờ lâu hơn phần lớn UE |
| 13 | `throughput_to_mean` | Throughput demand-normalized so với cell mean |
| 14 | `wait_to_deadline` | Wait so với target P99 đang active |
| 15 | `last_prb_share` | Tỷ lệ PRB UE nhận ở slot trước |

Các feature rank dùng tie-aware percentile để tránh tạo thứ tự giả khi nhiều UE có giá trị bằng nhau.

## 3. Action

Actor xuất hai giá trị liên tục trong `[0,1]` cho mỗi UE:

```text
priority_score_i
prb_demand_score_i
```

Trong full-control PPO:

1. PPO chấm điểm toàn bộ 1.200 UE.
2. Projector lấy Top-64 theo priority.
3. Mỗi UE được chọn nhận tối thiểu 1 PRB.
4. PRB còn lại được chia theo `prb_demand_score`.
5. Tổng grant phải đúng 273 PRB.

Projector chỉ enforce feasibility; nó không tự thêm HARQ UE hoặc UE chờ lâu.

## 4. Reward v2

Positive reward được chuẩn hóa và tổng trọng số bằng 1. Profile `balanced`:

```text
0.40 * throughput_score
0.15 * fairness_score
0.10 * service_score
0.05 * deficit_service_score
0.15 * PF_utility_score
0.10 * low_throughput_score
0.05 * urgency_service_score
```

Shaping bổ sung:

```text
+ 0.02 * fairness_progress
+ 0.03 * PF_utility_progress
```

Penalty:

```text
- starvation penalty
- P99 tail-risk penalty
- max-wait risk penalty
- population-wide wait-pressure penalty
- Lagrange constraint penalties khi train
```

### Ý nghĩa các tín hiệu mới

- `PF_utility_score`: ưu tiên tăng throughput của UE có throughput thấp vì lợi ích log giảm dần.
- `low_throughput_score`: P10 throughput demand-normalized chia cho cell mean; phạt việc bỏ lại nhóm đáy.
- `urgency_service_score`: thưởng khi truyền thành công cho UE vừa thiếu throughput vừa gần deadline.
- `population_wait_risk`: tạo gradient trước khi P99 hoặc max wait vượt hẳn constraint.

## 5. KPI chính

Reward không dùng để claim kết quả. KPI đánh giá gồm:

- cell goodput;
- throughput score;
- Jain fairness;
- P10/PF-related score;
- starvation rate theo successful delivery;
- P95/P99 wait;
- maximum successful-delivery wait;
- scheduling-only wait;
- HARQ failures/drops;
- inference latency;
- balanced score, worst KPI gap và Pareto dominance.

## 6. Constraint chính thức

```text
starvation rate = 0
P99 successful-delivery wait <= 50 slot
max successful-delivery wait <= 60 slot
Jain fairness >= 0.60
```

Goodput là objective cần tối đa sau khi policy kiểm soát được các constraint trên.
