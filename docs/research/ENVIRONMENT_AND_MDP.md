# ScaleMAC-RL — Môi trường, MDP và biến mạng hiện tại

> **Snapshot hành vi:** v0.12.0, sau Dynamic CQI và CSI reporting.  
> **Mục đích:** mô tả đúng môi trường đang dùng để train/evaluate nhánh PPO full-control, phân biệt rõ biến nào đang cố định, biến nào có thể cấu hình, biến nào thay đổi theo slot và phần nào của 5G NR vẫn chưa được mô phỏng.

## 0. Cách đọc tài liệu này

Tài liệu dùng các nhãn sau:

| Nhãn | Ý nghĩa |
|---|---|
| **LOCKED** | Giá trị đang được khóa trong protocol nghiên cứu hiện tại; không nên đổi nếu đang muốn so sánh công bằng với các round cũ. |
| **FLEX-PLAN** | Có thể thay từ plan/CLI hiện tại mà không sửa logic môi trường. |
| **FLEX-CONFIG** | Có field trong `ScaleMacConfig`, nhưng runner nghiên cứu hiện tại chưa expose trực tiếp hoặc đang hard-code giá trị khác. |
| **DYNAMIC** | State thay đổi trong lúc episode chạy. |
| **DERIVED** | Không cấu hình trực tiếp; được tính từ state/config khác. |
| **META-ONLY** | Có lưu trong simulator nhưng hiện chưa ảnh hưởng tới channel/reward/transition. |
| **NOT-MODELED** | Chưa có trong simulator hiện tại. |

**Source of truth** cho snapshot này là:

- `scalemac_rl/config.py`
- `scalemac_rl/env.py`
- `scalemac_rl/projector.py`
- `scalemac_rl/scripts/train_ppo.py`
- `scalemac_rl/scripts/run_reward_study.py`
- `configs/reproducibility/round_09_tjs_repeatability.json`

> **Lưu ý quan trọng:** `configs/mvp.json` vẫn chứa cấu hình hybrid/reward cũ để tương thích lịch sử. Nhánh nghiên cứu hiện tại không lấy nó làm protocol chính. Protocol active gần nhất là command được tạo bởi `run_reward_study.py` cho Round 09.

---

## 1. Môi trường hiện tại đang mô phỏng cái gì?

ScaleMAC-RL hiện là một **fast training surrogate** cho bài toán downlink scheduling ở một cell, không phải full 3GPP PHY/MAC simulator.

Mental model của một slot:

```text
Static UE profile
(CQI, demand, speed metadata)
          │
          ▼
Per-UE state ──► observation 1200 × 16
                       │
                       ▼
                 PPO full-control
          priority + PRB-demand / UE
                       │
                       ▼
              validity projector
          Top-64 + exact 273 PRBs
                       │
                       ▼
           CQI spectral efficiency
                       │
                       ▼
        Bernoulli HARQ success/fail
                       │
                       ▼
 delivered bits + wait + EWMA + HARQ
                       │
                       ▼
         T + Jain + Service reward
                       │
                       └──── next slot
```

Trong protocol hiện tại:

- một gNB, một cell;
- downlink only;
- 1.200 UE active;
- 273 PRB mỗi scheduling decision;
- PPO chấm toàn bộ 1.200 UE;
- projector lấy tối đa 64 UE;
- không candidate filtering;
- không forced HARQ;
- traffic full-buffer;
- true CQI heterogeneous và có thể **temporally correlated**;
- scheduler có thể nhìn `reported_cqi` bị periodic/delay/noise thay vì true CQI;
- reward active đã chốt là Throughput + Jain + Service.

---

## 2. Bảng nhanh các thông số quan trọng nhất

| Biến | Giá trị active hiện tại | Trạng thái | Ý nghĩa thực tế trong simulator |
|---|---:|---|---|
| Số gNB | 1 | **LOCKED** | Một scheduler cho một cell. |
| Số cell | 1 | **LOCKED** | Chưa có inter-cell interference/handover. |
| Hướng truyền | Downlink | **LOCKED** | Không mô phỏng UL scheduling. |
| `num_ues` | 1200 | **LOCKED / FLEX-PLAN ở trainer chung** | Toàn bộ 1.200 UE đều active/eligible. |
| `num_prbs` | 273 | **LOCKED** trong trainer hiện tại | Tổng PRB được cấp chính xác mỗi slot. |
| `max_selected_ues` | 64 | **LOCKED** | Top-K tối đa 64 UE/slot. |
| Scheduling decisions | 1 / slot | **LOCKED** | Mỗi `env.step()` là một scheduling decision. |
| `episode_slots` | 5000 | **FLEX-PLAN** | Round 09 train/validation dùng 5.000 slot/episode. |
| Traffic | Full-buffer | **LOCKED** | Queue được refill ngay sau mỗi slot. |
| `full_buffer_base_bytes` | 1,000,000 B | **FLEX-CONFIG** | Queue target cơ sở trước khi nhân demand factor. |
| `packet_size_bytes` | 1500 B | **FLEX-CONFIG nhưng hiện chưa operational** | Có trong config nhưng hiện không dùng trong công thức delivery/queue transition. |
| True CQI | 1–15 | **DYNAMIC / FLEX-PLAN** | Static hoặc correlated; Slow Dynamic CQI là realism profile chính. |
| CQI mix | 30% low / 40% medium / 30% high | **FLEX-CONFIG** | 360 / 480 / 360 UE khi `num_ues=1200`. |
| Demand mix | 40% low / 40% medium / 20% high | **FLEX-CONFIG** | 480 / 480 / 240 UE. |
| Demand factors | 0.5 / 1.0 / 2.0 | **LOCKED trong code** | Scale queue target và throughput normalization. |
| HARQ | bật | **FLEX-CONFIG** | Mỗi selected UE có success/fail ngẫu nhiên. |
| `target_bler` | 0.10 | **FLEX-CONFIG**, trainer active chưa expose | Success probability hiện là 0.90 cho mọi selected UE. |
| `max_harq_retransmissions` | 3 | **FLEX-CONFIG** | Sau lần fail thứ 4 thì pending bị drop/reset. |
| `starvation_threshold_slots` | 64 | **FLEX-PLAN** | Starved = không có successful delivery trong ≥64 slot. |
| P99 target | 50 slot | **LOCKED protocol** | KPI/constraint reference hiện tại. |
| Max-wait target | 60 slot | **LOCKED protocol** | KPI/constraint reference hiện tại. |
| `ewma_alpha` | 0.02 | **FLEX-CONFIG** | Memory của short-term throughput. |
| CSI report | perfect / periodic | **FLEX-PLAN** | Scheduler có thể nhận CQI theo period/delay/error. |
| `speed_mps` | {0, 1.5, 5, 15, 25} | **META-ONLY** | Hiện chưa điều khiển CQI. |
| Reward active | 1/3 T + 1/3 Jain + 1/3 Service | **LOCKED** | Không có reward thứ tư trong protocol hiện tại. |

---

## 3. CQI — biến cần hiểu rõ nhất trước Dynamic CQI

### 3.1 CQI hiện được tạo như thế nào?

Mỗi UE có một CQI nguyên từ **1 đến 15**. Population được chia thành ba nhóm:

| Nhóm | CQI range | Tỷ lệ | Số UE với 1200 UE |
|---|---:|---:|---:|
| Low CQI | 1–5 | 30% | 360 |
| Medium CQI | 6–10 | 40% | 480 |
| High CQI | 11–15 | 30% | 360 |

Trong từng nhóm, CQI được sample đều bằng integer RNG rồi shuffle toàn population.

### 3.2 CQI có thay đổi theo thời gian không?

**Có thể.** Từ v0.11, `cqi_mode=correlated` dùng một quá trình mean-reverting quanh heterogeneous CQI anchor. Slow Dynamic CQI (`rho=0.97`, `sigma=0.35`, max ΔCQI=1) là realism profile chính; static vẫn giữ làm regression baseline và faster Dynamic CQI là stress test.

`speed_mps` vẫn chỉ là metadata; mobility chưa trực tiếp sinh path loss/SINR/CQI.

### 3.3 CQI ảnh hưởng tới transmission hiện tại ra sao?

CQI index chọn một spectral-efficiency abstraction hard-coded. Sau đó simulator tính:

```text
bits_per_PRB = 12 subcarriers × 14 OFDM symbols × spectral_efficiency × 0.86
```

`0.86` là overhead abstraction 14%.

| CQI | Spectral efficiency | Attempted bits / PRB / slot | Expected delivered bits / PRB với BLER=0.10 |
|---:|---:|---:|---:|
| 1 | 0.1523 | 22.00 | 19.80 |
| 2 | 0.2344 | 33.87 | 30.48 |
| 3 | 0.3770 | 54.47 | 49.02 |
| 4 | 0.6016 | 86.92 | 78.23 |
| 5 | 0.8770 | 126.71 | 114.04 |
| 6 | 1.1758 | 169.88 | 152.89 |
| 7 | 1.4766 | 213.34 | 192.01 |
| 8 | 1.9141 | 276.55 | 248.89 |
| 9 | 2.4063 | 347.66 | 312.90 |
| 10 | 2.7305 | 394.50 | 355.05 |
| 11 | 3.3223 | 480.01 | 432.01 |
| 12 | 3.9023 | 563.80 | 507.42 |
| 13 | 4.5234 | 653.54 | 588.19 |
| 14 | 5.1152 | 739.04 | 665.14 |
| 15 | 5.5547 | 802.54 | 722.29 |

**Điều này có nghĩa:** CQI cao làm một PRB mang được nhiều attempted bits hơn.

### 3.4 Nhưng BLER hiện có phụ thuộc CQI không?

**Không.** Đây là simplification lớn.

Nếu HARQ bật:

```text
success = random() >= target_bler
```

với `target_bler=0.10`, nên mọi selected UE đều có xác suất success 90%, bất kể CQI 1 hay CQI 15.

Do đó hiện tại CQI tác động trực tiếp lên **capacity/attempted bits**, nhưng không tác động lên **reliability probability**.

Đây chưa phải chuỗi thực tế kiểu:

```text
SINR → CQI → MCS → TBS → BLER
```

### 3.5 CQI là FLEX ở mức nào?

| Thành phần CQI | Hiện tại | Muốn đổi cần gì? |
|---|---|---|
| Tỷ lệ low/medium/high | **FLEX-CONFIG** | Sửa `ScaleMacConfig` hoặc expose qua runner. |
| CQI range 1–5 / 6–10 / 11–15 | Hard-coded | Sửa code sampling. |
| Efficiency table 15 mức | Hard-coded | Sửa code `_CQI_EFFICIENCY`. |
| True CQI của từng UE | Static hoặc correlated | Đã FLEX bằng `cqi_mode` và các tham số rho/sigma/update/max-delta. |
| CQI correlation theo thời gian | Đã có | `cqi_temporal_correlation` + innovation std. |
| CQI reporting period/delay/error | Đã có | `csi_report_*`; scheduler thấy `reported_cqi`. |
| Mobility → CQI | Chưa có | Cần position/channel model. |

---

## 4. Traffic, demand và queue

### 4.1 Demand profile

Mỗi UE được gán một `demand_factor` static:

| Demand class | Factor | Tỷ lệ | Số UE / 1200 |
|---|---:|---:|---:|
| Low | 0.5 | 40% | 480 |
| Medium | 1.0 | 40% | 480 |
| High | 2.0 | 20% | 240 |

Queue target được tạo bởi:

```text
queue_target_bytes = 1,000,000 × demand_factor
```

nên ba mức queue target là:

```text
0.5 MB, 1.0 MB, 2.0 MB
```

### 4.2 Full-buffer thực sự hoạt động thế nào?

Sau một transmission thành công, simulator có trừ delivered bytes khỏi queue, nhưng ngay lập tức:

```text
queue_bytes = queue_target_bytes.copy()
```

Nghĩa là queue được refill hoàn toàn mỗi slot. Vì vậy:

- UE luôn có data để gửi;
- không có packet arrival process;
- không có empty queue;
- không có burst traffic;
- queue feature hiện gần như static và chủ yếu phản ánh demand class.

### 4.3 `packet_size_bytes = 1500` có tác dụng không?

Hiện tại **chưa có tác dụng operational trong transition**. Field này tồn tại trong config nhưng `_execute_grant()` tính trực tiếp delivered bits từ PRB × efficiency; không cắt thành packet 1500 B, không có packet completion/deadline/drop theo packet.

Do đó không nên claim simulator hiện tại là packet-level traffic simulator.

---

## 5. HARQ hiện tại mô phỏng đến đâu?

Mỗi UE có:

- `harq_pending`;
- `harq_retx_count`;
- `last_success`.

Nếu selected UE fail:

1. `harq_retx_count += 1`;
2. nếu count ≤ 3 → `harq_pending=True`;
3. nếu count > 3 → pending bị clear, counter reset và tăng `harq_drops`.

Nếu success:

- pending clear;
- retx count reset;
- `time_since_service=0`.

Trong nhánh PPO-only hiện tại:

- **không forced HARQ retransmission**;
- HARQ pending chỉ là một feature để PPO nhìn thấy;
- policy có quyền bỏ qua pending UE;
- projector không chèn HARQ UE vào Top-K.

### HARQ còn thiếu so với hệ thống thật

Hiện chưa mô phỏng:

- nhiều HARQ process song song;
- feedback timing/delay;
- ACK/NACK timing theo numerology;
- redundancy version;
- soft combining;
- giữ nguyên TB để retransmit;
- CQI/MCS-dependent BLER.

Nói chính xác hơn, đây là **HARQ state abstraction**, chưa phải full HARQ protocol model.

---

## 6. State nội bộ của environment

Mỗi UE có các state chính:

| State | Loại | Ý nghĩa |
|---|---|---|
| `cqi` | Static profile | Channel-quality class hiện tại, 1–15. |
| `demand_factor` | Static profile | 0.5 / 1 / 2. |
| `speed_mps` | META-ONLY | Mobility metadata, chưa tác động channel. |
| `queue_bytes` | Full-buffer state | Luôn refill về target mỗi slot. |
| `queue_target_bytes` | Static derived | `base_bytes × demand_factor`. |
| `ewma_throughput_bits` | **DYNAMIC** | Short-term delivered throughput memory. |
| `cumulative_delivered_bits` | **DYNAMIC** | Tổng delivered bits từ đầu episode. |
| `time_since_service` | **DYNAMIC** | Slot kể từ successful delivery gần nhất. |
| `time_since_schedule` | **DYNAMIC** | Slot kể từ lần được schedule gần nhất. |
| `harq_pending` | **DYNAMIC** | Có transmission fail chưa được success lại. |
| `harq_retx_count` | **DYNAMIC** | Số fail liên tiếp trong HARQ abstraction. |
| `eligible` | Hiện static | Hiện tất cả UE luôn `True`. |
| `last_grant` | **DYNAMIC** | PRB grant ở slot trước. |
| `last_success` | **DYNAMIC** | Slot trước có successful delivery hay không. |

Điểm cần nhớ: `time_since_service` và `time_since_schedule` **khác nhau**. UE được schedule nhưng transmission fail thì scheduling wait reset, còn service wait vẫn tăng.

---

## 7. Observation mà PPO nhìn thấy — 1200 × 16

PPO không nhận raw state nguyên xi. Mỗi UE được chuyển thành vector 16 feature:

| # | Feature | Công thức / range hiện tại | Ý nghĩa |
|---:|---|---|---|
| 0 | `cqi` | `reported_CQI / 15` | Scheduler-visible channel quality; có thể stale/noisy. |
| 1 | `queue` | `queue_bytes / max_queue`, clip [0,1] | Backlog relative; trong full-buffer gần như static. |
| 2 | `demand` | `demand_factor / 2` | Demand class normalized. |
| 3 | `ewma_throughput` | EWMA / max EWMA trong cell | Relative recent throughput. |
| 4 | `time_since_service` | wait / 64, clip [0,2] | Successful-delivery waiting age. |
| 5 | `harq_pending` | 0/1 | Pending HARQ abstraction. |
| 6 | `harq_retx_count` | count / 3, clip [0,1] | Retransmission pressure. |
| 7 | `eligible` | 0/1 | Hiện luôn 1 cho mọi UE. |
| 8 | `throughput_deficit` | deficit so với cell mean sau demand normalization | UE đang thiếu throughput tương đối. |
| 9 | `service_deficit` | wait / expected service cycle, clip [0,2] | UE chờ lâu so với cycle hợp lý. |
| 10 | `cqi_rank` | tie-aware percentile [0,1] | Rank của **reported CQI**, không nhất thiết true CQI. |
| 11 | `throughput_deficit_rank` | `1 - percentile(normalized throughput)` | Cao = thuộc nhóm underserved. |
| 12 | `wait_rank` | percentile wait | Cao = chờ lâu hơn nhiều UE khác. |
| 13 | `throughput_to_mean` | normalized throughput / cell mean / 2 | Throughput relative-to-cell. |
| 14 | `wait_to_deadline` | wait / 50, clip [0,2] | Khoảng cách tới P99 target. |
| 15 | `last_prb_share` | last grant / 273 | PRB share ở slot trước. |

Với 1.200 UE và Top-K 64, expected service cycle thô là:

```text
ceil(1200 / 64) = 19 slots
```

nên `service_deficit` bắt đầu vượt 1 khi UE chờ khoảng 19 slot.

---

## 8. Action của PPO và projector

Actor xuất hai số liên tục cho **mỗi UE**:

```text
priority_score_i ∈ [0,1]
prb_demand_score_i ∈ [0,1]
```

Action tensor có shape:

```text
1200 × 2
```

Trong `ppo_only`:

1. tất cả 1.200 UE đều là candidate;
2. sort theo `priority_score`;
3. lấy Top-64;
4. mỗi selected UE nhận tối thiểu 1 PRB;
5. còn `273 - 64 = 209` PRB được chia theo `prb_demand_score + 1e-3` bằng largest-remainder allocation;
6. tổng grant luôn đúng 273 PRB.

Projector hiện chỉ đảm bảo feasibility. Nó **không**:

- ép HARQ UE;
- ép UE gần starvation;
- dùng CQI để sửa priority;
- lọc candidate;
- thêm rule scheduling khác.

Do đó selection behavior trong nhánh này thực sự đến từ policy.

---

## 9. PHY abstraction: từ PRB tới delivered bits

Với UE selected:

```text
attempted_bits_i
= granted_PRBs_i × bits_per_PRB(CQI_i)
```

Sau đó:

```text
success_i ~ Bernoulli(0.90)
```

Nếu success:

```text
delivered_bits_i = attempted_bits_i
```

Nếu fail:

```text
delivered_bits_i = 0
HARQ state updated
```

### Đơn vị goodput

Environment log `cell_goodput_bits` theo **bits per simulator slot**.

Hiện chưa có slot duration/numerology được gắn vào model, vì vậy **không nên tự đổi trực tiếp sang Mbps** nếu chưa bổ sung một định nghĩa thời gian vật lý cho slot.

---

## 10. Reward active hiện tại — chính xác là gì?

Sau Round 09, reward study protocol active khóa toàn bộ reward phụ/penalty về 0 và chỉ dùng ba positive component:

```text
R = 1/3 × ThroughputScore
  + 1/3 × FairnessScore
  + 1/3 × ServiceScore
```

### 10.1 Throughput score

```text
throughput_score = cell_goodput / slot_oracle_expected_goodput
```

Oracle chỉ dùng để normalization. Nó:

- chọn các CQI mạnh nhất theo Top-K contract;
- cho mỗi selected UE 1 PRB;
- dồn PRB còn lại vào UE mạnh nhất;
- nhân success probability 0.90.

### 10.2 Fairness score

Reward fairness là mix:

```text
fairness_score
= 0.60 × cumulative Jain
+ 0.40 × short-term EWMA Jain
```

Nó không chỉ là final Jain KPI.

### 10.3 Service score

```text
delay_penalty
= 0.40 × starvation_rate
+ 0.35 × mean_wait_score
+ 0.25 × near_deadline_rate

service_score = 1 - delay_penalty
```

Trong protocol hiện tại:

- starvation threshold = 64 slot;
- near-deadline bắt đầu ở `0.60 × 50 = 30 slot`;
- `mean_wait_score = mean(clip(wait / 64, 0, 1))`.

### 10.4 Những reward term hiện tồn tại trong code nhưng **không active**

Round 09 set coefficient bằng 0 cho:

- deficit-service;
- PF utility;
- low-throughput;
- urgency-service;
- fairness progress;
- PF progress;
- starvation penalty;
- P99 deadline-risk penalty;
- max-wait penalty;
- population-wait penalty;
- Lagrange constraint penalties.

Các metric vẫn có thể được tính/log để phân tích, nhưng không tạo gradient nếu coefficient bằng 0.

---

## 11. Delay, starvation và các KPI cần phân biệt

### Successful-delivery wait

```text
time_since_service
```

Chỉ reset khi transmission **thành công**.

### Scheduling wait

```text
time_since_schedule
```

Reset ngay khi UE được selected, kể cả transmission fail.

### Starvation

```text
starved_i = time_since_service_i >= 64
```

Đây là delivery starvation, không phải scheduling starvation.

### P95/P99/max wait

Tính trực tiếp trên population `time_since_service` ở từng slot.

Protocol reference hiện tại:

```text
starvation = 0
P99 wait <= 50 slot
max wait <= 60 slot
```

Những threshold này là mục tiêu đánh giá/reference. Trong Round 09, penalty weight tương ứng đã bằng 0 nên chúng không phải hidden rule ép policy.

---

## 12. Những thứ hiện có tên trong code nhưng dễ hiểu nhầm

### `speed_mps`

Có sample từ:

```text
0, 1.5, 5, 15, 25 m/s
```

nhưng **không ảnh hưởng CQI, BLER, position hay handover**. Hiện chỉ là metadata.

### `packet_size_bytes = 1500`

Có trong config nhưng chưa dùng để phân packet trong transition.

### `eligible`

Có trong state/observation/projector, nhưng environment hiện reset toàn bộ UE thành eligible và không có logic làm UE inactive. Vì thế nhánh active hiện có 1.200/1.200 UE eligible.

### `mvp.json`

Là cấu hình lịch sử/general MVP, không phải active Round-09 protocol. Đừng dùng các reward weight/hybrid safety setting trong file đó để mô tả thí nghiệm hiện tại.

---

## 13. Các biến mạng thật mà environment **chưa có**

| Nhóm | Chưa mô phỏng |
|---|---|
| Radio carrier | carrier frequency, bandwidth thực, SCS/numerology, slot duration, TDD pattern |
| Propagation | distance, path loss, shadowing, fast fading, Doppler |
| Link quality | SINR, RSRP/RSRQ, interference, measurement noise |
| CQI process | time correlation, reporting periodicity, delay, quantization error |
| Link adaptation | MCS selection thật, TBS table, coding rate, CQI↔MCS↔BLER mapping |
| MIMO | layers, rank, PMI, precoder, beamforming |
| Interference | multi-cell, inter-cell interference, neighboring gNB load |
| Mobility | position trajectory, cell-edge movement, handover |
| Traffic | Poisson/bursty arrivals, finite queues, packet completion/drop |
| QoS | 5QI, GBR/non-GBR, multiple QoS flows, packet delay budget, priority level |
| HARQ chi tiết | process IDs, timing, RV, combining, feedback delay |
| Upper layers | RLC/PDCP behavior, segmentation/reassembly |

Điều này không làm environment “sai”; nó xác định đúng **scope của surrogate hiện tại**.

---

## 14. Biến nào có thể đổi ngay mà không viết Dynamic CQI?

### Có thể đổi qua plan/CLI hiện tại

- episode length;
- training steps/rollout;
- seed/profile seed;
- starvation threshold;
- reward weights;
- PPO hyperparameter;
- validation setup.

### Có field config nhưng current research runner chưa expose trực tiếp

- CQI population fractions;
- demand fractions;
- BLER;
- max HARQ retransmissions;
- EWMA alpha;
- full-buffer base bytes;
- packet size;
- HARQ enabled/disabled.

### Hiện hard-coded trong active trainer/environment path

- 273 PRB;
- Top-K 64;
- CQI group ranges;
- CQI efficiency table;
- demand factors 0.5/1/2;
- bits/PRB abstraction 12 × 14 × 0.86;
- CQI static trong slot transition.

---

## 15. Baseline cần khóa khi bắt đầu Dynamic CQI

Khi chuyển sang Dynamic CQI, để biết tác động đến từ channel chứ không phải thay đổi khác, nên giữ nguyên:

```text
1 gNB / 1 cell / downlink
1200 UE
273 PRB
Top-K = 64
PPO full-control
16-feature observation trước khi quyết định có thêm feature mới hay không
2-action/UE: priority + PRB demand
Full-buffer traffic
HARQ abstraction hiện tại
Reward = T + Jain + Service, equal-third
PPO hyperparameter đã dùng ở Round 09
```

Và chỉ thay một cơ chế:

```text
CQI_i = static
```

thành một process có thời gian:

```text
CQI_i(t+1) phụ thuộc CQI_i(t) + channel dynamics
```

**Phần này mới là kế hoạch, chưa được implement trong snapshot tài liệu này.**

---

## 16. Tóm tắt một câu cho từng biến quan trọng

- **CQI:** hiện là static per-UE quality index; ảnh hưởng bits/PRB nhưng chưa ảnh hưởng BLER.
- **Demand:** static UE class dùng để scale queue target và fairness normalization.
- **Queue:** full-buffer, refill mỗi slot; chưa phải traffic arrival queue thật.
- **PRB:** 273 resource units/slot; luôn được dùng hết khi có UE eligible.
- **Top-K:** PPO chỉ có thể serve tối đa 64 UE/slot.
- **Priority action:** quyết định UE nào vào Top-64.
- **Demand action:** quyết định chia 209 PRB còn lại giữa Top-64.
- **HARQ pending:** báo cho PPO biết UE đã fail; projector không bắt buộc retransmit.
- **EWMA throughput:** memory ngắn hạn của throughput, alpha=0.02.
- **Service wait:** số slot từ successful delivery gần nhất; là biến chính cho starvation/delay.
- **Starvation:** wait ≥64 slot.
- **Reward:** chỉ T + Jain + Service ở protocol đã chốt.
- **Speed:** metadata, chưa làm channel thay đổi.

---

## 17. Kết luận về mức độ “thật” hiện tại

Environment hiện tại đủ để nghiên cứu **scheduler learning dynamics** và trade-off giữa throughput–fairness–service trong một population lớn 1.200 UE, vì PPO thực sự điều khiển selection + PRB allocation và không được heuristic cứu.

Nhưng phần radio/channel còn là abstraction mạnh:

```text
static CQI + fixed efficiency + constant BLER
```

Do đó bước hợp lý tiếp theo không phải thêm reward nữa, mà là tăng realism có kiểm soát bắt đầu từ **Dynamic CQI**, sau đó mới tới CQI→MCS→BLER, traffic arrivals, HARQ timing, QoS và mobility.


## 12. True CQI và Reported CSI từ v0.12.0

Environment tách hai khái niệm:

- `cqi`: true instantaneous CQI, dùng cho capacity/attempted bits và oracle;
- `reported_cqi`: CQI mà scheduler nhìn thấy trong observation.

Ở `csi_report_mode=perfect`, hai giá trị trùng nhau mỗi slot. Ở `periodic`, measurement được tạo theo `csi_report_period_slots`, tới scheduler sau `csi_report_delay_slots`, và có thể thêm `csi_report_error_std` trước khi quantize/clip về 1..15.

Điều này làm environment có partial observability theo thời gian mà **không tăng observation width**: feature CQI và CQI-rank cũ chỉ chuyển sang dùng reported CSI. Đây là nền để sau này đánh giá PPO feed-forward so với RPPO.
