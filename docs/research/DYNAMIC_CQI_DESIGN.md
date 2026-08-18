# Dynamic CQI Design — v0.11.0

## Mục tiêu

v0.11.0 là bước tăng realism đầu tiên sau khi reward được khóa ở:

`R = 1/3 Throughput + 1/3 Jain + 1/3 Service`.

Round này **chỉ thay channel process**. PPO, 16 observation features, 2 action/UE, Top-K=64, 273 PRB, full-buffer traffic và HARQ abstraction giữ nguyên.

## CQI mới hoạt động thế nào?

Mỗi UE vẫn bắt đầu với heterogeneous CQI anchor 1..15 như baseline cũ. Với `cqi_mode=correlated`, simulator giữ một latent CQI liên tục và cập nhật:

`x(t+1) = anchor + rho * (x(t) - anchor) + epsilon`, với `epsilon ~ N(0, sigma^2)`.

Sau đó latent CQI được clip về [1,15], quantize thành integer CQI và rate-limit theo `cqi_max_delta_per_update`.

Ý nghĩa:

- `rho` cao → channel thay đổi chậm, có memory lớn;
- `sigma` cao → channel biến động mạnh hơn;
- `update_interval_slots` → số slot giữa hai lần channel update;
- `max_delta_per_update` → giới hạn bước nhảy CQI trong một update.

Channel innovation dùng RNG stream riêng, nên bật Dynamic CQI không làm đổi chuỗi HARQ randomness chỉ vì tiêu thụ thêm random numbers.

## Ba case Round 10

| Case | Mode | rho | sigma | Max ΔCQI/update | Ý nghĩa |
|---|---|---:|---:|---:|---|
| Static baseline | static | — | 0 | 0 | Baseline trước v0.11 |
| Slow Dynamic CQI | correlated | 0.97 | 0.35 | 1 | Channel chậm, tương quan mạnh |
| Faster Dynamic CQI | correlated | 0.80 | 0.90 | 3 | Channel biến động nhanh hơn |

Cả ba dùng cùng seed/profile, cùng PPO và cùng T–J–S reward để cô lập tác động của channel dynamics.

## Metric mới được log

- `mean_cqi`
- `mean_cqi_std`
- `mean_cqi_abs_change_per_slot`
- `mean_cqi_changed_fraction`

Các metric này được ghi vào training/validation CSV và báo cáo Round 10 cùng với goodput, Jain, starvation, P99 và max-wait.

## Những gì v0.11.0 CHƯA làm

- BLER vẫn cố định 10% và chưa phụ thuộc CQI.
- Chưa có CQI→MCS link adaptation thật.
- Chưa có CSI reporting delay/error.
- `speed_mps` chưa điều khiển CQI.
- Chưa có MIMO/RI/PMI.
- Agent chưa chọn số transmission layers.

Hướng sau này của Joint Link Adaptation + Scheduler là để agent quyết định đồng thời **UE + PRB + MCS + số spatial/transmission layers** khi MIMO được đưa vào system model.

## Câu hỏi nghiên cứu Round 10

> T–J–S PPO scheduler có giữ được goodput, fairness và service-delay khi CQI chuyển từ static sang temporally correlated channel hay không?
