# CSI Reporting Design — v0.12.0

## Mục tiêu

v0.12.0 thêm lớp **CSI reporting** giữa true channel và scheduler. Round này không đổi reward, PPO, action space, traffic hay HARQ. Toàn bộ case dùng **Slow Dynamic CQI** của Round 10.

Luồng mới:

```text
True CQI(t)
    │
    ├──► PHY abstraction / attempted bits
    │
    └──► CSI measurement ── period ── delay ── error ──► Reported CQI(t)
                                                        │
                                                        ▼
                                                PPO observation
```

Điểm quan trọng: `cqi` trong environment tiếp tục là **true CQI**. PPO không còn bắt buộc nhìn trực tiếp giá trị này; hai observation feature `CQI` và `CQI_RANK` dùng `reported_cqi`.

## Biến mới

| Biến | Giá trị mặc định | Ý nghĩa |
|---|---:|---|
| `csi_report_mode` | `perfect` | `perfect` = scheduler nhìn true CQI hiện tại; `periodic` = qua reporting pipeline. |
| `csi_report_period_slots` | 1 | Chu kỳ tạo CSI measurement. |
| `csi_report_delay_slots` | 0 | Số slot từ measurement đến lúc report tới scheduler. |
| `csi_report_error_std` | 0.0 | Độ lệch chuẩn Gaussian, tính theo CQI-index trước quantization/clipping 1..15. |

CSI measurement dùng RNG stream riêng, nên thêm measurement noise không làm thay đổi true CQI trajectory hoặc HARQ RNG chỉ vì tiêu thụ thêm random numbers.

## Metric mới

- `mean_reported_cqi`
- `mean_csi_abs_error`
- `max_p95_csi_abs_error`
- `mean_csi_stale_fraction`
- `mean_csi_report_age_slots`
- `mean_csi_report_generated_rate`
- `mean_csi_report_delivered_rate`

## Round 11 — bốn case có kiểm soát

1. **Perfect CSI baseline**: report tức thời, không lỗi.
2. **Periodic only**: period=4, delay=0, error=0.
3. **Periodic + delay**: period=4, delay=2, error=0.
4. **Periodic + delay + noise**: period=4, delay=2, error std=0.75 CQI-index.

Bốn case dùng cùng seed, Slow Dynamic CQI, T–J–S equal-third và cùng PPO. Nhờ vậy ta tách lần lượt tác động của periodicity, delay và measurement error.

## Vì sao chưa đổi sang RPPO ngay?

CSI delay làm bài toán có tính **partial observability theo thời gian**: reported CQI hiện tại có thể cũ, trong khi true channel tiếp tục tiến hóa. Đây là lý do RPPO/recurrent policy trở thành một ứng viên hợp lý.

Tuy nhiên Round 11 vẫn giữ PPO feed-forward để đo **độ khó mà CSI reporting tự nó tạo ra**. Nếu đổi RPPO cùng lúc, ta sẽ không biết cải thiện đến từ memory hay từ thay đổi environment.

## Những gì vẫn chưa làm

- Chưa có MCS action.
- BLER vẫn cố định 10%, chưa phụ thuộc true CQI/MCS.
- Chưa có explicit CSI-RS/PUCCH/PUSCH resource overhead.
- Chưa có RI/PMI hoặc transmission-layer selection.
- Chưa có mobility-driven CQI.

Bước realism kế tiếp dự kiến là **Reported CQI → MCS decision; True CQI + MCS → BLER**, sau đó mới mở checkpoint tối ưu policy/reward/architecture.
