# Round 08 — Xác nhận đa seed cho Urgency và Deficit

## Kết luận kỹ thuật hiện tại

Không ứng viên mới nào ổn định ở cả ba seed; nên quay lại T–J–S làm active reward.

## Mean ± std theo profile

| Profile | Stable | Goodput | Jain | Starvation | P99 | Max wait |
|---|---:|---:|---:|---:|---:|---:|
| Baseline T–J–S equal-third | 2/3 | 67,106 ± 49,951 | 0.2187 ± 0.1505 | 31.56% ± 54.66 điểm % | 1695.3 ± 2861.9 | 1696.0 ± 2861.3 |
| Ứng viên cân bằng: Urgency hold-Throughput | 1/3 | 38,150 ± 50,344 | 0.1549 ± 0.1853 | 63.11% ± 54.66 điểm % | 3346.7 ± 2863.7 | 3347.0 ± 2863.1 |
| Ứng viên delay: Deficit giữ nhóm T+S | 1/3 | 90,962 ± 75,396 | 0.0903 ± 0.0491 | 50.17% ± 47.59 điểm % | 3349.0 ± 2859.6 | 3349.0 ± 2859.6 |
