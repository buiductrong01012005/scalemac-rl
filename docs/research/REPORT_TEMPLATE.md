# Báo cáo thí nghiệm ScaleMAC-RL

## 1. Câu hỏi nghiên cứu

- Policy nào đang được kiểm tra?
- Thay đổi duy nhất so với vòng trước là gì?
- Giả thuyết kỳ vọng là gì?

## 2. Cấu hình

```text
version:
commit:
checkpoint:
seed/profile seed:
num UEs:
PRBs / Top-K:
observation schema:
reward profile:
PPO hyperparameters:
training steps:
validation protocol:
```

## 3. Learning trajectory

| Step | Core reward | Final reward | Goodput | Fairness | Starvation | P99 | Max wait |
|---:|---:|---:|---:|---:|---:|---:|---:|

## 4. Final comparison

| Scheduler | Goodput | Fairness | Starvation | P99 | Max wait | Balanced score | Pareto |
|---|---:|---:|---:|---:|---:|---:|---|

## 5. Reward decomposition

- throughput component:
- PF component:
- low-throughput component:
- urgency component:
- service/fairness components:
- tail/max/population wait penalties:
- Lagrange penalties:

## 6. Kết luận

- Giả thuyết được hỗ trợ hay bác bỏ?
- Policy học scheduling hay đang dựa vào một thành phần rule/filter?
- KPI nào là bottleneck?
- Có cần train thêm không, hay cần đổi input/reward/hyperparameter?

## 7. Bước tiếp theo

Chỉ đề xuất một thay đổi chính cho vòng sau.
