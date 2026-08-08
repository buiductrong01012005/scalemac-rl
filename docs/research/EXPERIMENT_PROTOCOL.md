# Protocol đánh giá và tái lập

## 1. Tách train, checkpoint selection và final evaluation

- Training seed không được dùng để claim robustness.
- Validation dùng để chọn checkpoint và cập nhật Lagrange controller.
- Final evaluation dùng checkpoint đã freeze và cùng scenario seed cho mọi scheduler.

## 2. Checkpoint phải giữ lại

```text
latest
best_reward
best_lowest_violation
best_feasible
best_tradeoff
milestone_100k
milestone_200k
milestone_300k
```

Không chỉ đánh giá một checkpoint. Reward cao nhất có thể khác trade-off tốt nhất.

## 3. Baseline

Mọi PPO phải so với:

- RR;
- PF;
- Max-CQI;
- Rule-only;
- PPO-only candidate-128 trước đây;
- full-control PPO hiện tại.

## 4. KPI báo cáo

Báo cáo cả mean và worst-case qua seed:

- goodput;
- Jain fairness;
- starvation rate;
- P99 wait;
- max wait;
- low-percentile throughput;
- inference P50/P95/P99/max;
- balanced score;
- worst KPI gap;
- Pareto dominated hay không.

## 5. Tối thiểu số seed

- Diagnostic: 1 seed, dùng để debug và quan sát learning curve.
- Decision gate: ít nhất 3 seed.
- Claim chính: 5 seed hoặc hơn, kèm từng seed riêng.

## 6. Provenance

Mỗi bảng evaluation phải giữ:

```text
evaluation_protocol_hash
scenario_hash
checkpoint_sha256
checkpoint_training_reward_version
observation feature count
candidate mode
scheduler mode
projector contract
```

## 7. Không kết luận từ reward đơn lẻ

Reward có thể thay đổi do curriculum target hoặc penalty. Báo cáo luôn:

```text
core reward
training/constrained reward
final-target reward
reward component decomposition
KPI thật
```
