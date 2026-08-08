# Kế hoạch thí nghiệm PPO Full Control

## Mục tiêu

Tìm một policy không dùng rule và không candidate filter, có thể tự trade-off:

```text
goodput ↑
fairness ↑
P99/max wait ↓
starvation ↓
inference latency ↓
```

## Thí nghiệm A — Set PPO v2 trực tiếp 1.200 UE

Lệnh chính:

```powershell
python -m scalemac_rl.scripts.train_full_control_ppo_v2 --profile balanced
```

Cấu hình:

- random initialization;
- 300.032 environment steps;
- một stage 1.200 UE;
- actor nhìn toàn bộ 1.200 UE;
- rollout 256;
- gamma 0.999;
- learning rate và entropy giảm tuyến tính;
- checkpoint tại khoảng 100k, 200k, 300k.

Đây là phép thử thẳng nhất cho câu hỏi full control.

## Thí nghiệm B — Curriculum nhưng vẫn full control

```powershell
python -m scalemac_rl.scripts.train_full_control_curriculum_v2
```

Curriculum:

```text
128 → 256 → 600 → 1200 UE
```

Ở mỗi stage, PPO vẫn nhìn và chọn từ toàn bộ UE của stage; không có heuristic candidate filter.

Mục đích là phân biệt hai nguyên nhân thất bại:

- reward/input chưa đủ;
- exploration ở action space 1.200 UE quá khó.

## Thí nghiệm C — Short profile tuning

```powershell
python -m scalemac_rl.scripts.run_full_control_tuning --steps 100096
```

Profiles:

- `balanced`: trade-off chính;
- `fairness`: tăng PF/P10 và population wait pressure;
- `stable`: update nhỏ hơn để tránh policy collapse.

Chỉ chạy tuning khi profile chính không cải thiện hoặc dao động mạnh.

## Ma trận ablation đề xuất

| Nhóm | Biến | Giá trị |
|---|---|---|
| Input/embedding | schema | 16-feature observation with unchanged 64-d shared set encoder |
| Reward | PF level | 0 / 0.15 / 0.20 |
| Reward | P10 weight | 0 / 0.10 / 0.15 |
| Reward | wait pressure | 0 / 0.08 / 0.10 |
| PPO | gamma | 0.995 / 0.999 |
| PPO | clip | 0.08 / 0.10 / 0.20 |
| PPO | LR | 5e-5 / 7.5e-5 / 1e-4 |
| PPO | rollout | 256 / 512 |
| Architecture | encoder | Set MLP / Recurrent Set / Sorted CNN |

Không chạy toàn bộ grid cùng lúc. Mỗi vòng chỉ thay một nhóm biến.

## RPPO

Recurrent Set PPO là kiến trúc ưu tiên sau baseline v2:

```text
shared per-UE encoder
→ hidden state riêng theo UE
→ set pooling
→ priority head + PRB-demand head
```

Lý do: starvation, HARQ và service cycle là hiện tượng theo thời gian. Tuy nhiên observation v2 đã chứa EWMA và wait, nên RPPO chỉ đáng triển khai sau khi biết baseline feed-forward còn thiếu gì.

## CNN

Không dùng Conv1D trực tiếp theo UE ID vì UE liền kề trong mảng không có quan hệ vật lý. Sorted-CNN chỉ hợp lý khi UE được sort theo một khóa có ý nghĩa, ví dụ:

```text
wait rank → throughput deficit → HARQ → CQI
```

CNN sẽ là thí nghiệm sau RPPO, không phải baseline đầu tiên.

## Tiêu chí dừng sớm

Dừng hoặc đổi profile nếu sau khoảng 100k–150k bước:

- starvation vẫn cao và không có xu hướng giảm;
- P99/max wait bám episode length;
- fairness giảm liên tục trong khi dual multiplier tăng;
- policy entropy collapse quá sớm;
- reward tăng nhưng KPI thật xấu đi.

## Sau khi train

Chọn checkpoint theo thứ tự ưu tiên:

```powershell
$ppoFull = if (Test-Path .\artifacts\full_control_v2_balanced_best_feasible.pt) {
    ".\artifacts\full_control_v2_balanced_best_feasible.pt"
} elseif (Test-Path .\artifacts\full_control_v2_balanced_best_tradeoff.pt) {
    ".\artifacts\full_control_v2_balanced_best_tradeoff.pt"
} else {
    ".\artifacts\full_control_v2_balanced_best_lowest_violation.pt"
}
```

Đánh giá unified:

```powershell
python -m scalemac_rl.scripts.run_unified_evaluation `
  --ppo-full-checkpoint $ppoFull `
  --num-ues 1200 `
  --slots 5000 `
  --seed 1701 `
  --seeds 1 `
  --profile-seed 1701 `
  --output .\artifacts\full_control_v2_unified.csv `
  --manifest-output .\artifacts\full_control_v2_unified_manifest.csv
```

Tạo báo cáo Markdown tự động:

```powershell
python -m scalemac_rl.scripts.build_full_control_report `
  --training .\artifacts\full_control_v2_balanced_training.csv `
  --validation-summary .\artifacts\full_control_v2_balanced_validation_summary.csv `
  --checkpoint-manifest .\artifacts\full_control_v2_balanced_checkpoint_manifest.csv `
  --evaluation .\artifacts\full_control_v2_unified_tradeoff.csv `
  --output .\docs\reports\full_control_v2_research_report.md
```
