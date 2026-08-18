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

## v0.9.3 — Round 06 coordinate ablation and Round 07 screen

Round 06 completed all six small coordinate moves around the equal-third
Throughput–Jain–Service point.

Observed deterministic pattern:

- keeping Service at one third allowed both a mild Throughput tilt and a mild
  Jain tilt to remain starvation-free;
- reducing Service to 0.2667 caused collapse in both tested directions;
- increasing Service to 0.40 remained stable only when Throughput stayed at one
  third; reducing Throughput to 0.2667 still collapsed;
- stochastic training metrics remained much fairer than deterministic
  validation, reinforcing the action-alignment finding from Round 03;
- the equal-third point remains the balanced reference, while
  `0.40 Throughput / 0.2667 Jain / 0.3333 Service` is retained only as a
  throughput-oriented local alternative.

Interpretation: Throughput provides the useful-transmission anchor, Service
provides coverage/wait shaping, and Jain redistributes successful throughput
when the first two signals remain sufficiently strong. This is an empirical
one-seed local result, not a universal coefficient threshold.

Round 07 therefore stops tuning the same three coefficients and adds one new
positive component at a time with four equal 0.25 coefficients. The screened
components are deficit service, PF utility, low-throughput, and urgency service.

## v0.9.4 — Integrated Round 07 fourth-component study

Round 07 is intentionally expanded before execution so all first-order questions
about a fourth reward component are answered in one controlled batch. Each new
component is observed under equal-quarter addition, 0.40 component dominance, and
an anchor-preserving substitution that keeps Throughput and Service at 0.30 while
reducing Jain to 0.10.

This design separates three explanations: the component has no useful marginal
signal; the component is useful only as shaping and fails when dominant; or the
component appears to fail only because the Throughput and Service anchors were
diluted. All 12 cases use the same one-seed environment, PPO architecture and
100,096-step budget. Dataset/Pareto work remains out of scope.

## v0.9.5 — Comprehensive fourth-component geometry

Round 07 now covers 32 controlled cases: equal and heavy introductions, three single-anchor holds, and three pair/group-preserving substitutions for each remaining positive reward. The analysis is intentionally kept in one round so anchor dependence, group dependence and component comparison can be interpreted together.


## v0.9.6 — Round 07 completed and interpreted

Round 07 completed all 32 fourth-component cases under the same 100,096-step,
one-seed protocol. Ten cases are stable by deterministic coverage/tail criteria,
three are borderline, four late-collapse after an intermediate safe checkpoint,
and fifteen fully collapse.

The strongest balanced case is `urgency_service_hold_throughput`
(T/J/S/U = 0.25/0.20/0.20/0.35): 97,877 bit/slot, Jain 0.3248, zero
starvation, P99 46 and max wait 48. `deficit_service_anchor_preserving` is the
strongest fairness/tail-delay alternative: Jain 0.2921, P99 43 and max wait 44,
with a 4.90% goodput reduction versus equal-third.

Mechanism findings:

- deficit service provides useful per-UE credit when two baseline anchors remain
  strong, but collapses as a dominant or weakly anchored objective;
- PF utility is stable only under a Jain anchor and still yields low deterministic
  fairness;
- low-throughput percentile has a P10=0 dead-zone, so coefficient tuning cannot
  recover a zero raw signal;
- urgency service is promising but non-monotonic: equal-quarter and group T+S
  late-collapse while the 0.35 hold-Throughput geometry remains stable;
- stochastic training Jain remains much higher than deterministic validation
  Jain across successful and failed cases.

Decision: run a small common-seed confirmation of equal-third, urgency hold-T,
and deficit group T+S. Do not start another broad sweep, architecture change or
PPO tuning before this confirmation.


## v0.9.7 — Round 07 documentation made reusable

The Round 07 numerical evidence and research decision are unchanged. The main
report was rewritten so a future reader can reconstruct the experiment without
remembering the case-name convention. It now states the reward formulas, explains
all eight regimes, describes what each of the 32 cases changed, highlights top-three
Stable KPI values, and separates four decision classes: winner, secondary
candidate, stable but low-marginal-value cases, and components to park/redesign.

The previous technical report is retained as an appendix. The next experimental
step remains a common-seed confirmation of equal-third, urgency hold-Throughput,
and deficit group Throughput+Service.

## v0.9.8 — Round 08 common-seed confirmation

Round 08 freezes the reward search to three profiles and tests repeatability rather
than adding another component. The equal-third Throughput–Jain–Service baseline,
`urgency_service_hold_throughput`, and `deficit_service_anchor_preserving` are each
run with seeds 1701, 2701 and 3701. Within a case, the training seed, static-profile
seed and validation seed are identical; all other environment, PPO, architecture,
step-budget and deterministic validation settings remain fixed.

The runner now supports case-level common overrides so a single auditable plan can
express common-seed comparisons without duplicating plan files. Round 08 exports
per-seed final metrics, validation trajectories, stability labels, paired deltas,
and profile-level mean/std in CSV plus a reader-facing HTML and Markdown summary.

Decision rules are fixed before execution: retain Urgency only if it is stable at
all three seeds, repeatedly improves deterministic fairness, and keeps at least 98%
of baseline mean goodput; retain Deficit only as a delay-sensitive profile if tail
improvements repeat and the goodput trade-off is acceptable. Otherwise return to
equal-third T–J–S. PPO, Beta, architecture, reward definitions and coefficient
micro-sweeps remain deferred.

## Round 08 — multi-seed confirmation

Round 08 completed 9/9 cases (3 profiles × 3 seeds). T–J–S equal-third was Stable in 2/3 seeds; Urgency and Deficit were each Stable in 1/3. The fourth-component candidates are therefore not confirmed as robust.

A new reproducibility concern was identified: the Round-07 candidate formulas rerun with seed 1701 produced materially different outcomes in Round 08 despite matching core commands. This must be diagnosed before another reward sweep. The next experiment should first repeat an identical T–J–S configuration within a controlled runtime and record software/CPU/thread/determinism metadata.


## Round 09 — reproducibility diagnostic

Round 08 did not confirm a fourth reward component across seeds, so T–J–S equal-third is retained as the active reward anchor. Before Dynamic CQI, Round 09 runs three identical T–J–S seed-1701 repeats and records runtime/RNG/model fingerprints plus pairwise trajectory divergence. This separates reward selection from reproducibility diagnosis.

## v0.10.1 — local execution handoff

Round 09 remains unchanged scientifically. A local PowerShell runner was added so the controlled reproducibility diagnostic can be executed/resumed on the user's machine without creating a Kaggle notebook. Kaggle notebooks should only be prepared when explicitly requested.

## Round 09 — reproducibility diagnostic

Three identical local CPU runs of T–J–S seed 1701 reproduced exactly. Training CSV numeric columns, RNG hashes, initial/final model hashes and deterministic validation KPI were identical across all repeats. The local runtime was Windows / Python 3.11.15 / NumPy 2.4.6 / PyTorch 2.13.0+cpu.

This separates two issues that were previously conflated:
1. repeatability within an identical runtime: confirmed;
2. robustness across seeds/profiles and sensitivity across runtimes: still open.

Decision: freeze T–J–S as the active reward and move to Dynamic CQI rather than further reward-component exploration.

## v0.11.0 — Dynamic CQI begins

Reward exploration is closed with T–J–S as the active objective. The first realism increase is a controlled temporally correlated CQI process around each UE's heterogeneous anchor. Static mode remains available as an exact baseline. Round 10 compares static, slow-correlated, and faster-correlated CQI without changing PPO, architecture, traffic, BLER, or HARQ control.

Future Joint Link Adaptation + Scheduler work is expected to extend the action space from UE/PRB decisions toward UE + PRB + MCS + transmission-layer selection after MIMO/CSI support is introduced.
