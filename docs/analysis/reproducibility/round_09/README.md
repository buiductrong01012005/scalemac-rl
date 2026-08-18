# Round 09 — Reproducibility Diagnostic

Mục tiêu: kiểm tra ba lần chạy hoàn toàn giống nhau của T–J–S seed 1701 trong cùng local runtime.

Kết quả: 3/3 repeat trùng tuyệt đối về training trajectory, RNG/model hashes và final deterministic KPI.

Kết luận:
- local CPU pipeline hiện tại reproducible;
- không cần tiếp tục deterministic-lock diagnostic;
- khác biệt giữa các round trước cần được xem như cross-runtime/session sensitivity;
- T–J–S được giữ làm reward nền;
- bước tiếp theo là Dynamic CQI.

Mở `reproducibility_analysis.html` để đọc bản dễ hiểu.
