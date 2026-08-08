# Decision Register

| ID | Quyết định | Lý do | Trạng thái |
|---|---|---|---|
| D01 | Giữ projector | Cần action hợp lệ; projector không được tự scheduling UE | Giữ |
| D02 | Bỏ oldest-UE rule khỏi nhánh chính | Rule che năng lực PPO và gánh safety cho Hybrid cũ | Đã áp dụng v0.8 |
| D03 | Bỏ candidate filter khỏi nhánh full-control | Cần kiểm tra PPO thực sự xử lý 1.200 UE | Đã áp dụng v0.8 |
| D04 | Không dùng PF imitation | Muốn quan sát PPO học từ đầu | Đã áp dụng v0.8 |
| D05 | Thêm rank/ratio features | Giá trị tương đối trong cell hữu ích hơn UE ID hoặc giá trị tuyệt đối | Đã áp dụng v0.8 |
| D06 | Thêm PF/P10/urgency reward | Jain toàn cell có credit assignment yếu | Đã áp dụng v0.8 |
| D07 | Gamma 0.999 và LR/entropy decay | Hậu quả starvation dài hạn, cần update ổn định | Đang thử |
| D08 | RPPO sau feed-forward baseline | Tránh đổi input, reward và architecture cùng lúc | Chờ kết quả v0.8 |
| D09 | CNN không chạy theo UE ID | Quan hệ lân cận UE ID là giả | Giữ |
