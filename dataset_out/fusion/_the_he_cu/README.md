# Tạo phẩm THẾ HỆ CŨ — không dùng

`s3_corpus.2026-08-12.csv` sinh ngày 12-08, TRƯỚC nhiều lần dựng lại bộ nhãn.
Chốt chặn `s3_signals.attach` đo được **1.019/59.690 hàng cùng `image` nhưng KHÁC
`label` (1,71% > ngưỡng 0,1%)** — tức tín hiệu đang chấm sai chữ.

Dời khỏi đường chạy vì hai lẽ:

1. `make_combined_batch` có sẵn nhánh chạy KHÔNG cần S3; tín hiệu S3 chỉ dùng để XẾP
   HẠNG độ nghi, không phải để chọn mẫu.
2. Với mục tiêu ƯỚC LƯỢNG PRECISION, lấy mẫu **ngẫu nhiên phân tầng** mới cho ước
   lượng không chệch. Xếp theo độ nghi làm lệch mẫu về phía ô xấu — tốt cho việc SĂN
   LỖI, sai cho việc ĐO TỶ LỆ.

Muốn dùng lại thì sinh mới, đừng sửa tệp này:

    .venv/bin/python -m pipeline.consensus_fusion.score_s3 --all
