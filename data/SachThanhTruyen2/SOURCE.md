# Sách Các Thánh Truyện — tháng 2 (STT2.pdf)
Kho ngữ liệu CHÍNH của luận văn (mã sách trong pipeline: `SachThanhTruyen2`, tiền tố crop `yen2`).
Chữ thảo viết tay (Maiorica), 320 trang PDF = 160 trang Nôm 9 cột + 160 trang quốc ngữ đối diện (có text layer).
Ảnh 1712×2708, ≈110 px/chữ. Đầu ra pipeline ở `prepared/SachThanhTruyen2/`, `dataset_out/`.

## Ghi chú 20/09 06:30
- `quocngu_pages/` + `INDEX.tsv` là ảnh trang QN trích lại từ chính PDF (bản sao; pipeline vẫn đọc PDF trực tiếp). Tên `facing_nom_page` trong INDEX (page_0001, page_0003…) KHÔNG trùng cách đặt tên trang của pipeline (`prepared/*/pages/page_0012…`, đánh theo số trang sách) → không dùng INDEX để ghép với dataset_out.
