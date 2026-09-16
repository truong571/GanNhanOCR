# Khai xuất xứ — quyết định khối glyph

Tệp này là **bằng chứng người**. `pipeline.remediation.glyph_fix` từ chối chạy nếu không có nó.
Mỗi quyết định phải khai đủ: **ai quyết · nhìn gì · ngày nào · quyết cái gì**.

> Dự án này đã một lần nhầm phán quyết MÁY thành phán quyết NGƯỜI, và phải huỷ toàn bộ số liệu
> dựng trên đó — precision GOLD 97,98%, Fisher p=5,4e-8, κ=0,13, tất cả thành vô giá trị
> (`docs/KE_HOACH_TONG_THE_2026-08-22.md` §0). Tệp này tồn tại để chuyện đó không lặp lại.

---

## QĐ-01 · âm "người" → 𠊚 (U+2029A)

| | |
|---|---|
| **Người quyết** | truongmdn (chủ nhiệm đề tài) |
| **Ngày** | 2026-08-25 |
| **Đã nhìn gì** | toàn bộ **2.014 ảnh crop** của các ô âm "người" đang bị giữ lại, xem trong `docs/nguoi_2144_standalone.html` (ảnh nhúng thẳng, không phải mẫu), cùng **49 ô mốc** đã được xác nhận trước đó |
| **Quyết định** | mọi ô mang âm "người" đang bị giữ lại **và có ảnh crop** được gán nhãn **𠊚 (U+2029A)**, đưa lên tầng GOLD |
| **Phạm vi** | 2.014 ô. **Không** áp cho 130 ô không có ảnh crop — người không nhìn được thì không có phán quyết |
| **Không đụng tới** | 12 ô GOLD âm "người" đang mang chữ khác (昆 ×8, 辞 ×2, 匕 ×1, 命 ×1). Đó là quyết định riêng, phải xem riêng |

### Vì sao khối này cần đến người

"người" là âm phổ biến **nhất** cả ba cuốn — 2.281 ô — nhưng **94% bị giữ lại**, trong khi tỷ lệ
rớt chung toàn ngữ liệu là 31%.

Nguyên nhân **không phải** thiếu từ điển: từ điển kê 20 ứng viên cho "người", trong đó có 𠊚, và
37/49 ô đã lọt GOLD đều dùng đúng chữ ấy. Nguyên nhân là **bộ OCR Nôm không đọc nổi glyph đó** —
nó nhả ra 㝵 (1.183 ô), 早 (223), 昇 (193), 身 (126), 𭔿 (79), 旱 (53) và một đuôi dài 157 ô lẻ.
Vì chữ nó đọc không phải cách đọc của "người", luật S1∩S2 không bao giờ khớp.

**Kiểm nhất quán trước khi quyết:** 𠊚 xuất hiện 38 lần trong cả ngữ liệu, **toàn bộ đều với âm
"người"** — không dùng lẫn cho âm nào khác. Mã điểm đối chiếu khớp chính xác với 37 ô mốc
(U+2029A, không phải U+2029B 𠊛 — hai chữ này chỉ khác nhau một mã điểm).

### Các đường tự động đã thử và đã bác trước khi giao cho người

| đã thử | kết quả |
|---|---|
| Unihan `kVietnamese` | 8.611/8.655 cặp đã có sẵn; 44 cặp mới cứu **0 ô** |
| hvdic.thivien.net (bảng Hán-Việt) | 15.272/15.275 cặp đã có; cứu **0 ô** |
| khôi phục dấu thanh Quốc ngữ | **0 ô** khôi phục được duy nhất |
| cầu tự dạng hai chiều (T7) | 1,91x < ngưỡng 2,0x → giữ một chiều |
| dị thể Unihan | 73 ô, không chạm khối này |
| mô hình/dịch vụ đọc ngoài | khảo sát 16 tác nhân, **0 ứng viên sống sót phản biện** |

Không còn đường máy nào. Đó là lý do khối này được đưa cho mắt người.

---

## Ghi cho lần sau

Thêm quyết định mới thì thêm một mục `QĐ-nn` ở đây **và** một khối trong
`config/quyet_dinh_glyph.yaml` trỏ `xuat_xu` về tệp này. Không có mục ở đây thì module từ chối chạy.

---

## Phụ lục a — Phán quyết QĐ-01a và ký bảng quy ước (2026-09-17)

| | |
|---|---|
| **Người quyết** | truongmdn (chủ nhiệm đề tài) |
| **Ngày** | 2026-09-17 |
| **Đã nhìn gì** | Xem trực tiếp các trang HTML kiểm tra crop: `dataset_out_v3/ky/qd01a_review.html`, `dataset_out_v3/ky/di_the_review.html`, `dataset_out_v3/ky/corpus_readings_review.html` |
| **Nội dung** | 1. **QĐ-01a (14 ô):** Phán quyết 14/14 ô `bo` (bỏ khỏi lớp người cưỡng bức; 2 ô trôi lệch âm sang `mà`/`con` và 12 ô mã ≠ 𠊚 độc lập kiểm FALSE).<br>2. **Dị thể (14 mục):** Ký chuẩn 13/14 cặp; hoãn mục `dt_5171_5176` (共/其 "cùng") cho Khối B-4.<br>3. **Corpus readings (8 mục):** Ký 3 mục áp đảo (`cr_7121_vồ`, `cr_50e5_nhiều`, `cr_4fc2_hế`); 5 mục còn lại giữ `cho_ky`. |

