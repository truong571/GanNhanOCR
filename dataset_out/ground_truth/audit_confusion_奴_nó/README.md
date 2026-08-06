# Mẻ audit lớp nhầm lẫn — `奴` / âm "nó"

**55 ô**, chấm khoảng nửa giờ.

## Câu hỏi cần trả lời

Mẻ audit GOLD đầu tiên tìm được 3 lỗi, trong đó **2 lỗi là cùng cặp `奴` /
"nó"**. Cặp này xuất hiện **312 lần** trong GOLD và
là nhãn duy nhất được dùng cho âm đó. Nên câu hỏi thật là:

> 312 hàng đó đúng hay sai — toàn bộ?

Hai khả năng, hệ quả khác hẳn nhau:

- **Sai hệ thống** → 312 nhãn hỏng (~0,6% GOLD), phải sửa bằng bảng
  nhầm lẫn ở bước 7, và đây là lỗi lớn nhất tìm được cho tới giờ.
- **Hai verdict trước bị nhầm** → precision GOLD thực ra là 116/117 = 99,1%, không có gì
  phải sửa.

## ĐỌC KỸ TRƯỚC KHI CHẤM

Trong mẻ này **nhiều ô cố ý mang cùng một chữ**, xen lẫn các ô chữ khác lấy ngẫu nhiên.
Điều đó là chủ ý của thiết kế.

**Hãy chấm từng ô độc lập.** Đừng vì đã kết luận ở một ô mà đánh y hệt cho các ô sau —
nếu làm vậy, kết quả sẽ chỉ lặp lại phán đoán đầu tiên của bạn chứ không đo được gì. Cũng
đừng cố cho ra kết quả "nhất quán": hoàn toàn có thể một số ô đúng và một số ô sai.

Nếu không đọc ra chữ, chọn **không chắc** — đừng đoán. Ô "không chắc" bị loại khỏi phép
tính chứ không bị tính là sai.

## Cách chấm

1. Mở `audit_001.html` trong trình duyệt.
2. Mỗi ô: crop + glyph tham chiếu + ngữ cảnh trang + âm + ứng viên từ điển. Bấm:
   **correct** / **wrong_label** (crop đúng 1 chữ nhưng gán sai chữ) /
   **wrong_image** (crop cắt hỏng, dính 2 chữ, nhầm ô) / **unsure**.
3. Xong bấm **Download JSON**, lưu thành `verdicts_001.jsonl` ngay trong thư mục này.

## Mẻ này quyết được tới đâu (nhóm mục tiêu, n = 30, 95%)

| Kết quả | Kết luận |
|---------|----------|
| 0 ô sai | tỷ lệ lỗi của lớp ≤ **100.0%** → bác bỏ giả thuyết "sai hệ thống" |
| 30 ô sai | tỷ lệ lỗi ≥ **90.5%** → xác nhận sai hệ thống, sửa cả 312 hàng |
| lẫn lộn | lỗi phụ thuộc ngữ cảnh, cần soi theo sách/trang |

## Hai nhóm trong mẻ

| Nhóm | n | Dân số | Dùng để |
|------|--:|-------:|---------|
| `class_target` | 30 | 310 hàng chưa chấm của lớp | **tỷ lệ lỗi của lớp** |
| `control_gold` | 25 | 49,776 hàng GOLD còn lại | phá hiệu ứng mỏ neo; thêm bằng chứng cho precision GOLD |

Các ô đã chấm ở mẻ trước đã được loại, không chấm lại.
