# Mẻ audit GOLD — verdict NGƯỜI đầu tiên

Tổng **200 ô**, chấm một buổi. Dân số GOLD = 50,286 hàng.

## Vì sao mẻ này tồn tại

846 verdict đang có trong repo đều do MÁY chấm (`source: "ai_vision"`). Không thể dùng
chúng để đặt ngưỡng cho chính tín hiệu máy — đó là lập luận vòng tròn. Mẻ này tạo thước đo
độc lập đầu tiên.

## Cách chấm

1. Mở `audit_001.html` (và các phần sau nếu có) trong trình duyệt.
2. Mỗi ô hiện: crop + glyph tham chiếu + ngữ cảnh trang + âm QN + ứng viên từ điển.
   Bấm 1 trong 4:
   - **correct** — nhãn đúng với chữ trong ảnh
   - **wrong_label** — crop cắt đúng một chữ, nhưng nhãn gán sai chữ đó
   - **wrong_image** — crop cắt hỏng: dính 2 chữ, mất nét, hoặc nhầm ô
   - **unsure** — không đủ căn cứ để kết luận
   Tiến độ tự lưu trong trình duyệt.
3. Chấm HẾT rồi bấm **Download JSON**, lưu thành `verdicts_001.jsonl` **ngay trong thư
   mục này**.

Quan trọng: chấm theo đúng thứ tự hiện ra, **đừng bỏ ô khó**. Bỏ chọn lọc sẽ phá tính
ngẫu nhiên của tầng `srs` và làm hỏng ước lượng precision.

## Hai tầng — đọc kỹ trước khi dùng số

| Tầng | n | Cách rút | Được dùng để |
|------|--:|----------|--------------|
| `srs` | 120 | Ngẫu nhiên đơn giản từ GOLD | **Ước lượng precision + CI. Đây là số báo cáo được.** |
| `active_lowmargin` | 80 | 80 hàng `s3_head_margin` thấp nhất | Đo AUC, hiệu chỉnh ngưỡng S3. **Tuyệt đối không tính precision.** |

Tầng `active_lowmargin` được chọn CHỦ ĐÍCH vì nghi ngờ cao, nên tỷ lệ lỗi trong đó cao hơn
hẳn dân số. Gộp nó vào phép tính precision sẽ cho ra con số thấp giả. `design_weight` của
tầng này để trống chính là để chặn việc gộp nhầm.

Người chấm không phân biệt được hai tầng: thứ tự hiển thị đã xáo trộn chung.

## Mẻ này đo được tới đâu (tầng `srs`, n = 120, độ tin cậy 95%)

| Số lỗi tìm thấy | Cận dưới precision |
|----------------:|-------------------:|
| 0 | 0.0000 |
| 2 | 0.0030 |
| 5 | 0.0166 |

Nói thẳng: n=120 **không đủ** để chứng minh mệnh đề "GOLD precision 98%" — kể cả khi
không tìm thấy lỗi nào, cận dưới cũng chỉ đạt 0.0%. Mẻ này là thước đo để chỉnh bước
3, **không phải** mẻ nghiệm thu cho luận văn. Mẻ nghiệm thu đầy đủ (n≈450-600) là việc của
giai đoạn audit chính thức.

## Sau khi chấm xong

```
.venv/bin/python -m pipeline.ground_truth estimate \
    --verdicts dataset_out/ground_truth/audit_gold_human \
    --manifest dataset_out/ground_truth/audit_gold_human/manifest.jsonl \
    --design srs --p0 0.97
```
