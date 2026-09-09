# Datasheet

## Động cơ
Gán nhãn tự động ở mức ký tự cho văn bản chữ Nôm **chép tay bút lông, thể hành-thảo**,
bằng cách khai thác **bản phiên âm Quốc ngữ song song in kèm** làm giám sát yếu — thay cho
việc gán tay từng chữ.

## Thành phần
- **54,156** ô có nhãn cấp ký tự · **10,369** ô chỉ có chú giải âm
- **1,660** lớp chữ phân biệt · **1,353** âm Quốc ngữ phân biệt
- **447** trang · **4,011** cột · **3** cuốn: stt11, stt2, stt4
- chất lượng ảnh: `blank` 40 (0.06%) · `bleed` 4,090 (6.34%) · `ok` 59,954 (92.92%) · `truncated` 441 (0.68%)

## Quy trình thu thập
Ảnh trang → dò cột → OCR chữ Nôm (dịch vụ HCMUS) + OCR Quốc ngữ (VietOCR, chạy cục bộ) →
căn chỉnh chữ↔âm bằng quy hoạch động có dải → hợp nhất ba tín hiệu thành tier.
Toàn bộ **tất định tới từng byte**; chạy lại hai lần cho kết quả trùng khít.

## 🔴 Giới hạn — đọc trước khi dùng

1. **Chưa có phép đo precision nào còn hiệu lực.** Số cũ đã bị tước tư cách bằng chứng
   (nguồn phán quyết là máy chấm, không phải người).
2. **Ba cuốn cùng MỘT thể loại** (truyện thánh Công giáo). Ngoại suy sang Nôm văn học
   hay hành chính **chưa được kiểm chứng**.
3. **Chữ Nôm tự tạo có thể bị hụt**: chỉ **6,24%** (3,380 ô)
   nằm ngoài khối CJK cơ bản, so với
   **4,21%** ở ngữ liệu NomNaOCR. Chưa rõ do pipeline bóc mất bộ thủ hay do Nôm Công
   giáo thế kỷ XIX vốn chuộng dạng giản.
4. **Chưa chuẩn hoá dị thể.** Cùng một chữ có thể xuất hiện dưới nhiều mã
   (`徳`/`德`, `别`/`別`). Ứng viên đã lọc ở `docs/UNG_VIEN_CHUAN_HOA_DI_THE.csv`,
   **chưa áp dụng**.
5. **481 ô có ảnh hỏng** (`usable_image=0`) vẫn nằm trong bộ — nhãn có thể
   đúng, ảnh thì không dùng được.
6. **207 ô nằm trên 3 trang không đủ 9 cột** (`page_cot_lech=1`).
   Bố cục trang luôn 9 cột, nên thiếu cột nghĩa là phép ghép cột Nôm↔Quốc ngữ trên
   trang đó có thể đã trượt một nhịp. Cờ chỉ nêu sự việc, không kết luận nhãn sai.

7. **Không có recall.** Bộ này chỉ chứa ô đã gán được nhãn; phần bị bỏ không nằm ở đây.
8. **Chia tách: 0/447 trang nằm ở hai phía.** 175 ô có
   lớp chữ không mặt trong `train` — hệ quả của việc chia tách trung thực theo trang.
   Lọc bằng `label_in_train` khi đánh giá (cột này RỖNG ở tầng SYLLABLE, xem README).

## Khuyến nghị dùng
Dùng được: huấn luyện mô hình, thăm dò, làm điểm khởi đầu để chấm tay.
**Chưa dùng được**: trích dẫn như dữ liệu đã kiểm chứng, hoặc làm chuẩn đánh giá.

*Sinh tự động từ `labels.csv` ngày 2026-09-09 · commit `2f0b9dc116`*
