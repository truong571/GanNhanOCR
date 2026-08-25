# Datasheet

## Động cơ
Gán nhãn tự động ở mức ký tự cho văn bản chữ Nôm khắc gỗ, bằng cách khai thác **bản dịch
Quốc ngữ song song in kèm** làm giám sát yếu — thay cho việc gán tay từng chữ.

## Thành phần
- **50,156** ô có nhãn cấp ký tự · **6,911** ô chỉ có chú giải âm
- **1,583** lớp chữ phân biệt · **1,344** âm Quốc ngữ phân biệt
- **444** trang · **3,985** cột · **3** cuốn: stt11, stt2, stt4
- chất lượng ảnh: `blank` 40 (0.07%) · `bleed` 3,581 (6.28%) · `ok` 53,062 (92.98%) · `truncated` 384 (0.67%)

## Quy trình thu thập
Ảnh trang → dò cột → OCR chữ Nôm (dịch vụ HCMUS) + OCR Quốc ngữ (VietOCR, chạy cục bộ) →
căn chỉnh chữ↔âm bằng quy hoạch động có dải → hợp nhất ba tín hiệu thành tier.
Toàn bộ **tất định tới từng byte**; chạy lại hai lần cho kết quả trùng khít.

## 🔴 Giới hạn — đọc trước khi dùng

1. **Chưa có phép đo precision nào còn hiệu lực.** Số cũ đã bị tước tư cách bằng chứng
   (nguồn phán quyết là máy chấm, không phải người).
2. **Ba cuốn cùng MỘT thể loại** (truyện thánh Công giáo). Ngoại suy sang Nôm văn học
   hay hành chính **chưa được kiểm chứng**.
3. **Chữ Nôm tự tạo có thể bị hụt**: chỉ **1,63%** ô nằm ngoài khối CJK cơ bản, so với
   **4,21%** ở ngữ liệu NomNaOCR. Chưa rõ do pipeline bóc mất bộ thủ hay do Nôm Công
   giáo thế kỷ XIX vốn chuộng dạng giản.
4. **Chưa chuẩn hoá dị thể.** Cùng một chữ có thể xuất hiện dưới nhiều mã
   (`徳`/`德`, `别`/`別`). Ứng viên đã lọc ở `docs/UNG_VIEN_CHUAN_HOA_DI_THE.csv`,
   **chưa áp dụng**.
5. **424 ô có ảnh hỏng** (`usable_image=0`) vẫn nằm trong bộ — nhãn có thể
   đúng, ảnh thì không dùng được.
6. **Không có recall.** Bộ này chỉ chứa ô đã gán được nhãn; phần bị bỏ không nằm ở đây.
7. **Chia tách neo ở mức CỘT, không phải TRANG** — 360/444 trang có cột ở nhiều phía, nên
   chỉ số đo bằng `split` sẵn có là **cận trên**. Xem README.
7. **Chia tách neo ở mức CỘT, không phải TRANG** — 360/444 trang có cột ở nhiều phía.
   Chỉ số đo bằng `split` sẵn có là **cận trên**. Xem README.

## Khuyến nghị dùng
Dùng được: huấn luyện mô hình, thăm dò, làm điểm khởi đầu để chấm tay.
**Chưa dùng được**: trích dẫn như dữ liệu đã kiểm chứng, hoặc làm chuẩn đánh giá.

*Sinh tự động từ `labels.csv` ngày 2026-08-25 · commit `bfec0ac78b`*
