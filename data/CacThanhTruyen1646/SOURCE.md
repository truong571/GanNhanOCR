# Các Thánh Truyện 1646 (CacThanhTruyen_1646.pdf) — CHƯA XÁC ĐỊNH
- 78 tờ đôi (~156 trang), ~1650×2350 px/trang, chữ thảo cùng kiểu Maiorica như STT; PDF đảo ngược toàn bộ: tờ 0 = tr.322–323 … tờ 77 = tr.168–169 (đo OCR số trang 20/09 07:40).
- Không có quốc ngữ. Đã so ảnh với 448 trang STT: KHÔNG trùng bản scan (sim max 0,68 < nền 0,86) nhưng có thể trùng NỘI DUNG (config/pipeline.yaml ghi: các bản "Các Thánh Truyện" là tập con của "Sách Các Thánh Truyện").
- Việc cần làm trước khi dùng: OCR vài trang → tra vào dataset_out xem có trùng tháng 2/4/11 không; nếu là tháng khác thì phải tìm bản QN tương ứng.
- Khả năng ra dataset kiểu STT: CHƯA — thiếu QN.

## Thẩm định (20/09 07:40)
- **Dải trang & thứ tự (đã OCR số trang 78 tờ):** số trang in đầu trang chạy **323 → 168**, giảm đều 2 trang/tờ (tờ 0 = 322–323, tờ 14 = 294–295, tờ 49 = 224, tờ 77 = 168). Tức PDF chỉ bị **đảo ngược toàn bộ**, không xáo trộn; đảo lại là đọc được. Khẳng định "từ 312 đến 323" là sai.
- **Nguồn "Harvard":** không có căn cứ trong tệp — metadata PDF: title "Các thánh truyện V3.pdf", author "Francis Nguyen", Microsoft Print to PDF 05/03/2024. Chưa xác minh được kho lưu trữ gốc.
- **"Tháng 12 / Thánh Phanxicô Xaviê ở tr.322–323":** là kết luận do máy đọc chữ thảo, CHƯA kiểm chứng độc lập (không chạy được OCR Nôm cục bộ; cần API kim). Cách kiểm chắc: OCR 3–5 trang qua pipeline rồi tra tên thánh vào QN của STT2/4/11; hoặc người biết Nôm đọc tiêu đề truyện.
- Đã kiểm trước: KHÔNG trùng bản scan STT (sim 0,68 < nền 0,86). Trùng nội dung hay không vẫn mở.

## Đo đạc (20/09 23:00)
- **Harvard: ĐÚNG** — trang 168 (tờ 77 trái) là trang trắng có dấu "HARVARD COLLEGE LIBRARY". Rút lại nhận xét "không có căn cứ" trước đó.
- **Không trùng nội dung STT2/4/11 (đã đo):** API kimhannom trả "User account is not active" nên OCR cục bộ bằng NomNaOCR finetuned (`NomNaOCR/weights/finetuned_CRNNxCTC.h5`, cắt cột bằng chiếu dọc, nhận dạng theo đoạn ~5 chữ). Đối chứng dương (3 trang STT có trong tham chiếu): max-ratio từng cột trung vị 0,51–0,56; nền (sách khác): 0,20–0,25. Năm trang CTT (168, 246, 247, 322, 323): trung vị 0,23–0,26, LCS dài nhất 2–5 chữ → mức nền ⇒ văn bản không nằm trong 3 tháng đã có.
- "Tháng 12 / Thánh Phanxicô Xaviê": chưa kiểm chứng (không ảnh hưởng kết luận không trùng).
- ⚠️ Phát hiện kèm: tài khoản OCR HCMUS (SN_OCR_USERNAME) đang **không hoạt động** (401 "User account is not active") → pipeline bước 1 không OCR được sách mới cho tới khi kích hoạt lại.
