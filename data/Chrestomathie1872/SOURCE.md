# Chrestomathie Cochinchinoise (Abel des Michels, Paris 1872)

## 1. Thông tin văn bản
- **Tên tác phẩm:** *Chrestomathie cochinchinoise : recueil de textes annamites* (Tuyển tập văn liệu Nam Kỳ)
- **Tác giả / Khảo cứu:** Abel des Michels (Giáo sư Trường Ngôn ngữ Đông phương, Paris)
- **Nguồn văn bản gốc:** Trích tuyển từ công trình *Chuyện đời xưa* (Contes du temps passé, 1866) của Pétrus Trương Vĩnh Ký cùng nhiều truyện cổ tích, ngụ ngôn, ca dao dân gian Việt Nam thế kỷ 19.
- **Nhà xuất bản:** Maisonneuve et Cie, Paris, 1872.
- **Nguồn lưu trữ số hóa:** Thư viện Quốc gia Pháp (Bibliothèque nationale de France - Gallica).
- **Mã định danh Gallica Ark:** `ark:/12148/bpt6k5812244p` (Trọn bộ 173 canvas).

## 2. Cấu trúc ấn bản & Dữ liệu lưu trữ
Toàn bộ sách gồm 173 canvas ảnh scan chất lượng cao (1500px) lưu tại `data/Chrestomathie1872/pages/`:
1. **Phần chữ Nôm in thạch bản (Canvases 106 đến 171):**
   - Chữ Nôm in thạch bản phỏng theo chữ viết tay bút lông chân thư mực đậm của người Việt.
   - Có dấu khuyên son/vòng tròn ngắt câu, nét chữ rất sắc nét và chuẩn mực.
   - Nội dung bao gồm hàng chục câu chuyện văn xuôi, truyện tiếu lâm, ngụ ngôn và tục ngữ ca dao.
2. **Phần chữ Quốc ngữ & Pháp văn (Canvases 27 đến 105):**
   - Trọn vẹn toàn bộ văn bản chữ Quốc ngữ cổ của Pétrus Trương Vĩnh Ký phiên âm đối chiếu.
   - Bản dịch nghĩa tiếng Pháp chuẩn xác của Abel des Michels.

## 3. Ý nghĩa đối với bài toán OCR Chữ Nôm
- **Ngữ liệu văn xuôi quý hiếm:** Hầu hết các bản Nôm có phiên âm Quốc ngữ hiện nay đều là thơ lục bát (Kiều, Lục Vân Tiên). `Chrestomathie 1872` cung cấp kho dữ liệu **văn xuôi khẩu ngữ đời thường (Chuyện đời xưa)**, giúp mô hình OCR và NLP học được cấu trúc câu văn xuôi đa dạng của tiếng Việt thế kỷ 19.
- **Khớp 1-1 tuyệt đối:** Bản Nôm và Quốc ngữ được in cùng trong một ấn bản học thuật của Abel des Michels và Trương Vĩnh Ký, đảm bảo tính nhất quán 100% về mặt từ vựng và phiên âm.

## Thẩm định (20/09 06:30)
- Cùng sách, cùng văn bản (Abel des Michels 1872, Chuyện đời xưa của Trương Vĩnh Ký): Nôm thạch bản canvas 104–170, QN canvas **29–56 (~28 trang)**, canvas 57–101 là **bản dịch Pháp** — nên `quocngu_pages/` 79 canvas chỉ có ~28 trang QN.
- Văn xuôi theo truyện (I, II, …), không có số câu/cột tương ứng → 1-1 ở mức TRUYỆN. Pipeline hiện tại ghép cột↔cột (9 cột STT) không áp dụng thẳng; cần căn chỉnh chuỗi QN của cả truyện vào các cột Nôm. Có thể làm nhưng là công đoạn mới.
