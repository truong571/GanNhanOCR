# Kim Vân Kiều Tân Truyện (Ấn bản song ngữ 1884–1885)

## 1. Thông tin tư liệu
- **Tên tác phẩm:** Kim Vân Kiều tân truyện (金雲翹新傳)
- **Tác giả:** Nguyễn Du
- **Khảo cứu, phiên âm Quốc ngữ và dịch Pháp văn:** Abel des Michels (Giáo sư Trường Ngôn ngữ Đông phương, Paris)
- **Nhà xuất bản:** Ernest Leroux, Paris, 1884–1885
- **Đơn vị lưu trữ nguồn số hóa:** Thư viện Quốc gia Pháp (Bibliothèque nationale de France - Gallica)

## 2. Cấu trúc ấn bản & Dữ liệu tải về
Toàn bộ ấn bản gồm 3 tập được số hóa đồng bộ và lưu trữ tại `data/KimVanKieu1884/`:

1. **Bản Nôm viết tay in thạch bản (Tome II, 2e partie - 1885):**
   - **Thư mục lưu trữ:** `data/KimVanKieu1884/pages/` (171 canvas, `canvas_0001.jpg` đến `canvas_0171.jpg`)
   - **Mã định danh Gallica Ark:** `ark:/12148/bpt6k5453029r`
   - **Đặc điểm thể hiện:** Toàn bộ 3.254 câu Kiều được chép bằng bút lông theo lối chân thư mỹ thuật của người Việt thế kỷ 19, in thạch bản (lithography) sắc nét trên giấy tây khổ lớn, phân chia trang rõ ràng.

2. **Bản Quốc ngữ đối chiếu & Pháp văn - Tập 1 (Tome I - 1884):**
   - **Thư mục lưu trữ:** `data/KimVanKieu1884/quocngu_pages/vol1/` (322 canvas, `canvas_0001.jpg` đến `canvas_0322.jpg`)
   - **Mã định danh Gallica Ark:** `ark:/12148/bpt6k5439461n`
   - **Nội dung:** Lời tựa, khảo cứu ngữ pháp, văn bản Quốc ngữ phiên âm chuẩn mực đối chiếu song song với bản dịch tiếng Pháp cho các câu từ 1 đến 1.500.

3. **Bản Quốc ngữ đối chiếu & Pháp văn - Tập 2 (Tome II, 1ère partie - 1884):**
   - **Thư mục lưu trữ:** `data/KimVanKieu1884/quocngu_pages/vol2/` (309 canvas, `canvas_0001.jpg` đến `canvas_0309.jpg`)
   - **Mã định danh Gallica Ark:** `ark:/12148/bpt6k54394659`
   - **Nội dung:** Tiếp tục văn bản Quốc ngữ phiên âm đối chiếu song song với tiếng Pháp cho các câu từ 1.501 đến 3.254, kèm bảng từ vựng và chú thích dị bản cuối sách.

## 3. Ý nghĩa đối với Ground Truth OCR Chữ Nôm
- **Khớp 1-1 tuyệt đối:** Khác với các bản Nôm chép tay khác thường bị lệch dị bản khi đối chiếu với các bản Quốc ngữ hiện đại thế kỷ 20-21, bản 1884 của Abel des Michels có bản Nôm và bản Quốc ngữ được soạn thảo đồng thời, cùng một hệ thống phiên âm và đối chiếu chuẩn xác từng dòng từng chữ.
- **Hình ảnh scan gốc làm bằng chứng:** Có đầy đủ cả ảnh scan trang Nôm lẫn ảnh scan trang in Quốc ngữ cổ, đáp ứng tiêu chuẩn khắt khe nhất về tính minh bạch của nhãn dữ liệu (ground truth).

## 4. Cấu trúc thư mục & Ground Truth TSV
```
data/KimVanKieu1884/
├── pages/                              # 171 canvas Nôm in thạch bản chữ bút lông
├── quocngu_pages/
│   ├── vol1/                          # 322 canvas Quốc ngữ đối chiếu Tập 1 (câu 1-1500)
│   └── vol2/                          # 309 canvas Quốc ngữ đối chiếu Tập 2 (câu 1501-3254)
├── kim_van_kieu_1884_quoc_ngu.tsv     # Toàn bộ 3.254 câu chuẩn (verse_id, tome, meter, qn, nom)
└── SOURCE.md                           # Tài liệu này
```

## Thẩm định (20/09 06:30, đo bằng ảnh + OCR)
- ✅ Ảnh Nôm 171 canvas (Tome II 2e partie) = in thạch bản chữ bút lông, ~150 px/chữ, có SỐ CÂU in mỗi 5 câu; QN cùng sách nằm trong `quocngu_pages/vol1|vol2` (631 canvas), trang QN (số câu mỗi 5 câu) xen kẽ trang dịch Pháp → ≈ một nửa là QN. Đây là cặp 1-1 đúng nghĩa, cùng thiết lập với LucVanTien1883 (cần Tesseract/VietOCR trang QN).
- ⚠️ **Canvas Nôm xếp NGƯỢC thứ tự đọc** (sách đóng phải→trái, Gallica quét từ bìa Tây): OCR số câu: canvas 5 ≈ câu 3.220–3.240, canvas 30 ≈ 2.730, canvas 90 ≈ 1.525, canvas 150 ≈ 330, canvas 165 ≈ 40. Ghép trang phải đảo thứ tự (`page = 167 − canvas`, đúng 163/163). Bố cục: 10 cột = 10 cặp lục bát (6 trên/8 dưới). Nôm đếm 3.256 câu vs QN 3.253 (nhảy 1581→1585) → lệch chưa định vị (21/09).
- ❌ `kim_van_kieu_1884_quoc_ngu.tsv` KHÔNG phải phiên âm 1884: trùng 3.254/3.254 câu (cả Nôm lẫn QN) với bản **1871 Liễu Văn Đường** trên nomfoundation.org. OCR thử trang QN 1884 câu 343–350: ≥3/8 câu khác ("Khuôn lĩnh dầu phụ tấc thành" ≠ "Khuôn thiêng dù phụ", "như dầu" ≠ "như ru", "Khi buổi mới" ≠ "Trong buổi mới"). Không dùng làm GT; chỉ làm gợi ý.
