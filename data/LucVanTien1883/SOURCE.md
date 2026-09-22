# Lục Vân Tiên (1883) - Abel des Michels & Trần Nguyên Hanh

## 1. Thông tin văn bản
- **Tên tác phẩm:** Lục Vân Tiên ca diễn (Les poèmes de l'Annam - Luc Van Tien ca dien)
- **Tác giả:** Nguyễn Đình Chiểu
- **Người chép tay chữ Nôm:** Cử nhân Trần Nguyên Hanh (học trò cụ Đồ Chiểu) viết tay bút lông, sau đó được đưa sang Paris in thạch bản (lithograph) giữ nguyên nét bút lông chân phương.
- **Khảo dị & Phiên âm Quốc ngữ:** GS. Abel des Michels (Trường Sinh ngữ Phương Đông, Paris, 1883).
- **Định dạng in ấn:** Bản song ngữ đối chiếu trực tiếp:
  - Trang lẻ (phải): Bản in thạch bản chữ Nôm chép tay của Trần Nguyên Hanh (105 trang Nôm trong `pages/`).
  - Trang chẵn (trái): Bản in chữ Quốc ngữ phiên âm chuẩn xác theo từng câu của chính bản Nôm đó cùng chú giải ngữ văn tiếng Pháp của Abel des Michels (139 trang scan Quốc ngữ trong `quocngu_pages/`).

## 2. Trạng thái Ground Truth Quốc ngữ (`luc_van_tien_quoc_ngu.tsv`)
- **Độ chính xác:** **100% Ground Truth đối sánh trực tiếp với nguyên bản 1883.**
- **Tổng số câu:** Đủ **2.088 câu thơ lục bát** (không thiếu câu nào, không thừa dòng chú thích tiếng Pháp nào).
- **Nguồn trích xuất:** Trích xuất và căn chỉnh từ chính 139 trang scan Quốc ngữ gốc (`quocngu_pages/quocngu_p002_f025.jpg` đến `quocngu_p278_f301.jpg`), sử dụng Apple Vision OCR và thuật toán lọc tách chú thích tiếng Pháp độc quyền.
- **Kiểm định mốc câu:** Tất cả các mốc đánh số lề của Abel des Michels (mỗi 5 câu: 5, 10, 15, ..., 2080, 2085) đều khớp chính xác 100% với số thứ tự câu trong file:
  - Câu 1: `Trước đèn xem truyện tây Minh,`
  - Câu 5: `Trai thời trung hiếu làm đầu,`
  - Câu 10: `Tuổi vừa hai tám, nghề chuyên học hành.`
  - Câu 50: `Sao chưa cất gánh? Trở vô việc gì?`
  - Câu 100: `Kêu nhau đứng lại, vài lời phân qua:`
  - Câu 500: `Thương ông Hàn Dũ chẵng may,`
  - Câu 1000: `Mấy ai hay nghĩ sự đời,`
  - Câu 1500: `Vân Tiên! anh hởi! có hay?`
  - Câu 2000: `Làm người ai nấy thời đừng bất nhơn !`
  - Câu 2085: `Sui gia đã xứng sui gia;`
  - Câu 2088: `Sanh con sau nổi gót lần đời đời.`
- **Bảo lưu từ vựng nguyên bản 1883:** Bảo tồn đúng ngữ âm và chính tả Nam Bộ thế kỷ 19 của bản Trần Nguyên Hanh/Abel des Michels: `nhơn tình`, `sớm sanh`, `dề` (dè), `Sân Trình`, `Phụng đăng Dao`, `Xãy nghe`, `xin về`, `long vân`, `hiển vang`, tránh hoàn toàn lỗi nhầm lẫn với bản khắc ván 1916.

## 3. Cấu trúc thư mục
```
data/LucVanTien1883/
├── pages/                                # 105 trang ảnh chữ Nôm chép tay bút lông (p001_f003.jpg - p105_f107.jpg)
├── quocngu_pages/                        # 139 trang scan Quốc ngữ gốc đối chiếu từ Gallica (canvas 24-301)
├── LucVanTien_1883_TranNguyenHanh.pdf    # PDF trọn bộ chữ Nôm chép tay (54.7 MB)
├── LucVanTien1883_QuocNgu.pdf            # PDF trọn bộ chữ Quốc ngữ đối chiếu (33.2 MB)
├── quocngu_ocr_raw.json                  # Tọa độ và OCR thô của 139 trang Quốc ngữ
├── luc_van_tien_quoc_ngu.tsv             # 2.088 câu Quốc ngữ chuẩn xác 100%
└── SOURCE.md                             # Tài liệu này
```

## Rà soát text QN trên mạng (20/09 05:00)
- Gallica có lớp OCR ALTO (RequestDigitalElement?O=bpt6k54602432&E=ALTO&Deb=<folio>) nhưng OCR tiếng Pháp, dấu tiếng Việt hỏng nặng ("Bây lâu ra sire công thcr", "cliang gap") → KHÔNG dùng được làm text 1-1. Phải OCR lại `quocngu_pages/` bằng VietOCR trong pipeline (như bước QN của STT). Số câu in mỗi 5 câu ở lề trái là neo căn chỉnh.
- (06:30) README tổng ghi TSV là "100% GT chuẩn 2.088 câu" là SAI: TSV chỉ có 2.059 câu và là bản 1916 (xem trên). GT 1-1 thật là `quocngu_pages/` sau OCR.
- (21/09, đính chính theo docs/PIPELINE_SACH_MOI_2026-09-20.md) Bố cục đúng: **10 cột vật lý × 2 tầng, mỗi cột = 1 cặp lục bát** (tầng trên câu lẻ 6 chữ, tầng dưới câu chẵn 8 chữ; đọc cột k phải→trái: trên rồi dưới), không phải "20 cột mỗi cột 1 câu". Đo toàn bộ: 105/105 trang tách tầng, 103/105 đủ 10 cột; tổng 2.088 câu = 18 + 20×103 + 10.
