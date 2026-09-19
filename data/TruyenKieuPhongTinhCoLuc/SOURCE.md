# Truyện Kiều - Bản Phong Tình Cổ Lục (R.987 / NLVNPF-0221)

## 1. Thông tin văn bản
- **Mã ký hiệu:** R.987 (Thư viện Quốc gia Việt Nam) / NLVNPF-0221 (Nom Foundation).
- **Tên tác phẩm:** Đoạn trường tân thanh (phía trước có đề tựa *Phong tình cổ lục*).
- **Hình thức:** Bản chép tay bút lông (bản cảo thủ bút) cực kỳ sắc nét trên giấy bản cổ truyền, có dấu son chu sa chấm câu ngắt nhịp và phê bình của người xưa.
- **Quy mô:** 116 tờ đôi / 120 trang ảnh số hóa chất lượng cao (~2000 x 3000px).
- **File PDF tạo sẵn:** `TruyenKieu_PhongTinhCoLuc.pdf` (70.4 MB, 120 trang).

## 2. Về văn bản Quốc ngữ: Khung tham chiếu (`reference_kieu_1872.tsv`)
- **Vai trò file:** `reference_kieu_1872.tsv` là **Khung tham chiếu (Reference Frame)** gồm 3.239 câu thơ Lục bát chuẩn của ấn bản Kiều 1872, được giữ lại để làm cột mốc định vị câu khi gán nhãn.
- **Cảnh báo về Dị bản (Variant Alert):** File này **KHÔNG PHẢI** là bản phiên âm chuẩn 100% từng chữ của R.987. Bản chép tay R.987 là một dị bản chép tay độc lập có nhiều chữ Nôm và âm Quốc ngữ khác biệt rõ rệt so với bản in mộc bản 1872:
  - **Câu 2:**
    - R.987: 𡦂才𡦂**命**窖羅**恄**饒 (*Chữ tài chữ **mệnh** khéo là **ghét** nhau*)
    - Bản 1872: 𡦂才𡦂**色**窖󰑼󰡒僥 (*Chữ tài chữ **sắc** khéo là **cợt** nhau*)
  - **Câu 8:**
    - R.987: **cổ lục** ... sử **thanh** (古錄 ... 青)
    - Bản 1872: **có lục** ... sử **xanh** (固錄 ... 撑)
  - **Câu 18:**
    - R.987: **Mỗi người** mỗi vẻ (每𠊛每䘕)
    - Bản 1872: **Một người** một vẻ (𠬠𠊛𠬠䘕)
- **Quy trình bắt buộc khi tạo Dataset huấn luyện:**
  - Không được dùng trực tiếp `reference_kieu_1872.tsv` làm nhãn Ground Truth ký tự cho R.987.
  - Phải chạy bước **Căn chỉnh dị bản (Variant Alignment)**: Dùng bounding box ký tự Nôm cắt từ ảnh của R.987 so sánh đối chiếu với khung tham chiếu để ghi nhận đúng dị bản viết tay (chữ *mệnh* thay vì *sắc*, chữ *ghét* thay vì *cợt*).

## 3. Cấu trúc thư mục
```
data/TruyenKieuPhongTinhCoLuc/
├── pages/                           # 120 trang ảnh quét thủ bút Nôm (p001.jpg - p120.jpg)
├── TruyenKieu_PhongTinhCoLuc.pdf   # PDF 120 trang hoàn chỉnh (70.4 MB)
├── reference_kieu_1872.tsv         # Khung tham chiếu 3.239 câu (để căn chỉnh dị bản)
├── metadata.json                   # Metadata thư viện số NLVNPF
├── README_tai_ve.md               # Hướng dẫn tải về
└── SOURCE.md                        # Tài liệu này
```
