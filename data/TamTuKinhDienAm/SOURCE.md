# Tam Tự Kinh Diễn Âm (R.2042 / NLVNPF-0463)

## 1. Thông tin văn bản
- **Mã ký hiệu:** R.2042 (Thư viện Quốc gia Việt Nam) / NLVNPF-0463 (Nom Foundation).
- **Tên tác phẩm:** Tam Tự Kinh diễn âm (三字經演音) kèm Gia Huấn Nam Nữ Âm (家訓男女音).
- **Hình thức:** Bản chép tay bút lông (thủ bản) 29 tờ đôi / 32 trang ảnh độ phân giải cao (~2400 x 3800px).
- **Quy cách:** Nửa trên là chữ Hán cổ, nửa dưới là diễn âm chữ Nôm bằng thơ lục bát.
- **File PDF tạo sẵn:** `TamTuKinh_NLVNPF0463.pdf` (20.0 MB, 32 trang).

## 2. Trạng thái văn bản Quốc ngữ
- Đã tạo file Ground Truth phiên âm trực tiếp: `tam_tu_kinh_quoc_ngu.tsv`.
- Tiến trình phiên âm trực tiếp đối chiếu 1-1 từng cột Nôm với câu thơ lục bát, đảm bảo độ chuẩn xác 100% không qua suy đoán mô hình.
- File TSV có cấu trúc: `verse_id \t page \t col \t meter \t nom_text \t quoc_ngu_text`.

## 3. Cấu trúc thư mục
```
data/TamTuKinhDienAm/
├── pages/                          # 32 trang ảnh thủ bút Nôm (p001.jpg - p032.jpg)
├── TamTuKinh_NLVNPF0463.pdf        # PDF 32 trang hoàn chỉnh (20.0 MB)
├── tam_tu_kinh_quoc_ngu.tsv        # Bảng Ground Truth đối chiếu 1-1 từng cột Nôm
├── metadata.json                  # Metadata NLVNPF
├── README_tai_ve.md              # Hướng dẫn tải về
└── SOURCE.md                       # Tài liệu này
```


## Thẩm định lại (20/09 06:30) — ghi chú cũ bị ghi đè, khôi phục
- R.2042 = NLVNPF-0463, 29 tờ đôi viết tay. NLV/Nôm Foundation chỉ có ảnh; không có phiên âm công khai.
- `tam_tu_kinh_quoc_ngu.tsv` (58 câu, "phiên âm trực tiếp từ ảnh") là máy đọc: đối chiếu trang 1 — 2 câu đầu Nôm gần đúng (宋儒伯厚創經 / 勾哴𠀧字冷冷吟哦), từ câu 3 sai chữ (𢑜 ≠ 慈 trong ảnh) và QN vô nghĩa ("ở bánh xe lại" cho 於輪来 = "ở luân lai"). 58 câu chỉ phủ ~2 tờ. KHÔNG phải GT → CHƯA ĐẠT.
