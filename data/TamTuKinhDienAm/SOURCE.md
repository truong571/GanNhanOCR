# Tam Tự Kinh Diễn Âm (R.2042 / NLVNPF-0463)

## 1. Thông tin văn bản
- **Mã ký hiệu:** R.2042 (Thư viện Quốc gia Việt Nam) / NLVNPF-0463 (Nom Foundation).
- **Tên tác phẩm:** Tam Tự Kinh diễn âm (三字經演音) kèm Gia Huấn Nam Nữ Âm (家訓男女音).
- **Hình thức:** Bản chép tay bút lông (thủ bản) 29 tờ đôi / 32 trang ảnh độ phân giải cao (~2400 x 3800px).
- **Quy cách:** Nửa trên là chữ Hán cổ, nửa dưới là diễn âm chữ Nôm bằng thơ lục bát.
- **File PDF tạo sẵn:** `TamTuKinh_NLVNPF0463.pdf` (20.0 MB, 32 trang).

## 2. Trạng thái văn bản Quốc ngữ
- File .tsv mẫu tạm thời (26 câu) đã được **dọn dẹp và xóa bỏ** để giữ thư mục sạch sẽ và tránh ngộ nhận là ground-truth hoàn chỉnh.
- Bản diễn âm Nôm trong văn bản này có khoảng 196 câu thơ lục bát cộng thêm phần Gia huấn. Để phục vụ gán nhãn OCR sau này, cần thực hiện phiên âm trọn vẹn từ chính 32 trang scan thủ bút này.

## 3. Cấu trúc thư mục
```
data/TamTuKinhDienAm/
├── pages/                       # 32 trang ảnh thủ bút Nôm (p001.jpg - p032.jpg)
├── TamTuKinh_NLVNPF0463.pdf     # PDF 32 trang hoàn chỉnh (20.0 MB)
├── metadata.json               # Metadata NLVNPF
├── README_tai_ve.md           # Hướng dẫn tải về
└── SOURCE.md                    # Tài liệu này
```
