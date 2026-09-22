# GanNhanOCR — Web Presentation Dashboard
## Nền Tảng Trực Quan Hoá Quy Trình & Kết Quả Bộ Dữ Liệu OCR Hán Nôm Cổ (107k+ Ký Tự)
### Phục vụ trình diễn & bảo vệ luận văn Thạc sĩ Công nghệ Thông tin

---

## 1. Mục Đích
Ứng dụng web được thiết kế đặc biệt nhằm phục vụ buổi **Bảo vệ luận văn Thạc sĩ**, giúp Hội đồng chấm luận văn và người tham dự:
1. **Nắm bắt trực quan dòng chảy dữ liệu:** Trực quan hoá luồng 6 bước tự động 100% từ ảnh quét thô (scan) $\to$ phân đoạn cột $\to$ phát hiện hộp CenterNet $\to$ gióng hàng quy hoạch động Banded-DP $\to$ sửa lỗi nhầm hệ thống $\to$ giải cứu thông minh Self-Training $\to$ đóng gói xuất bản bộ dataset chuẩn quốc tế.
2. **Trải nghiệm soi bản thảo thực tế (Interactive Manuscript Inspector):** Khả năng soi trực tiếp từng trang sách scan thật của 4 danh tác cổ (**Lục Vân Tiên 1883**, **Kim Vân Kiều 1884**, **Chrestomathie 1872**, **Sách Thánh Truyện**), xem các khung bounding box phủ lên chữ, di chuột để xem ảnh crop phóng to, âm đọc Quốc ngữ, chữ Nôm và mã Unicode.
3. **Tra cứu & lọc mẫu ký tự nhanh:** Tra cứu hơn 107.000 mẫu chữ theo âm Quốc ngữ, chữ Nôm, xem ảnh crop thật và kiểm chứng độ sạch.
4. **Luận chứng khoa học tin cậy:** Trình bày bảng đối chiếu phương pháp đề xuất với baseline (tăng tỷ lệ đếm đúng số chữ lên >78-90%, giảm 4 lần tỷ lệ cắt vào thân chữ, 18/18 invariants kiểm định toàn vẹn PASS).

---

## 2. Hướng Dẫn Khởi Chạy Nhanh

### Cách 1: Khởi chạy máy chủ đầy đủ (Khuyên Dùng khi trình diễn trên máy)
Sử dụng thư viện chuẩn của Python (không cần cài thêm bất kỳ package nào):

```bash
# Từ thư mục gốc của repository GanNhanOCR:
python3 web/server.py
```
Hoặc chỉ định cổng tùy chọn:
```bash
python3 web/server.py 8080
```

Mở trình duyệt truy cập: **`http://localhost:8080`**

*Tính năng khi chạy máy chủ:*
- Kết nối trực tiếp vào toàn bộ 107.786 dòng nhãn thật trong `dataset/`.
- Phục vụ ảnh scan trang HD gốc từ `prepared/<Book>/pages/`.
- Phục vụ ảnh crop ký tự thật từ `dataset/<Book>/gold/` và `dataset/<Book>/syllable/`.
- Tìm kiếm tức thời theo mọi từ khoá Quốc ngữ / Hán Nôm trên toàn bộ cơ sở dữ liệu.

---

### Cách 2: Khởi chạy độc lập / Ngoại tuyến (Offline / Standalone)
Trong trường hợp phòng bảo vệ không có môi trường chạy Python hoặc cần mở nhanh:
- Mở trực tiếp tệp `web/index.html` bằng trình duyệt (Chrome, Safari, Edge, Firefox).
- Ứng dụng tự động kích hoạt chế độ **Demo Standalone** sử dụng cơ sở dữ liệu tuyển chọn được đóng gói sẵn trong `web/sample_data.json`.
- Mọi tính năng chuyển Tab, soi trang bản thảo với bounding box, tra cứu mẫu chữ và bảng số liệu đều hoạt động mượt mà 100%.

---

## 3. Cấu Trúc Thư Mục `web/`

```
web/
├── index.html            # Giao diện đơn trang (Single Page App) hiện đại
├── style.css             # Hệ thống thiết kế học thuật, Dark/Light mode, Noto Serif / Inter
├── app.js                # Toàn bộ logic tương tác: BBox Inspector, Stepper, Gallery, Fallback
├── server.py             # Máy chủ Python nội bộ phục vụ API và static assets
├── sample_data.json      # Cơ sở dữ liệu mẫu độc lập (dành cho chế độ offline)
└── README.md             # Tài liệu hướng dẫn sử dụng này
```

---

## 4. Các Điểm Nhấn Nên Trình Bày Trước Hội Đồng
1. **Chỉ số ấn tượng:** Chỉ ra thanh KPI Banner đầu trang: **107.786 ký tự**, **4 bộ tác phẩm**, **100% tự động**, **0 quy tắc can thiệp thủ công**.
2. **Soi bản thảo:** Vào tab **"Soi Bản Thảo Trực Tiếp"**, chọn *Lục Vân Tiên* hoặc *Kim Vân Kiều*, di chuột qua các câu thơ lục bát để chứng minh thuật toán phân tách ranh giới câu 6/8 hoàn hảo, các hộp chữ ôm sát nét mực.
3. **Tra cứu âm Quốc ngữ:** Vào tab **"Tra Cứu Ký Tự Dataset"**, gõ chữ *"kieu"*, *"tien"*, *"thang"*, *"ve"* để cho Hội đồng thấy chất lượng các ảnh crop đạt chuẩn GOLD không bị đứt nét hay loang viền.
4. **Đối chiếu số liệu:** Vào tab **"Luận Chứng & Số Liệu"** để bảo vệ luận điểm về việc giảm tỷ lệ cắt phạm thân chữ từ 10.6% xuống 2.6% nhờ Pitch Decoding.
