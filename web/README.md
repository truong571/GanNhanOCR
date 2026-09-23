# GanNhanOCR — Công cụ minh họa và trực quan hóa dữ liệu
## Đề tài Luận văn Thạc sĩ Công nghệ Thông tin
### Hệ thống hỗ trợ gán nhãn tự động văn bản Hán Nôm cổ (111.525 ký tự)

---

## 1. Mục đích ứng dụng
Ứng dụng web được xây dựng nhằm hỗ trợ buổi báo cáo bảo vệ Luận văn Thạc sĩ, giúp Hội đồng và người tham dự dễ dàng theo dõi:
1. **Quy trình xử lý dữ liệu (6 giai đoạn):** Thể hiện tuần tự các bước từ ảnh quét tài liệu gốc $\to$ tiền xử lý khử nhiễu $\to$ định vị ký tự bằng CenterNet & Pitch Decoding $\to$ gióng hàng song ngữ bằng quy hoạch động Banded-DP $\to$ kiểm kê & hiệu chỉnh lỗi $\to$ kiểm soát biên & đối soát dị bản $\to$ đóng gói tập ngữ liệu chuẩn.
2. **Trực quan hóa bản quét (Manuscript Inspector):** Trực tiếp hiển thị các trang tài liệu gốc độ nét cao của 4 tác phẩm tiêu biểu (*Lục Vân Tiên 1883*, *Kim Vân Kiều 1884*, *Chrestomathie 1872*, *Sách Thánh Truyện*), phủ tọa độ các hộp bao ký tự theo màu quy ước mức tin cậy, xem chi tiết ảnh trích xuất, chữ Nôm, âm đọc Quốc ngữ và mã Unicode.
3. **Tra cứu mẫu ký tự:** Tìm kiếm tức thời theo âm đọc hoặc tự dạng chữ Nôm trên tập dữ liệu thực nghiệm hơn 111.000 ký tự.
4. **Kết quả đánh giá thực nghiệm:** Trình bày bảng so sánh định lượng với phương pháp cơ sở (tỷ lệ đếm đúng số chữ trên cột đạt 78.4% – 90.0%, giảm 4 lần tỷ lệ cắt phạm vào nét chữ, vượt qua toàn bộ 18 tiêu chí kiểm tra tính toàn vẹn dữ liệu).

---

## 2. Hướng dẫn khởi chạy

### Cách 1: Khởi chạy máy chủ API (Khuyên dùng khi báo cáo trên máy tính cá nhân)
Sử dụng thư viện chuẩn của Python (`http.server.ThreadingHTTPServer`), không cần cài đặt thêm thư viện ngoài:

```bash
# Từ thư mục gốc của dự án:
.venv/bin/python web/server.py 8088
```

Mở trình duyệt truy cập: **`http://localhost:8088`**

*Ưu điểm:*
- Kết nối trực tiếp vào toàn bộ 111.525 bản ghi nhãn trong thư mục `dataset/`.
- Phục vụ ảnh quét trang gốc và ảnh trích xuất ký tự độ phân giải cao.
- Tốc độ tra cứu tức thời nhờ nạp sẵn chỉ mục vào bộ nhớ RAM.

---

### Cách 2: Mở trực tiếp tập tin tĩnh (Khi trình chiếu tại phòng máy không có Python)
- Sao chép toàn bộ thư mục `web/` sang USB hoặc máy trình chiếu.
- Nhấp đúp mở trực tiếp tệp `web/index.html` bằng trình duyệt web bất kỳ (Chrome, Safari, Edge, Firefox).
- Ứng dụng tự động chuyển sang chế độ dữ liệu mẫu sử dụng tệp `web/sample_data.json` được gói sẵn.

---

## 3. Cấu trúc thư mục `web/`

```
web/
├── index.html            # Giao diện chính (chuẩn mực học thuật, hỗ trợ Sáng / Tối)
├── style.css             # Định dạng phong cách tài liệu khoa học, phông chữ Noto Serif / Inter
├── app.js                # Logic điều khiển: Trực quan hóa bản quét, Stepper quy trình, Tra cứu
├── server.py             # Máy chủ HTTP / REST API nội bộ bằng thư viện chuẩn Python
├── sample_data.json      # Cơ sở dữ liệu mẫu độc lập cho chế độ ngoại tuyến
└── README.md             # Tài liệu thuyết minh này
```

---

## 4. Các điểm nhấn khi thuyết minh trước Hội đồng
1. **Quy mô và tính tự động:** Nêu bật bảng tổng hợp số liệu thực nghiệm với 111.525 ký tự được gán nhãn hoàn toàn theo quy tắc thuật toán, không can thiệp thủ công (`quyet_dinh_nguoi = 0`).
2. **Khả năng định vị ký tự:** Chuyển sang mục **"3. Trực quan hóa bản quét"**, chọn một trang bất kỳ của *Lục Vân Tiên* hoặc *Kim Vân Kiều* để minh họa giải thuật giải mã nhịp (Pitch Decoding) giúp phân tách chính xác các vị trí chữ dính mực mà không cắt phạm vào nét chữ.
3. **Chất lượng dữ liệu trích xuất:** Vào mục **"4. Tra cứu mẫu ký tự"**, tra cứu các âm phổ biến (ví dụ: *kieu*, *tien*, *troi*, *nguoi*) để minh chứng chất lượng ảnh cắt ký tự mức 1 (GOLD).
4. **Đánh giá thực nghiệm:** Vào mục **"5. Kết quả & Đánh giá"** để trình bày các chỉ số cải thiện so với phương pháp cơ sở và tính toàn vẹn của tập dữ liệu giao nộp.
