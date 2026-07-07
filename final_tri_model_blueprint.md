# Bản Thiết kế Cuối cùng (Final Blueprint): Hệ thống Đồng thuận 3 Mô hình OCR Hán Nôm

> [!IMPORTANT]
> Đây là tài liệu thiết kế kiến trúc chuẩn và chi tiết nhất, được rà soát kỹ lưỡng dựa trên đặc thù hoạt động của từng mô hình. **Điểm mấu chốt được cập nhật:** Qwen-2.5-VL sẽ xử lý toàn bộ một trang (Full Page) để lấy ngữ cảnh, thay vì xử lý từng ảnh crop nhỏ.

---

## 1. Định hình Đội hình 3 Mô hình (Tri-Model OCR)

Hệ thống sẽ dựa vào 3 mô hình hoạt động ở 3 "tầm nhìn" khác nhau để triệt tiêu hoàn toàn điểm yếu của nhau:

1. **Model 1: Kimhannom API (Tầm nhìn Cấu trúc)**
   - **Đầu vào:** Ảnh Full Trang.
   - **Đầu ra:** Văn bản (Text) + **Toạ độ vị trí (Bounding Box)** của từng chữ.
   - **Vai trò:** Vạch ra bộ khung layout (chia cột, cắt chữ) cho toàn bộ hệ thống.

2. **Model 2: Qwen-2.5-VL qua OpenRouter (Tầm nhìn Ngữ cảnh Toàn cục)**
   - **Đầu vào:** Ảnh Full Trang.
   - **Đầu ra:** Dòng văn bản liền mạch (Chỉ Text, không có Bounding Box).
   - **Đặc điểm:** Do được xem cả trang, Qwen sẽ hiểu ngữ cảnh câu văn (Semantic Context). Nó sẽ **không bao giờ mắc lỗi đồng âm** (như Kimhannom) vì nó hiểu nghĩa của cả câu để chọn đúng chữ. Mức độ thông minh ngôn ngữ là cao nhất.

3. **Model 3: Fine-tuned PaddleOCR ch_tra (Tầm nhìn Thị giác Vi mô)**
   - **Đầu vào:** Từng Ảnh Crop nhỏ (Cắt ra từ toạ độ của Kimhannom).
   - **Đầu ra:** 1 Ký tự (Text).
   - **Đặc điểm:** Chạy offline siêu tốc. Do chỉ tập trung nhìn 1 chữ bị cắt rời, nó không bị ảo giác (hallucination) bởi ngữ cảnh. Nó đánh giá thuần tuý dựa trên đường nét (Strokes).

---

## 2. Quy trình Thực thi Chuẩn (Luồng Dữ liệu)

### Bước 1: Trích xuất Dữ liệu Đa tầng (Multi-level Extraction)
* Gửi ảnh gốc `page_X.png` lên Kimhannom API $\rightarrow$ Thu về danh sách các ký tự `[K1, K2, ..., Kn]` kèm toạ độ khung chữ (BBoxes).
* Gửi ảnh gốc `page_X.png` lên Qwen-2.5-VL API $\rightarrow$ Yêu cầu: *"Hãy nhận diện văn bản Hán Nôm trong ảnh này theo thứ tự từ phải sang trái, từ trên xuống dưới"*. Thu về chuỗi văn bản dài `[Q1, Q2, ..., Qm]`.
* Dùng toạ độ của Kimhannom, cắt ảnh thành các crop nhỏ. Đưa từng crop vào mô hình PaddleOCR (đã fine-tune) chạy offline $\rightarrow$ Thu về danh sách ký tự `[P1, P2, ..., Pn]` (Danh sách này ánh xạ 1-1 với vị trí của Kimhannom).

### Bước 2: Can chỉnh Đa chuỗi (Multi-sequence Alignment)
Do Qwen đọc cả trang nên số lượng chữ có thể lệch so với Kimhannom. Ta sử dụng thuật toán **Banded DP (Levenshtein)** đang có sẵn của dự án để can chỉnh:
* **Neo (Anchor) thứ 1:** Can chỉnh chuỗi Quốc Ngữ (Sách dịch) với chuỗi Kimhannom `[K1...n]` (Dự án đang làm rất tốt việc này).
* **Neo (Anchor) thứ 2:** Can chỉnh chuỗi Quốc Ngữ với chuỗi Qwen `[Q1...m]`.
* **Kết quả:** Tại một vị trí âm tiết Quốc Ngữ (Ví dụ âm: *"gia"*), hệ thống trích xuất được 1 hàng biểu quyết:
  $\rightarrow$ `Quốc Ngữ = "gia" | Dict(gia) = {家, 價...} | Kimhannom = 價 | PaddleOCR = 家 | Qwen-VL = 家`

### Bước 3: Luật Đồng Thuận Thép (Consensus Logic)

Đưa hàng biểu quyết trên vào hàm `decide_label`. Các tầng phân loại như sau:

> [!TIP]
> **TẦNG STRICT GOLD (Vàng nguyên chất):** 
> *Điều kiện:* `Kimhannom` == `PaddleOCR` == `Qwen-VL` VÀ Ký tự đó có nằm trong `Dict(Quốc Ngữ)`.
> *Hành động:* Tin tưởng 100%, ghi thẳng vào Dataset mức GOLD mà không cần xem lại.

> [!TIP]
> **TẦNG CONTEXT GOLD (Vàng ngữ cảnh - Sửa lỗi Kimhannom):**
> *Điều kiện:* `Kimhannom` sai (bị lỗi đồng âm), nhưng `PaddleOCR` (nhìn nét) == `Qwen-VL` (nhìn ngữ cảnh) VÀ Ký tự đó có nằm trong `Dict(Quốc Ngữ)`.
> *Hành động:* Lấy kết quả của PaddleOCR/Qwen làm nhãn, ghi vào Dataset mức GOLD. Đây chính là giá trị lớn nhất của kiến trúc 3 model này!

> [!WARNING]
> **TẦNG SILVER (Gỡ gạc bằng NomEncoder):**
> *Điều kiện:* PaddleOCR và Qwen-VL cãi nhau, hoặc chữ Nôm thuần Việt (vay mượn âm/nghĩa) từ điển không chứa.
> *Hành động:* Đánh thức mô hình `NomEncoder` nội bộ (ResNet+ArcFace) để tính Cosine Similarity của ảnh crop với thư viện ảnh chuẩn (FontDiffusion). Lấy chữ có điểm Cosine cao nhất. Nếu đạt ngưỡng an toàn, xếp vào SILVER.

> [!CAUTION]
> **TẦNG REVIEW (Bất đồng nghiêm trọng):**
> *Điều kiện:* Cả 3 model ra 3 chữ khác nhau, NomEncoder điểm Cosine thấp, hoặc không model nào khớp từ điển.
> *Hành động:* Bỏ trống nhãn, tống vào danh sách REVIEW cho chuyên gia soi thủ công.

---

## 3. Lưu ý Hạng mục: Fine-tune PaddleOCR (ch_tra) bằng dữ liệu dự án

Để mô hình **PaddleOCR (Mô hình 3)** thực sự sắc bén khi nhìn ảnh Crop, việc huấn luyện nó là cực kỳ quan trọng.

1. **Kéo Data Dự án (Cực kỳ quan trọng):**
   - Chỉ được lấy những crop đã đạt chuẩn `STRICT GOLD` (sau khi Qwen và Kimhannom đồng ý) hoặc `GOLD` gốc của dự án. Không lấy data ở tầng `REVIEW` để tránh làm bẩn tập train.
   - Chuyển file `labels.csv` thành file `train.txt` theo đúng chuẩn của PaddleOCR (Ví dụ: `dataset_out/gold/book1_page1_c01_001.png\t家`).

2. **Kéo Data Internet (Mở rộng vốn từ Nôm):**
   - Cào bộ dữ liệu **NomNaOCR** trên HuggingFace. File này có cả triệu crop ảnh chữ Nôm. Trộn 50% data dự án + 50% data NomNaOCR.

3. **Cập nhật Từ điển Ký tự (Tử huyệt):**
   - Mở file `ppocr/utils/dict/japan_dict.txt` hoặc `ch_tra_dict.txt` của PaddleOCR.
   - Thêm toàn bộ danh sách chữ Nôm thuần Việt (có trong file `Dict/QuocNgu_SinoNom_TongHop3.csv` của anh/chị) vào cuối file này.
   - Chỉnh sửa file config `.yml` của Paddle: sửa `character_dict_path` trỏ về file từ điển mới, sửa số lượng `num_classes` bằng tổng số dòng trong file.

4. **Kịch bản Huấn luyện:**
   - Dùng trọng số pre-trained `ch_PP-OCRv4_rec_train` (đã học tiếng Trung Phồn thể).
   - Huấn luyện trên máy có GPU (Kaggle/Colab). Do ảnh crop rất nhỏ (32x32 hoặc 48x48), thời gian train sẽ rất nhanh.

**TỔNG KẾT:** Với quy trình này, Qwen-VL lo việc **hiểu ngữ nghĩa cả trang**, PaddleOCR lo việc **nhìn nét từng chữ nhỏ**, và Kimhannom lo **vạch toạ độ**. Lỗ hổng "Đồng âm nhầm chữ" sẽ bị chặn đứng hoàn toàn!
