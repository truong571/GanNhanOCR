# LAB NGHIÊN CỨU NÂNG CAO: ENHANCED SELF-TRAINING V3
## Kết Hợp Phương Án 1 (Two-Stage SE-ResNet) & Phương Án 2 (Stroke Enhancement & Calligraphy Augmentation)

**Thư mục:** `/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/enhanced_self_training_v3`

---

## 1. Đặt Vấn Đề & Động Lực Kết Hợp

Hạn chế lớn nhất của bài toán gán nhãn tự động Hán Nôm chép tay là **sự không tương thích miền (Domain Gap)** của các mô hình OCR tổng quát bên ngoài:
1. **Về mặt dữ liệu (Data Domain)**: OCR ngoài được huấn luyện trên mộc bản (ván khắc gỗ) hoặc bản in hiện đại. Chúng hoàn toàn xa lạ với thể viết tay hành-thảo, mực loang, và độ phản quang của giấy điệp/giấy sắc thế kỷ 19.
2. **Về mặt mô hình (Model Architecture)**: Các mô hình OCR ngoài thiếu cơ chế chú ý (Channel Attention) để khu biệt các nét chấm, nét phẩy, bộ thủ đặc thù của chữ Nôm, dễ gây ra hiện tượng *Correlated Error* (nắn chữ Nôm về chữ Hán in gần hình).

👉 **Giải pháp đột phá: Kết hợp đồng thời Phương án 1 và Phương án 2:**
- **Phương án 1 (Two-Stage SE-ResNet)**: Tận dụng kết quả giải cứu từ Vòng 1 để mở rộng tập huấn luyện lên **54.094 ảnh sạch** (1.331 lớp chữ Nôm). Thiết kế mạng `EnhancedNomOCRNet` tích hợp Residual Blocks và Squeeze-and-Excitation (SE) Attention.
- **Phương án 2 (Stroke Enhancement & Augmentation)**: Tích hợp module xử lý hình thái học mô phỏng các hiệu ứng ngọn bút lông:
  - Tăng tương phản thích nghi (CLAHE) cân bằng nền giấy cổ.
  - Phép *Dilation* ngẫu nhiên mô phỏng ngọn bút đẫm mực, nét dày loang.
  - Phép *Erosion* ngẫu nhiên mô phỏng ngọn bút lướt nhanh, mực khô xước nét (khát bút).
  - Biến dạng góc viết & đàn hồi mô phỏng sự phóng khoáng của thể hành-thảo.

---

## 2. Cấu Trúc Thư Mục Lab

| Tệp | Mô tả |
|---|---|
| **`prepare_expanded_labels.py`** | Script tự động hợp nhất 51.922 nhãn GOLD và 2.172 nhãn RESCUED -> 54.094 nhãn sạch. |
| **`labels_expanded.csv`** | Bảng nhãn mở rộng 83.239 dòng (54.094 dòng `is_train_v2=True`). |
| **`train_enhanced_ocr.py`** | Script PyTorch huấn luyện 20 epochs, tích hợp `StrokeEnhancer` và `EnhancedNomOCRNet`. |
| **`pack_for_kaggle.py`** | Script đóng gói thành tệp zip tải lên Kaggle. |
| **`enhanced_ocr_kaggle.zip`** | *(78.15 MB)* Gói dữ liệu nén sẵn sàng kéo thả lên Kaggle. |
| **`kaggle_run_enhanced_pipeline.ipynb`** | Notebook Jupyter chạy 1-click trên Kaggle GPU (Tesla T4/P100) trong ~10 phút. |
| **`evaluate_enhanced.py`** | Script đánh giá kết quả sau khi tải `enhanced_results.zip` từ Kaggle về. |
| **`enhanced_preview.html`** | Báo cáo trực quan kèm ảnh crop chữ viết tay bút lông thực tế. |

---

## 3. Hướng Dẫn Chạy Trên Kaggle GPU (3 Bước Đơn Giản)

### Bước 1: Tạo Dataset trên Kaggle
1. Vào [Kaggle Datasets](https://www.kaggle.com/datasets) -> Bấm **New Dataset**.
2. Kéo thả tệp `enhanced_ocr_kaggle.zip` vào, đặt tên là `enhanced-nom-ocr` rồi bấm **Create**.

### Bước 2: Chạy Notebook
1. Vào [Kaggle Code](https://www.kaggle.com/code) -> Bấm **New Notebook**.
2. Bấm **File -> Import Notebook** -> Chọn tệp:
   `lab/enhanced_self_training_v3/kaggle_run_enhanced_pipeline.ipynb`
3. Ở panel bên phải (**Input**): Bấm **Add Input** -> Chọn dataset `enhanced-nom-ocr` vừa tạo.
4. Ở panel bên phải (**Settings**): Mục **Accelerator** chọn **GPU T4 x2** (hoặc P100).
5. Bấm **Run All** (thời gian chạy ~10-12 phút).

### Bước 3: Tải Kết Quả & Xem Báo Cáo Cục Bộ
1. Sau khi chạy xong, ở panel bên phải (**Output**), tải tệp **`enhanced_results.zip`** về.
2. Đặt tệp đó vào thư mục:
   `/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/lab/enhanced_self_training_v3/enhanced_results.zip`
3. Chạy lệnh đánh giá trên terminal Mac:
   ```bash
   .venv/bin/python lab/enhanced_self_training_v3/evaluate_enhanced.py
   ```
4. Mở tệp `lab/enhanced_self_training_v3/enhanced_preview.html` trên trình duyệt để kiểm tra ảnh crop và số liệu!
