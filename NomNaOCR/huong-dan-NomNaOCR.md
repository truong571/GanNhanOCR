# Hướng dẫn đầy đủ: Làm ra model OCR Hán-Nôm (NomNaOCR) và sử dụng

> Dành cho việc nhận dạng tài liệu Hán-Nôm chép tay/mộc bản, viết dọc, đọc phải → trái.

---

## 0. Điều cần biết trước

NomNaOCR **KHÔNG phải** một package cài xong gọi 1 dòng là chạy. Nó là một **pipeline nghiên cứu** gồm nhiều mảnh (detection + recognition + xử lý ảnh dọc). Muốn "có model để dùng", bạn phải tự lắp ráp từ code notebook + weights có sẵn.

**Nguồn:**
- Repo chính: https://github.com/ds4v/NomNaOCR
- Repo mở rộng: https://github.com/ds4v/NomNaOCRpp (NomNaOCR++)
- Dataset: https://www.kaggle.com/datasets/quandang/nomnaocr
- Weights: link Google Drive trong README của repo chính

---

## 1. Kiến trúc pipeline

Gồm 2 giai đoạn nối tiếp:

1. **Text Detection** — tìm vùng chữ trong ảnh → dùng **DBNet** (train qua hệ sinh thái PaddleOCR).
2. **Text Recognition** — đọc chữ trong từng vùng → 2 model tốt nhất:
   - `CRNNxCTC.ipynb` — Sequence Accuracy cao nhất
   - `SC-CNNxTransformer_finetune.ipynb` — Character Accuracy & CER tốt nhất

Nhận dạng ở **mức chuỗi (sequence level)**, không phải từng ký tự rời → giữ được ngữ nghĩa cả cột chữ.

---

## 2. ⚠️ Xử lý chữ VIẾT DỌC (bắt buộc với ảnh dạng này)

PPOCRLabel/DBNet mặc định detect chữ theo phương NGANG. Với tài liệu Hán-Nôm viết dọc:

1. Chạy `rotated_generator.py` → tạo ảnh đã xoay 90°.
2. Đưa ảnh xoay vào PPOCRLabel → dự đoán bounding box.
3. Chạy `unrotated_convertor.py` → xoay box về lại phương dọc.

Tùy tài liệu, xoay ±90° hoặc cả hai chiều.

---

## HAI CON ĐƯỜNG

### Con đường A — Dùng lại model có sẵn (nhanh, thử ngay)

Phù hợp khi chỉ cần đọc vài ảnh, chấp nhận độ chính xác baseline.

1. Clone repo + tải weights từ Google Drive (link trong README).
2. Detection: dùng weights DBNet của họ + bước xử lý xoay ảnh (mục 2).
3. Recognition: load weights CRNNxCTC hoặc SC-CNN×Transformer từ notebook.
4. Ghép inference, chạy trên ảnh của bạn.

**Lưu ý:** model train chủ yếu trên MỘC BẢN IN (Truyện Kiều, Lục Vân Tiên, Đại Việt Sử Ký Toàn Thư). Nếu ảnh của bạn là CHỮ VIẾT TAY THẢO THƯ, kết quả có thể kém → cân nhắc Con đường B.

---

### Con đường B — Train/Fine-tune model riêng (chậm, tốt cho chữ của bạn)

#### Bước 1 — Môi trường
- Clone repo + tải dataset Kaggle.
- Cài PaddleOCR (detection + PPOCRLabel) và TensorFlow/Keras (notebook recognition).
- Dùng GPU (Colab/Kaggle GPU nếu không có máy mạnh).

#### Bước 2 — Gán nhãn dữ liệu của bạn
- Dùng PPOCRLabel (repo đóng gói sẵn `annotators.zip` và `composer.zip`).
- Ảnh dọc: `rotated_generator.py` → gán box → `unrotated_convertor.py`.
- Kết quả: mỗi trang cắt thành các "patch" (mỗi cột = 1 dòng text) kèm nhãn.

#### Bước 3 — Train Text Detection (DBNet)
- Dùng config DBNet trong folder Text detection.
- Khởi tạo từ weights DBNet pretrained → fine-tune trên dữ liệu của bạn.

#### Bước 4 — Train Text Recognition
- Mở `CRNNxCTC.ipynb` hoặc `SC-CNNxTransformer_finetune.ipynb`.
- Chiến lược 2 giai đoạn (theo tác giả gốc):
  1. **Pretrain** trên tập Synthetic Nom String của dataset IHR-NomDB.
  2. **Fine-tune** trên dữ liệu thật (NomNaOCR + dữ liệu của bạn).
- **Quan trọng:** dictionary ký tự phải bao phủ đủ chữ Nôm trong tài liệu của bạn.

#### Bước 5 — Ghép pipeline end-to-end
Ảnh mới → xoay → DBNet detect cột chữ → cắt từng cột → recognition đọc → ghép kết quả theo thứ tự PHẢI → TRÁI.

#### Bước 6 — Đánh giá
Đo bằng Character Accuracy, Sequence Accuracy, CER trên tập validate.

---

## 3. Khuyến nghị thực tế

- **Chỉ cần đọc vài ảnh:** chạy Con đường A trước để xem baseline. Nếu kém → **fine-tune** (KHÔNG train from scratch) trên vài trăm–vài nghìn patch chữ viết tay cùng phong cách.
- **Quen PaddleOCR hơn:** cân nhắc hướng fine-tune PP-OCRv5 (bài arXiv 10/2025, arxiv.org/html/2510.04003v1) — có pipeline tiền xử lý ảnh, chuyển LMDB, fine-tune với hyperparameter tối ưu cho Hán-Nôm.
- **Ảnh chữ thảo/viết tay, scan nhiễu là trường hợp KHÓ NHẤT** — kể cả model fine-tune cũng cần nhiều dữ liệu cùng phong cách chữ mới đạt độ chính xác tốt.

---

## 4. Checklist nhanh

- [ ] Clone repo + tải weights + dataset
- [ ] Cài PaddleOCR + TensorFlow
- [ ] Thử Con đường A trên ảnh của bạn (baseline)
- [ ] Nếu kém: gán nhãn dữ liệu của bạn qua PPOCRLabel (nhớ xoay ảnh dọc)
- [ ] Fine-tune recognition (pretrain synthetic → fine-tune dữ liệu thật)
- [ ] Ghép pipeline end-to-end + xử lý thứ tự đọc phải→trái
- [ ] Đánh giá bằng CER / Sequence Accuracy
