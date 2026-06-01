# Huấn luyện embedding-model chữ Nôm (thay DINOv2 cho S3)

Mục tiêu: thay DINOv2 zero-shot (đã chứng minh không dùng được —
`REPORT_dinov2_unsuitable.md`) bằng một model **học riêng** để embedding sao cho
**cùng-chữ thì gần, khác-chữ thì xa**, *bắc cầu* được giữa ảnh woodblock thật và
glyph FontDiffusion. Khi đó S3 (xếp hạng ứng viên bằng cosine) sẽ phân biệt được
→ bật lại SILVER.

---

## 1. Dữ liệu — ĐÃ CÓ SẴN, không cần gán nhãn thêm

| Nguồn | Số lượng | Vai trò |
|---|---|---|
| `dataset_out/gold/*.png` + `labels.csv` | **50.502 crop** (1.859 lớp chữ) | ảnh woodblock thật, đã gán nhãn (OCR∩từ điển) |
| `gannhanocr-fd/U+*.png` | **89.898 glyph** | ảnh tham chiếu sạch, nhãn = Unicode |

Phân bố lớp (long-tail): 854 lớp có ≥5 mẫu, 575 lớp ≥10. Chữ hiếm ít mẫu → **đây
là lý do chọn metric-learning + lấy glyph FD làm "mỏ neo" cho MỌI lớp** (kể cả lớp
0 crop vẫn có 1 glyph FD định nghĩa).

---

## 2. Nguyên lý

Học hàm `f(ảnh) → vector 256-D (L2-norm)` sao cho:
- `f(crop chữ X)` ≈ `f(FD glyph chữ X)`  (kéo gần — *bắc cầu miền* woodblock↔sạch)
- `f(chữ X)` xa `f(chữ Y≠X)`              (đẩy xa — *phân biệt chữ*)

Glyph FD đóng vai **anchor/prototype** của mỗi lớp; crop thật được kéo về anchor
của đúng chữ. Đây chính là chỗ FontDiffusion phát huy: nó cho 1 glyph sạch/ổn định
cho mọi chữ.

---

## 3. Model + Loss (khuyến nghị)

- **Backbone:** ResNet-18 (≈11,7M tham số) hoặc EfficientNet-B0. Đầu ra → FC 256,
  **L2-normalize**. (Nhẹ, thừa sức cho glyph; không cần ViT.)
- **Ảnh vào:** grayscale → 3 kênh, **128–160px** (chữ Nôm không cần 224).
- **Loss — chọn 1:**
  1. **SupCon / NT-Xent (khuyến nghị):** mỗi batch trộn crop + FD glyph; cặp dương
     = cùng chữ (kể cả crop↔FD), âm = khác chữ. Tự nhiên xử lý long-tail + open-set.
  2. **ArcFace** (head phân loại 1.859 lớp + margin): mạnh, nhưng lớp hiếm học kém
     và khó thêm chữ mới → kém hợp hơn (1).
  3. **Triplet (anchor=FD glyph, pos=crop cùng chữ, neg=crop khác chữ):** đơn giản,
     hiệu quả; dùng hard-negative mining.
- **Thư viện gợi ý:** `torch`, `torchvision`/`timm` (backbone),
  `pytorch-metric-learning` (SupCon/Triplet/miner sẵn có).

---

## 4. Augmentation (mô phỏng woodblock — quan trọng để bắc cầu miền)

Áp cho cả crop lẫn FD glyph: xoay ±5°, co giãn 0.9–1.1, dịch nhẹ, **erode/dilate**
(nét đậm/mảnh), nhiễu muối-tiêu + Gauss, mô phỏng thấm mực/đứt nét, nhị phân
ngưỡng ngẫu nhiên. Mục đích: ép embedding **bất biến với phong cách in**, chỉ giữ
**đặc trưng định danh chữ**.

---

## 5. Chia tập (tránh rò rỉ)

- **Train/val theo TRANG hoặc theo chữ**, không trộn ngẫu nhiên cùng trang.
- Giữ riêng ~50–100 chữ **không xuất hiện khi train** để đo **open-set** (model phải
  khớp được chữ mới qua glyph FD của nó).
- Val dùng để hiệu chuẩn ngưỡng cosine của SILVER.

---

## 6. Tài nguyên & chạy Kaggle P100 — **CÓ, thoải mái**

**P100 = 16 GB VRAM, ~9,3 TFLOPS FP32.** Bài toán này NHỎ so với P100.

| Hạng mục | Ước tính trên P100 |
|---|---|
| VRAM (ResNet-18, 160px, batch 256, AMP) | ~4–6 GB / 16 GB → **dư** |
| 1 epoch (≈50k crop + FD, batch 256) | **~3–5 phút** |
| Hội tụ (30–50 epoch) | **~1,5–3 giờ** |
| Quota Kaggle | 30 GPU-giờ/tuần, tối đa 12h/phiên → **xong trong 1 phiên** |

Mẹo tốc độ trên P100: **mixed precision (AMP)**, ảnh 128–160px, `num_workers=2–4`,
nén dữ liệu trước (resize sẵn về 160px, lưu .npy/.webp) để khỏi nghẽn I/O.

**Chuẩn bị cho Kaggle:**
1. Tạo **Kaggle Dataset** gồm: `gold/` (50k crop, resize 160px cho nhẹ), `labels.csv`,
   và thư mục FD glyph **chỉ các chữ xuất hiện** (~vài nghìn file, không cần cả 90k).
2. Notebook bật **Accelerator = GPU P100**.
3. `pip install timm pytorch-metric-learning` (Kaggle có sẵn torch).
4. Train → lưu `model.pt` (chỉ backbone+FC) ra Output.

---

## 7. Quy trình từng bước (checklist)

1. **Build cặp dữ liệu:** từ `labels.csv` lập danh sách (crop_path, char); map char→FD
   glyph. (Có thể viết `evaluation/ver_new/make_pairs.py`.)
2. **Dataset/Augment:** `Dataset` trả (ảnh, char_id); collate trộn crop+FD; augment §4.
3. **Model:** backbone → FC256 → L2-norm.
4. **Loss/Miner:** SupCon (hoặc Triplet + hard-miner).
5. **Train:** AdamW lr 3e-4, cosine schedule, AMP, 30–50 epoch, batch 256.
6. **Eval mỗi epoch:** **chạy lại đúng T1/T2/T3** trong `REPORT_dinov2_unsuitable.md`
   (chỉ thay `_embed` bằng model mới) — theo dõi T3 top-1 tăng.
7. **Export** `model.pt` + script `embed.py`.

---

## 8. Tiêu chí NGHIỆM THU (so trực tiếp với DINOv2)

| Test | DINOv2 (hỏng) | Model mới phải đạt |
|---|---|---|
| T1 cùng vs khác (glyph sạch) | 0,95 vs 0,90 (trùng) | tách rõ, khác < ~0,5 |
| T2 cùng vs khác (crop thật) | 0,889 vs 0,877 | chênh **≥ ~0,2** |
| **T3 retrieval top-1** | **0,0%** | **≥ 80–95%** |

Đạt 3 mốc này ⇒ S3 đáng tin ⇒ **bật lại SILVER**, hiệu chuẩn ngưỡng trên val, đo
label-error per-tier trên mẫu audit tách riêng.

---

## 9. Tích hợp vào S3 (đổi tối thiểu)

Trong `evaluation/ver_new/visual_signal.py`: thay lời gọi `DINOv2Ranker._embed`
bằng `NomEmbedder.embed` (model mới). Giữ nguyên `consensus.py` (S3 vẫn trả
top_char/cosine/margin). Bật lại bằng `build_dataset.py --use-s3`. Khi production
ổn, port `NomEmbedder` vào `core/ranking/` thay DINOv2.

---

**Tóm tắt câu hỏi của bạn:** Có, **chạy Kaggle P100 thoải mái** (~1,5–3h, dùng <6GB
trong 16GB), dữ liệu đã đủ (50k crop + 90k glyph), không cần gán nhãn thêm. Việc
chính là viết script cặp dữ liệu + train SupCon/Triplet ResNet-18 + nghiệm thu bằng
3 test đã có.
