# Huấn luyện encoder Nôm trên Kaggle (ResNet + ArcFace, embedding 256-D)

Encoder này sinh embedding cho tầng **S3** (so khớp crop viết tay ↔ glyph tham chiếu).
File chạy: **`kaggle_train.py`** — một file self-contained (gộp model + ArcFace +
dataset + train loop + push HuggingFace), checkpoint tương thích với `infer.py`/`visual_signal.py`.

---

## 1. Có cần train trên các ảnh tương đồng đã sinh không?  → **CÓ, bắt buộc.**

Ảnh tương đồng (`gannhanocr-fd`) **không phải tùy chọn** — chúng là phần lõi của việc học,
vì 3 lý do (đều đo được):

1. **Cầu nối miền (quan trọng nhất).** ArcFace kéo embedding của *crop thật* và *glyph
   tương đồng* CÙNG một chữ về gần nhau. Nhờ đó lúc suy luận, so crop với glyph tham chiếu
   mới có nghĩa. Không train cùng glyph → tham chiếu glyph gần như vô dụng (DINOv2 zero-shot:
   retrieval **0%**).
2. **Phủ đuôi dài / singleton.** Mỗi lớp chữ được bảo đảm ≥1 mẫu kể cả chữ hiếm ít/không có
   crop thật.
3. **Mở open-set / zero-shot.** Khả năng nhận diện chữ chưa-thấy đến từ việc encoder đã học
   embedding glyph.

**Bằng chứng (đo trên dự án này):**

| Cấu hình tham chiếu | retrieval@1 (VAL) |
|---|---:|
| glyph-only (chỉ ảnh tương đồng) | **82,5%** |
| crop thật | 90,2% |
| kết hợp | 89,4% |
| Chữ CHƯA train (char-disjoint), glyph-only | **83,8%** |

→ Glyph tương đồng một mình đã đạt 82–84% (DINOv2 = 0%). Muốn **chứng minh** điều này, chạy
ablation: `kaggle_train.py --exclude-glyphs` (train crop-only) rồi so T2/T3 — tầng glyph sẽ
sụp. **Khuyến nghị: GIỮ glyph trong train (mặc định).**

> Cách trộn dữ liệu (đã làm sẵn trong `prepare_data.py`): mỗi lớp = các crop GOLD thật
> (giữ split train/val/test) + **1 glyph `fd` tương đồng** (luôn `split=train`) + (tùy chọn)
> glyph đa-font. Train tất cả CÙNG nhau dưới ArcFace.

---

## 2. Ví dụ chạy đầy đủ (4 bước)

### Bước A — LOCAL: dựng index + đóng gói cho Kaggle
```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
.venv/bin/python evaluation/ver_new/nom_classifier/prepare_data.py     # -> index.csv, classes.json
.venv/bin/python evaluation/ver_new/nom_classifier/pack_for_kaggle.py  # -> kaggle_pkg/
```
Tạo `kaggle_pkg/` gồm: `images/{crop,fd,font}/*.png` + `index.csv` + `classes.json` + `kaggle_train.py`.

### Bước B — Upload `kaggle_pkg/` lên Kaggle dưới dạng **một Dataset** (vd tên `nom-embed-data`).

### Bước C — KAGGLE Notebook (GPU **P100**, Internet **ON**, Add-ons → Secrets: đặt **`HF_TOKEN`**)
```python
!cp /kaggle/input/nom-embed-data/kaggle_pkg/kaggle_train.py .
!python kaggle_train.py --epochs 40 --arch resnet34 --img 160 \
        --hf-repo <user>/nom-embed --resume
```
- `--root` tự dò mount Kaggle (tìm `index.csv`); không cần chỉ tay.
- Mỗi epoch đẩy `best.pt`/`last.pt` lên `<user>/nom-embed` → **không mất khi Kaggle reset 12h**;
  chạy lại cùng lệnh (có `--resume`) là tiếp tục.
- ResNet34 @160px, batch 256, AMP ≈ **2–4 giờ / 40 epoch** trên P100.

### Bước D — LOCAL: kéo checkpoint về cho S3
```bash
huggingface-cli download <user>/nom-embed best.pt --local-dir nom-embed/
# rồi: build_dataset.py --use-s3 (S3 tự dùng nom-embed/best.pt)
```

---

## 3. Kết quả mong đợi (in ở cuối log)

```
epoch 40/40 loss=1.83 val_top1=0.94 (210s)
done. best val_top1=0.94 -> /kaggle/working/best.pt
ACCEPTANCE (test): T2 separation +0.29 (same 0.80/diff 0.51) ·
  T3 crop->FD retrieval 76.5% on 200 chars  (cần T2>=~0.20, T3>=~0.80; DINOv2 = +0.01 / 0%)
```
**Đạt** khi `T2 separation ≥ ~0.20` và `T3 ≥ ~0.80` (DINOv2: +0.01 / 0% → không dùng được).

---

## 4. Tham số hữu ích

| Cờ | Ý nghĩa |
|---|---|
| `--arch resnet18\|resnet34\|resnet50` | backbone (mặc định resnet34; resnet18 nhẹ hơn) |
| `--img 128\|160` | cỡ ảnh vào (160 cho chất lượng, 128 cho tốc độ) |
| `--epochs 40 --batch 256` | P100 vừa đủ |
| `--no-pretrained` | train from scratch khi Kaggle tắt Internet (không tải ImageNet weights) |
| `--resume` | tiếp tục từ `last.pt` (local hoặc kéo từ `--hf-repo`) |
| `--exclude-glyphs` | **ablation**: train crop-only để chứng minh glyph cần thiết |

Checkpoint lưu các khóa `backbone / embed_dim / img / arch / classes` — `infer.py` và
`visual_signal.py` nạp trực tiếp, không cần chỉnh.
