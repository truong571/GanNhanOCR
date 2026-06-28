# train_crop — Bộ ĐỊNH VỊ & TÁCH CHỮ Hán–Nôm dính nhau (CenterNet anchorless)

Phát hiện **tâm chữ** (anchorless center-point detection, ResNet34 + FPN, tuỳ chọn DCNv2)
để cắt từng ký tự trên trang Hán–Nôm ván khắc/viết tay, rồi **ràng buộc về đúng N ký tự**
(N = số âm tiết Quốc ngữ đã biết của cột) bằng prune-by-confidence + **Seam Carving**.

> Đây là bản hoàn thiện của module "Bước 2 (CenterNet)" trong lộ trình luận văn. Quy trình
> chuẩn: **pretrain trên MTH/TKH (chữ Hán ván khắc, ~1.08M box) → fine-tune trên Nôm (445 trang)**.

---

## 1. Vì sao center-point tách được chữ dính
Mỗi ký tự = **một điểm** (tâm). Hai tâm kề nhau luôn tách biệt về toạ độ dù nét mực dính tới
đâu → không cần anchor box, không NMS-IoU (vốn dễ gộp nhầm 2 chữ dính). Tách hộp = **Max-Pool
3×3** tìm cực trị địa phương trên heatmap.

**Kiến trúc:** Ảnh → ResNet34(ImageNet) → FPN (P2 ở stride 4) → 3 head:
Heatmap(1, sigmoid, bias −2.19) · Size(2:[w,h]) · Offset(2:[dx,dy]).
**Loss** = FocalLoss(heatmap) + 0.1·L1(size) + 1.0·L1(offset).

---

## 2. Cấu trúc file
| File | Vai trò |
|------|---------|
| `data_centernet.py` | Dataset + sinh nhãn Gaussian/size/offset; đọc VOC-XML, JSON manifest, **MTH/TKH .txt** (`read_mth_items`) |
| `model_centernet.py` | ResNet34 + FPN, DCNv2 tuỳ chọn, 3 head |
| `train_centernet.py` | Loss + train/validate (AMP/AdamW/CosineLR), lưu best theo VAL F1, đẩy HuggingFace |
| `infer_centernet.py` | Ràng buộc N + **Seam Carving** + crop; adapter pipeline (`boxes_for_page`, `column_boxes`, `make_valley_split`) |
| `build_manifest.py` | Dựng manifest train từ `labels.csv` theo tier (GOLD/SILVER/SYLLABLE) |
| `build_mth_pretrain.py` | Đóng gói MTH/TKH (downscale + scale box) → `mth_manifest.json` |
| `pack_for_kaggle.py` | Đóng gói code + ảnh → `kaggle_pkg/` để upload Kaggle |
| `kaggle_train.ipynb` | Notebook train trên Kaggle GPU (pretrain MTH → fine-tune Nôm) |
| `KAGGLE.md` | Hướng dẫn Kaggle chi tiết |
| `detector_r34.best.pt` | Ckpt demo (train local 20 epoch, VAL F1 ≈ 0.68) |
| `ket_qua_centernet.pdf` | PDF kết quả minh hoạ |

---

## 3. Cài đặt
Dùng `.venv` của repo (torch ≥ 2.x, torchvision, opencv, matplotlib, reportlab). Kiểm tra nhanh:
```bash
.venv/bin/python -c "import torch, cv2, torchvision; print('ok', torch.__version__)"
```
Trọng số ResNet34 ImageNet tải tự động (cache torch hub) — chạy offline được nếu đã cache.

---

## 4. Chạy thử nhanh (không cần GPU/dữ liệu)
```bash
.venv/bin/python train_crop/data_centernet.py  --selftest
.venv/bin/python train_crop/model_centernet.py --selftest
.venv/bin/python train_crop/train_centernet.py --smoke
.venv/bin/python train_crop/infer_centernet.py --smoke
```

---

## 5. Dữ liệu
| Nguồn | Vị trí | Dùng cho |
|-------|--------|----------|
| Nôm (445 trang, bbox + tier) | `evaluation/ver_new/dataset_out/labels.csv` + `prepared/*/pages/*.png` | fine-tune |
| MTH/TKH (3199 trang, 1.08M box) | `MTH/TKHMTH2200/{MTH1000,MTH1200,TKH}` (repo root) | pretrain |
| Kuzushiji-Kanji (140k ảnh, 3832 lớp) | `data/kkanji2/` | pretrain Recognizer (Bước 4–5) |

Manifest Nôm có sẵn: `evaluation/ver_new/char_detector/detect_manifest.json` (≈66k box).
Dựng lại theo tier khác: `build_manifest.py --tiers GOLD,SILVER,SYLLABLE`.

---

## 6. Huấn luyện

### 6a. Local (macOS MPS / CPU — chỉ để thử, chậm)
```bash
.venv/bin/python train_crop/train_centernet.py \
    --manifest evaluation/ver_new/char_detector/detect_manifest.json \
    --img 768 --epochs 12 --batch 4 --out train_crop/detector_r34.pt
```

### 6b. Kaggle GPU (KHUYẾN NGHỊ — xem KAGGLE.md)
1. Đóng gói (tạo `train_crop/kaggle_pkg/` + ảnh MTH downscale):
   ```bash
   .venv/bin/python train_crop/pack_for_kaggle.py            # code + ảnh Nôm
   .venv/bin/python train_crop/build_mth_pretrain.py         # + mth_manifest + mth_images
   (cd train_crop/kaggle_pkg && zip -rq ../kaggle_pkg.zip .)
   ```
2. Upload **1 file** `train_crop/kaggle_pkg.zip` lên Kaggle = 1 Dataset (title hợp lệ, vd `nom char det r34`).
3. Notebook GPU **T4**, Internet **On**, Secret `HF_TOKEN`. Import `kaggle_train.ipynb` → sửa `<user>`/`mdnt571` → **Save Version → Commit**.
   - Cell A: **pretrain MTH 35 epoch** → `mthv2_pretrain.best.pt`.
   - Cell B: **fine-tune Nôm** `--img 1024 --lr 5e-5 --dcn`, warm-start từ pretrain, đẩy HF.

**Đạt:** mỗi epoch in `VAL F1 .. cnt-err ..`; mục tiêu **F1 ≥ ~0.85, cnt-err ~0**.

---

## 7. Suy luận 1 cột → đúng N crop
```bash
.venv/bin/python train_crop/infer_centernet.py \
    --ckpt train_crop/detector_r34.best.pt --image cot.png --n 9 \
    --split seam --carve --out crops_out
```

## 8. Sinh PDF kết quả
```bash
.venv/bin/python train_crop/make_report_pdf.py \
    --ckpt train_crop/detector_r34.best.pt \
    --manifest evaluation/ver_new/char_detector/detect_manifest.json \
    --out train_crop/ket_qua_centernet.pdf
```

## 9. Tích hợp pipeline chính (align_production --reseg detector)
`CenterNetDetector` tương thích `evaluation/ver_new/char_detector/detector_infer.py`:
```python
from infer_centernet import CenterNetDetector
det = CenterNetDetector("detector_r34.best.pt", split_method="seam")
page_boxes = det.boxes_for_page(page_bgr)                              # detector 1 lần/trang
col = det.column_boxes(page_boxes, x_range, N, gray_image=page_gray)   # → đúng N hộp [x1,y1,x2,y2]
```
Hoặc dùng seam làm callback cho `count_constrained.constrain_to_count`:
```python
from infer_centernet import make_valley_split
constrain_to_count(col_boxes, N, valley_split=make_valley_split(page_gray, "seam"))
```

---

## 10. Ghi chú kỹ thuật
- Output stride = 4; kích thước ảnh chia hết 32.
- AMP chỉ bật trên CUDA; MPS/CPU chạy fp32.
- DCNv2 (`--dcn`) tự fallback Conv thường nếu torchvision không có `DeformConv2d`.
- **P100 KHÔNG chạy được** với torch mới của Kaggle (sm_60) → dùng **T4**.
- Tham chiếu: Objects as Points (Zhou 2019); HRCenterNet (IEEE BigData 2020); TKH/MTHv2 (Ma ICFHR 2020).
