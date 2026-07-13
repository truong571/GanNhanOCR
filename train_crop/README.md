# train_crop — Định vị & tách chữ Hán–Nôm dính nhau (CenterNet anchorless)

Phát hiện **tâm chữ** (anchorless center-point detection, ResNet34 + FPN, tuỳ chọn DCNv2)
để cắt từng ký tự trên trang Hán–Nôm ván khắc/viết tay, rồi **ràng buộc về đúng N ký tự**
(N = số âm tiết Quốc ngữ đã biết của cột) bằng prune-by-confidence + **Seam Carving**.

Quy trình: **pretrain MTH/TKH (chữ Hán ván khắc, ~1.08M box) → fine-tune trên Nôm (445 trang)**.
Kết quả tham chiếu: VAL box-F1 ≈ **0.84**, recall ≈ 0.94; crop sau ràng buộc N **~99% sạch**.

## 1. Vì sao center-point tách được chữ dính
Mỗi ký tự = **một điểm** (tâm). Hai tâm kề nhau luôn tách biệt về toạ độ dù nét mực dính tới
đâu → không cần anchor box, không NMS-IoU. Tách hộp = **Max-Pool 3×3** tìm cực trị heatmap.

**Kiến trúc:** Ảnh → ResNet34(ImageNet) → FPN (P2 ở stride 4) → 3 head:
Heatmap(1, sigmoid, bias −2.19) · Size(2:[w,h]) · Offset(2:[dx,dy]).
**Loss** = FocalLoss(heatmap) + 0.1·L1(size) + 1.0·L1(offset).

## 2. File
| File | Vai trò |
|------|---------|
| `data_centernet.py` | Dataset + nhãn Gaussian/size/offset + augmentation; đọc VOC-XML, JSON, **MTH/TKH .txt** |
| `model_centernet.py` | ResNet34 + FPN, DCNv2 tuỳ chọn, 3 head |
| `train_centernet.py` | Loss + train/validate (AMP/AdamW/CosineLR), lưu best theo VAL F1, đẩy HuggingFace |
| `infer_centernet.py` | Ràng buộc N + Seam Carving + crop; adapter pipeline (`boxes_for_page`, `column_boxes`, `make_valley_split`) |
| `build_manifest.py` | Dựng manifest train từ `labels.csv` theo tier (GOLD/SILVER/SYLLABLE) |
| `build_mth_pretrain.py` | Đóng gói MTH/TKH (downscale + scale box) → `mth_manifest.json` |
| `pack_for_kaggle.py` | Đóng gói code + ảnh → `kaggle_pkg/` để upload Kaggle |
| `kaggle_train.ipynb` | Notebook train trên Kaggle GPU (pretrain MTH → fine-tune Nôm) |

> Model (`detector_r34.best.pt`) và dữ liệu KHÔNG nằm trong folder này: tải model từ
> HuggingFace, dữ liệu xem mục 5.

## 3. Cài đặt & chạy thử
```bash
.venv/bin/python train_crop/data_centernet.py  --selftest
.venv/bin/python train_crop/model_centernet.py --selftest
.venv/bin/python train_crop/train_centernet.py --smoke
.venv/bin/python train_crop/infer_centernet.py --smoke
```

## 4. Huấn luyện trên Kaggle GPU (khuyến nghị)
> Model v1 ĐÃ train sẵn (F1 0.84) và wire vào pipeline — chỉ làm mục này nếu muốn train lại.

1. **Gói upload** đã dựng sẵn: `nom-char-det-r34_kaggle.zip` (ở repo root, ~1 GB).
   Dựng lại từ đầu nếu cần:
   ```bash
   .venv/bin/python train_crop/pack_for_kaggle.py        # code + ảnh Nôm
   .venv/bin/python train_crop/build_mth_pretrain.py     # + mth_images + mth_manifest
   (cd train_crop/kaggle_pkg && zip -rq ../../nom-char-det-r34_kaggle.zip .)
   ```
2. **Upload** `nom-char-det-r34_kaggle.zip` lên Kaggle = 1 Dataset (title hợp lệ, vd `nom char det r34`).
3. Notebook GPU **T4** (⚠️ KHÔNG P100), Internet **On**, Secret `HF_TOKEN`. Import
   `kaggle_train.ipynb` → **Save Version → Commit**:
   - Cell A: pretrain MTH (tự bỏ qua nếu đã có trên HF).
   - Cell B: fine-tune Nôm `--img 1024 --lr 2.5e-4 --dcn` (cấu hình chốt; img 1280/aug-mạnh đã thử,
     KHÔNG tốt hơn — model plateau ~0.84).

**Đạt** khi VAL F1 ≥ ~0.84 (trần thật do GT thiếu tier REVIEW; crop sau ràng buộc N ~99.5% sạch).

**Local (chỉ để thử, chậm):**
```bash
.venv/bin/python train_crop/train_centernet.py \
    --manifest train_crop/detect_manifest.json \
    --img 768 --epochs 12 --batch 4 --out detector.pt
```

## 5. Dữ liệu (ở ngoài folder này)
| Nguồn | Vị trí | Dùng cho |
|-------|--------|----------|
| Nôm (445 trang, bbox+tier) | `dataset_out/labels.csv` + `prepared/*/pages/*.png` | fine-tune |
| Manifest Nôm dựng sẵn | `train_crop/detect_manifest.json` | fine-tune |
| MTH/TKH (3199 trang) | `MTH/TKHMTH2200/{MTH1000,MTH1200,TKH}` (repo root) | pretrain |

## 6. Suy luận + tích hợp pipeline
Tải model rồi cắt 1 cột thành đúng N crop:
```bash
huggingface-cli download mdnt571/nom-char-det detector_r34.best.pt --local-dir train_crop/
.venv/bin/python train_crop/infer_centernet.py --ckpt train_crop/detector_r34.best.pt \
    --image cot.png --n 9 --split seam --carve --out crops_out
```
Ghép vào `align_production --reseg detector` (tương thích `detector_infer.py`):
```python
from infer_centernet import CenterNetDetector
det = CenterNetDetector("detector_r34.best.pt", split_method="seam")
page_boxes = det.boxes_for_page(page_bgr)
col = det.column_boxes(page_boxes, x_range, N, gray_image=page_gray)   # → đúng N hộp [x1,y1,x2,y2]
```

## 7. Ghi chú
- Output stride = 4; ảnh chia hết 32. AMP chỉ bật trên CUDA. DCNv2 tự fallback Conv nếu không có.
- **P100 không chạy được** với torch mới của Kaggle (sm_60) → dùng **T4**.
- Công cụ đánh giá/so sánh model nằm ở `train_crop_eval/` (ngoài folder này).
- Tham chiếu: Objects as Points (Zhou 2019); HRCenterNet (BigData 2020); TKH/MTHv2 (Ma ICFHR 2020).
