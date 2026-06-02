# nom_classifier — Embedding model chữ Nôm (thay DINOv2 cho S3)

Train một embedding-model **phân biệt được chữ Nôm** (ResNet-18 + ArcFace) trên
cặp *crop GOLD ↔ glyph FontDiffusion cùng chữ*, để **bật lại tầng SILVER** trong
pipeline. DINOv2 zero-shot đã chứng minh KHÔNG dùng được
(`../REPORT_dinov2_unsuitable.md`: cosine 0,91 giữa 2 chữ khác nhau, retrieval 0%).

## File
| File | Vai trò |
|---|---|
| `prepare_data.py` | Dựng `index.csv` (path,label,unicode,split,source) + `classes.json` từ `dataset_out/labels.csv` (crop GOLD) + `gannhanocr-fd/` (glyph FD). |
| `model.py` | `NomEmbedder` (ResNet-18 → embedding 256, L2) + `ArcMargin` (head train). |
| `dataset.py` | `NomDataset` — nạp ảnh + augment mô phỏng mộc bản (xoay/co giãn/erode-dilate/nhiễu). |
| `train.py` | Vòng train (AMP, AdamW, cosine LR) → `checkpoints/best.pt`. |
| `eval_discrim.py` | Chạy lại test T2/T3 (như DINOv2) để **nghiệm thu**. |
| `infer.py` | `NomEncoder` — nạp checkpoint + embed ảnh; drop-in thay DINOv2 trong `../visual_signal.py`. |

## Dữ liệu (đã có sẵn, không cần gán thêm)
- **GOLD crop** thật (51.195, ~1.591 lớp) — từ `dataset_out/gold/`.
- **Glyph FD** (1 ảnh/chữ) — từ `gannhanocr-fd/` → bắc cầu miền + phủ lớp hiếm.

## Chạy LOCAL (sinh index trước khi upload)
```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
.venv/bin/python evaluation/ver_new/nom_classifier/prepare_data.py
# -> index.csv, classes.json, stats.json
```

## Chạy KAGGLE (GPU **P100**)
1. **Tạo Kaggle Dataset** gồm (giữ cấu trúc thư mục):
   - `dataset_out/gold/` (crop), `dataset_out/labels.csv`
   - `gannhanocr-fd/` (chỉ cần các chữ xuất hiện cũng được — vài nghìn file)
   - `evaluation/ver_new/nom_classifier/` (toàn bộ code + `index.csv`, `classes.json`)
2. **Notebook** → Settings → Accelerator = **GPU P100**.
3. Cell:
   ```python
   import sys; sys.path.append('/kaggle/input/<dataset>/evaluation/ver_new/nom_classifier')
   ROOT='/kaggle/input/<dataset>'        # gốc để path trong index.csv trỏ tới
   !python train.py --root "$ROOT" --epochs 35 --batch 256 --img 128
   !python eval_discrim.py --root "$ROOT" --ckpt checkpoints/best.pt
   ```
   *(Nếu path trong index.csv không khớp ROOT của Kaggle, chạy lại `prepare_data.py --root "$ROOT"` trên Kaggle để sinh index đúng.)*

**Tài nguyên P100 (16 GB):** ResNet-18 @128px, batch 256, AMP → ~4–6 GB; ~3–5
phút/epoch; 30–40 epoch ≈ **1,5–3 giờ** (trong quota 12h/phiên · 30h/tuần).

## Nghiệm thu (so trực tiếp với DINOv2)
| Test | DINOv2 (hỏng) | Mục tiêu |
|---|---|---|
| T2 tách cùng/khác chữ (crop thật) | +0,012 | **≥ +0,20** |
| T3 retrieval top-1 (crop→glyph) | 0,0% | **≥ 80%** |

Đạt 2 mốc ⇒ embedding đáng tin.

## Cắm vào S3 (sau khi đạt)
Trong `../visual_signal.py`: thay `DINOv2Ranker._embed` bằng `NomEncoder.embed_gray`
(file `infer.py`). Giữ nguyên `consensus.py`. Bật lại bằng
`build_dataset.py --use-s3`, hiệu chuẩn ngưỡng `τ/δ` trên val. Khi ổn, port
`NomEncoder` vào `core/ranking/`.
