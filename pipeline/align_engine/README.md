# pipeline/align_engine — Engine Step-2 (bản PRODUCTION đóng băng)

Đây là **bản production tự chứa** của bước dựng dataset (align + consensus + tách/cắt
chữ + gán nhãn + xuất chuẩn). Là code **chạy thật** của `run_pipeline.sh` Step 2.

## Nguyên tắc
- **Chỉ phụ thuộc package chính**: `pipeline/`, `core/`, `train_crop/`. **KHÔNG import gì
  từ `evaluation/`.** Sửa/xoá `evaluation/` (thư mục nghiên cứu) không ảnh hưởng ở đây.
- Đây là **bản copy đóng băng** tách ra từ `evaluation/ver_new/` (nơi vẫn để thử nghiệm).
  Khi cải tiến engine đã kiểm chứng ở `evaluation/ver_new`, **đồng bộ tay** sang đây.

## File
| File | Vai trò |
|------|---------|
| `build_dataset.py` | Entrypoint Step 2: PASS1 align mọi trang → PASS2 cắt crop + labels.csv + summary |
| `to_standard.py` | Entrypoint Step 2b: xuất HF imagefolder + Frictionless + Croissant |
| `align_production.py` | Banded DP align cột, chọn reseg (midpoint/valley/detector) |
| `anchor_align.py` · `consensus.py` · `bbox_fix.py` | Neo dict · 3-signal tier · frame-offset + tighten box |
| `visual_signal.py` | S3: embedder Nôm + FD glyph + **crop-protos** (đọc `data/index.csv`) |
| `char_detector/detector_infer.py` | Wrapper detector CenterNet (`train_crop/`) — reseg detector + seam |
| `nom_classifier/{infer,model}.py` | NomEncoder (embedder) cho S3 |
| `data/index.csv` | Chỉ mục crop-proto (đường dẫn trỏ `dataset_out/…` ở repo root) |
| `s3_calibration.json` | Điểm vận hành S3 (isotonic per-tier) |

## Data (vị trí trung lập, ngoài evaluation/)
- Encoder ckpt: `nom-embed/best.pt` (repo root)
- FD glyph cache: theo `config/pipeline.yaml` → `paths.fd_cache_universal`
- **Output**: `dataset_out/` ở **repo root** (crop + labels + chuẩn). `evaluation/ver_new/dataset_out`
  là symlink trỏ về đây (để script nghiên cứu cũ vẫn chạy).

## Chạy
```bash
./run_pipeline.sh --step 2          # qua run_pipeline (khuyến nghị)
# hoặc trực tiếp:
.venv/bin/python -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --use-s3 --reseg detector
.venv/bin/python -m pipeline.align_engine.to_standard
```
Smoke nhanh (không đụng dataset_out thật): thêm `--limit 1 --out /tmp/smoke`.
