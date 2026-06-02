# Chạy classifier Nôm trên Kaggle (GPU P100) — hướng dẫn từng bước

## A. Chuẩn bị gói (chạy 1 lần ở MÁY)
```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
.venv/bin/python evaluation/ver_new/nom_classifier/prepare_data.py     # index.csv, classes.json
.venv/bin/python evaluation/ver_new/nom_classifier/pack_for_kaggle.py  # -> kaggle_pkg/
```
`kaggle_pkg/` chứa **mọi thứ tự-chứa**: `images/{crop,fd}/*.png`, `index.csv`
(đường dẫn tương đối), `classes.json`, và toàn bộ code. ~50–60k ảnh nhỏ (~100–150 MB).

## B. Tạo Kaggle Dataset
1. https://www.kaggle.com/datasets → **New Dataset**.
2. Kéo-thả/nén `kaggle_pkg/` (giữ nguyên cấu trúc thư mục bên trong). Đặt tên, ví dụ
   slug **`nom-crops`**. Create.
   - *(Hoặc dùng Kaggle CLI: `kaggle datasets create -p kaggle_pkg`.)*
3. Sau khi tạo, dataset mount tại **`/kaggle/input/nom-crops/`**.

## C. Tạo Notebook + bật GPU
1. **New Notebook** → panel phải **Add Input** → chọn dataset `nom-crops`.
2. **Settings**: Accelerator = **GPU P100** · **Internet = ON**
   *(cần Internet để tải trọng số ResNet-18 ImageNet ~45 MB lần đầu; nếu không bật
   được Internet thì thêm `--no-pretrained`… — xem mục E).*

## D. Các cell chạy
```python
# Cell 1 — đường dẫn + kiểm GPU
ROOT = "/kaggle/input/nom-crops"     # đổi theo slug của bạn
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
```
```python
# Cell 2 — TRAIN (P100, ~1.5–3h cho 35 epoch)
!cd {ROOT} && python train.py \
    --root {ROOT} --index {ROOT}/index.csv --classes {ROOT}/classes.json \
    --out /kaggle/working/checkpoints \
    --epochs 35 --batch 256 --img 128 --workers 2
```
```python
# Cell 3 — NGHIỆM THU (so với DINOv2 0%)
!cd {ROOT} && python eval_discrim.py \
    --root {ROOT} --index {ROOT}/index.csv \
    --ckpt /kaggle/working/checkpoints/best.pt
```
Checkpoint `best.pt`/`last.pt` nằm ở `/kaggle/working/checkpoints/` → tab **Output**
để tải về (hoặc Save Version để giữ).

## E. Mẹo & xử lý sự cố
- **Internet OFF** (không tải được pretrained): sửa `train.py` gọi `NomEmbedder(pretrained=False)`
  hoặc thêm cờ — vẫn train được nhưng hội tụ chậm hơn, nên ưu tiên bật Internet.
- **OOM**: giảm `--batch 128` hoặc `--img 112`.
- **Chậm I/O**: `--workers 2` (Kaggle 2 CPU); ảnh đã nhỏ nên thường ổn.
- **Hết 12h/phiên**: `train.py` lưu `last.pt` mỗi epoch → Save Version rồi chạy tiếp
  (load `last.pt` để resume — hoặc giảm epoch).
- **Quota**: P100 ~30 GPU-giờ/tuần; 1 lần train nằm gọn trong đó.

## F. Đọc kết quả (nghiệm thu)
| Test | DINOv2 (hỏng) | Đạt khi |
|---|---|---|
| T2 separation (cùng−khác) | +0,012 | **≥ +0,20** |
| T3 retrieval top-1 | 0,0% | **≥ 80%** |

Đạt 2 mốc ⇒ embedding đáng tin → tải `best.pt` về, cắm `NomEncoder` (`infer.py`)
vào `../visual_signal.py` thay DINOv2, bật lại SILVER bằng `build_dataset.py --use-s3`.
(Chi tiết tích hợp: `README.md`.)
