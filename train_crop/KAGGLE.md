# Train trên Kaggle GPU (vì local MPS rất chậm)

**Phần nào train ở đâu:**

| Phần | Chạy ở đâu | Vì sao |
|------|-----------|--------|
| **Pretrain MTHv2/TKH** (~3.000 ảnh, 1M+ box) | **Kaggle GPU (bắt buộc)** | Quá nặng cho MPS; cùng miền ván khắc → tăng độ chính xác nhiều nhất |
| **Train / fine-tune** trên 445 trang Nôm | **Kaggle GPU (nên)** | T4/P100 nhanh hơn MPS ~10–20×; AMP tự bật |
| **Smoke test** (build/forward/decode) | Local | Nhẹ, kiểm tra code đúng |
| **Suy luận + sinh PDF** | Local | Chỉ vài trang, không cần GPU |

> Tóm lại: **mọi `train_centernet.py` đều nên chạy trên Kaggle**. Local chỉ smoke + infer + PDF.

---

## Bước 1 — Đóng gói (local, 1 phút)
```bash
.venv/bin/python test/pack_for_kaggle.py        # -> test/kaggle_pkg/  (+ test/kaggle_pkg.zip)
```
Gói gồm: các module `.py` + `images/` (trang downscale 1280px) + `detect_manifest.json`
(đường dẫn tương đối + box đã scale). Mặc định tái dùng ảnh đã downscale sẵn nên rất nhanh.

*(Tuỳ chọn — chọn tier dữ liệu khác)* dựng lại manifest từ `labels.csv` rồi đóng gói:
```bash
.venv/bin/python test/build_manifest.py --tiers GOLD,SILVER,SYLLABLE --out test/m.json  # ~66k box
.venv/bin/python test/pack_for_kaggle.py --manifest test/m.json --force-downscale
```

## Bước 2 — Upload
Lên https://www.kaggle.com/datasets → **New Dataset** → kéo thả cả thư mục
`test/kaggle_pkg/` → đặt tên (vd. `nom-char-det-r34`) → Create.

## Bước 3 — Notebook GPU
1. **New Notebook** → Settings: Accelerator **GPU T4 x2** (hoặc P100), Internet **ON**.
2. **Add data** → dataset vừa tạo (và MTHv2/TKH nếu có).
3. **File → Import Notebook** → chọn `test/kaggle_train.ipynb` → **Run All**.

Hoặc chạy tay trong 1 cell:
```python
import glob, os, shutil
PKG = os.path.dirname(glob.glob('/kaggle/input/**/train_centernet.py', recursive=True)[0])
for f in glob.glob(PKG + '/*'):
    dst = '/kaggle/working/' + os.path.basename(f)
    shutil.copytree(f, dst, dirs_exist_ok=True) if os.path.isdir(f) else shutil.copy(f, dst)
os.chdir('/kaggle/working')
```
```bash
# (tuỳ chọn) pretrain MTHv2 trước:
!python train_centernet.py --voc-img <MTHv2/img> --voc-xml <MTHv2/xml> \
    --img 768 --epochs 30 --batch 8 --out /kaggle/working/mthv2_pretrain.pt

# train/fine-tune trên Nôm:
!python train_centernet.py --manifest detect_manifest.json \
    --img 768 --epochs 40 --batch 8 --workers 2 --val-frac 0.1 --val-pages 44 \
    --out /kaggle/working/detector_r34.pt \
    # --init /kaggle/working/mthv2_pretrain.best.pt   # nếu đã pretrain
    # --hf-repo <user>/nom-char-det                   # nếu muốn đẩy HuggingFace
```

## Bước 4 — Lưu ckpt (chống mất khi phiên reset)
- **Cách 1 — Output Kaggle:** ckpt nằm ở `/kaggle/working/detector_r34.best.pt`; tải từ
  panel **Output** bên phải.
- **Cách 2 — HuggingFace (khuyến nghị):** Add-ons → Secrets → thêm `HF_TOKEN` (quyền write),
  truyền `--hf-repo <user>/nom-char-det`. Trainer tự đẩy `detector_r34.best.pt` mỗi lần F1
  cải thiện + lúc kết thúc. Kéo về: `huggingface-cli download <user>/nom-char-det detector_r34.best.pt`.

## Bước 5 — Dùng ckpt ở local (suy luận + PDF)
```bash
cp detector_r34.best.pt test/
.venv/bin/python test/infer_centernet.py --ckpt test/detector_r34.best.pt --image cot.png --n 9
.venv/bin/python test/make_report_pdf.py --ckpt test/detector_r34.best.pt \
    --manifest evaluation/ver_new/char_detector/detect_manifest.json \
    --out test/ket_qua_centernet.pdf
```

## Đạt hay chưa? (trainer tự in mỗi epoch)
| Chỉ số VAL | ĐẠT | Chưa → lặp |
|---|---|---|
| **box F1 @IoU0.5** | ≥ ~0.85 | < 0.8 |
| **median count-err / trang** | ~0 | ≥ 1 |

Chưa đạt → theo thứ tự rẻ→đắt: (1) pretrain MTHv2 rồi `--init`, (2) tăng `--epochs 60–80`
hoặc `--img 1024`, (3) bật `--dcn` (DCNv2 bám nét cong), (4) lọc nhãn sạch hơn.

**Tài nguyên:** P100 16GB thừa cho ResNet34+FPN batch 8 @768px; 40 epoch / 400 trang ≈ 1–2h.
Mốc ngoài: HRCenterNet ~0.81 IoU trên MTHv2.
