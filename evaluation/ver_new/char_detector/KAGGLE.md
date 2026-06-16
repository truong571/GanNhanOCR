# Train the character detector on Kaggle (train → đánh giá → lặp → dùng)

Mục tiêu: detector anchorless ràng-buộc-N để **cắt đúng ranh giới chữ** trên cột
diverged (heuristic không-train đã chứng minh không làm được — `seg_smart_ab.py`).
Trainer tự **báo F1/đếm trên tập val** nên train xong biết ngay **đạt hay chưa**.

## 0. Đã chuẩn bị sẵn (local)
```bash
.venv/bin/python evaluation/ver_new/char_detector/bootstrap_boxes.py    # detect_manifest.json (66.630 box / 445 trang)
.venv/bin/python evaluation/ver_new/char_detector/pack_for_kaggle.py     # kaggle_det_pkg/ (193 MB)
```
`kaggle_det_pkg/` gồm: `images/` (trang downscale 1280px) + `detect_manifest.json`
(đường dẫn tương đối + box đã scale) + `train_centernet.py` + `count_constrained.py`.

## 1. Upload `kaggle_det_pkg/` lên Kaggle dưới dạng **một Dataset** (vd `nom-char-det`).

## 2. Kaggle Notebook (GPU **P100/T4**, Internet ON)
Cách dễ nhất: **import `train_detector_kaggle.ipynb`** (có sẵn trong gói) → Run All.
Hoặc chạy tay:
```python
!cp /kaggle/input/nom-char-det/* . -r
!python train_centernet.py --manifest detect_manifest.json --img 768 \
        --epochs 40 --batch 8 --val-frac 0.1 --out detector.pt \
        --hf-repo <username>/nom-char-det     # (tùy chọn) đẩy HuggingFace
```
Mỗi epoch in: `VAL F1 .. P .. R .. count-err ..`. Lưu `detector.pt` (last) +
`detector.best.pt` (val-F1 cao nhất).

**Lưu HuggingFace (khuyến nghị — khỏi mất khi phiên reset):** Add-ons → Secrets →
thêm secret **`HF_TOKEN`** (token quyền write); truyền `--hf-repo <user>/nom-char-det`.
Trainer đẩy `detector.best.pt` lên HF **mỗi lần F1 cải thiện + lúc kết thúc** (giống
encoder của bạn). Kéo về máy: `huggingface-cli download <user>/nom-char-det detector.best.pt`.

## 3. ĐÁNH GIÁ — đạt hay chưa?
| Chỉ số (trên val) | ĐẠT (đáng wire) | Chưa đạt → lặp |
|---|---|---|
| **box F1 @IoU0.5** | ≥ ~0.85 | < 0.8 |
| **median count-err / cột** | ~0 | ≥ 1 |

(So mốc ngoài: HRCenterNet ~0.81 IoU trên MTHv2 ván khắc.)

## 4. KHÔNG đạt → train mới (theo thứ tự rẻ→đắt)
1. **Pretrain TKH/MTHv2 rồi finetune** (mạnh nhất — cùng miền ván khắc chữ Hán):
   thêm dataset `HCIILAB/TKH_MTH_Datasets_Release`, train detector trên đó trước,
   rồi `--init tkh_detector.pt` khi train trên Nôm.
2. **Tăng epoch** (60–80) / **ảnh lớn hơn** `--img 1024` (chữ to hơn, dễ tách).
3. **Lọc nhãn sạch hơn:** `bootstrap_boxes.py --complete-only` (chỉ cột không-REVIEW)
   để bớt box thiếu — ít nhiễu nhãn hơn (đánh đổi: ít data hơn).
4. Heatmap/aug tinh chỉnh trong `train_centernet.py` (radius, MixUp, background class).

## 5. Đạt → kéo về + dùng (plug-and-play)
```bash
# tải detector.best.pt từ Kaggle Output về:
cp detector.best.pt evaluation/ver_new/char_detector/detector.pt
# verify đường inference (đã tự dò checkpoint):
.venv/bin/python evaluation/ver_new/char_detector/detector_infer.py --smoke
# build dataset DÙNG detector (count-constrained) — nhớ --out để A/B, đừng đè bản chuẩn:
.venv/bin/python evaluation/ver_new/build_dataset.py --use-s3 --reseg detector --out dataset_out_det
```
`align_production` tự: chạy detector 1 lần/trang → lọc box theo cột → `constrain_to_count(N)`
→ map về từng chữ. Thiếu `detector.pt` thì **tự fallback midpoint** (an toàn).

## 6. Đo cải thiện crop dính (so trước/sau)
```bash
# crop-audit PDF của bản detector để soi mắt:
.venv/bin/python evaluation/ver_new/crop_audit_pdf.py --dataset dataset_out_det
# số: chạy lại harness merged-crop trên bản detector (kỳ vọng two_blob giảm & MLS ~0.68):
#   so 3 cách (midpoint / valley / detector) bằng cùng metric trong seg_valley_n_ab.py
```
Mốc thành công cuối: trên cột diverged, **MLS ~0.68** (mức glyph sạch) và **two_blob
thấp** — điều heuristic không đạt (valley: MLS 0.44).

---
**Tài nguyên Kaggle:** P100 16GB thừa cho ResNet18-CenterNet batch 8 @768px; ~40
epoch trên 400 trang ≈ 1–2h. Smoke local đã verify build+forward+backward+decode+
count-constraint. Trích dẫn: HRCenterNet (IEEE BigData 2020), Objects as Points
(Zhou 2019), TKH/MTHv2 (Yang IEEE Access 2018 / Ma ICFHR 2020).
