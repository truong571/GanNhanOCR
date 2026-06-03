# Cải tiến embedder Nôm (v1 → v2) + lộ trình tiếp

Mục tiêu: tăng **T3 retrieval** (v1 = 76,5%) để SILVER chính xác hơn & gỡ thêm REVIEW.

## Đã làm (v2 — re-train là có ngay)

| # | Cải tiến | File | Vì sao |
|---|---|---|---|
| 1 | **Multi-font references**: render mỗi chữ trong 6 font Nôm (HanaMin A/B, NomNaTong, Khai, HAN NOM A/B) → **+9.493 mẫu** (crop 51.195 + FD 1.591 + **font 9.493**) | `prepare_data.py` | Mỗi lớp giờ có ~7 tham chiếu sạch (thay vì 1 FD) → **boost lớn cho 554 singleton + bất biến phong cách in** |
| 2 | **Augmentation mạnh hơn**: thêm elastic warp + perspective + cutout (ngoài xoay/co/erode-dilate/nhiễu) | `dataset.py` | Mô phỏng cong/nhoè/đứt nét mộc bản → khử khoảng cách miền tốt hơn |
| 3 | **Backbone lớn hơn**: mặc định **ResNet-34** (`--arch resnet18/34/50`), ảnh **160px** | `model.py`, `train.py` | Nhiều capacity + chi tiết nét hơn |
| 4 | **Warmup + cosine**, epochs 40, lưu `arch` vào checkpoint | `train.py`, `infer.py` | Hội tụ ổn; `infer.py` tự đọc đúng backbone |

## Cách re-train (Kaggle P100)
```bash
# ở máy: dựng lại index (đã có multi-font) + đóng gói
.venv/bin/python evaluation/ver_new/nom_classifier/prepare_data.py
.venv/bin/python evaluation/ver_new/nom_classifier/pack_for_kaggle.py     # gồm images/font/
# Kaggle (P100): batch 192 cho resnet34@160 (resnet34 nặng hơn -> giảm batch nếu OOM)
!python train.py --root $ROOT --index $ROOT/index.csv --classes $ROOT/classes.json \
    --out /kaggle/working/checkpoints --arch resnet34 --img 160 --batch 192 --epochs 40
!python eval_discrim.py --root $ROOT --index $ROOT/index.csv --ckpt /kaggle/working/checkpoints/best.pt
```
Tải `best.pt` về `nom-embed/` → `visual_signal.py` tự dùng → `build_dataset.py --use-s3` ra SILVER tốt hơn.

## Lộ trình tiếp (chưa làm — theo impact)

1. **Hard-negative mining bằng `SinoNom_Similar_Dic_v2`** *(impact cao)* — các chữ **dễ nhầm** (lookalike) chính là thứ S3 phải phân biệt (vd 韋 vs 喡). Batch-sampler nhóm chữ-tương-tự vào cùng batch, hoặc thêm triplet-loss với negative = chữ similar → ép model tách đúng cặp khó.
2. **Self-training trên REVIEW** *(impact cao, vòng lặp)* — dùng model gán nhãn crop REVIEW; lấy mẫu **confidence cao** (s3_cosine ≥ ngưỡng) bổ sung vào train → retrain → lặp. Tăng dần dữ liệu + gỡ thêm REVIEW.
3. **TTA khi infer** *(rẻ)* — embed nhiều view augment của crop rồi trung bình → cosine ổn định hơn ở `visual_signal.py`.
4. **Sub-center ArcFace / k=3** *(vừa)* — chịu nhiễu nhãn (mẫu xấu) tốt hơn cho lớp đông.
5. **Hiệu chuẩn ngưỡng SILVER `τ/δ`** *(rẻ)* — quét trên tập val (crop có nhãn) để chọn `τ/δ` đạt precision mục tiêu, thay vì đặt tay 0.62/0.06.

## Đo tiến bộ
Sau mỗi vòng chạy `eval_discrim.py` → so **T2 separation** & **T3 retrieval** với v1 (0,29 / 76,5%). Mục tiêu T3 **≥ 85–90%**.
