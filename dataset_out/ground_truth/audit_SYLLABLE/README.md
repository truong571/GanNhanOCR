# Mẻ audit SYLLABLE — hướng dẫn chấm

Mục tiêu: đo PRECISION tier SYLLABLE (chưa từng đo — audit GOLD 100% là GOLD). Mẫu 300 crop,
lấy NGẪU NHIÊN phân tầng theo rule (không thiên lệch), design_weight quy về dân số 6751.

## Chấm (giống audit GOLD đã làm)
1. Mở lần lượt `audit_001.html`, `audit_002.html`, ... trong trình duyệt.
2. Mỗi ô: crop + glyph tham chiếu (nếu có) + ngữ cảnh trang + âm QN + ứng viên từ điển.
   Bấm 1 trong: **correct** (nhãn đúng) / **wrong_label** (crop đúng chữ nhưng nhãn sai) /
   **wrong_image** (crop cắt lỗi/dính/nhầm ô) / **unsure** (không chắc). Tiến độ tự lưu.
3. Chấm hết mọi batch → **Download JSON** → lưu thành `verdicts_001.jsonl`, `verdicts_002.jsonl`...
   ĐẶT NGAY trong thư mục này.

## Đo sau khi chấm
```
.venv/bin/python -m pipeline.ground_truth estimate \
    --verdicts dataset_out/ground_truth/audit_SYLLABLE --manifest dataset_out/ground_truth/audit_SYLLABLE/manifest.jsonl --design stratified --p0 0.90
```
→ precision SYLLABLE + CI95 (Wilson/Clopper-Pearson) + per-rule + acceptance.

## Phân tầng (design_weight = N_h / n_h)
- nghia_consensus: n=300 / N=6751 (w=22.503)
