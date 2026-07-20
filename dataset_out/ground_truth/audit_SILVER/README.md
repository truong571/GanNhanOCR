# Mẻ audit SILVER — hướng dẫn chấm

Mục tiêu: đo PRECISION tier SILVER (chưa từng đo — audit GOLD 100% là GOLD). Mẫu 750 crop,
lấy NGẪU NHIÊN phân tầng theo rule (không thiên lệch), design_weight quy về dân số 10856.

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
    --verdicts /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/dataset_out/ground_truth/audit_SILVER --manifest /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/dataset_out/ground_truth/audit_SILVER/manifest.jsonl --design stratified --p0 0.90
```
→ precision SILVER + CI95 (Wilson/Clopper-Pearson) + per-rule + acceptance.

## Phân tầng (design_weight = N_h / n_h)
- s1_inter_s3_out_of_dict: n=40 / N=188 (w=4.7)
- s2_inter_s3_corrected: n=605 / N=9147 (w=15.119)
- s3_head_bank_consensus: n=105 / N=1521 (w=14.486)
