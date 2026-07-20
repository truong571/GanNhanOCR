# Mẻ audit GOLD — 846 (ĐÃ CHẤM XONG + re-anchor)

Mẻ human-audit tier GOLD, n=846 (SRS, p0=0.97, accept ≤17 defect — xem plan.json).

## Trạng thái: HOÀN TẤT
- **Verdicts (thô):** `verdicts_001.jsonl` … `verdicts_006.jsonl` (TRONG chính thư mục này) —
  846 phán quyết: 819 correct / 19 wrong_label / 6 wrong_image / 2 unsure.
- **Re-anchor sang dataset fresh 3-sách:** `../verdicts_reanchored.csv` (825/846 matched theo
  book+page+bbox-IoU; 6 label_changed; 15 orphan). Nếu chạy lại:
  `python -m pipeline.ground_truth.reanchor_verdicts --verdicts dataset_out/ground_truth/audit_gold`
- **Precision GOLD đo được:** 97.08% (799/823 matched non-unsure), Wilson95 [95.70, 98.03];
  sau demote confusion 㝵/người → **98.00%** (báo cáo `../report.json`, `FLOW_TONG_THE_CHOT_*.md`).

## File
- `manifest.jsonl` (846 item, thế hệ CŨ) · `plan.json` (thiết kế) · `audit_001-006.html` (đã chấm).

## Lưu ý
Manifest ở đây neo theo image-path thế hệ cũ; dùng `../verdicts_reanchored.csv` khi làm việc với
dataset hiện tại. Các mẻ SILVER/SYLLABLE (`../audit_SILVER`, `../audit_SYLLABLE`) neo thẳng trên
`labels_final.csv` fresh — verdicts của chúng LƯU TRONG chính thư mục mẻ đó (không phải cấp cha này).
