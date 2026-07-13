# Giai đoạn 2 — Nâng chuẩn SOTA (phiếu bầu độc lập thật)

Tầng **pipeline chính** hợp nhất các phiếu **có tương quan** (kim, qwen, nomnaocr —
cùng đọc một crop) với các kênh **độc lập thật** (S3 glyph-verifier, từ điển QN) theo
đúng cách SOTA 2024–2026, thay cho "3 phiếu ngang" ngây thơ.

Chỉ `numpy + scipy + pandas` — thống kê tự cài, không hộp đen. Đây là **khung ổn định,
test đầy đủ**; các mô hình độc lập nặng là **kênh dữ liệu plug vào** (xem Drivers).

## Bốn thành phần

| Module | Vai trò |
|---|---|
| `independence.py` | **Kish n_eff** — số phiếu ĐỘC LẬP HIỆU DỤNG thật (phiếu tương quan ≪ số phiếu) |
| `fusion.py` | **stacking hiệu chỉnh** — logistic (IRLS) fit trên nhãn human-audit + isotonic PAV |
| `gating.py` | **gate bất đối xứng** — thăng GOLD khó (đa kênh), giáng REVIEW dễ (1 tín hiệu) |
| `qwen_verifier.py` | **verifier mù** — lineup MCQ chống sycophancy + đảo thứ tự chống position-bias + abstain |

## Năm kênh bằng chứng (channels.py)

Trọng số theo **loại bằng chứng khác gốc**, không theo số phiếu:

| Kênh | Role | Độc lập pixel? | Quy tắc |
|---|---|---|---|
| kim | proposer | ✗ | đề xuất nhãn+toạ độ; không tự xác nhận |
| qwen | verifier | ✗ | **mù** (không show nhãn kim); có thể abstain; không overrule một mình |
| nna_lobo | demote_only | ✗ | bất đồng → giáng; đồng ý không cộng điểm |
| **s3** | **must_pass** | ✓ | kênh độc lập duy nhất; **bắt buộc pass để thăng GOLD** |
| dict | prior | ✓ | tone-canon; không bao giờ thăng một mình |

## Lệnh

```bash
PY=.venv/bin/python
# độ độc lập hiệu dụng của các phiếu OCR (báo trong luận văn)
$PY -m pipeline.consensus_fusion neff --votes votes.csv --truth label
# fit fuser hiệu chỉnh trên nhãn audit, dự đoán + gate mọi crop
$PY -m pipeline.consensus_fusion fuse --features channels.csv --label-col y \
     --score-cols s3,dict,qwen_agree,nna_agree
# minh hoạ tổng hợp (không cần dữ liệu)
$PY -m pipeline.consensus_fusion demo
```

`demo` in ra ví dụ then chốt: 3 phiếu nhưng nna echo kim (φ≈0,77) → **n_eff≈2** — đúng
kết quả "Nine Judges, Two Effective Votes" (arXiv:2605.29800) trên chính setup này.

## Định dạng dữ liệu kênh (schema plug-in)

Mỗi kênh nặng xuất một CSV `crop_id, score[, vote][, flag_*]`; `fuse` gộp theo `crop_id`:
- `score` ∈ [0,1] hoặc raw (fuser tự chuẩn hoá); NaN = kênh vắng (thành feature riêng).
- `vote` = ký tự dự đoán (cho n_eff).
- `flag_qwen_abstain / flag_quality_flag / flag_qwen_disagree / flag_nna_disagree /
  flag_nna_echoes_kim` = boolean cho gate.

## Drivers — cách sinh kênh nặng (chạy riêng, KHÔNG trong khung này)

Khung này chỉ hợp nhất; các mô hình độc lập chạy offline rồi xuất CSV:

1. **CTC forced-alignment cross-check** (kraken 5.x `ForcedAlignmentTaskModel`,
   arXiv:2508.07904): fine-tune recognizer dòng dọc trên line+chuỗi Nôm đã align →
   forced-align lấy span từng ký tự độc lập với detector → flag crop IoU<0.5. Xuất
   `crop_id, flag_ctc_mismatch`.
2. **NomNaOCR-LOBO** (fix vòng tròn ở [[tri-consensus-golive-verification]]): 3 model
   leave-one-book-out (train 2 sách, vote sách còn lại), vai chỉ-demote. Xuất
   `crop_id, nna_vote, flag_nna_disagree`.
3. **Qwen-VL-235B blind-MCQ**: dùng `qwen_verifier.verify(true_label, distractors, ask)`
   với `distractors` = top-k láng giềng S3, `ask` = gọi DashScope trên crop+render. Xuất
   `crop_id, qwen_agree, flag_qwen_abstain, flag_qwen_disagree`. (~$4–15 cho 445 trang.)
4. **S3 head-logit toàn corpus** (đóng bug scoring visual_signal.py vòng 3): chấm mọi
   crop → `crop_id, s3` (mở khoá cả PPI ở Giai đoạn 0).
5. **Loại**: GOT-OCR2 / PaddleOCR-VL (page-model, không inventory Nôm; mọi VLM <25%
   char-F1 trên cổ văn — arXiv:2605.11960). Ghi rationale, không tích hợp.

## Gate (bất đối xứng có chủ đích)

```
promote_gold  iff  P >= tau_promote (đặt cho ≥99% precision held-out)
                   AND s3 >= 0.29   (điểm vận hành 85% recall / 3.5% FAR — đo vòng 3)
                   AND qwen không abstain
                   AND không cờ crop-quality
demote_review iff  qwen đọc-mù ra chữ khác ổn định
                   OR s3 dưới sàn 0.15 và nna chỉ echo kim
                   OR nna_lobo cãi
                   OR dict-implausible không có glyph support
```

## Test

```bash
.venv/bin/python -m pipeline.consensus_fusion.selftest   # 44 assertions, exit 0 = pass
```

Kiểm: n_eff (3 độc lập→~3, 3 trùng→~1, hỗn hợp→giữa, công thức Kish); fuser học được
tín hiệu (AUC), xử lý NaN, isotonic đơn điệu + hiệu chỉnh đúng; gate 7 kịch bản
(promote cần đủ mọi cổng, demote bất đối xứng khi P cao); verifier mù (lineup không lộ
nhãn kim, parse glyph/letter/number, confirm/disagree/abstain, sycophancy guard);
integration fit+gate trên `s3_cosine` thật của labels.csv.

## Phụ thuộc & vị trí

`numpy`, `scipy`, `pandas`. Đầu ra: `dataset_out/fusion/`. Hiệu chỉnh dùng nhãn từ
Giai đoạn 0 (`pipeline/ground_truth`); giáng/thăng ghi lại như một lớp trên
`labels_remediated.csv` của Giai đoạn 1.
