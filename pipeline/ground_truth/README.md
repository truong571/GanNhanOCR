# Giai đoạn 0 — Tạo ground truth

Tầng **pipeline chính** (không phải `evaluation/`) tạo ra con số duy nhất mà pipeline
tự động không thể tự tạo: **đo độ chính xác nhãn GOLD/usable bằng người, đối chiếu bản
scan gốc, kèm khoảng tin cậy bảo vệ được trước hội đồng**.

Mọi thống kê tự cài bằng `numpy` + `scipy` (không `sklearn`, không `cleanlab`) nên không
có hộp đen — mỗi khoảng tin cậy là một phương pháp có tên, trích dẫn được.

## Luồng

```
rank      chấm điểm nghi vấn mọi crop usable        -> labels_ranked.csv
plan      tính cỡ mẫu acceptance / CI (thống kê thuần)
sample    rút mẫu phân tầng (hoặc SRS)               -> sample_*.csv
grid      dựng công cụ audit HTML mù + manifest      -> audit_*.html + manifest.jsonl
(người)   auditor gán nhãn từng crop, bấm Xuất       -> verdicts.jsonl
estimate  precision + Wilson/CP CI + acceptance + PPI -> report.json
```

## Lệnh (chạy từ gốc repo)

```bash
PY=.venv/bin/python

# 1. Xếp hạng nghi vấn toàn bộ 68k crop usable
$PY -m pipeline.ground_truth rank

# 2. Tính cỡ mẫu: claim "precision >= 97%"
$PY -m pipeline.ground_truth plan --p0 0.97 --p-assumed 0.985
#   -> Acceptance (SRS): n=846, accept nếu defects <= 17  (one-sided 95% LB = 0.97)
#   -> Wilson +/-1%: n=1139 ;  +/-0.5%: n=4482

# 3a. Mẫu SRS cho CLAIM acceptance "≥97%"
$PY -m pipeline.ground_truth sample --n 846 --design srs
# 3b. Mẫu phân tầng cho ƯỚC LƯỢNG headline ±1% (oversample tầng rủi ro)
$PY -m pipeline.ground_truth sample --n 1150 --design stratified

# 4. Dựng công cụ audit mù (chia batch 150 item/file cho nhẹ trình duyệt)
$PY -m pipeline.ground_truth grid --sample dataset_out/ground_truth/sample_stratified.csv

# 5. (Người mở audit_*.html, gán nhãn, bấm "Xuất verdicts.jsonl")

# 6. Ước lượng
$PY -m pipeline.ground_truth estimate \
    --verdicts verdicts.jsonl \
    --manifest dataset_out/ground_truth/manifest.jsonl \
    --p0 0.97 --design stratified
```

## Hai thiết kế mẫu — dùng đúng chỗ

| Mục tiêu | Design | Ước lượng dùng |
|---|---|---|
| Tuyên bố "precision ≥ 97% (one-sided 95%)" | **SRS** (`--design srs`) | `acceptance` (Clopper–Pearson một phía) |
| Con số headline ± khoảng tin cậy | **stratified** | `weighted_precision` (Horvitz–Thompson + FPC) |

SRS bắt buộc cho acceptance vì mọi crop có xác suất chọn bằng nhau. Mẫu phân tầng
oversample các tầng rủi ro (dup-defect, similar-bridge, s3-low, head-bank) để tiêu ngân
sách nơi lỗi ẩn, rồi `design_weight = N_h/n_h` đưa ước lượng về tổng thể **không thiên
lệch** — nên đừng đọc bound acceptance SRS trên mẫu phân tầng (code đã cảnh báo).

## Công cụ audit mù

Mỗi item hiện đúng 3 thứ: **crop được gán · vị trí trên scan (khung bbox đỏ) · glyph
tham chiếu** (font-render theo nhãn), kèm ký tự nhãn + âm + ứng viên từ điển. Mọi thứ gây
thiên lệch (tier, rule, sách, điểm S3, stratum, suspicion) **bị giấu**, chỉ ghi vào
`manifest.jsonl` để `estimate` ghép lại sau. Bốn phím tắt:

- `1` **đúng** — ảnh là một glyph sạch VÀ nhãn khớp
- `2` **sai nhãn** — ảnh là glyph sạch NHƯNG nhãn sai ký tự (co-error/similar-bridge)
- `3` **sai ảnh** — crop cắt lỗi/dính/glyph hàng xóm (lỗi AE-1/F1)
- `4` **không chắc** — không đọc được / biến thể mơ hồ

Verdict lưu localStorage, xuất `verdicts.jsonl`. Hai trục (`sai nhãn` vs `sai ảnh`) tách
đúng hai họ lỗi để đo riêng.

## Xếp hạng nghi vấn (suspicion)

Điểm `suspicion` ∈ [0,1] là **prior để ưu tiên audit**, KHÔNG phải xác suất lỗi — chỉ
người mới đo được sự thật. Tổ hợp noisy-OR từ các tín hiệu đã kiểm chứng trong 3 vòng
đánh giá; tầng `stratum` (loại trừ lẫn nhau, ưu tiên từ cao xuống):
`dup_defect` › `similar_bridge_lowcos` › `similar_bridge` › `silver_headbank` ›
`s3_low` › `quality_flag` › `gold_direct` › `other`.

## PPI (tuỳ chọn, đang tự bỏ qua có chủ đích)

PPI siết CI bằng một surrogate máy, nhưng **chỉ hợp lệ khi surrogate phủ ≥90% cả tập
audit lẫn tổng thể**. `s3_cosine` rỗng trên toàn bộ 47k dòng GOLD-direct (phủ ~30%) nên
code **tự bỏ qua PPI** kèm ghi chú — không phát ra số sai lệch. Muốn bật PPI: cung cấp
surrogate phủ toàn tổng thể (ví dụ S3 head-logit chấm cho mọi crop) rồi truyền qua
`surrogate_col`.

## Thống kê (stats.py) — đối chiếu độc lập trong selftest

`wilson_ci` · `clopper_pearson_ci` · `cp_lower_bound` (một phía) · `acceptance_plan`
(tái lập đúng n=846, c=17 cho p0=0.97) · `required_n_for_halfwidth` · `ppi_mean_ci`
(Angelopoulos et al. 2023) · `stratified_mean_ci` (có FPC).

## Test

```bash
.venv/bin/python -m pipeline.ground_truth.selftest   # 60 assertions, exit 0 = pass
```

Kiểm: thống kê đối chiếu `scipy` độc lập + giá trị sách; `suspicion` đối chiếu census
(dup_bbox=701, cross_col=1686, similar_bridge=3856, union=2321); sampling xác định +
tổng design_weight ≈ N; grid mù (không rò tier/rule) + manifest đủ trường ẩn; estimate
đúng cả nhánh PPI-đủ-phủ lẫn PPI-bỏ-qua, acceptance ACCEPT/REJECT.

## Phụ thuộc

`numpy`, `scipy`, `pandas`, `Pillow`, `PyYAML` (đọc `config/pipeline.yaml`); từ điển tuỳ
chọn qua `core.text.dictionary.load_qn_to_nom`. Đầu ra mặc định:
`dataset_out/ground_truth/`.
