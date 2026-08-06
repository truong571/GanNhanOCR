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

## Tín hiệu S3 corpus-wide (`s3_signals.py`, thêm 2026-08-03)

`build_dataset.maybe_s3()` CỐ Ý bỏ bước so ảnh mỗi khi `ocr_char` đã khớp từ điển — đúng
luật `s1_inter_s2_direct`, tức **94,2% GOLD**. Hệ quả: cột `s3_cosine` chỉ phủ **5,8%
GOLD**, phần lớn nhãn GOLD chưa từng được đối chiếu với ảnh lần nào.

`consensus_fusion/score_s3.py --all` đã chấm bù ra `dataset_out/fusion/s3_corpus.csv`
(6 tín hiệu, phủ 100% mọi hàng chấm được) nhưng kết quả chỉ chảy vào tầng fuse ở bước 6.
`s3_signals.attach()` nối nó ngược lại khung labels, nên `rank`/`sample`/`grid` mới nhìn
thấy. Đo được sau khi nối: hàng "chưa từng soi ảnh" **47.382 → 18**, và lộ ra tầng
`head_disagree` **5.525 hàng** (đầu ArcFace xếp một chữ KHÁC cao điểm hơn chính nhãn).

Kèm hai chốt an toàn: join báo lỗi nếu `s3_corpus.csv` lệch thế hệ với `labels.csv`
(cùng `image` nhưng khác `label`), và nhãn nằm ngoài 1591 lớp ArcFace vẫn được tính là
"đã chấm" nhờ `bank_cos`/`mls` thay vì bị xếp nhầm vào nhóm mù.

Tắt bằng `--no-s3-signals` nếu muốn tái lập cách xếp hạng cũ.

## Mẻ audit GOLD hai tầng (`make_gold_batch.py`)

```bash
$PY -m pipeline.ground_truth.make_gold_batch          # mặc định 120 SRS + 80 margin thấp
```

Hai mục tiêu xung khắc về thống kê nên phải rút RIÊNG, gắn nhãn RIÊNG bằng `audit_batch`:

| Tầng               | Cách rút                                           | Dùng để                                                    |
|--------------------|----------------------------------------------------|------------------------------------------------------------|
| `srs`              | ngẫu nhiên đơn giản từ GOLD, `design_weight = N/n` | **precision + CI — số báo cáo được**                       |
| `active_lowmargin` | `s3_head_margin` thấp nhất, `design_weight` rỗng   | AUC / hiệu chỉnh ngưỡng — **không bao giờ tính precision** |

`estimate` tự LOẠI mọi hàng `design_weight` rỗng ra khỏi precision và báo rõ số bị loại;
gộp nhầm sẽ cho precision thấp giả vì tầng chủ đích cố ý giàu lỗi. Người chấm không phân
biệt được hai tầng: `audit_order` xáo trộn chung và `audit_batch` chỉ nằm trong manifest.

`--verdicts` nhận cả một THƯ MỤC (gộp mọi `verdicts*.jsonl`), không chỉ một file.

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

| Mục tiêu                                   | Design                   | Ước lượng dùng                                |
|--------------------------------------------|--------------------------|-----------------------------------------------|
| Tuyên bố "precision ≥ 97% (one-sided 95%)" | **SRS** (`--design srs`) | `acceptance` (Clopper–Pearson một phía)       |
| Con số headline ± khoảng tin cậy           | **stratified**           | `weighted_precision` (Horvitz–Thompson + FPC) |

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
`dup_defect` › `head_disagree` › `similar_bridge_lowcos` › `similar_bridge` ›
`silver_headbank` › `s3_low` › `quality_flag` › `gold_direct` › `other`.

`head_disagree` chỉ xuất hiện khi khung đã đi qua `s3_signals.attach()`. Trọng số của nó
(`r_head_disagree`, `r_bank_low`) là **tiên nghiệm CHƯA kiểm định** — sức phân biệt thật
trên nhãn NGƯỜI chưa từng được đo, mọi số AUC đang lưu hành đều đo trên verdict MÁY. Chúng
chỉ đổi thứ tự ưu tiên audit, không bao giờ đổi `tier`/`label`.

## PPI — đã mở khoá được (2026-08-03)

PPI siết CI bằng một surrogate máy, nhưng **chỉ hợp lệ khi surrogate phủ ≥90% cả tập
audit lẫn tổng thể**. `s3_cosine` rỗng trên toàn bộ 47k dòng GOLD-direct (phủ ~32%) nên
PPI trước đây **luôn tự bỏ qua**.

Sau khi gắn `s3_corpus`, `s3_bank_cos` / `s3_head_cos` phủ **90,2%** dân số chưa chấm —
vừa qua ngưỡng, PPI chạy được. `estimate --surrogate auto` (mặc định) tự chọn cột phủ rộng
nhất và IN RA độ phủ của từng cột để việc chọn là minh bạch, không âm thầm.

Lưu ý 90,2% là sát ngưỡng 90%: phần thiếu là 6.809 hàng tier SYLLABLE, vốn mang nhãn âm
tiết chứ không phải chữ đơn nên **không thể** chấm S3 theo thiết kế. Nếu dân số usable đổi
thành phần, PPI có thể tụt xuống dưới ngưỡng và tự bỏ qua trở lại — đó là hành vi đúng.

## Thống kê (stats.py) — đối chiếu độc lập trong selftest

`wilson_ci` · `clopper_pearson_ci` · `cp_lower_bound` (một phía) · `acceptance_plan`
(tái lập đúng n=846, c=17 cho p0=0.97) · `required_n_for_halfwidth` · `ppi_mean_ci`
(Angelopoulos et al. 2023) · `stratified_mean_ci` (có FPC).

## Test

```bash
.venv/bin/python -m pipeline.ground_truth.selftest   # 99 assertions, exit 0 = pass
bash scripts/run_all_selftests.sh                   # 321 toàn repo, so với mốc đã chốt
```

Kiểm: thống kê đối chiếu `scipy` độc lập + giá trị sách; `suspicion` đối chiếu census đo
trên `labels.csv` hiện tại (lớp trùng lặp đã đóng: dup_bbox=0, cross_col=0, union=0 —
lịch sử trong `docs/census_history.md`); sampling xác định + tổng design_weight ≈ N; grid
mù (không rò tier/rule) + manifest đủ trường ẩn; estimate đúng cả nhánh PPI-đủ-phủ lẫn
PPI-bỏ-qua, acceptance ACCEPT/REJECT.

Thêm từ 2026-08-03: `s3_signals` (gắn tín hiệu, chống lệch thế hệ corpus, nhãn ngoài từ
vựng ArcFace vẫn tính là đã chấm), `make_gold_batch` (hai tầng không giao nhau, trọng số
đúng, thứ tự hiển thị đã trộn), `estimate` LOẠI mẫu chủ đích khỏi precision (kiểm bằng ví
dụ số: 38/40 chứ không phải 48/60 nếu gộp nhầm), nạp verdict từ cả thư mục, và tương thích
ngược của `suspicion` khi không có tín hiệu S3.

## Phụ thuộc

`numpy`, `scipy`, `pandas`, `Pillow`, `PyYAML` (đọc `config/pipeline.yaml`); từ điển tuỳ
chọn qua `core.text.dictionary.load_qn_to_nom`. Đầu ra mặc định:
`dataset_out/ground_truth/`.
