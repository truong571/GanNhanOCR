# KHỐI B-1 · THẾ HỆ v3 — CNN ÂM TIẾT 5-FOLD OUT-OF-FOLD TRÊN KAGGLE

`KhoiB/v3/` là bản **sửa** của `KhoiB/` theo thẩm định K1 (16/09, `lab/tham_dinh_2026-09-16/k1_b1.py`
→ `dataset_out_v3/khoi_b/k1_b1_tham_dinh.json`). **Không** sửa tệp nào trong `KhoiB/` cũ. **Không** huấn luyện trên Mac —
chỉ đóng gói để chạy Kaggle GPU y như lần trước.

## 1. Khác bản cũ ở 4 điểm K1 (+2 chỗ nhỏ)

| # | Bản cũ `KhoiB/` (K1 đo được) | Bản v3 `KhoiB/v3/` |
|---|---|---|
| 1 | `nom_idx` = `groupby(book,page,column).cumcount()` trên `labels_final` CŨ không có `nom_idx` (`train_oof_cnn.py:131-132`) → lệch nom_idx thật ở **7.865/82.629 ô (9,52%)**, 741/4.021 cột; join sang v3 bằng khoá này sai `ocr_char` **7.389 ô** | đọc thẳng `nom_idx`/`syl_idx` của `labels_final.csv` v3; `train_oof_cnn_v3.py` **đối chiếu khoá csv ↔ npz từng dòng** (assert), khoá `(book,page,column,nom_idx)` duy nhất |
| 2 | không lưu mô hình từng fold → B-2 (phát xạ cho hộp bất kỳ trong DP) không dùng được | lưu `models/fold{k}.pt` = `{"state","classes","fold","fold_formula",…}` (cùng khuôn `lab/.../model.pt`); API `load_fold_model()` + `predict_logprobs()` trong script |
| 3 | chỉ `p_visual` (âm CŨ), `argmax`, `max_prob`; `p_visual=0` cho 4.843 ô âm ∉ lớp **không phân biệt** với 1.032 ô p làm tròn 0,0000 (`:239`, `:258`) | `probs_oof.npz` = **đủ vector** `LP` (N×C float16, log-softmax; `P = exp(LP)`); csv có `p_syl_v3` theo **âm v3**, để **trống** khi âm ∉ lớp, ghi `%.6g`; thêm `top5`, `p_left/p_right` (âm ô kề `syl_idx∓1` cùng cột) |
| 4 | crop cắt từ bbox CŨ (`lab/gan_nhan_2026-09-13/crops.npz`, 82.780) — v3 đổi bbox **12.505** ô có p (10.337 usable) | `crops_v3.npz`: 83.239 crop từ bbox v3 bằng **đúng** `cut()` của lab (import trực tiếp, pad 0,08, 64×64); kiểm: 69.917/69.917 ô bbox không đổi → crop **byte-identical** với bản cũ |
| + | fold = `md5("{book}_{page}") % 5` | fold = `int(md5(f"{book}\|{page}").hexdigest(),16) % 5` (ghi trong `summary.json.fold_formula` và trong từng `.pt`); trang/fold **80/85/88/104/91** (448 trang) |
| + | lớp = âm ≥5 ô **split=train** ∧ GOLD/SYLLABLE (706) | lớp = âm ≥5 ô tier ∈ {GOLD, SYLLABLE} **toàn bộ** (không dùng split — v3 không còn cột `split`) → **804 lớp** (không phải 706: 695/706 lớp cũ nằm trong 804); `--classes KhoiB/p_visual_oof_results/classes.json` nếu muốn ép đúng 706 lớp cũ để so sánh |

Giữ nguyên: kiến trúc CNN (4 khối conv 32-64-128-256 + emb 256), AdamW 2e-3, OneCycle, label-smoothing 0,1, lấy mẫu cân bằng
1/√n, tăng cường affine ±8%/±3px, 15 epoch cố định, val **chỉ in** (không chọn mô hình → không rò), fp16 khi CUDA.

Số kỳ vọng trên bộ v3 (đo từ `dataset_out/labels_final.csv` md5 `1a2d8e0c…`, 83.239 dòng): ô GOLD+SYLLABLE 70.326, trong 804 lớp 69.176;
train/val theo fold: 56.773/12.403 · 56.178/12.998 · 55.559/13.617 · 53.006/16.170 · 55.188/13.988; OOF dự đoán mọi ô của fold
(14.948 · 15.639 · 16.491 · 19.414 · 16.747 = 83.239); âm ∉ lớp 3.350 ô (REVIEW 2.199, CHAR_A 923, SYL 197, CHAR_B 31);
`image_md5` GOLD+SYL xuất hiện ở ≥2 fold: 0.

## 2. Tệp trong `KhoiB/v3/`

| Tệp | Vai trò |
|---|---|
| `make_crops_v3.py` | Mac: cắt `crops_v3.npz` (17 s). Đã chạy — `crops_v3.npz` 76,8 MB, ok 83.239/83.239 |
| `crops_v3.npz` | `X (83239,64,64) uint8`, `ok`, khoá `book/page/column/nom_idx/syl_idx/bbox`, `labels_md5` |
| `train_oof_cnn_v3.py` | script huấn luyện (Kaggle/MPS/CPU); `--smoke` = kiểm mã 300 ô/1 epoch (~10 s) |
| `pack_for_kaggle_v3.py` | Mac: nén `KhoiB_v3_kaggle_dataset.zip` (kiểm md5/khoá trước khi nén) |
| `KhoiB_v3_kaggle_dataset.zip` | **81,3 MB**: `crops_v3.npz` + `labels_final.csv` + `train_oof_cnn_v3.py` + `MANIFEST.json` |
| `MANIFEST.json` | md5 từng tệp + git HEAD lúc đóng gói (notebook đối chiếu lại trên Kaggle) |
| `kaggle_run_khoib_v3.ipynb` | notebook 4 bước (cấu trúc như `KhoiB/kaggle_run_khoib.ipynb`) |
| `p_visual_oof_v3_results/` | **chỗ thả kết quả** (giải nén `p_visual_oof_v3_results.zip` tải từ Kaggle) |

## 3. Bốn bước trên Kaggle (~12–15 phút GPU T4/P100)

1. **Tải dữ liệu:** kaggle.com → *Datasets* → *+ New Dataset* → kéo `KhoiB/v3/KhoiB_v3_kaggle_dataset.zip` → tên `khoib-v3-data` → *Create*.
2. **Notebook + GPU:** *+ New Notebook* → *Accelerator: GPU T4 x2* (hoặc P100) → *+ Add Input* → chọn `khoib-v3-data`.
3. **Chạy:** *File → Upload Notebook* → `KhoiB/v3/kaggle_run_khoib_v3.ipynb` → *Run All*. Cell 1 đối chiếu md5 với `MANIFEST.json`
   (phải in `KHỚP` ×3); cell 2 huấn luyện 5 fold (mỗi fold ≈2,5–3 phút); cell 3 kiểm đủ 9 tệp và in `summary.json`.
4. **Tải về:** *Output* → `p_visual_oof_v3_results.zip` (≈ 60–90 MB: `models/` 5×~6 MB + `probs_oof.npz` ≈ 60 MB + csv ≈ 12 MB)
   → giải nén vào **`KhoiB/v3/p_visual_oof_v3_results/`** sao cho có
   `p_visual_oof_v3.csv`, `probs_oof.npz`, `summary.json`, `classes.json`, `models/fold0..4.pt`.

## 4. Schema kết quả

`p_visual_oof_v3.csv` (83.239 dòng, thứ tự = `labels_final.csv`):
`book,page,column,nom_idx,syl_idx,bbox,fold,tier,tier_v3,syllable_v3,syl_in_classes,ok,p_syl_v3,argmax,max_prob,top5,p_left,p_right`
— `p_syl_v3` = P(âm v3 | crop) out-of-fold (**trống** nếu `syl_in_classes=0`); `top5` = JSON `[[âm, p]×5]`;
`p_left`/`p_right` = P(âm của ô `syl_idx−1` / `syl_idx+1` cùng cột | crop) (trống nếu không có ô kề hoặc âm kề ∉ lớp).

`probs_oof.npz`: `LP` (83.239 × C float16, log-softmax OOF; `P = np.exp(LP.astype(np.float32))`; NaN khi `ok=0`), `classes`, `fold`, `ok`,
khoá `book/page/column/nom_idx`, `labels_md5`.

`summary.json`: `fold_formula`, `pages_per_fold`, `folds[]` (train/val/pred, `val_top1/top5`, giây), `by_fold_oof`, `by_tier_v3`, `by_tier`
(top-1/top-5, p50/p10 của `p_syl_v3`, % p<0,05, % argmax đúng ∧ P≥0,9), **`e2b`** (AUC P(âm mình) của ô CHAR_A vs P(âm ô kề) — loại kề cùng âm;
% p<0,05 hai phía; theo fold), `b3_gate_review` (REVIEW: `argmax == âm v3 ∧ max_prob ≥ 0,9 / 0,8`).

Dùng lại mô hình cho B-2 (hộp bất kỳ, ví dụ hộp OCR thô như `lab/tham_dinh_2026-09-16/dp_vis5.py`):
```python
import sys; sys.path.insert(0, "KhoiB/v3"); import train_oof_cnn_v3 as T
net, classes, dev = T.load_fold_model("KhoiB/v3/p_visual_oof_v3_results/models/fold2.pt")   # fold của TRANG đang xét
LP = T.predict_logprobs(net, X_boxes, dev)        # X_boxes (M,64,64) uint8 cắt bằng lab cut() → (M, C) log P(âm | crop)
```
Chọn `fold = T.page_fold(book, page)` để mô hình **chưa từng thấy trang đó** (out-of-fold thật).

## 5. Đã kiểm trên Mac (không huấn luyện thật)

- `make_crops_v3.py`: 83.239/83.239 ok, 0 bbox lỗi, 0 trang thiếu ảnh, 17 s; 69.917/69.917 ô bbox không đổi → byte-identical với `lab/.../crops.npz`.
- `train_oof_cnn_v3.py --smoke` (301 ô/15 cột/15 trang, 5 fold, 1 epoch, MPS): 0,14 phút; ra đủ `classes.json`, `models/fold0..4.pt`,
  `probs_oof.npz`, `p_visual_oof_v3.csv`, `summary.json` đúng schema; nạp lại `fold2.pt` dự đoán lại = `LP` đã lưu (sai số float16 ≤ 0,002). Đầu ra khói đã xoá.
- `pack_for_kaggle_v3.py`: md5 npz ↔ csv khớp, khoá khớp 83.239/83.239, zip 81,3 MB.

## 6. K4 — mã B-2 / B-3 (16/09, sau khi có kết quả Kaggle trong `p_visual_oof_v3_results/`)

| Việc | Mã | Chạy |
|---|---|---|
| B-2 phát xạ ảnh trong DP | `pipeline/align_engine/visual_emission.py` (VisualEmitter: fold theo trang = md5("book\|page")%5, hộp OCR thô, `cut()` == lab byte-identical, trung tính log 0,5, `cost_ij = text + λ·min(−logP, 12)`, khe +λ·8 — chép dp_vis5.py); `anchor_align.realign_column/posterior_matches(cost_ij=, cost_del=, cost_ins=)` (mặc định KHÔNG đổi: ops+posterior y hệt HEAD trên 4.029 cột); `build_dataset --visual-emission` (mặc định TẮT; config `step2.visual_emission`; thiếu mô hình → cảnh báo, chạy văn bản thuần; `--strict` dừng) → cột `p_visual_syl, visual_fold, visual_argmax, visual_max_p` ở CUỐI labels.csv + sidecar `labels_trace_visual.csv` + `summary.json.visual_emission`; export thêm 4 cột ấy vào `labels_trace.csv` chỉ khi có | `.venv/bin/python -m pipeline.align_engine.build_dataset --reseg detector --visual-emission --out <DS_OUT khác>` |
| B-3 cổng thị giác REVIEW→SYL | `pipeline/tools/visual_syl_gate.py` (chỉ đo, không đổi nhãn) | `.venv/bin/python -m pipeline.tools.visual_syl_gate` → `visual_syl_candidates.csv`, `visual_syl_gate_report.json` |
| Benchmark gác B-2 | `bench_b2.py` (đường mã sản xuất, mô hình OOF thật) | `.venv/bin/python KhoiB/v3/bench_b2.py [--set all]` → `bench_b2_results*.json`, log `bench_b2_*.log` |
| So build toàn bộ | `b2_compare_builds.py` | `b2_build_compare.json`, `b2_full_build/` (summary 2 bên, `labels_trace_visual.csv`, log) |
| Selftest | `pipeline.align_engine.visual_emission_selftest` (35), `pipeline.tools.visual_syl_gate_selftest` (13) — đã vào `scripts/run_all_selftests.sh` (mốc 1035 → 1083) | |

**Số đo (mô hình Kaggle 804 lớp, val top-1 0,775, E2b AUC 0,945; labels_md5 khớp `1a2d8e0c…`):**

- Benchmark rụng-1-chữ, cột m=n (đường mã sản xuất, neo LOO cap 2,0 của engine — dp_vis5 dùng cap 0):

| Tập | Kịch bản | +corpus (neo LOO cap 2,0) | +corpus+ảnh λ=0.25 | λ=1,0 |
|---|---|---|---|---|
| val+test 525 cột | drop_char | 76/10657 (0.71%) · khe 87.4% | **28/10657 (0.26%) · khe 95.4%** | 236/10652 (2.22%) · khe 89.1% |
| | drop_syl | 77/10657 (0.72%) · khe 86.5% | 29/10657 (0.27%) · khe 94.9% | 223/10653 (2.09%) · khe 87.8% |
| | none | 0/11182 (0.00%) · khe 100.0% | 0/11182 (0.00%) · khe 100.0% | 82/11174 (0.73%) · khe 98.5% |
| all 2547 cột | drop_char | 337/51111 (0.66%) · khe 88.1% | **113/51111 (0.22%) · khe 96.2%** | 823/51104 (1.61%) · khe 89.8% |
| | drop_syl | 352/51111 (0.69%) · khe 88.1% | 137/51111 (0.27%) · khe 95.2% | 797/51100 (1.56%) · khe 89.5% |
| | none | 0/53658 (0.00%) · khe 100.0% | 5/53657 (0.01%) · khe 100.0% | 268/53630 (0.50%) · khe 98.9% |

  Điều kiện chấp nhận (none không tệ hơn ∧ drop_char tốt hơn): **PASS** trên val+test; trên `all` FAIL đúng 1 cột
  (`stt11/page_0108/c3`, 5 cặp): dòng QN có token dính `mìnhép`, ảnh nói hộp 4–7 = ép/thịt/một/ngày (p 0,78–0,95) tức
  kênh ảnh ĐÚNG, "đáp án đường chéo" sai — xem `none_deviations` trong `bench_b2_results_all.json`. λ ≥ 1 hại (none sai 0,50–0,73%).
- Build toàn bộ 448 trang `--no-crops` (văn bản thuần vs +ảnh): cột đổi đường ghép **235** / 4.030, cặp đổi
  403 mỗi bên (DANH_MUC ước ≤≈50 ô — thực tế lớn hơn ≈8×); 90% cặp đổi nằm trong REVIEW (362/403 bị phá, 350/403 tạo),
  216/235 cột là cột n_ocr ≠ n_qn; cặp từ điển xác nhận bị phá 23, tạo 15; ô đổi tier_v3 68
  ({'REVIEW->SYL': 22, 'CHAR_B->CHAR_A': 11, 'SYL->CHAR_B': 2, 'CHAR_A->CHAR_B': 11, 'SYL->REVIEW': 5, 'REVIEW->CHAR_B': 3, 'CHAR_B->REVIEW': 9, 'CHAR_A->REVIEW': 2, 'CHAR_A->SYL': 1, 'REVIEW->CHAR_A': 1, 'CHAR_B->SYL': 1}); usable 70,368 → 70,381. p_visual_syl trung vị của cặp ảnh tạo = 0,024
  (ảnh chủ yếu "bác" cặp cũ chứ không "xác nhận" cặp mới). **Không thăng tier** — cờ mặc định tắt.
- B-3 (crop bbox v3, `p_visual_oof_v3.csv`): REVIEW tier_v3 12,897 → loại QĐ-01 đã dùng được 36,
  not_plausible 188, âm ∉ lớp 2,199 → đủ điều kiện 10,662;
  argmax == âm 5,808; **qua cổng p ≥ 0,9: 1,025** (≥ 0,8: 1,827) — DANH_MUC ước 894–961.
  FAR CHAR_A (p âm kề ≥ 0,9 ∧ p mình < 0,5 — trùng với argmax == kề ∧ p_kề ≥ 0,9 vì p ≥ 0,9 ⇒ argmax): 678/46,717
  = 1.45% Wilson [0.01347, 0.01564] (ở 0,8: 1248); độ nhạy cổng trên CHAR_A 11,871/47,033.
  Chờ Khối C (100–150 ô qua cổng, 0/150 lỗi) rồi mới bàn rule `visual_syl_gate`.

## 7. K5 — chạy B-2/B-3 với mô hình OOF thật + báo cáo Khối B (16/09)

Báo cáo tổng hợp: `docs/BAO_CAO_KHOI_B_2026-09-16.md`. Tệp K5 trong thư mục này:

| Tệp | Nội dung |
|---|---|
| `BENCH_B2.md` (sinh bởi `bench_b2_md.py` từ `bench_b2_results*.json`) | benchmark gác đầy đủ: none/drop_char/drop_syl/**drop2_char** × λ ∈ {0; 0,25; 0,5; 1,0} × {val+test 525, all 2.547 cột}, kèm giây; `bench_b2.py` thêm drop2_char + thời gian |
| `b2_build_compare.json`, `b2_build_compare_p60.json`, `b2_changed_cells.py/.json`, `b2_full_build/` | build `--limit 60` (180 trang: +0,116 s/trang) và FULL 448 trang (276 → 320 s, +16%) văn bản thuần vs `--visual-emission`; 235 cột/403 cặp đổi, 19 ô usable đổi âm, 1 ô QĐ-01 pending |
| `E1_E2_OOF.md`, `e1_e2_oof.py/.json` | E1 top-1/top-5/hạng/p50 theo tier_v3 × sách × fold; E2b theo fold/sách; đối chiếu 1.030/1.846 ↔ 1.025/1.827 (khớp sau khi trừ 36+188 ô bị loại) |
| `visual_syl_candidates.csv`, `visual_syl_gate_report.json`, `b3_sample.html` (`b3_sample_html.py`) | B-3 chạy trên `dataset_out/labels_final.csv`; 20 crop qua cổng + ngữ cảnh để nhìn nhanh |
| `k5_register_shift.py`, `k5_register_shift_null.py`, `k5_register_shift_thr{0.8,0.5}_len3.{json,csv}`, `k5_stt11_p0126_c6_crops_nom4-14.png` | phát hiện chuỗi "thanh ghi trượt" trong bộ giao nộp: 114 chuỗi/404 ô (371 usable, 8 QĐ-01) ở p≥0,8, đối chứng hoán vị 1–3 chuỗi; ví dụ nhìn mắt stt11/page_0126/c6 |
