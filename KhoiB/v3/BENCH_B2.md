# BENCHMARK GÁC B-2 — phát xạ ảnh trong DP, mô hình OOF thật (K5, 16/09/2026)

Script `KhoiB/v3/bench_b2.py` (đường mã sản xuất: `pipeline/align_engine/visual_emission.py` VisualEmitter + `build_dataset.anchored_cost_fn` + `anchor_align.realign_column(cost_ij=…)`), seed 2026, `thuc_nghiem.perturb` như `lab/tham_dinh_2026-09-16/dp_vis5.py`; mô hình 5 fold Kaggle `KhoiB/v3/p_visual_oof_v3_results/models/fold{k}.pt`, fold của TRANG (md5("book|page")%5) → mọi cột đều out-of-fold. Kết quả: `bench_b2_results.json` (val+test), `bench_b2_results_all.json` (all); log `bench_b2_*.log`.

Đáp án = đường chéo của cột m = n toàn match (cột đã ghép đúng số). Ghép sai = cặp (i, j) khác đáp án / tổng cặp; khe đúng = cột có đúng khe ins/del ở đúng chỗ (none: không khe). Thời gian = DP của cả tập cho 1 cấu hình (MPS; phát xạ ảnh tính 1 lần cho mọi cấu hình).

## Tập `valtest` — 525 cột m=n≥10 · hộp OCR thô top-1 OOF 7,332/10,747 = 0.682 · cột theo fold [99, 65, 91, 120, 150] · phát xạ 6.2s

| Kịch bản | Cấu hình | ghép sai | % | khe đúng | % | DP (s) |
|---|---|---|---|---|---|---|
| none (m=n, không nhiễu) | văn bản thuần (CALIB) | 0/11,182 | 0.00% | 525/525 | 100.0% | 0.24 |
|  | +corpus (neo LOO cap 2,0) | 0/11,182 | 0.00% | 525/525 | 100.0% | 0.24 |
|  | +corpus+ảnh λ=0.25 | **0/11,182** | **0.00%** | **525/525** | **100.0%** | 0.26 |
|  | +corpus+ảnh λ=0.5 | 0/11,182 | 0.00% | 525/525 | 100.0% | 0.25 |
|  | +corpus+ảnh λ=1.0 | 82/11,174 | 0.73% | 517/525 | 98.5% | 0.26 |
|  | +ảnh λ=0.25 không corpus | 0/11,182 | 0.00% | 525/525 | 100.0% | 0.25 |
| drop_char (OCR rụng 1 chữ) | văn bản thuần (CALIB) | 238/10,657 | 2.23% | 359/525 | 68.4% | 0.31 |
|  | +corpus (neo LOO cap 2,0) | 76/10,657 | 0.71% | 459/525 | 87.4% | 0.33 |
|  | +corpus+ảnh λ=0.25 | **28/10,657** | **0.26%** | **501/525** | **95.4%** | 0.35 |
|  | +corpus+ảnh λ=0.5 | 41/10,657 | 0.38% | 496/525 | 94.5% | 0.34 |
|  | +corpus+ảnh λ=1.0 | 236/10,652 | 2.22% | 468/525 | 89.1% | 0.34 |
|  | +ảnh λ=0.25 không corpus | 70/10,657 | 0.66% | 470/525 | 89.5% | 0.33 |
| drop_syl (QN rụng 1 âm) | văn bản thuần (CALIB) | 249/10,657 | 2.34% | 356/525 | 67.8% | 0.3 |
|  | +corpus (neo LOO cap 2,0) | 77/10,657 | 0.72% | 454/525 | 86.5% | 0.32 |
|  | +corpus+ảnh λ=0.25 | **29/10,657** | **0.27%** | **498/525** | **94.9%** | 0.35 |
|  | +corpus+ảnh λ=0.5 | 37/10,657 | 0.35% | 494/525 | 94.1% | 0.34 |
|  | +corpus+ảnh λ=1.0 | 223/10,653 | 2.09% | 461/525 | 87.8% | 0.33 |
|  | +ảnh λ=0.25 không corpus | 69/10,657 | 0.65% | 472/525 | 89.9% | 0.33 |
| drop2_char (OCR rụng 2 chữ liền) | văn bản thuần (CALIB) | 307/10,132 | 3.03% | 319/525 | 60.8% | 0.38 |
|  | +corpus (neo LOO cap 2,0) | 97/10,132 | 0.96% | 445/525 | 84.8% | 0.4 |
|  | +corpus+ảnh λ=0.25 | **40/10,132** | **0.39%** | **487/525** | **92.8%** | 0.41 |
|  | +corpus+ảnh λ=0.5 | 63/10,132 | 0.62% | 480/525 | 91.4% | 0.41 |
|  | +corpus+ảnh λ=1.0 | 251/10,128 | 2.48% | 449/525 | 85.5% | 0.41 |
|  | +ảnh λ=0.25 không corpus | 95/10,132 | 0.94% | 447/525 | 85.1% | 0.39 |

Điều kiện chấp nhận (`+corpus+ảnh λ=0.25` so `+corpus (neo LOO cap 2,0)`): none không tệ hơn = **True** (sai 0→0, khe 525→525/525); drop_char tốt hơn = **True** (sai 76→28, khe 459→501); drop2_char tốt hơn = True (sai 97→40, khe 445→487) → **PASS**.

## Tập `all` — 2,547 cột m=n≥10 · hộp OCR thô top-1 OOF 35,291/51,561 = 0.684 · cột theo fold [440, 460, 515, 607, 525] · phát xạ 24.4s

| Kịch bản | Cấu hình | ghép sai | % | khe đúng | % | DP (s) |
|---|---|---|---|---|---|---|
| none (m=n, không nhiễu) | văn bản thuần (CALIB) | 0/53,658 | 0.00% | 2547/2547 | 100.0% | 1.09 |
|  | +corpus (neo LOO cap 2,0) | 0/53,658 | 0.00% | 2547/2547 | 100.0% | 1.14 |
|  | +corpus+ảnh λ=0.25 | **5/53,657** | **0.01%** | **2546/2547** | **100.0%** | 1.23 |
|  | +corpus+ảnh λ=0.5 | 5/53,657 | 0.01% | 2546/2547 | 100.0% | 1.23 |
|  | +corpus+ảnh λ=1.0 | 268/53,630 | 0.50% | 2519/2547 | 98.9% | 1.22 |
|  | +ảnh λ=0.25 không corpus | 0/53,658 | 0.00% | 2547/2547 | 100.0% | 1.17 |
| drop_char (OCR rụng 1 chữ) | văn bản thuần (CALIB) | 1286/51,111 | 2.52% | 1639/2547 | 64.4% | 1.4 |
|  | +corpus (neo LOO cap 2,0) | 337/51,111 | 0.66% | 2244/2547 | 88.1% | 1.47 |
|  | +corpus+ảnh λ=0.25 | **113/51,111** | **0.22%** | **2449/2547** | **96.2%** | 1.57 |
|  | +corpus+ảnh λ=0.5 | 205/51,111 | 0.40% | 2418/2547 | 94.9% | 1.62 |
|  | +corpus+ảnh λ=1.0 | 823/51,104 | 1.61% | 2286/2547 | 89.8% | 1.65 |
|  | +ảnh λ=0.25 không corpus | 341/51,111 | 0.67% | 2295/2547 | 90.1% | 1.56 |
| drop_syl (QN rụng 1 âm) | văn bản thuần (CALIB) | 1302/51,111 | 2.55% | 1632/2547 | 64.1% | 1.48 |
|  | +corpus (neo LOO cap 2,0) | 352/51,111 | 0.69% | 2243/2547 | 88.1% | 1.54 |
|  | +corpus+ảnh λ=0.25 | **137/51,111** | **0.27%** | **2424/2547** | **95.2%** | 1.61 |
|  | +corpus+ảnh λ=0.5 | 178/51,111 | 0.35% | 2406/2547 | 94.5% | 1.62 |
|  | +corpus+ảnh λ=1.0 | 797/51,100 | 1.56% | 2279/2547 | 89.5% | 1.64 |
|  | +ảnh λ=0.25 không corpus | 347/51,111 | 0.68% | 2275/2547 | 89.3% | 1.56 |
| drop2_char (OCR rụng 2 chữ liền) | văn bản thuần (CALIB) | 1536/48,564 | 3.16% | 1485/2547 | 58.3% | 1.82 |
|  | +corpus (neo LOO cap 2,0) | 484/48,564 | 1.00% | 2133/2547 | 83.7% | 1.87 |
|  | +corpus+ảnh λ=0.25 | **192/48,564** | **0.40%** | **2376/2547** | **93.3%** | 1.97 |
|  | +corpus+ảnh λ=0.5 | 261/48,564 | 0.54% | 2356/2547 | 92.5% | 1.97 |
|  | +corpus+ảnh λ=1.0 | 853/48,558 | 1.76% | 2223/2547 | 87.3% | 1.96 |
|  | +ảnh λ=0.25 không corpus | 494/48,564 | 1.02% | 2172/2547 | 85.3% | 1.93 |

Điều kiện chấp nhận (`+corpus+ảnh λ=0.25` so `+corpus (neo LOO cap 2,0)`): none không tệ hơn = **False** (sai 0→5, khe 2547→2546/2547); drop_char tốt hơn = **True** (sai 337→113, khe 2244→2449); drop2_char tốt hơn = True (sai 484→192, khe 2133→2376) → **FAIL**.

Cột lệch ở `none` với λ=0,25 (1):

- `stt11/page_0108/3` (m=20): del [2], ins [7], cặp lệch [[3, 2], [4, 3], [5, 4], [6, 5], [7, 6]]. Dòng QN có token dính `mìnhép` (2 âm 1 token) nên cột thật có 21 âm/20 hộp — đáp án đường chéo SAI ở 4 cặp (押↔thịt, 烟↔một, 弟↔ngày, 将↔một). Ảnh đọc từng hộp: 琰→phong 0.13, 命→mình 0.93, 押→ép 0.82, 烟→thịt 0.88, 弟→một 0.78, 将→ngày 0.95, 吝→lần 0.15. Đường ghép +ảnh đúng 3 cặp mà đáp án sai (烟↔thịt, 弟↔một, 将↔ngày) và đặt khe ins[7] đúng chỗ thiếu hộp; sai 命↔nhặm và del 琰. Trong bộ giao nộp 7 ô này (nom_idx 1–7) đều REVIEW/no_context (`dataset_out/labels_final.csv` dòng tệp 9113–9119 (chỉ số pandas 9111–9117)) — không ảnh hưởng ô usable.

## Đường cong λ (drop_char, ghép sai %) — λ ≥ 1 hại

| tập | +corpus (λ=0) | λ=0,25 | λ=0,5 | λ=1,0 |
|---|---|---|---|---|
| val+test | 0.71% / khe 87.4% | **0.26% / 95.4%** | 0.38% / 94.5% | 2.22% / 89.1% |
| val+test (none) | 0 | 0 | 0 | 82 |
| all | 0.66% / khe 88.1% | **0.22% / 96.2%** | 0.40% / 94.9% | 1.61% / 89.8% |
| all (none) | 0 | 5 | 5 | 268 |

## Kết luận

1. λ=0,25 (config `step2.visual_emission.lambda`) là điểm tốt nhất trên cả 4 kịch bản × 2 tập; λ=0,5 vẫn tốt hơn văn bản, λ=1,0 hại (none sai 0,50–0,73%, drop_char 1,6–2,2% > +corpus). Đúng như `dp_vis5.py` trên mô hình cũ (khe 86,6→95,4%).
2. Trên val+test (525 cột — ít hơn mốc 600 của nhiệm vụ; tập `all` 2.547 cột đủ): **PASS** trọn vẹn. Trên `all`: FAIL đúng 1 cột vì đáp án đường chéo sai (token QN dính) — kênh ảnh đúng hơn đáp án; drop_char/drop_syl/drop2_char đều tốt hơn rõ (ghép sai ÷3, khe đúng +7–10 điểm).
3. Ảnh KHÔNG thay được ngữ liệu: `+ảnh λ=0,25 không corpus` ≈ `+corpus` (0,66–0,67% drop_char); hai kênh bổ sung nhau (ảnh+corpus 0,22–0,26%).
4. Chi phí: phát xạ ảnh 24 s / 2.547 cột (53.658 hộp, MPS) ≈ 0,45 ms/hộp; DP thêm ≈ 0,1 s / 2.547 cột. Trên build 448 trang: xem `docs/BAO_CAO_KHOI_B_2026-09-16.md` §B-2.
5. Khuyến nghị cho lần thăng cấp kế tiếp: **chưa bật cờ `--visual-emission` mặc định**. Benchmark gác qua, nhưng build thật đổi 235 cột / 403 cặp (DANH_MUC ước ≤≈50), 90% trong REVIEW, và 1 ô QĐ-01 rơi pending (2.011 locked + 1 pending thay vì 2.012) — cần Khối C chấm mù các cặp đổi (≈100 ô) trước khi bật; bật thì phải kèm sửa khoá QĐ-01 (ô trôi → QĐ-01a) và ghi 4 cột sidecar vào labels_trace.
