# BÁO CÁO KHỐI B — KÊNH ẢNH (B-1 … B-5), 16/09/2026

**Phạm vi:** `DANH_MUC_SUA_DOI_CUOI_2026-09-16.md §1B` (B-1..B-5). Khối B là *đo*, không đổi bộ giao nộp:
`dataset_out/labels_final.csv` md5 `1a2d8e0ca0a1199f11ff514135051c15` và `re-dataset/labels.csv` md5
`cd530ec985288a0bf25edacb600ea3ca` **không đổi** trước/sau (§7). Mọi đầu ra ở `KhoiB/`, `KhoiB/v3/`, `KhoiB/h2/`,
`dataset_out_v3/khoi_b/`, `lab/tham_dinh_2026-09-16/`. Chuỗi việc: K1 (thẩm định B-1 cũ) → B-1' (đóng gói v3, chạy Kaggle)
→ K4 (mã B-2/B-3) → K5 (báo cáo này: chạy B-2/B-3 với mô hình OOF thật, E1/E2, tổng hợp B-4/B-5).

## 0. Tóm tắt 10 dòng

| # | Kết luận | Số | Nguồn |
|---|---|---|---|
| B-1 | B-1 cũ (`KhoiB/`) **không dùng được** để nối v3: khoá `nom_idx` là cumcount | lệch 7.865/82.629 ô (9,52%), 741/4.021 cột; join sai `ocr_char` 7.389 ô | `dataset_out_v3/khoi_b/k1_b1_tham_dinh.json` |
| B-1' | CNN âm tiết 5-fold OOF trên bộ v3 (Kaggle, 804 lớp, 22,6 phút GPU) | val top-1 **0,775** / top-5 0,918; OOF mọi ô 74,42% (79.889 ô ∈ lớp); **E2b AUC 0,9446** (fold 0,940–0,949) — điều kiện B-1 "≈0,95" ĐẠT | `KhoiB/v3/p_visual_oof_v3_results/summary.json`, `KhoiB/v3/E1_E2_OOF.md` |
| B-2 gác | Benchmark rụng chữ, λ=0,25: ghép sai ÷3, khe đúng 88→96% | val+test 525 cột **PASS**; all 2.547 cột FAIL đúng 1 cột do đáp án sai (QN dính token) | `KhoiB/v3/BENCH_B2.md` |
| B-2 thật | Build 448 trang `--visual-emission` so văn bản thuần | 235 cột / 403 cặp đổi (DANH_MUC ước ≤≈50 → **8×**); 90% cặp đổi trong REVIEW; 19 ô usable đổi ÂM; 1 ô QĐ-01 rơi pending; +43 s/448 trang (+16%) | `KhoiB/v3/b2_build_compare.json`, `b2_changed_cells.json` |
| B-2 quyết định | **Không bật cờ mặc định** ở lần thăng cấp kế tiếp; giữ sidecar | | §2.4 |
| B-3 | Cổng REVIEW→SYL `argmax == âm ∧ p ≥ 0,9` | đủ điều kiện 10.662; qua cổng **1.025** (≥0,8: 1.827); DANH_MUC ước 894–961; FAR CHAR_A **1,45%** [1,35; 1,56] | `KhoiB/v3/visual_syl_gate_report.json`, `visual_syl_candidates.csv`, `b3_sample.html` |
| Phát hiện mới | Chuỗi "thanh ghi trượt" trong bộ giao nộp (hộp lệch chữ OCR ≥3 ô liền) | 114 chuỗi / 110 cột / **404 ô (371 usable, 8 ô QĐ-01)** ở p≥0,8; ví dụ đã nhìn mắt: `stt11/page_0126/c6` nom 5–12 (ô QĐ-01 nom 5 crop là 羅 'là', không phải 𠊚) | `KhoiB/v3/k5_register_shift_thr0.8_len3.{json,csv}`, `k5_stt11_p0126_c6_crops_nom4-14.png` |
| B-4 | Hàng đợi lớp cho người ký (H2-lite) | **122 lớp** (44 E3-MỘT/966 ô + 10 mờ/174 + 68 khác/291 = 1.431 ô GOLD thiểu số); 37 lớp HAI giữ; 119 lớp ngoài hàng đợi/495 ô | `KhoiB/h2/lop_queue_summary.json`, `lop_review.html` (2.040 thumb), `lop_decisions_template.csv` |
| B-5 | Khoá ô thay khoá cột (mô phỏng `--lock-scope cell/cell_fallback`) | detector trong cột từng khoá 58,5% → 96,8% (cell) / 94,3% (cell_fallback); nhưng `cell` tạo 29 cặp IoU≥0,5 cùng cột, F1 40→98 → **giữ mặc định `col`**; `cell_fallback` là ứng viên | `KhoiB/v3/b5/summary_b5.json` |
| Người | 4 việc người (§6): ký 122 lớp, Khối C mẻ 600 + tầng 100–150 ô qua cổng + tầng chuỗi trượt, QĐ-01a 1 ô, quyết định cell_fallback | | §6 |

---

## 1. B-1 · CNN âm tiết out-of-fold

### 1.1 Thẩm định K1 bản cũ `KhoiB/` (`lab/tham_dinh_2026-09-16/k1_b1.py` → `dataset_out_v3/khoi_b/k1_b1_tham_dinh.json`)

| Nghi vấn | Đo được | Hệ quả |
|---|---|---|
| (1) `nom_idx` = `groupby(book,page,column).cumcount()` trên `labels_final` CŨ (`KhoiB/train_oof_cnn.py:131-132`) | lệch nom_idx thật ở **7.865/82.629 ô (9,52%)**, 741/4.021 cột; join sang v3 bằng khoá này sai `ocr_char` **7.389** ô, `syllable` 8.083 | khoá join sai ở mọi cột có del/khe → không nối được với v3 |
| (2) không lưu mô hình fold | đúng: chỉ có `p_visual_oof.csv` | B-2 (hộp bất kỳ) không dùng được |
| (3) chỉ `p_visual/argmax/max_prob` của âm CŨ; `p_visual = 0` cho 4.843 ô âm ∉ 706 lớp lẫn với 1.032 ô p làm tròn 0 | đúng | B-3 với âm v3 chỉ dùng khi argmax == âm mới |
| (4) crop từ bbox CŨ | v3 đổi bbox 12.505 ô có p (10.337 usable), IoU p50 0,71 | p không còn ứng với crop v3 |
| fold, split | fold = md5("book_page")%5 tái lập 82.780/82.780; 0 trang thuộc 2 fold; 0 md5 ô ở ≥2 fold; val top-1 theo fold khớp summary (0,7723/0,7616/0,7698/…) | phần *huấn luyện* đúng — chỉ hỏng ở khoá và bbox |

→ **Không sửa `KhoiB/` cũ**; làm lại trong `KhoiB/v3/` (README §1).

### 1.2 B-1' — bản v3 (`KhoiB/v3/`, chạy Kaggle)

- Dữ liệu: `crops_v3.npz` 83.239 crop cắt bằng đúng `lab/gan_nhan_2026-09-13/thi_giac_am_tiet.cut()` (`:53`, pad 0,08, 64×64) từ bbox v3; 69.917/69.917 ô bbox không đổi → crop byte-identical bản cũ. Khoá `(book,page,column,nom_idx)` đọc thẳng từ `labels_final.csv` (assert từng dòng, `train_oof_cnn_v3.py`).
- Fold = `int(md5(f"{book}|{page}").hexdigest(),16) % 5` (ghi trong `summary.json.fold_formula` và từng `models/fold{k}.pt`), trang/fold 80/85/88/104/91.
- Lớp = âm ≥5 ô tier ∈ {GOLD, SYLLABLE} toàn bộ → **804 lớp** (695/706 lớp cũ ⊂ 804). Kiến trúc/siêu tham số giữ nguyên; val chỉ in, không chọn mô hình.
- Kết quả (`summary.json`): 22,57 phút (T4, fp16); train/val theo fold 56.773/12.403 · 56.178/12.998 · 55.559/13.617 · 53.006/16.170 · 55.188/13.988; **val top-1 0,775** (0,7739/0,7821/0,7726/0,7808/0,7654), top-5 0,918; `labels_md5` = `1a2d8e0c…` khớp bộ giao nộp; kiểm nạp lại `fold*.pt` dự đoán lại = `LP` (max|Δ| 0,008 float16).

### 1.3 E1/E2 out-of-fold (`KhoiB/v3/E1_E2_OOF.md`, script `e1_e2_oof.py`, kiểm csv↔npz max|Δp| 6e-7)

| tier_v3 | n ∈ lớp | top-1 | top-5 | p50 | %p<0,05 | §3 (706 lớp, trang test) |
|---|---|---|---|---|---|---|
| CHAR_A | 47.033 | **77,31%** | 91,77% | 0,698 | 13,2% | 77,8% |
| CHAR_B | 3.280 | 76,19% | 90,18% | 0,641 | 14,8% | 67,6% |
| SYL | 18.878 | **78,13%** | 92,04% | 0,696 | 12,8% | 81,9% |
| REVIEW | 10.698 | **54,62%** | 71,70% | 0,196 | 35,6% | 45,6% |
| tất cả | 79.889 | 74,42% | 89,08% | 0,645 | 16,2% | — |

Theo sách: stt11 76,3% > stt2 73,8% > stt4 73,2% (REVIEW stt11 thấp nhất 50,9%). Theo fold: OOF top-1 73,4–75,2%, E2b AUC 0,9395–0,9486.
**E2b toàn bộ 0,9447** (47.033 ô đúng / 85.914 âm kề sai; p<0,05 bắt 91,1% sai, oan 13,2%) — §3 đo 0,946 chỉ trên trang test;
nay đo trên mọi trang, out-of-fold, ổn định giữa fold và sách → điều kiện B-1 **đạt**.

---

## 2. B-2 · Phát xạ ảnh trong DP

### 2.1 Mã (K4, đã có selftest 35+13 trong `scripts/run_all_selftests.sh`, mốc 1035 → 1083)

`pipeline/align_engine/visual_emission.py`: `VisualEmitter` nạp lười mô hình theo fold của trang (`page_fold` `:54`), `cut_box` `:69` chép
`cut()` lab (pad 0,08), `emission_from_LP` `:225` (trung tính log 0,5 cho âm ∉ lớp / hộp không cắt được), `cost_matrix` `:255`
`λ·min(−logP, 12)`, `gap_costs` `:267` khe = COST_DEL/INS + λ·8 (chép `lab/tham_dinh_2026-09-16/dp_vis5.py:63,101-104`).
`anchor_align.realign_column(cost_ij=…)` `:90-155` và `posterior_matches(cost_ij=…)` `:232` — mặc định không truyền → ops/posterior y hệt HEAD
(K4 kiểm 4.029/4.029 cột `cols.pkl`). `build_dataset --visual-emission` `:960-967` (mặc định TẮT, config `step2.visual_emission`
`config/pipeline.yaml:98-103`), thiếu mô hình → in lý do, chạy văn bản thuần (`:1057`), `--strict` dừng; 4 cột sidecar ở CUỐI `labels.csv`
chỉ khi bật (`:1546`) + `labels_trace_visual.csv` + `summary.json.visual_emission`.

### 2.2 Benchmark gác (`KhoiB/v3/BENCH_B2.md`, `bench_b2.py`, mô hình OOF thật, fold theo trang)

| tập | kịch bản | +corpus (neo LOO cap 2,0) | **+corpus+ảnh λ=0,25** | λ=0,5 | λ=1,0 |
|---|---|---|---|---|---|
| val+test 525 cột | none | 0/11.182 · khe 100% | **0 · 100%** | 0 · 100% | 82 (0,73%) · 98,5% |
| | drop_char | 76 (0,71%) · 87,4% | **28 (0,26%) · 95,4%** | 41 · 94,5% | 236 (2,22%) · 89,1% |
| | drop_syl | 77 (0,72%) · 86,5% | **29 (0,27%) · 94,9%** | 37 · 94,1% | 223 · 87,8% |
| | drop2_char | 97 (0,96%) · 84,8% | **40 (0,39%) · 92,8%** | 63 · 91,4% | 251 · 85,5% |
| all 2.547 cột | none | 0/53.658 · 100% | **5 (0,01%) · 2.546/2.547** | 5 | 268 (0,50%) · 98,9% |
| | drop_char | 337 (0,66%) · 88,1% | **113 (0,22%) · 96,2%** | 205 · 94,9% | 823 (1,61%) · 89,8% |
| | drop_syl | 352 (0,69%) · 88,1% | **137 (0,27%) · 95,2%** | 178 · 94,5% | 797 · 89,5% |
| | drop2_char | 484 (1,00%) · 83,7% | **192 (0,40%) · 93,3%** | 261 · 92,5% | 853 · 87,3% |

- Điều kiện chấp nhận (none không tệ hơn ∧ drop_char tốt hơn): **PASS** trên val+test; trên `all` FAIL đúng 1 cột
  `stt11/page_0108/c3` — QN có token dính `mìnhép` nên đáp án đường chéo sai 4 cặp, kênh ảnh đúng 3 cặp + đặt khe đúng chỗ thiếu hộp
  (7 ô đó đều REVIEW trong bộ giao nộp, `labels_final.csv` dòng tệp 9113–9119). λ ≥ 1 hại. Ảnh không thay được ngữ liệu (+ảnh không corpus ≈ +corpus).
- Thời gian: phát xạ 24 s / 53.658 hộp (MPS); DP +0,1 s / 2.547 cột.

### 2.3 Build thật (`--no-crops`, scratchpad; log/summary/sidecar chép về `KhoiB/v3/b2_full_build/`)

| build | trang | thời gian | cột đổi đường ghép | cặp đổi (mỗi bên) | ô cùng nom_idx đổi syl_idx | ô đổi tier_v3 | usable | QĐ-01 |
|---|---|---|---|---|---|---|---|---|
| `--limit 60` văn bản | 180 | 110,6 s | — | — | — | — | 27.195 | 813 locked / 0 pending |
| `--limit 60` +ảnh | 180 | 131,5 s (**+0,116 s/trang**) | 139/1.619 | 280–281 | 211 | 42 | 27.208 | 812 / **1 pending** |
| FULL văn bản | 448 | 276,4 s | — | — | — | — | 70.368 | 2.012 / 0 |
| FULL +ảnh | 448 | 319,7 s (**+43 s = +16%, 0,097 s/trang**) | **235/4.030** | **403** | 293 | **68** | 70.381 (+13) | 2.011 / **1 pending** |

Kiểm tái lập: FULL văn bản `--no-crops` so `dataset_out/labels.csv`: **0** cặp khác, 0 ô đổi tier (`b2_compare_builds.py`, md5 khác chỉ vì cột crop rỗng).
So `--limit 60` với `dataset_out` trên cùng 180 trang **không dùng được** (1.525 ô đổi tier chỉ vì ngữ liệu LOO nhỏ hơn) → mọi số trên là cùng-ngữ-liệu.

Mổ xẻ 403 cặp đổi (`KhoiB/v3/b2_changed_cells.json`): cặp bị phá theo tier_v3 REVIEW 362 / CHAR_B 20 / SYL 17 / CHAR_A 4; cặp tạo REVIEW 350 / SYL 36 /
CHAR_B 14 / CHAR_A 3; 216/235 cột là cột n_ocr ≠ n_qn; cặp từ điển (CHAR_A+CHAR_B) bị phá 24, tạo 17; p_visual_syl trung vị của cặp tạo **0,024**
(chỉ 5,2% ≥ 0,9) — ảnh chủ yếu *bác* cặp cũ chứ không *xác nhận* cặp mới. **19 ô usable đổi ÂM mà vẫn usable** (GOLD 12, SYLLABLE 7), trong đó
có ô đúng (連 trên→lên p 0,83; 於 nước→ở 0,83; 5 ô 不/石 vậy→làm 0,68–0,99) và ô sai rõ (嗔 đền→xin với p 0,0; 女 mà→nữa p 0,08) →
không thể áp tự động. Ma trận tier_v3: REVIEW→SYL 22, CHAR_B→CHAR_A 11, CHAR_A→CHAR_B 11, CHAR_B→REVIEW 9, SYL→REVIEW 5, …

**Ô QĐ-01 rơi pending** `stt11/page_0126/c6` nom 5 (`b2_full_build/qd01a_pending_vis.csv`): ảnh nói crop là 'là' p 0,94; đối chiếu OOF
(`p_visual_oof_v3.csv`) và **nhìn mắt** (`KhoiB/v3/k5_stt11_p0126_c6_crops_nom4-14.png`): cột 21 hộp / 22 âm, detector có hộp 'là' mà OCR sót,
OCR có 色 'đã' mà detector sót → hộp nom 5–12 lệch 1 so chữ OCR: crop nom 5 (khoá QĐ-01, "byte-identical") là 羅 **'là'**, nom 6 mới là 𠊚
người, … nom 12 'đã' là 𡗶 trời. 8 ô usable (GOLD 5, SYLLABLE 3) sai crop trong bộ giao nộp, gồm 1 ô QĐ-01. Kênh ảnh bắt đúng; DP văn bản
(luật hộp `legacy_locked_col`) không thể thấy.

### 2.4 Quyết định B-2

**Chưa bật `--visual-emission` mặc định cho lần thăng cấp kế tiếp.** Lý do (số): benchmark gác qua, nhưng trên hộp thật đổi 235 cột/403 cặp
(8× ước lượng ≤≈50 của DANH_MUC), 19 ô usable đổi âm chưa ai nhìn, 1 ô QĐ-01 trôi (vi phạm ràng buộc "khoá QĐ-01" nếu áp), lợi ích usable
chỉ +13. Điều kiện bật: §6.2. Giữ nguyên: cờ tắt, config, sidecar; mã ở nhánh sản xuất không đổi hành vi khi tắt (0/4.029 cột lệch).

---

## 3. B-3 · Cổng thị giác REVIEW → SYL (chỉ đo — `pipeline/tools/visual_syl_gate.py`)

Chạy trên `dataset_out/labels_final.csv` + `p_visual_oof_v3.csv` (join 83.239/83.239, `syllable` lệch 0, `labels_md5` khớp; chặn schema cumcount cũ).

| bước | số |
|---|---|
| tier_v3 REVIEW | 12.897 |
| − đã dùng được (QĐ-01 khoá → GOLD) | 36 |
| − âm không hợp lệ (`is_plausible_qn_syllable`) | 188 |
| − âm ∉ 804 lớp | 2.199 |
| = đủ điều kiện | **10.662** |
| argmax == âm ghép | 5.808 |
| **qua cổng p ≥ 0,9** | **1.025** (stt2 357 · stt4 403 · stt11 265; rule no_context 1.021, low_posterior 4) |
| qua cổng p ≥ 0,8 | 1.827 (640 · 700 · 487) |
| theo `tier_goc` | toàn rỗng (ô REVIEW v3 không có tier gốc) |

- Đối chiếu `summary.json.b3_gate_review` 1.030 / 1.846: đếm trên MỌI 12.897 ô REVIEW; trong 36+188 ô bị loại có 5 (≥0,9) / 19 (≥0,8) qua cổng
  → 1.030−5 = **1.025**, 1.846−19 = **1.827** — **khớp** (`E1_E2_OOF.md` §đối chiếu).
- DANH_MUC ước 894–961 → thực 1.025 (+1,46% usable nếu Khối C cho phép).
- **FAR ước trên CHAR_A** (mẫu số 46.717 ô có âm kề khác âm mình ∈ lớp): (a) p_kề ≥ 0,9 ∧ p_mình < 0,5 = **678 = 1,45% Wilson [1,35%; 1,56%]**;
  (b) argmax == kề ∧ p_kề ≥ 0,9 = 678 (trùng (a) vì p ≥ 0,9 ⇒ argmax); ở 0,8: 1.248 = 2,67% [2,53; 2,82]. Độ nhạy cổng trên CHAR_A 11.871/47.033 = 25,2%.
  FAR này là **cận trên**: một phần 678 ô là hộp lệch thật (§4) chứ không phải mô hình oan.
- Âm qua cổng nhiều nhất: một 50, ra 32, vua 29, hai 26, thầy 25, là 22, sau 20 (`visual_syl_gate_report.json`).
- 20 crop ngẫu nhiên (seed 2026) qua cổng ≥0,9 + ngữ cảnh ±2 ô → `KhoiB/v3/b3_sample.html` (`b3_sample_html.py`); 4 ô đầu nhìn nhanh
  (𥙩 lấy, 鬼 quỷ, 丕 vậy đúng; 𠰺 dạy crop cụt đáy) — chỉ để nhìn, không phải chấm.
- **KHÔNG áp vào tier.** `visual_syl_candidates.csv` (1.827 dòng, `gate_09/gate_08`, `rule_de_xuat = visual_syl_gate`) là đầu vào cho tầng Khối C.

---

## 4. Phát hiện ngoài kế hoạch: chuỗi "thanh ghi trượt" trong bộ giao nộp

`KhoiB/v3/k5_register_shift.py`: ô i "trượt" nếu argmax(crop i) == âm ô `syl_idx±1` cùng cột ∧ p ≥ ngưỡng ∧ âm kề ≠ âm mình; chuỗi = ≥3 ô liền cùng chiều.
Mô hình nhầm ngẫu nhiên gần như không tạo chuỗi: ô "trượt" đơn lẻ 2.197/83.239 (2,64%); đối chứng hoán vị argmax **trong từng cột** (giữ tỉ lệ/cột, 3 lần, seed 0) cho **3 / 2 / 1** chuỗi ≥3 so với **114** thật (`KhoiB/v3/k5_register_shift_null.py`).

| ngưỡng | chuỗi | cột | ô | usable | QĐ-01 | box_source ô | n_det ≠ n_ocr (cột) |
|---|---|---|---|---|---|---|---|
| p ≥ 0,8 | 114 (dài 3: 71, 4: 26, 5: 15, 6: 2) | 110 | **404** | 371 (GOLD 267, SYL 104) | **8** | detector 251, legacy_locked_col 123, midpoint 22, qd01_locked 8 | 47/110 |
| p ≥ 0,5 | 444 | 405 | 1.755 | 1.598 | 35 | detector 1.012, legacy 643, midpoint 65, qd01 35 | 153/405 |

Ví dụ đã nhìn mắt: `stt11/page_0126/c6` (§2.3). Không đặc thù cột khoá (36/110 cột legacy so 40,7% cột toàn bộ) — là lệch hộp↔chữ OCR khi
detector và OCR sót *khác* glyph (n_det = n_ocr nhưng khác tập). Đây là lớp lỗi crop mà DP văn bản không thấy; **đề nghị thêm tầng Khối C**
"ô trong chuỗi trượt" (371 ô usable, p≥0,8) và nếu ≥50% xác nhận thì hạ REVIEW theo chuỗi (việc người + 1 rule mới, chưa làm).

---

## 5. B-4 · Hàng đợi lớp (H2-lite) và B-5 · Khoá ô

### 5.1 B-4 (`KhoiB/h2/xay_hang_doi_lop.py` → `lop_review.html` 5,7 MB / 2.040 thumb, `lop_decisions_template.csv`, `lop_queue_info.csv`, `lop_queue_summary.json`)

Lớp = (sách, âm, mã đa số, mã thiểu số) trên GOLD v3; 3 nguồn gộp 309 → loại dị thể đã ký 23, E3-HAI 37 (giữ nguyên), corpus_reading 3, QĐ-01 2,
không còn GOLD 3, TN không nhìn giống 119 (ngoài hàng đợi, 495 ô) → **hàng đợi 122 lớp**:

| nhóm | lớp | ô GOLD thiểu số | ô đa số |
|---|---|---|---|
| 1 · E3 MỘT hình (ưu tiên 1) | 44 | 966 | 4.694 |
| 2 · E3 mờ | 10 | 174 | 414 |
| 3 · TN nhìn giống / Cap_sai | 68 | 291 | 2.161 |
| tổng | **122** | **1.431** | 7.269 |

DANH_MUC ước ≈60 quyết định (51 nhầm/1.159 ô + 9 mờ/202) → thực 122 lớp/1.431 ô vì gộp thêm nguồn TN v3 + Cap_sai. Máy chỉ *sắp* (k-means HOG),
không quyết; HOG bacc v3 là ý kiến thứ hai. `pipeline/tools/apply_lop_decisions.py` đọc CSV người ký → `decisions.yaml` có `xuat_xu`; chạy thử
giả định "ký toàn bộ theo E3" (`apply_lop_report_gia_dinh_e3.json`): 112 mục lop_nham, 1.257 ô GOLD bị ghi đè, 174 ô mờ → REVIEW — **chưa áp**, chờ ký.

### 5.2 B-5 (`KhoiB/v3/b5/summary_b5.json`; mã `build_dataset --lock-scope {col,cell,cell_fallback}` `:659-723`, mặc định `col` không đổi, selftest 250/250)

| cấu hình (FULL 448 trang, có crop) | detector trong cột từng khoá (usable) | AE-1 | IoU≥0,5 cùng cột | F1 (dòng/nhóm) | bbox/md5 QĐ-01 = cũ | ô usable đổi bbox |
|---|---|---|---|---|---|---|
| `col` (v3 hiện hành) | 58,5% | 0 | 0 | 40/20 | 2.012/2.012 | — |
| `cell` | **96,8%** | 0 | **29** | **98/49** | 2.012/2.012 | 6.411 |
| `cell_fallback` (111 cột lùi legacy) | 94,3% | 0 | 0 | 88/44 | 2.012/2.012 | 5.316 |

Hộp 3 nhánh trên 2.012 ô khoá: bbox = cũ byte 77,3%, IoU ≥ 0,5 94,2%, lệch 5,8% (117 ô, 48 lệch đúng 1 ô). **Khuyến nghị: KHÔNG đổi mặc định** —
`cell` thuần tạo 29 cặp IoU≥0,5 cùng cột (dây chuyền lệch 1 ô); `cell_fallback` sạch cặp cùng cột nhưng F1 +48 dòng và 5.316 ô usable đổi bbox
→ chỉ đổi sau Khối C có tầng `box_source=detector` trong cột từng khoá. Phát hiện §4 (8 ô QĐ-01 trong chuỗi trượt) cho thấy "crop QĐ-01 giữ byte"
không đồng nghĩa "crop QĐ-01 đúng glyph" — B-5 không giải quyết được lớp lỗi này, cần ảnh (E2b) + người.

---

## 6. Việc người phải làm tiếp và điều kiện dùng B-2/B-3

### 6.1 Việc người

| # | Việc | Đầu vào | Đầu ra | Công |
|---|---|---|---|---|
| N1 | Ký 122 lớp (ưu tiên nhóm 1: 44 lớp/966 ô) | `KhoiB/h2/lop_review.html` | `lop_decisions_template.csv` điền `quyet/nguoi_ky/ngay/xuat_xu` → `apply_lop_decisions.py` → `decisions.yaml` | 1–2 buổi |
| N2 | Khối C mẻ ~600 ô mù (C-2) **+ tầng 100–150 ô qua cổng B-3** (`visual_syl_candidates.csv`, gate_09) **+ tầng ≈100 ô chuỗi trượt** (`k5_register_shift_thr0.8_len3.csv`, ưu tiên chuỗi ≥4 và ô QĐ-01) | sha256 `labels_final.csv` | Wilson theo tầng; B-3: 0/150 lỗi | 3–4 buổi |
| N3 | QĐ-01a: xem 1 ô `stt11/page_0126/c6` nom 5 (+7 ô kề) — quyết crop 𠊚 nằm ở hộp nào | `k5_stt11_p0126_c6_crops_nom4-14.png` | `config/qd01a_decisions.csv` | 10 phút |
| N4 | Xem 19 ô usable đổi âm của B-2 (`b2_changed_cells.json` → `resyl_usable_both_sample`) để định tính "ảnh đúng bao nhiêu" | | ghi vào báo cáo Khối C | 20 phút |
| N5 | Quyết `--lock-scope cell_fallback` hay giữ `col` sau khi có tầng detector-trong-cột-khoá của Khối C | `b5/summary_b5.json` | | — |

### 6.2 Điều kiện để đưa B-2 / B-3 vào bộ giao nộp

| kênh | điều kiện đo được | trạng thái |
|---|---|---|
| B-3 `visual_syl_gate` (REVIEW→SYL, chỉ nhãn âm) | tầng 100–150 ô qua cổng trong mẻ Khối C: **0/150 lỗi** (cận trên Wilson 2,0%); FAR CHAR_A ≤ 2% (hiện 1,45%); rule ghi `xuat_xu = visual_syl_gate@labels_md5`; mã chữ tự động = 0 | chờ N2 |
| B-2 `--visual-emission` | (i) 19 ô usable đổi âm + ≈50 cặp đổi ngoài REVIEW được người xem, ảnh đúng ≥ 90%; (ii) sửa khoá QĐ-01 để ô trôi → QĐ-01a (không pending trong build); (iii) 4 cột sidecar vào `labels_trace.csv` (đã có khi bật); (iv) chạy lại benchmark gác sau mỗi lần đổi mô hình; (v) usable không giảm, `check_consistency` 3/3 | chờ N2/N3 |
| Cả hai | mô hình OOF phải là bản `labels_md5` == bộ đang build (`VisualEmitter` kiểm; đổi bộ → huấn luyện lại 5 fold trên Kaggle ≈23 phút) | có |

---

## 7. Kiểm bất biến

- `md5 dataset_out/labels_final.csv` = `1a2d8e0ca0a1199f11ff514135051c15` trước và sau; `re-dataset/labels.csv` = `cd530ec985288a0bf25edacb600ea3ca` trước và sau.
- `git status` không có tệp nào trong `dataset_out/`, `re-dataset/`, `prepared/`, `docs/EVIDENCE_INDEX.md` đổi (mọi build vào scratchpad `--no-crops`).
- Tệp mới/đổi của K5: `KhoiB/v3/{bench_b2.py (+drop2_char, +giây), bench_b2_md.py, BENCH_B2.md, bench_b2_results*.json, bench_b2_*.log, b2_compare_builds.py (+--restrict-to-vis-pages), b2_build_compare*.json, b2_changed_cells.py/.json, b2_full_build/, e1_e2_oof.py, E1_E2_OOF.md, e1_e2_oof.json, k5_register_shift.py, k5_register_shift_null.py, k5_register_shift_thr*.{json,csv}, k5_stt11_p0126_c6_crops_nom4-14.png, b3_sample_html.py, b3_sample.html, visual_syl_candidates.csv, visual_syl_gate_report.json}`, `docs/BAO_CAO_KHOI_B_2026-09-16.md`. Không commit.
