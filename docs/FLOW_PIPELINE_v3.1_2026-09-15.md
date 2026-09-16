# FLOW PIPELINE END-TO-END THEO KẾ HOẠCH v3.1 — SAU KHI RÀ TỪNG BƯỚC

**2026-09-15, sửa 18:40 sau phản biện** · Rà `docs/KE_HOACH_NANG_CAP_GAN_NHAN_FINAL.md` bản **3.1-Calibrated** (viết lại 15/09 14:08 theo
`DANH_GIA_KE_HOACH_NANG_CAP_2026-09-15.md`). Cách rà: 9 lượt độc lập (A0–A6, bảng nghiệm thu, và một lượt lập
bản đồ mã đường chạy hiện hành tới từng dòng), mỗi lượt mở mã + chạy đo trên `cols.pkl`/`labels_final.csv`
Phản biện: 65 phát hiện chặn/lớn qua 130 phiếu bác bỏ thử → **0 bị bác** (5 chặn, 13 lớn giữ mức); bản flow này
sau đó được 2 lượt phê bình độc lập (lăng kính sự thật mã/số; lăng kính logic theo dõi một ô QĐ-01 qua 20 nút) — 4
mâu thuẫn nội tại + 15 lỗi nhỏ đã sửa, ghi ở §7. Phần 1 là kết quả rà; phần 2–4 là **flow hoàn chỉnh** từ PDF đến bằng chứng,
từng nút ghi `[GIỮ]` (mã hiện có, không đổi) · `[SỬA]` · `[MỚI]` · `[BỎ]`, kèm `file:line` và phép kiểm.

---

## 1. RÀ v3.1 — CÁI ĐÚNG, CÁI CÒN HỔNG

**Đúng (tái lập):** hằng số CALIB (0/2,5/6,7/5,1/8,6 — khe giả 346→54, lệch chéo 1,63→0,82%); T=1,0; đệ quy LOO
chi phí 2,0 (khe đúng 64→85–86%, ghép sai 3,14→1,3–1,4%); `tier_v3_exact` **đúng logic** với
`thuc_nghiem.py:541-551` (0 cặp khác tier, 70.108 ô); lệnh xuất `qd01_cells` ra đúng 2.014 ô (100% 𠊚, 100% GOLD);
danh sách "không làm" (COM, Pitch-DP, resolve_overlap, đổi tên tier) đúng; `evidence()` đã được gọi sẵn ở
`run_pipeline.sh:675`.

**Hổng — phải sửa trước khi thực thi** (xếp theo mức nguy hiểm):

| # | Chỗ | Vấn đề | Số đo | Sửa (đã đưa vào flow §3) |
|---|---|---|---|---|
| 1 | A0-2 / A5-1 | Khoá QĐ-01 theo `(bbox, image_md5)` **tự vỡ**: A3 đổi bbox 336–675 ô QĐ-01; md5 còn đổi khi hàng xóm đổi hộp (carve theo prev/next); chỉ số `_NNN` trong tên ảnh là **enumerate theo TRANG** (`build_dataset.py:431`) nên đổi ở ≈656/2.014 ô QĐ-01 (≈27,8k/82,6k ô toàn bộ) chỉ vì cột trước thêm/bớt cặp | khoá `(book,page,column,'người',thứ tự)` cũng không đủ: 337 cột có ≥2 ô QĐ-01 | khoá bền = `(book,page,column,nom_idx)` — có sẵn trong ops (`anchor_align.py:157`) nhưng `_pair_new` vứt (`align_production.py:458-461`); giữ 2.007–2.012/2.014 qua CALIB+neo; 2–7 ô còn lại → **QĐ-01a** (người xem) |
| 2 | A4 ↔ A5 ↔ A6 | Khoá QĐ-01 **không nói nằm ở đâu**. Dưới tier_v3, 2.014 ô rơi CHAR_A 1.178 (nhãn **㝵**) / CHAR_B 281 / SYL 512 / REVIEW 41; `glyph_fix.CO_THE_GAN` (`:49`) không gồm GOLD/SYLLABLE → không gán; 41 ô REVIEW không có crop nếu khoá sau PASS 2 → 1.973 ≠ 2.014 | | khoá **trong build, PASS 1c, sau tier_v3, trước PASS 2**; `glyph_fix` chuyển sang chế độ *kiểm* (assert), không gán mới |
| 3 | A6-2 | "`--crop-review` nếu tắt S3" là **bẫy**: `glyph_fix` gán theo ÂM + có ảnh → 218 ô REVIEW 'người' chưa ai nhìn được gán 𠊚 → **2.232 ≠ 2.014**, vi phạm `NGUOI_CHAM_QUYET_DINH.md:20` | | không cần `--crop-review` khi khoá nằm trong build (ô khoá → GOLD → được cắt) |
| 4 | dòng 4 ↔ A3 | Nguyên tắc "**không thay đổi md5 crop**" mâu thuẫn A3: A3 đổi bbox **11.537/64.525 ô usable (17,9%)**, cận trên md5 đổi 14.117 (21,9%); QĐ-01 336 ô | tái lập bbox 82.780/82.780 | bỏ nguyên tắc ấy; thay bằng "md5 đổi được **liệt kê**; **crop QĐ-01 giữ nguyên byte** (cột có ô khoá chạy luật hộp cũ + đóng băng `bbox/prev/next_cu`)"; cờ `--box-rule legacy` = **trọn gói** (thr 0,3 + ±0,5w + đường cũ) để lùi |
| 5 | A3 | Chỉ đặc tả \|G\|=\|Q\| (74% cột); 26% cột / 15.879 ô bỏ ngỏ | nhánh \|G\|=\|OCR\|≠\|Q\| (673 cột, 10.054 ô): `hộp[nom_idx]` đúng **98%**, ép `hộp[j]` chỉ 45% | 3 nhánh (§3 N9) |
| 6 | A3 | Lọc x "\|x − x_col\| > 0,10·w" mơ hồ (hiểu theo tâm cột thì vứt 27% hộp thật); cửa sổ hiện ±0,5w (`infer_centernet.py:247-249`) | sweep: ±0,10w **không** tăng M==N (91,8 vs 91,9%); **±0,25w tốt nhất** (93,2%) | `DETECTOR_XMARGIN = 0.25`, lọc trước `enforce_count` |
| 7 | A1 | `posterior_matches` **chưa có trong engine** (chỉ ở lab); v3.1 bỏ `cost_fn` → posterior không cùng chi phí neo với DP | không cùng chi phí: 70.108 → 69.717 (−391, vượt ±300) | port với `cost_fn`; posterior và Viterbi dùng **cùng** closure |
| 8 | A2 | Không nói `pair_pages` lấy match nào | chỉ lấy match *confirmed* → đệ quy **vô tác dụng** (3,13%/64%); lợi ích đến từ 62–75% neo là chữ ∉ R(âm) (而/làm 287 trang, 妃/bà 102) | lấy **mọi** match, khoá `(ocr_char, âm.lower())`, trang = `(book,page)`; neo **không** đổi cờ `confirmed` |
| 9 | A4 | Không nói bỏ/giữ L2 (848 GOLD cầu ngược), L3 (112), L5 (10.369); thiếu chốt `is_plausible_qn_syllable` (19 ô SYL âm 'r1'/'0') và `AM_DA_QUYET`; không định nghĩa LOO cho `bigram/corpus` | bỏ LOO → bigram tự khẳng định 35.808 ô → usable **81.755** (lỗi 82k của bản 3.0 quay lại) | tier_v3 **thay** decide_label + L2 + L3 + L5; L1 giữ có cổng; feats LOO đúng `Corpus.feats` |
| 10 | A5 | Dị thể gộp 4 cặp đều **về mã hiếm** (徳313→德163, 别147→別84, 爲364→為191, 廪84→廩46), không theo quy tắc; E3 nói (stt2, vì) 為/爲 là **hai hình** → gộp toàn cục sai đơn vị; `(cùng,其)→共` ghi đè 158 ô GOLD là phán quyết **người** nhưng v3.1 bỏ `xuat_xu` | | hai cột `label` (quan sát) + `label_canonical` (theo yaml có `xuat_xu`, `book`); không đổi `label` |
| 11 | A5 | L1 "2-gram ngoài ô" chưa định nghĩa — 3 cách định nghĩa cho 264/198/293 · 376/245/134 · 498/161/70 | | định nghĩa cố định (§3 N13); đo lại trên bản build |
| 12 | A6 | `check_consistency` **4/4 không thể đạt** ở `dataset_out_v3`: `update_bang_so_lieu.py:29`, `rebuild_proto_index.py:35`, `evidence()` (`run_pipeline.sh:55-57`) ghi cứng `dataset_out`; gọi `evidence()` trong lần chạy v3 sẽ **ghi đè khối HIEN_HANH** và xoá dấu vết bộ 64.525 | | thêm bước **thăng cấp** (§3 N26) |
| 13 | A0 | Không có bước **tái lập HEAD trước khi đổi hằng số**; A6-1 đối soát sau khi đã đổi A1–A5 nên không tách được lỗi tái lập khỏi thay đổi chủ ý | bộ 64.525 chưa qua `run_pipeline.sh` (CHECKSUMS dừng 25/08) | N0 |
| 14 | ACC | "70.108 ± 300" lệch tâm và quá chặt; `thuc_nghiem.py rebuild calib` **vô cảm với engine** (`cmd_rebuild` ép `set_matrix(PROD)` `:76`) | bản build thật: +38 (usable hiện hành trong 7 cột không join) +~145 (BUG-1) +41 (QĐ-01 REVIEW→khoá) −19 (SYL không plausible) −10 ('người' ngoài khoá nhãn 㝵) ±L1 → **70.100–70.450**, cận trên ~70.500 | bảng §4 |

---

## 2. SƠ ĐỒ TỔNG THỂ

```
Data/STT*.pdf
  │ N1 extract [GIỮ]                pages/ · pages_denoised/ · detected/*_ocr_cache.json (kinhhannom, 9 cột) · transcriptions/*_qn_ocr_cache.json (VietOCR)
  ▼
build_dataset --out $DS_OUT --reseg detector --box-rule syl_index      (S3 TẮT)
  │ N3  PASS 1  [SỬA]  mỗi trang: cột Nôm → dòng QN (parse_v5, BUG-1 vá) → normalize_column → DP CALIB → hộp thô detector (thr 0,2, ±0,25w)
  │                    → col_states (chars, syllables, G, n_ocr/n_qn/n_det) — KHÔNG decide_label, KHÔNG S3
  │ N4  PASS 1b [MỚI]  pair_pages LOO từ mọi match lượt 1 → DP lại từng cột: cost_fn = min(base, 2,0) nếu cặp ≥2 trang khác
  │                    (KHÔNG ép cặp QĐ-01 — ô trôi → QĐ-01a) → posterior(cùng cost_fn, T=1,0) → gán hộp 3 nhánh (cột có ô khoá: luật cũ) → records
  │ N5  PASS 1c [MỚI]  bigram_pages/pair_pages LOO từ 1b → feats → tier_v3 → L1 có cổng 2-gram (syllable_raw) → corpus_readings →
  │                    decisions.yaml (label_canonical) → chốt is_plausible → KHOÁ QĐ-01 (qd01_cells + qd01a_decisions; ghi đè; bbox/prev/next_cũ)
  │                    → ô 'người' ngoài khoá: chỉ hạ REVIEW khi nhãn v3 == 㝵 (10 ô); 𠊚 giữ GOLD
  │ N6  SPLIT   [BỎ]   không chia train/val/test (ràng buộc 16/09) — README ghi công thức hash cũ
  │ N7  PASS 2  [SỬA]  cắt crop GOLD+SYLLABLE+pending (ô khoá: save_crop(bbox_cũ, prev_cũ, next_cũ) → md5 tất định) → labels.csv (+17 cột mới, cờ DÀY 0/1)
  │ N8  enrich_crop_quality [GIỮ]
  ▼
remediate [GIỮ] → confusion_fix [GIỮ, kỳ vọng 0] → s3_unwind [GIỮ, no-op tất định] → glyph_fix --mode kiem [SỬA] → assert QĐ-01 (locked+pending == 2.014; trước export pending == 0) → export [GIỮ]
  ▼
docs + xlsx [GIỮ] → doi_soat_the_he.py [MỚI] → QĐ-01a (người → config/qd01a_decisions.csv) → THĂNG CẤP: chạy lại DS_OUT=dataset_out → rm s3_proto_cache + update_bang_so_lieu [MỚI nối] → evidence() [GIỮ] → 1 commit → check_consistency 4/4
```

---

## 3. BẢNG NÚT CHI TIẾT

Ký hiệu: `$DS_OUT` = `dataset_out_v3` khi thử, `dataset_out` khi thăng cấp. Mọi nút có phép kiểm ở cột cuối.

### 3.0 Trước khi đổi bất kỳ hằng số nào

| # | Nút | Loại | Mã | Vào → ra | Kiểm |
|---|---|---|---|---|---|
| N0a | Biến `DS_OUT` xuyên suốt runner | [SỬA] | `run_pipeline.sh:52` thêm `DS_OUT="${DS_OUT:-dataset_out}"` (KHÔNG dùng tên `OUT_DIR` — va với `local OUT_DIR` `:489`); thay 13 chỗ ghi cứng `:55-57,:67,:69,:373,:397-398,:432-433,:451,:507,:642`; `:365` truyền `--out "$DS_OUT"` (build đã có `--out` `:316`); `REDATASET_DIR`/`FINAL_DIR` đặt dưới `$DS_OUT` khi ≠ `dataset_out`; `evidence()` **chỉ chạy khi `DS_OUT == dataset_out`**; `.gitignore` + `/dataset_out_v3/` | — | `grep -nE '(^|[^-A-Za-z_$/])dataset_out' run_pipeline.sh \| grep -vE '^[0-9]+:\s*#\|log "'` chỉ còn dòng 52; thêm `NONINTERACTIVE=1` (bỏ 4 `read -r -p`, BOOKS = cả 3, FRESH_OCR=0) để N0c/N19 chạy được không tay |
| N0b | Phát `nom_idx`/`syl_idx` (không đổi hành vi) | [SỬA] | `align_production.py:458-461` `_pair_new` thêm `"nom_idx": i, "syl_idx": p["syl_idx"]`; `build_dataset.py:433-441` record + `:583-600` labels + `:606-610` fields | ops (`anchor_align.py:157` đã có) → 2 cột mới | tier/rule/label/bbox/md5 không đổi |
| N0c | **Tái lập HEAD** | [MỚI] | `DS_OUT=dataset_out_v3 ./run_pipeline.sh` tại HEAD + N0a/N0b; script `pipeline/tools/doi_soat_the_he.py --old dataset_out/labels_final.csv --new dataset_out_v3/labels_final.csv` khoá `(book,page,column,bbox)` → `(tier,rule,label,image_md5)` | 3 tệp labels | **0 lệch** (lệch chỉ được phép ở ô SILVER/S3 và phải liệt kê); QĐ-01 = 2.014 **bắt buộc**; `time` thật ghi lại (mốc hiện có: build→build ≥10,3 phút, S3 ≈5,5–9 phút, selftest 722 ≈20 s) |
| N0d | `config/qd01_cells.csv` **khoá bền** | [MỚI] | sinh từ **bản người đã ký** `dataset_out/labels_final.csv` (`rule.startswith('quyet_dinh_nguoi')`) JOIN bản tái lập N0c theo `(book,page,column,bbox,image_md5)` để lấy `nom_idx/syl_idx`; cột `book,page,column,nom_idx,syl_idx,syllable,ocr_char,label,unicode,bbox_cu,prev_bbox_cu,next_bbox_cu,image_md5_cu,image_cu,tier_build,rule_build` (`bbox/prev/next_cu` để **tái tạo crop đúng byte**, không làm khoá; `prev/next` lấy từ `by_col` cũ `build_dataset.py:568-576`) | 2.014 dòng | `assert len == 2014`, 0 trùng khoá, 100% 𠊚, join 2.014/2.014 |
| N0e | Selftest 4 hàm + 2 | [MỚI] | `pipeline/phase1_engine_selftest.py`: `realign_column` (khớp / rụng 1 / thừa 1 / khe hai đầu / chạm băng / neo không đổi `confirmed`), `posterior_matches` (argmax == Viterbi khi cùng `cost_fn`; tổng p theo hàng ≤ 1), `tier_v3` (mỗi nhánh + chốt plausible + AM_DA_QUYET), `apply_am_sua_dau` (có/không 2-gram, hoà), `_pair_new` (3 nhánh hộp), `enforce_count` (M>N, M<N, rỗng); `scripts/run_all_selftests.sh:80 BASELINE_PASS=722+N` (cập nhật một lần ở ngày 6). **Ngày 1 chỉ viết test cho hàm đã có** (`realign_column`, `enforce_count`); test của `posterior_matches`/`tier_v3`/`_pair_new` 3 nhánh/L1 có cổng thêm đúng ngày làm nút đó | — | `python -m pipeline.phase1_engine_selftest` (lệnh `-m unittest <path>` chạy 0 test) |
| N0f | Vệ sinh | [MỚI] | `rm dict/dict` (BUG-5); `pipeline/tiers.py` gom `USABLE_TIERS` từ 5 nơi (`export_final_dataset.py:21`, `ground_truth/suspicion.py:48`, `publish/splits.py:26`, `remediation/census.py:24`, `s3_unwind.py:76` — nội dung khác, đặt `KEPT_TIERS` riêng) (BUG-6) | — | selftest xanh |

### 3.1 Trích xuất (không đổi)

| # | Nút | Loại | Mã | Vào → ra |
|---|---|---|---|---|
| N1 | setup + extract | [GIỮ] | `run_pipeline.sh:328-354` → `step0_setup`, `step1_extract.process_book:27-257` | PDF → `prepared/<sách>/pages/`, `pages_denoised/`, `detected/page_XXXX_ocr_cache.json` (kinhhannom, md5/pixel_hash, `coords_space=fullpage`), `transcriptions/page_XXXX_qn_ocr_cache.json` (VietOCR 2-pass) |
| N2 | Cache OCR | [GIỮ] | `core/ocr/ocr_api.ocr_page:645-715` | cache được đối chiếu `image_hash/pixel_hash` (StaleOCRCacheError); đổi hằng số hình học/detector **không** đụng cache |

### 3.2 build_dataset — PASS 1

| # | Nút | Loại | Mã | Vào → ra | Kiểm |
|---|---|---|---|---|---|
| N3a | Khởi tạo | [SỬA] | `build_dataset.main:313-402`: bỏ `--use-s3` khỏi `run_pipeline.sh:365` (S3 = 0 ô giao nộp; không nạp `VisualS3`, tiết kiệm 5,5–9 phút; index.csv ghi cứng `dataset_out/gold` không còn liên quan); thêm `--box-rule {legacy,syl_index}` (mặc định `syl_index`), `--cells config/qd01_cells.csv` | — | |
| N3b | Dòng QN (BUG-1) | [SỬA] | `pipeline/step2_align.py:80-82`: `pn = {k: _split_syllables(' '.join(v), qn_dict=qn_dict) for k, v in pn.items() if v}` (vá **một chỗ** trước cả hai `return :84/:88`; `_split_syllables` ở `parser_v5.py:34`) | `transcriptions/*_qn_ocr_cache.json` → `{line: [âm]}` | `stt4/page_0146`: 185 âm / 186 chữ, 8/9 cột khớp (hôm nay: 18 "âm" = dòng) |
| N3c | Cột Nôm + chuẩn hoá | [GIỮ] | `align_production._detect:39-100` (`nom_cols_hybrid`, `frame_offset`), `syllable_normalize.normalize_column` (`syllable_normalize.py:65`, gọi tại `align_production.py:509`); ghi **hai cột**: `syllable_ocr` = VietOCR nguyên văn (lower) và `syllable_raw` = sau normalize_column (~115 ô đã vá dấu trước DP, cờ `norm_fixed`), trước mọi L1 | | "ghi đè bản in = 0" đo trên `syllable_ocr` |
| N3d | Ma trận CALIB | [SỬA] | `anchor_align.py:33-39` → `0.0 / 2.5 / 6.7 / 5.1 / 8.6 / 8.6 / BAND_SLACK 2`; `realign_column(:81)` thêm `cost_fn=None` dùng ở `:127`, trả thêm `band_touched` (CALIB chạm biên 3/4.029 cột) | | `thuc_nghiem_ke_hoach.py matrix`: 54 / 0,82% |
| N3e | `posterior_matches` vào engine | [MỚI] | `anchor_align.py` sau `:172`: port `thuc_nghiem.py:118-173` với chữ ký `(nom_chars, syllables, qn_to_nom, similar_dict, T=1.0, cost_fn=None, band_slack)`; **cùng `cost_fn`** với Viterbi | | selftest argmax == Viterbi |
| N3f | Hộp thô detector | [SỬA] | `align_production.py:184` (ngay dưới `BOX_OVERLAP_FRAC`, ghi đè từ config như `build_dataset.py:349-350`) hằng `DETECTOR_THR = 0.2`, `DETECTOR_XMARGIN = 0.25`; `:252 DetectorInfer(thr=DETECTOR_THR)`; `char_detector/detector_infer.py:67` thêm `raw_column_boxes(page_boxes, x_range, x_margin)` = lọc tâm-x ∈ [x1−m·w, x2+m·w] (`infer_centernet.py:247-249`) + `_nms_vertical(0.45)` + sort y — **trước** `enforce_count` | ảnh trang → `G` mỗi cột, `n_det = len(G)` | cột OCR=QN, thr 0,2 ±0,25w: M==N 69,9→**93,2%**, M<N 27,1→1,9%, M>N 3,0→4,8% |
| N3g | PASS 1 vòng trang | [SỬA] | `build_dataset.py:404-447`: `align_page` (`align_production.py:469`) trả **`col_states`** = (line_id, cluster, syllables, syllable_raw, G, n_ocr, n_qn, n_det, matched) + ops lượt 1; **bỏ** `maybe_s3 :432` và `decide_label :433` khỏi vòng | trang → col_states | detector chạy **1 lần/trang** (0,4 s/trang ≈ 3 phút) |

### 3.3 PASS 1b — đệ quy + posterior + hộp

| # | Nút | Loại | Mã | Vào → ra | Kiểm |
|---|---|---|---|---|---|
| N4a | `pair_pages` LOO | [MỚI] | sau `:447`, trước `:449`: `pair_pages[(ocr_char, syllable.lower())] = {(book,page)}` từ **mọi** match lượt 1 (không chỉ confirmed — chỉ confirmed thì đệ quy vô tác dụng); ghi `$DS_OUT/pair_pages.json` | records lượt 1 → dict | ≥2 trang khác: ~23.400 ô được neo |
| N4b | DP lại từng cột | [MỚI] | closure `cost_fn(c, s)`: `min(base, ANCHOR_CAP=2.0)` nếu `len(pair_pages[(c, s.lower())] − {(book,page)}) ≥ 2`; **không ép** cặp QĐ-01 (ép làm đổi ghép ô lân cận mà không cờ; ô QĐ-01 trôi → QĐ-01a ở N5g); `is_confirmed` (`:156`) **không** bị neo đổi; chạy `realign_column` trên `col_states` (0,5 ms/cột ≈ 2 s) | col_states → ops cuối | `thuc_nghiem.py recursive` (sửa `:525` chi phí neo 2,0): khe đúng 85–86%, ghép sai 1,3–1,4%. Lưu ý: mọi crosstab trích dẫn (70.108; QĐ-01 1.178/281/512/41) đo ở cap **0** (`thuc_nghiem.py:575/:582`); cap 2,0 chênh ≤ +3 usable — lab đọc `ANCHOR_CAP` từ engine rồi tái sinh một lần |
| N4c | Posterior | [MỚI] | `posterior_matches(..., T=1.0, cost_fn=cost_fn)` → `p_register` từng cặp | | phân bố: p≥0,8 ở ~98,9% — **p chỉ gác cặp hoà**, ngữ cảnh mới là cổng (119/83.091 ô do p quyết) |
| N4d | **Gán hộp 3 nhánh** | [SỬA] | `_pick_reseg:387-421` + `_pair_new:452-466`: (1) `\|G\| == n_qn` → `bbox = G[syl_idx]`, `count_source=equal_qn`, `box_source=detector` (74% cột, 48.646 ô); (2) `\|G\| == n_ocr ≠ n_qn` → `bbox = G[nom_idx]`, `equal_ocr` (673 cột, 10.054 ô; benchmark DEL 98%); (3) còn lại → `enforce_count(G→n_qn)` + `_monotone_assign` như cũ (`:401-419`), `count_source=conflict`, `box_source` từng hộp ∈ {detector, split, midpoint} (≈326 cột, 8%). Phân bố ±0,25w ≈ equal_qn 3.024 cột (75%) · equal_ocr 680 (17%) · conflict 326 (8%). **Cột có ô khoá QĐ-01 → chạy trọn gói luật cũ cho cả cột** (không trộn hộp cũ của ô khoá với hộp mới của hàng xóm: 23 cặp sẽ trùng bbox → census AE-1 → `remediate.py:106-108` cách ly **cả hai**), `box_source=legacy_locked_col` — **cột có QĐ-01 = 1.640/4.030 cột, chứa 42,4% ô usable → hộp detector hiệu dụng 87,7%** (khoá ô thay trọn cột ≈96%, xem DANH_MUC B-5). `--box-rule legacy` = **trọn gói** (DETECTOR_THR 0,3 + ±0,5w + `:400-419`) vì `_DETECTOR` là singleton — selftest: legacy tái lập bbox `labels_final` 100% trên 5 trang | ops + G → bbox, box_source | `thuc_nghiem.py geo` tham số hoá (thr từ engine, `raw_column_boxes`, điều kiện \|G\|=\|Q\|, kịch bản INS+DEL): hộp[j] **96,6% CALIB / 98,4% CALIB+neo** trên 237/272 cột |
| N4e | Records | [SỬA] | `build_dataset.py:433-447`: record = {…, `nom_idx, syl_idx, syllable_ocr, syllable_raw, p_register, band_touched, n_ocr, n_qn, n_det, count_source, box_source, box_x_outlier`} (ngày 3 các cột hộp = '' cho tới N3f/N4d ngày 5); **mọi cột cờ ghi DÀY 0/1** (cột thưa qua pandas của remediate/confusion/glyph thành float '1.0' → join trượt; hoặc thêm vào DTYPE str ở 3 mô-đun); `idx` tên ảnh vẫn theo trang (giữ để không gãy `confusion_fix.py:105` join theo tên **trong cùng thế hệ**) | | |

### 3.4 PASS 1c — phân tầng, L1, quy ước, khoá QĐ-01

| # | Nút | Loại | Mã | Vào → ra | Kiểm |
|---|---|---|---|---|---|
| N5a | Thống kê LOO | [MỚI] | module mới `pipeline/align_engine/tier_v3.py`: chép `Corpus` (`thuc_nghiem.py:307-362`) — `pair_pages` (từ ops **cuối**), `bigram_pages[(c1,c2,s1,s2)]` (cặp match kề nhau `nom_idx+1 ∧ syl_idx+1`), `by_tone/by_all`; **mọi cờ trừ `{(book,page)}` hiện tại** | records → feats | không LOO → 81.755 ô (sai) |
| N5b | `feats` | [MỚI] | đúng `Corpus.feats:334-357`: `direct = c∈R(s)`; `sim_unique = \|sim(c)∩R\|==1`; `tone = ∃v≠s, strip_tone(v)==strip_tone(s), c∈R(v)`; `corpus2/4 = \|pair_pages−{pg}\| ≥ 2/4`; `bigram = ∃ 2-gram kề (trái hoặc phải) có ở trang khác`; thêm `dict_support = \|R(s)\|`, `context_evidence` = chuỗi cờ | | |
| N5c | `tier_v3` | [MỚI] | chép nguyên văn `thuc_nghiem.py:541-551`; **thay** `decide_label` (`consensus.py`), L2 cầu ngược (846 GOLD → SYL 629/REVIEW 217, chủ ý), L3 (`apply_cot_lech_cau_xuoi`, 110 ô → CHAR_B 83/REVIEW 22/SYL 5), L5 (`syllable_gate`, 48 ô SYLLABLE hiện hành rớt: 33 + 15); **chốt trước tier**: `is_plausible_qn_syllable(s)` (19 ô 'r1'/'0'/'1' → REVIEW `not_plausible`). **Không** chốt `AM_DA_QUYET` bao trùm (làm mất 243 ô 'người' ngoài khoá và hạ 49 ô mốc đã ký gồm 37 ô 𠊚 `s1_inter_s2_similar` + 12 `khong_dung_cho`) — xử lý riêng ở N5h | feats + p → `tier_v3` | crosstab = 47.482 / 3.338 / 19.288 / 12.983 (± ô mới; L2 848 → SYL 629 / REVIEW 217; L3 112 → 83/22/5) |
| N5d | Ánh xạ tier/rule | [MỚI] | `CHAR_A → tier GOLD, rule s1_inter_s2_direct` (**giữ tên** — 7 chỗ đọc rule đúng-bằng: `build_dataset.py:94`, `remediate.py:126`, `apply_verdicts.py:212`, `suspicion.py:132`, `make_combined_batch.py:288`, `sem_score.py:182`, `run_pipeline.sh:474`); `CHAR_A: label = ocr_char`; `CHAR_B-cầu → GOLD, rule **`s1_inter_s2_similar`** (giữ tên — 4/7 chỗ đọc đúng-bằng chuỗi này), label = chữ cầu duy nhất, ngữ cảnh ghi ở `context_evidence``; `CHAR_B-direct-p-thấp → GOLD, rule s1_inter_s2_direct_lowp, label = ocr_char, **không** làm neo`; `SYL → SYLLABLE, label='', rule syl_ctx:<bigram\|corpus4\|tone>`; `REVIEW → rule no_context \| low_posterior \| not_plausible`; cột `tier_v3` ghi riêng | | `confusion_fix/glyph_fix/s3_unwind/census/publish` đọc `(syllable,label,tier)` — không gãy |
| N5e | L1 có cổng | [SỬA] | `apply_am_sua_dau:106-160`: thêm `bigram_syl_pages` = 2-gram **ÂM–ÂM** dựng từ **`col_states[syllables_raw]`** theo `(book,page)` (độc lập DP — dựng từ records thì 2-gram qua khe `ins` bị đứt), LOO trang, hai phía cộng; chỉ đổi khi `n(âm mới) > n(âm gốc)`; hoà → giữ gốc, cờ `l1_tie`; **luôn giữ `syllable_ocr/syllable_raw`**; ghi `l1_support`; `anchors` (`gold_direct_anchors:89-98`) lấy từ CHAR_A, **không** gồm `_lowp`/corpus_readings | 755 ô hôm nay theo đúng định nghĩa này → **≈376 đổi / ≈245 giữ gốc / ≈134 hoà** (bộ 498/161/70 của báo cáo 15/09 thuộc định nghĩa khác, không tái sinh được) — đo lại trên build ±20 | crosstab `syllable ≠ syllable_raw` × `l1_support` |
| N5f | `corpus_readings` | [MỚI] | `config/decisions.yaml` mục `corpus_readings` (僥→nhiều, 無→vồ, 生→chẳng, … ≈8–12 cặp) — mỗi mục `xuat_xu` + `book` tuỳ chọn + `unicode`; áp: `(ocr_char, syllable_raw)` khớp → `tier GOLD, rule corpus_reading:<id>, dict_support='corpus'`; **không** đưa vào `qn_to_nom` trước PASS 1/1b (`:353`) và **không** tính vào `gold_direct_anchors` (`:89-99`) — tránh vòng tự khẳng định (chú thích `build_dataset.py:73-78`) | ~245 ô (số 'giữ gốc' của N5e) | |
| N5g | **Khoá QĐ-01** | [MỚI] | đọc `qd01_cells.csv` **rồi** `config/qd01a_decisions.csv` (phán quyết người cho ô trôi: khoá `book,page,column,nom_idx`, `quyet ∈ {giu_2029A, bo, khac:<chữ>}`, `nguoi_ky, ngay, xuat_xu`); join `(book,page,column,nom_idx)`: (i) khớp và `syllables[syl_idx].lower() == 'người'` → ghi đè `label=𠊚, unicode=U+2029A, tier=GOLD, rule=quyet_dinh_nguoi:qd01_cell_lock`, giữ `tier_goc/rule_goc`, `qd01_locked=1`, **`bbox = bbox_cu`, `prev/next = prev/next_bbox_cu`, `box_source=qd01_locked`**; (ii) khớp nom_idx nhưng ghép sang âm khác (đo không ép: 2 ô 'mà'/'con') hoặc thành khe (1–4 ô) → nếu có trong `qd01a_decisions` thì áp, không thì **pending**: `rule=quyet_dinh_nguoi:pending`, `qd01_locked=0`, tier_goc giữ, **ép cắt crop** bằng cả bbox mới lẫn `bbox_cu` để người nhìn; (iii) không khớp → QĐ-01b (lỗi, phải 0) | 2.014 → locked + pending | `assert locked + pending == 2014`, `pending ≤ 10`; trước export `pending == 0` |
| N5h | 'người' ngoài khoá | [MỚI] | `decisions.yaml` mục `lop_nham`: ô **không** khoá, âm 'người', **nhãn v3 == 㝵** → REVIEW `lop_nham:nguoi_2029A_vs_346B` (≈10 ô); ô v3 → 𠊚 (26 CHAR_B + 37 hiện hành) **giữ GOLD**; 12 ô `khong_dung_cho` giữ theo tier_v3, cờ `qd01_excluded=1`; còn lại 281 ô 'người' ngoài khoá (201 SYL, 41 CHAR_B, 1 CHAR_A; 239 không ảnh) đi theo tier_v3 với cờ `am_da_quyet_ngoai_khoa=1` để mẻ chấm ưu tiên; giữ `confusion_fixes.yaml` làm chốt kép | | `confusion_fix` báo demote **0** (đã chặn trong build); usable −10, không −243 |
| N5i | Dị thể → `label_canonical` | [MỚI] | `decisions.yaml` mục `di_the`: mỗi mục `{quan_sat, chuan, book?, quy_tac: 'da_so_ngu_lieu' \| 'unicode', xuat_xu}`; ghi cột **`label_canonical`**, **không đổi `label`** (H3: hình quan sát / mã chuẩn); đơn vị `(sách, âm)` — ví dụ (stt2, vì) 為/爲 là hai hình → không gộp | | `label ≠ label_canonical` đếm theo sách |
| N5j | `(cùng, 其) → 共` | [HOÃN] | là phán quyết người (ghi đè 158 ô GOLD; 73 ô `direct` theo chính từ điển) → chỉ vào yaml khi có `xuat_xu` = tệp người ký (khối B2/C), giống `glyph_fix.kiem_xuat_xu:56-69` | | |
| N5k | `flank_gold` | [MỚI] | sau tier: số ô kề (`syl_idx ± 1` cùng cột) có `tier_v3 == CHAR_A` ∈ {0,1,2} | | |

### 3.5 SPLIT, PASS 2, manifest

| # | Nút | Loại | Mã | Vào → ra | Kiểm |
|---|---|---|---|---|---|
| N6 | SPLIT theo trang | **[BỎ]** (ràng buộc 16/09: bộ giao nộp không chia train/val/test) | xoá `build_dataset.py:518-550` + cột `split/split_group/label_in_train`; README ghi công thức hash cũ `int(md5(book\|page),16)%100` để ai cần tự chia theo trang; 17 vị trí mã sửa theo `DANH_MUC_SUA_DOI_CUOI_2026-09-16.md` §3 | | — |
| N7 | PASS 2 cắt crop | [SỬA] | `:552-604`: `crop_tiers = {GOLD, SYLLABLE}` + ô `pending` (không SILVER vì S3 tắt; **không** cần `--crop-review`); ô `qd01_locked` gọi `save_crop(bbox_cu, prev_bbox=prev_bbox_cu, next_bbox=next_bbox_cu)` (md5 phụ thuộc **cả hộp hàng xóm** qua carve `:290-292, :572-576` — đóng băng cả ba thì md5 tất định); dọn thư mục đích trước khi cắt (hôm nay `silver/` 4.490 + `syllable/` 1.674 tệp mồ côi 25/08); `save_crop:273-310` giữ nguyên (pad 0,12 + carve + tighten) | | hai mức: `bbox == bbox_cu` **2.014/2.014 bắt buộc**; `md5 == md5_cu` kỳ vọng 2.014, ô lệch liệt kê vào QĐ-01a lý do 'carve' |
| N7b | Manifest | [SỬA] | `:606-647`: **`labels.csv` giao nộp cố định 12 cột** (`image,book,page,column,ocr_char,syllable,label,unicode,tier,rule,bbox,image_md5`), mọi cột chẩn đoán sang **`labels_trace.csv`** + `columns.csv` (xem `DANH_MUC_SUA_DOI_CUOI_2026-09-16.md` §2); nội bộ `dataset_out/labels_final.csv` giữ đủ cột. Cột chẩn đoán gồm **17 cột**: `syllable_ocr, syllable_raw, nom_idx, syl_idx, tier_v3, p_register, dict_support, context_evidence, l1_support, n_ocr, n_qn, n_det, count_source, box_source, flank_gold, label_canonical, qd01_locked` (+ `band_touched, norm_fixed, l1_tie, qd01_excluded, am_da_quyet_ngoai_khoa` — tất cả DÀY 0/1); `summary.json` thêm `det_thr, det_xmargin, matrix, T, phân bố count_source/box_source` | `labels.csv` 22 → ~39 cột | |
| N8 | `enrich_crop_quality` | [GIỮ] | `run_pipeline.sh:373` `--src-root "$DS_OUT"` → +3 cột | | cờ chỉ để tra, **không** gác cổng (hết phân biệt sau tái định tâm; không bắt lỗi sai glyph) |

### 3.6 Hậu xử lý

| # | Nút | Loại | Mã | Ghi chú | Kiểm |
|---|---|---|---|---|---|
| N9 | `remediate` | [SỬA nhẹ] | `run_pipeline.sh:395-400` (`census` + `apply --tau 0.62`, `s3_demote=False`); `remediate.py:78` bỏ `split` khỏi tuple bắt buộc; `census.py:60-72` thêm luật **cùng md5 trong cùng cột** (cặp 法/冉 lọt hôm nay) | bắt buộc (trùng crop); hôm nay no-op (0/0/0) | report 0 quarantine, 0 nhóm md5 trùng |
| N10 | `confusion_fix` | [GIỮ] | `:407-416` | chốt kép cho (người,㝵) ngoài khoá; **DEMOTABLE = {GOLD, SILVER}** không đụng ô khoá (label 𠊚) | demote = 0 |
| N11 | `s3_unwind` | [GIỮ] | `:428-434` chạy **vô điều kiện** (no-op tất định khi S3 tắt: 0 SILVER) để `s3_unwind_report.json` được ghi mới = 0 — nếu bỏ, báo cáo cũ 25/08 (readmitted 1.185) ở lại và `update_bang_so_lieu` đọc vào bảng VA_LOI khi thăng cấp | | report readmitted 0 / unwound 0 |
| N12 | `glyph_fix --mode kiem` | [SỬA] | `glyph_fix.py:88-121`: chế độ `khoa_o` — join `(book,page,column,nom_idx)` với `qd01_cells.csv` **và** `qd01a_decisions.csv`; **không gán mới** (`CO_THE_GAN` chỉ cho chế độ theo âm cũ); report `n_khop / n_pending / n_bbox_doi (0) / n_md5_doi`; `run_pipeline.sh:449-451` truyền `--mode kiem --cells --decisions` | QĐ-01a: người điền `qd01a_decisions.csv` rồi **rebuild** (không sửa tay CSV) | `run_pipeline.sh:474` đếm tiền tố `quyet_dinh_nguoi:` (locked + pending) == 2014; trước export pending == 0 |
| N13 | `assert_qd01()` | [MỚI] | tách `:466-481` thành hàm, gọi sau N9, N10, N12, N14 | | |
| N14 | export | [GIỮ] | `:506-508 export_final_dataset` (USABLE = GOLD + SYLLABLE; ảnh copy theo đường `<tier lúc build>/`), `:516 make_dataset_docs`, `:518 make_xlsx` | | `usable_image` |

### 3.7 Đối soát, người, thăng cấp, bằng chứng

| # | Nút | Loại | Mã | Kiểm |
|---|---|---|---|---|
| N15 | `doi_soat_the_he.py` | [MỚI] | `pipeline/tools/doi_soat_the_he.py --old dataset_out/labels_final.csv(+nom_idx từ N0c) --new dataset_out_v3/labels_final.csv --qd01 config/qd01_cells.csv` → `docs/DOI_SOAT_v3.md`: (B1) crosstab tier×rule cũ→mới; (B2) ô đổi âm ghép — **một cấu hình** CALIB + neo cap 2,0 LOO: GOLD ≈439 ± 20, REVIEW ≈780; (B3) ô mới/mất ≈ +660 / −210 (GOLD mất ≈18 — liệt kê từng ô; mốc "≤10" của báo cáo không đạt với neo); (B4) bbox đổi theo `count_source` (≈11.5k usable); (B5) md5 đổi (≤14,1k) + ô khoá có hàng xóm thật khác `prev/next_cu`; (B6) QĐ-01 locked/pending/bbox/md5; (B7) ≈1.290 GOLD→REVIEW liệt kê theo rule + dòng 'người ngoài khoá' (10); (B8) L1 crosstab | mọi số kỳ vọng ở §4 |
| N16 | **QĐ-01a** (người) | [MỚI] | người xem crop của ô pending (≤10 ô + 23 ô trùng hộp nếu chọn phương án trộn) — phán quyết ghi vào **`config/qd01a_decisions.csv`** (N5g đọc) + phụ lục a của `docs/NGUOI_CHAM_QUYET_DINH.md` làm `xuat_xu`; rồi **rebuild** (N19 tái lập được không cần tay) | pending → 0 |
| N17 | Thử 1 sách | [MỚI] | gọi thẳng `python -m pipeline.align_engine.build_dataset --config config/pipeline_stt11.yaml --limit 30 --out dataset_out_test --reseg detector` (runner tương tác, không truyền `--limit`); so với `labels_final` cùng trang **chỉ các cờ không phụ thuộc corpus** (`direct, sim_unique, tone, bbox, count_source`) — cờ `corpus/bigram` LOO trên 30 trang không so được | trước khi chạy 3 sách |
| N18 | Thời gian | [MỚI] | `run_pipeline.sh:654-675` `_t=$SECONDS` quanh mỗi `step_*`; `checkpoint()` `:382-393` ghi kèm giây; `evidence()` thêm dòng thời gian vào HIEN_HANH; chạy 2 lần (có/không detector cache) | thay mọi số "45–60 / 25–35 phút" |
| N19 | **Thăng cấp** | [MỚI] | sau N15/N16 đạt: `DS_OUT=dataset_out ./run_pipeline.sh` (tất định, không RNG → chạy lại sạch hơn rsync; `.FROZEN` gỡ có chủ ý); rồi **nối vào cuối `step_export`** (`run_pipeline.sh:518`, chỉ khi `DS_OUT == dataset_out`): S3 tắt → `rm pipeline/align_engine/s3_proto_cache.pkl` và **bỏ** `rebuild_proto_index` (chạy nó làm `index.csv` đổi mtime → repro_check R2 lệch với cache cũ) → `python -m pipeline.tools.update_bang_so_lieu` (đổi HEADER `:165-174` sang sha16 của `labels_final` để **một commit** là đủ) → `evidence()` `:675` (đã có) → commit → `check_consistency` (R1 đòi cây sạch) | `bash scripts/check_consistency.sh` **3/3** (phép proto-index bỏ cùng S3); `check_evidence.sh` khớp |
| N20 | Bộ giao nộp | [GIỮ] | `re-dataset/` (GOLD + SYLLABLE), `labels.xlsx`, DATASHEET; `re-dataset/check/` (bộ kiểm ngoài) chạy tay | |

---

## 4. BẢNG NGHIỆM THU SỬA LẠI (đo trên **bản build**, không phải mô phỏng)

| Chỉ tiêu | Hiện hành | Kỳ vọng v3.1 | Đo bằng | Ghi chú |
|---|---|---|---|---|
| Tái lập HEAD trước khi đổi gì | chưa từng | **0 lệch** theo `(book,page,column,bbox)` | N0c | điều kiện tiên quyết |
| Cột có khe giả / cặp lệch chéo (cột khớp số ≥10 chữ) | 346 / 1,63% | **54 / 0,82%** | `thuc_nghiem_ke_hoach.py matrix` **và** đếm trực tiếp trên `labels.csv` mới (ops del/ins trong cột `n_ocr == n_qn`) | `thuc_nghiem.py rebuild calib` **vô cảm với engine** (`:76` ép PROD) — chỉ dùng sau khi sửa `set_matrix(ENGINE)` |
| Khe đúng chỗ / ghép sai khi rụng 1 chữ (benchmark) | 60% / 3,85% | 85–86% / 1,3–1,4% | `thuc_nghiem.py recursive` (sửa `:525` chi phí neo 2,0) | mô phỏng có đáp án — ghi rõ "benchmark cols.pkl" |
| Ô dùng được | 64.525 | **70.100 – 70.450** (cận trên ~70.500; L3/L5 đã thay bằng v3) | crosstab `labels_final_v3` | không phải "70.108 ± 300" |
| GOLD hiện hành → REVIEW | — | ≈1.290 (similar ≈821, ngược ≈217, direct ≈136, L1 ≈67, cột lệch ≈22; QĐ-01 41 → khoá lại; 'người' ngoài khoá 10) | N15 B7 | các ô "yếu lộ diện" |
| QĐ-01 | 2.014 | **2.014 = locked + pending**, pending ≤ 10 (→ 0 trước export); **bbox 2.014/2.014 = bbox_cu (bắt buộc)**; md5 khớp kỳ vọng 2.014 với prev/next đóng băng, ô lệch liệt kê | N12 + N7 assert | |
| L1 | 755 đổi âm, ≈245 ghi đè (định nghĩa ÂM–ÂM LOO) | ghi đè bản in **0** (đo trên `syllable_ocr`); đổi âm chỉ khi `l1_support > 0` (≈376), hoà giữ gốc (≈134) | N15 B8 | định nghĩa 2-gram cố định N5e |
| Hộp detector đúng ở cột rụng chữ | 59,1% | **≥ 95%** trên cột \|G\|=\|Q\| (thr 0,2, ±0,25w, CALIB+neo) | `thuc_nghiem.py geo` tham số hoá | báo cả tỉ lệ cột phủ (≈91% thay 63%) |
| `count_source` / `box_source` | không có | phân bố ±0,25w: equal_qn ≈75% cột · equal_ocr ≈17% · conflict ≈8% · legacy_locked_col (cột có QĐ-01); midpoint chỉ còn trong `conflict`/legacy | `summary.json` | không mốc, chỉ báo cáo |
| md5 crop đổi | — | ≤ 14.117 ô usable, **liệt kê**; QĐ-01 0 | N15 B5 | thay nguyên tắc "không đổi md5" |
| Ghép sai theo tier (benchmark) | GOLD 1,3–1,7% | CHAR_A ≤0,6% · SYL ≤1,0% · CHAR_B ≤6% (có cờ) | `thuc_nghiem.py v3` | |
| Selftest | 722 | 722 + ≥15, 0 fail | `scripts/run_all_selftests.sh` | |
| Thời gian | không nguồn | số thật, 2 lần | N18 | |
| Chuỗi bằng chứng | 2/4 | **4/4** sau N19 | `check_consistency.sh` | không đạt được ở `dataset_out_v3` |

---

## 5. THỨ TỰ THỰC THI (ước lượng công)

| Ngày | Việc | Nút |
|---|---|---|
| 1 | N0a–N0f: `DS_OUT` + `NONINTERACTIVE`, nom_idx/syl_idx, **tái lập HEAD**, `qd01_cells.csv` (kèm prev/next_cu), selftest **chỉ cho hàm đã có**, BUG-5/6 | hàng rào |
| 2 | N3b–N3e: BUG-1, CALIB, `cost_fn`, `posterior_matches` port; chạy `thuc_nghiem_ke_hoach.py` + selftest | A1 |
| 3 | N3g + N4a–N4c: `col_states`, PASS 1b, posterior (cột hộp = '' tới ngày 5) | A2 |
| 4 | N5a–N5e, N5g, N5h: `tier_v3.py`, ánh xạ rule, L1 có cổng, khoá QĐ-01, (người,㝵) ngoài khoá; N17 thử 1 sách | A4/A5 |
| 5 | N3f + N4d + N7: detector thr/margin, 3 nhánh hộp, cột có QĐ-01 chạy luật cũ, `bbox/prev/next_cu` cho ô khoá, dọn thư mục crop, 17 cột; `geo` tham số hoá; selftest legacy 100% | A3 (nếu < 2 tuần: `--box-rule legacy` trọn gói) |
| 6 | N5f, N5i, N5k, N12, N13: `decisions.yaml` (corpus_readings, di_the → `label_canonical`), `qd01a_decisions.csv` (rỗng), `glyph_fix --mode kiem`, `assert_qd01`; `BASELINE_PASS` cập nhật | A5 |
| 7 | N15, N16, N18: chạy 3 sách vào `dataset_out_v3`, đối soát thế hệ, QĐ-01a (người, ~15 phút), thời gian | A6 |
| 8 | N19: thăng cấp vào `dataset_out`, `rebuild_proto_index` + `update_bang_so_lieu` + `evidence()`, `check_consistency` 4/4, commit | A6 |
| sau | Khối B (H1 `p_visual_register` out-of-fold; H2 cụm (sách, âm) → yaml có người ký, kể cả `(cùng,其)`), Khối C (600 ô mù), Khối D (F3g crops_v2, Pitch DP, tách từ dính) | ngoài v3.1 |

---

## 6. NHỮNG CÂU CHỮ TRONG v3.1 PHẢI SỬA

1. Dòng 4: bỏ "không thay đổi md5 crop" → "md5 đổi được liệt kê (N15); **crop QĐ-01 giữ nguyên byte**".
2. A0-2: khoá `(book,page,column,bbox,image_md5,label)` → `(book,page,column,nom_idx[,syl_idx])`; `bbox_cu/image_md5_cu` chỉ để giữ crop; sinh **sau** N0c.
3. A0: thêm "tái lập HEAD, 0 lệch" trước A1; selftest liệt đủ 6 hàm + ca kiểm; tên biến `DS_OUT`.
4. A1: `posterior_matches(..., cost_fn)` — port vào engine, cùng chi phí neo; `realign_column(cost_fn, forced)`.
5. A2: sửa "align_page trong build_dataset.py" → `align_production.py:469`; ghi rõ `pair_pages` từ **mọi** match, LOO `(book,page)`, không đổi `confirmed`; PASS 1b chạy trên `col_states`, không gọi lại detector; ghi `pair_pages.json`.
6. A3: ba nhánh (\|G\|=\|Q\| → `G[syl_idx]`; \|G\|=\|OCR\| → `G[nom_idx]`; còn lại → đường cũ + `conflict`); `DETECTOR_XMARGIN = 0.25` (không phải 0,10); `count_source/box_source` định nghĩa; `--box-rule legacy`; QĐ-01a.
7. A4: thêm định nghĩa 6 cờ **LOO**; ghi rõ tier_v3 **thay** decide_label + L2 + L3 + L5; chốt `is_plausible` + `AM_DA_QUYET`; tên `rule` ánh xạ; khoá QĐ-01 **sau** tier_v3, **trước** PASS 2.
8. A5: `label_canonical` thay vì gộp `label`; đơn vị `(sách, âm)`; mọi mục có `xuat_xu`; `(cùng,其)→共` chờ người ký; L1 định nghĩa 2-gram ÂM–ÂM trên `syllable_raw`, LOO, hoà → giữ gốc; `corpus_readings` không vào `qn_to_nom`; BUG-1 vá **một chỗ** `:80`.
9. A6: bỏ "`--crop-review` nếu tắt S3" (bẫy 2.232); thêm N15 đối soát theo `nom_idx`, N16 QĐ-01a, **N19 thăng cấp** (4/4 chỉ đạt ở `dataset_out`), `rebuild_proto_index` + `update_bang_so_lieu` nối vào `step_export` (`evidence()` đã có sẵn `:675`), `s3_unwind` chỉ khi `--use-s3`, thời gian thật.
10. Bảng §1: thay theo §4 ở trên (70.100–70.450; `rebuild calib` cần sửa `thuc_nghiem.py:76`; ≥95% trên cột \|G\|=\|Q\|; thêm dòng tái lập HEAD, md5 đổi, QĐ-01a, selftest, thời gian).

---

## 7. ĐÃ SỬA SAU 2 LƯỢT PHÊ BÌNH FLOW (18:40)

| Lỗi của bản flow 17:00 | Sửa |
|---|---|
| **[chặn]** "md5 QĐ-01 2.014/2.014 byte-identical" chỉ bằng `bbox_cu` — bất khả vì carve dùng hộp **hàng xóm** (`build_dataset.py:571-576`, `save_crop:290-292`) | N0d lưu `prev/next_bbox_cu`; N7 gọi `save_crop(bbox_cu, prev_cu, next_cu)`; assert hai mức (bbox bắt buộc, md5 kỳ vọng) |
| **[lớn]** trộn hộp cũ (ô khoá) với hộp mới A3 của hàng xóm → 23 cặp trùng bbox → `remediate.py:106-108` cách ly **cả ô QĐ-01** | cột có ô khoá chạy trọn gói luật cũ (`legacy_locked_col`) |
| **[lớn]** chốt `AM_DA_QUYET` bao trùm ở N5c → mất 243 ô, hạ 49 ô mốc đã ký; N5h thành vô nghĩa | bỏ chốt bao trùm; N5h chỉ hạ ô nhãn v3 == 㝵 (10 ô); usable −10 |
| **[lớn]** `--box-rule legacy` không lùi được vì `DETECTOR_THR` đổi toàn cục (`_DETECTOR` singleton `:252`) | legacy = trọn gói thr 0,3 + ±0,5w + đường cũ; selftest tái lập 100% |
| neo cứng QĐ-01 trong DP mâu thuẫn với nhánh (ii) và đổi ghép ô lân cận không cờ | bỏ ép; ô trôi → pending |
| QĐ-01a không có đường dữ liệu (N12 không gán, N19 chạy lại từ config) | `config/qd01a_decisions.csv` do N5g đọc |
| `assert_qd01` gãy ngay lần chạy đầu (ô pending chưa có rule) | pending mang `quyet_dinh_nguoi:pending`, đếm tiền tố; trước export pending == 0 |
| bỏ `s3_unwind` để báo cáo cũ 25/08 (readmitted 1.185) lọt vào `update_bang_so_lieu` | giữ, no-op tất định |
| N19 `rebuild_proto_index` làm R2 lệch (mtime index.csv vs cache cũ); 2 commit | rm cache khi S3 tắt, bỏ rebuild; HEADER sha16 → 1 commit |
| cột cờ thưa → pandas float `1.0` → join trượt | cột DÀY 0/1 hoặc DTYPE str |
| lịch ngày 1 viết test cho hàm chưa tồn tại | test theo ngày làm nút |
| L1 kỳ vọng 498/161/70 thuộc định nghĩa khác | 376/245/134 theo ÂM–ÂM LOO; 2-gram dựng từ `col_states` không phụ thuộc DP |
| `syllable_raw` sau normalize_column ≠ bản in (~115 ô) | thêm `syllable_ocr` |
| usable 70.300–70.700 (+131 sai: chỉ 38 usable; thiếu −19, −10) | 70.100–70.450 |
| số lệch nhỏ: 1.187 → 656 tên ảnh QĐ-01; 846/110 → 848/112; `run_all_selftests.sh:9` → `:80`; `BOX_OVERLAP_FRAC ~137` → `:184`; `normalize_column:509` → `syllable_normalize.py:65`; `build_dataset.py:78-83` → `:73-78`; `ocr_page:645-720` → `:645-715`; N3f số của ±0,10w → ±0,25w (93,2/4,8/1,9); N17 gọi thẳng `build_dataset`; N0a grep; N0d sinh từ bản ký; rule CHAR_B-cầu giữ `s1_inter_s2_similar` | đã sửa tại chỗ |
