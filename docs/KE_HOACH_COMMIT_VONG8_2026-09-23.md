# Kế hoạch commit — vòng 7 + vòng 8 GỘP (23/09/2026)

Trên HEAD `014a0733fa`, nhánh `main`. **Chưa commit gì.**
Thay thế `docs/KE_HOACH_COMMIT_VONG7_2026-09-23.md` (giữ tệp ấy làm hồ sơ, **đừng dùng để commit nữa**).
Báo cáo: `docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md` (vòng 7) ·
`docs/VONG8_CAN_CHINH_VA_CROP_2026-09-23.md` (vòng 8) · đề bài `docs/RA_SOAT_CAN_CHINH_2026-09-23.md`.

## Cổng chung — chạy TRƯỚC MỌI commit dưới đây

```bash
PY=.venv/bin/python
$PY -m pipeline.align_engine.book_layout_selftest      && \
$PY -m pipeline.phase1_engine_selftest                 && \
$PY -m pipeline.tools.ingest_lithograph_selftest       && \
$PY -m pipeline.tools.ingest_prose_selftest            && \
$PY -m pipeline.tools.ingest_ihr_selftest              && \
$PY -m pipeline.remediation.mechanism_gates_selftest   && \
$PY -m pipeline.align_engine.char_detector.pitch_decode --selftest && \
$PY scripts/measure/align_audit.py --selftest          && \
$PY scripts/measure/ihr_endtoend_eval.py --selftest    && \
$PY scripts/measure/verses_ref_fix.py --selftest       && \
$PY scripts/measure/code_facts.py --check
# Mốc phải thấy: 157/0 · 288/0 · 82/82 · 33 · 81/81 · 113/0 · 22/0 · OK · 24/24 · 19/19 · 18/18
# hồi quy STT: labels.csv md5 = 59e436d7641fa849bb6759868ac29259 (556 dòng, 42 cột), diff -rq 0 tệp
```

---

## C1 — engine: `books[].crop_source` (crop giao nộp cắt từ ảnh quét GỐC) — *vòng 7*

```
pipeline/align_engine/book_layout.py          (CROP_SOURCES, BookLayout.crop_source/.crop_from_original)
pipeline/align_engine/build_dataset.py        (_orig_index / load_original_page / _paper_bg; save_crop(img_orig, bin_path);
                                               PASS1 crop_src_by_book + fail fast; PASS2 nạp ảnh gốc + dọn crops_bin/)
pipeline/tools/enrich_crop_quality.py         (--bin-root, tự ưu tiên <src-root>/crops_bin)
pipeline/align_engine/book_layout_selftest.py (mục [10] crop_source)
pipeline/phase1_engine_selftest.py            (test_crop_source_original)
```

⚠️ Ba tệp `book_layout.py` / `build_dataset.py` / `book_layout_selftest.py` / `phase1_engine_selftest.py`
**mang thay đổi của CẢ hai vòng**. Nếu muốn C1 và C4 tách sạch thì phải `git add -p`; nếu không,
**gộp C1 + C4 thành một commit** (khuyến nghị: gộp — hai cơ chế đều đã được nghiệm thu cùng một lần chạy).

```
feat(engine): books[].crop_source — crop giao nộp cắt từ ảnh quét GỐC, bản đã xử lý sang crops_bin/

Hình học (pad, seam carve, tighten_box) vẫn tính trên ảnh đã xử lý rồi áp nguyên xi sang ảnh gốc:
labels.csv chỉ khác đúng cột image_md5 (đo 126/126 ô, 2 trang LVT1883); crops_bin/ trùng byte với
crop cũ nên enrich_crop_quality giữ nguyên crop_quality_flag/stray_ink/border_ink. Mặc định
'processed' -> STT byte-identical (md5 59e436d7641fa849bb6759868ac29259, diff -rq 0 tệp).
```

## C2 — config: bật `crop_source: original` cho 5 sách + trả lại B1' của KVK — *vòng 7 + 8*

```
config/pipeline_LucVanTien1883.yaml     config/pipeline_KimVanKieu1884.yaml
config/pipeline_KimVanKieu1884_b1.yaml  config/pipeline_Chrestomathie1872.yaml
config/pipeline_LucVanTien1916.yaml     config/pipeline_TruyenKieu1872.yaml
```

`config/pipeline.yaml` (3 sách STT) **KHÔNG đụng**. Trong `_b1` còn một thay đổi của vòng 8:
khối `run.verses_ref_fix` đã **bỏ dấu `#`** (tệp phiên âm Kiều 1871 đã có lại trên đĩa).

```
config: crop_source original cho 5 sách mới + bật lại B1' verses_ref_fix của KimVanKieu1884

Ảnh giao nộp giữ nền giấy bản quét (nền p90 255 -> 128..133; đo 6 ô mẫu, 0 FAIL).
Tệp thamchieu_kieu_1871_LieuVanDuong_phienam.json nằm trong thư mục PTCL nhưng là DỊ BẢN CỦA KVK,
không phải dữ liệu của bản chép tay -> giữ lại, B1' chạy lại như cũ.
```

## C3 — bộ đo: ảnh mẫu crop + rà chuẩn căn chỉnh — *vòng 7 + 8*

```
scripts/measure/crop_source_samples.py   (MỚI, vòng 7 — 2 bất biến)
scripts/measure/align_audit.py           (MỚI, đã có trong worktree TRƯỚC vòng 7 — 13 bất biến + đo trôi, 8 phép tự kiểm)
scripts/measure/README.md                (2 dòng bảng cho hai mô-đun trên)
```

## C4 — engine: 4 sửa CĂN CHỈNH — *vòng 8* (đề bài `RA_SOAT_CAN_CHINH` §4 #1/#2/#4 + §3.1)

```
pipeline/tools/ingest_ihr_book.py             (a) page_seq_step = 2*ceil(n/2); cờ verses_odd trong page_gate
pipeline/tools/ingest_ihr_selftest.py         (+17 phép: page_seq_step, parity, số câu không trùng)   -> 81/81
pipeline/align_engine/book_layout.py          (c) lithograph_gate(tier_rule=…) lấy luật 6/8 làm kỳ vọng
                                              (b) PROSE_DP_RATIO_MIN, COL_QN_COUNT_UNFIXED,
                                                  qn_count_flag_on / prose_dp_ratios / qn_count_unfixed_columns
pipeline/align_engine/align_production.py     (c) truyền tier_rule vào cổng; (b) gắn cờ vào pair + col_states;
                                              (d) pitch_target_count(n_rule=…) -> 'pitch_rule';
                                                  assign_boxes_pitch(mode=…)
pipeline/align_engine/build_dataset.py        (b) cột qn_count_unfixed vào labels.csv CHỈ khi có ô mang khoá;
                                                  PASS 1b giữ count_source 'pitch_rule'
pipeline/remediation/mechanism_gates.py       (e) cổng qn_count_unfixed + qn_count_mode() + --qn-count-gate;
                                                  COUNT_SOURCE_PITCH thêm 'pitch_rule'
pipeline/remediation/mechanism_gates_selftest.py (+19 phép: mục [6])                                  -> 113/0
pipeline/align_engine/book_layout_selftest.py (+24 phép: mục [11]; sửa 1 phép CŨ đã đổi hành vi có chủ ý) -> 157/0
pipeline/phase1_engine_selftest.py            (+18 phép: test_pitch_rule_and_qn_flag)                  -> 288/0
```

```
fix(align): parity số câu IHR + cờ qn_count_unfixed + cổng 6/8 + pitch_target_count theo luật

(a) ingest_ihr: seq += 2*ceil(n/2) — verse_pairs_for_page giả định first_seq LẺ, cộng số câu thô làm
    một trang lẻ đảo parity cho mọi trang sau. A11 61/988 -> 0 (LVT1916), 1.078/1.610 -> 0 (TK1872);
    số câu bị dùng hai lần 1 -> 0 ở cả hai bộ. Tập trang ghi ra không đổi.
(b) cờ CỘT qn_count_unfixed vào labels.csv (lithograph: n_qn != 14; prose: dp_ratio < 0,75) và cổng
    cơ chế (e) hạ ô GOLD của cột ấy xuống REVIEW (mặc định review cho lithograph, off cho prose/STT).
    Lý do chọn REVIEW chứ không GOLD_text_only: nhóm này chỉ đúng 52,2 % (n 209) / 49,0 % (n 51) trên
    NHÃN NGƯỜI, tức sai ở chính nhãn văn bản. Đo end-to-end: precision GOLD-ảnh 97,07 -> 97,96 %
    (LVT1916) và 98,49 -> 98,64 % (TK1872); lỗi "khác hẳn" 111 -> 31 và 19 -> 1 ô.
(c) lithograph_gate lấy LUẬT 6/8 thay num_syllables (do chính tesseract đếm) -> cột 13/15 âm trượt
    cổng thay vì đi lọt: page_ok 103/105 -> 77/105, 162/163 -> 137/163, 97/99 -> 81/99, 159/161 -> 154/161.
    Cổng chỉ ĐẾM (summary.json layout_gate), KHÔNG loại ô nào.
(d) pitch_target_count: khi có luật và kim đọc đúng 14 chữ thì N = 14 thay vì n_qn sai
    -> count_source 'pitch_rule' 364/350/237/67 ô. Trung lập theo mọi phép đo hiện có
    (box_ref_eval 24/24 PASS không đổi; crop_quality ±4 ô).

STT byte-identical: labels.csv md5 59e436d7641fa849bb6759868ac29259, 556 dòng, 42 cột (không có cột
mới), diff -rq 0 tệp. book_layout 157/0 · phase1 288/0 · ingest_ihr 81/81 · gates 113/0 · code_facts 18/18.
```

## C5 — lab: guard STT tuỳ chọn + `best_litho.pt` — *vòng 7*

```
lab/i5_detector_v2/train_kaggle.py            (--guard-stt-f1 <số|none>)
lab/i5_detector_v2/i5v2/trainer.py            (guard tuỳ chọn; LUÔN ghi best_litho.pt + warning trong ckpt)
lab/i5_detector_v2/build_notebook.py          (GUARD_STT, HF_REPO mặc định, in cả 2 ckpt)
lab/i5_detector_v2/kaggle_train_i5_v2.ipynb   (sinh lại)
lab/i5_detector_v2/kaggle_run_2026-09-22_ketqua.ipynb  (MỚI — output lần chạy 22/09 làm bằng chứng)
lab/i5_detector_v2/make_bundle.py             (i5v2_code.zip 39 KB)
lab/i5_detector_v2/apply_v2.sh                (nhận best_litho.pt, --allow-stt-drop, 6 config, thêm all-ihr)
lab/i5_detector_v2/README.md
```

⚠️ Kiểm `lab/i5_detector_v2/.gitignore` TRƯỚC khi `git add`: `bundle/`, `i5v2_bundle.zip` (243 MB) và
`i5v2_code.zip` **không được** vào git.

## C6 — PTCL: gỡ bản chép tay, GIỮ tệp phiên âm Kiều 1871 — *vòng 7 (gỡ) + vòng 8 (trả lại 1871)*

```
scripts/measure/measure.py            (PTCL ra khỏi ALL_BOOKS/ALL_STEPS)                       [v7]
scripts/measure/verses_ref_fix.py     (--ref không còn mặc định; thiếu tệp -> chỉ cách khôi phục) [v7]
run_pipeline.sh                       (thiếu tệp ref -> cảnh báo + bỏ B1')                      [v7]
scripts/measure/auto_precision.py     (v7: ref thiếu tệp -> bỏ qua có cảnh báo;
                                       v8: TRẢ LẠI ("Kieu1871_LVD", …) LÊN TRƯỚC trong CROSS_BOOKS[KVK].refs)
scripts/measure/qn_engine_compare.py  (v7: REFS lọc theo tệp tồn tại; v8: TRẢ LẠI REFS["KimVanKieu1884"])
```

```
chore(scope): bản chép tay PTCL ra khỏi phạm vi, GIỮ tệp phiên âm Kiều 1871 (dị bản của KVK)

Bốn tệp mô tả R.987 để nguyên trạng thái xoá trong worktree (KHÔNG git rm).
thamchieu_kieu_1871_LieuVanDuong_phienam.json chỉ nằm nhờ trong thư mục đó nhưng là tham chiếu dị bản
của KimVanKieu1884 -> giữ, và trả lại vào CROSS_BOOKS/REFS. Bằng chứng khôi phục đúng: số trôi KVK
trùng khít mốc công bố (n 31.866 · đúng 80,58 % · lệch vị trí 16 · 1/4.792 câu trôi).
Các nhánh "thiếu tệp -> cảnh báo, bỏ bước" của vòng 7 giữ nguyên.
```

## C7 — tài liệu

```
docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md   (MỚI; §4 thêm hộp "ĐÃ SỬA Ở VÒNG 8")
docs/VONG8_CAN_CHINH_VA_CROP_2026-09-23.md         (MỚI — báo cáo vòng 8)
docs/RA_SOAT_CAN_CHINH_2026-09-23.md               (MỚI — đề bài, rà soát 13 bất biến)
docs/KE_HOACH_COMMIT_VONG7_2026-09-23.md           (MỚI — hồ sơ; kế hoạch dùng THẬT là tệp này)
docs/KE_HOACH_COMMIT_VONG8_2026-09-23.md           (MỚI — tệp này)
docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md       (§10: hộp cảnh báo PTCL, bản vòng 8)
docs/CHAY_3_BO_CON_LAI_2026-09-23.md               (hộp cảnh báo đầu tệp, bản vòng 8)
docs/CHAY_KVK1884_B1_2026-09-22.md                 (lệnh B1' CHẠY LẠI ĐƯỢC)
docs/PIPELINE_FACTS.json                           (code_facts sinh lại — commit CÙNG C4)
```

## KHÔNG commit / KHÔNG đụng

- `data/` — kể cả 4 tệp PTCL đang ` D`: **để nguyên trạng thái xoá**, đừng `git rm`, đừng khôi phục.
  Tệp `thamchieu_kieu_1871_LieuVanDuong_phienam.json` đã có lại trên đĩa và **khớp HEAD** (không hiện
  trong `git status`) — không cần làm gì.
- `dataset/`, `dataset_out*/`, `prepared*/`, `measure_out/` (kể cả `measure_out/_vong8_before/` —
  bản sao nhãn "trước" để đối chiếu, xoá được).
- `config/pipeline.yaml` (STT).
- **`web/{README.md,app.js,index.html,style.css,server.py,sample_data.json}`** và **`nom-embed`**
  (submodule ` m`): có sẵn trong worktree TRƯỚC vòng 7, **KHÔNG thuộc hai vòng này** — tách commit
  riêng hoặc để nguyên. Đừng gộp vào C1–C7.

## Thứ tự

**C1+C4 → C2 → C3 → C5 → C6 → C7.** Chạy lại cổng chung sau C1+C4 và sau C2 (đặc biệt hồi quy STT).

## Nếu muốn hoàn nguyên từng sửa của C4 (không cần revert cả commit)

| Sửa | Tắt bằng |
|---|---|
| (e) hạ cấp cột đếm hỏng | `books[].qn_count_gate: off` trong config sách, hoặc `--qn-count-gate off` — cờ vẫn được ghi |
| (c) cổng theo luật 6/8 | bỏ đối số `tier_rule=` ở lời gọi `lithograph_gate` (`align_production.py`, 1 dòng) |
| (d) N theo luật | truyền `n_rule=None` ở `_pair_new_state` (1 dòng) |
| (a) parity IHR | `seq += len(verses)` ở `load_pages` (1 dòng) — sẽ làm A11 hỏng lại |
