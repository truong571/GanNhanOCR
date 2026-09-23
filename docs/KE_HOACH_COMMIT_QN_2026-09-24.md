# Kế hoạch commit — chữa nút thắt quốc ngữ (24/09/2026)

Nền: HEAD `8b6b59f112` (= `origin/main`). **Chưa commit gì.** Thiết kế + số đo: `docs/CHUA_QN_2026-09-24.md`.

## 0. Nguyên tắc của vòng này

* Mọi cờ mới **mặc định TẮT** ⇒ đường STT và mọi sách chưa khai cờ giữ nguyên hành vi, nguyên byte.
* Cột mới `qn_fix_kind` chỉ xuất hiện trong `labels.csv` của sách **đã khai** `qn_charfix: true`
  (đúng cơ chế đã dùng cho `qn_count_unfixed`) ⇒ `labels.csv` của STT giữ nguyên bộ cột cũ.
* Không sửa `data/`; `measure_out/` không commit.

## 1. Hồi quy (chạy trước và sau)

| Phép kiểm | Mốc (HEAD) | Sau khi sửa |
|---|---|---|
| STT 3 trang `labels.csv` (`build_dataset --config config/pipeline.yaml --qd01-cells none --force --pages pages3.csv`) | md5 `59e436d7641fa849bb6759868ac29259`, 556 dòng | **md5 y hệt, 556 dòng** |
| `bash scripts/run_all_selftests.sh` | 1176 passed, 9 failed | **1192 passed, 9 failed** (+16 phép mới của `qn_charfix`; KHÔNG FAIL mới) |
| `scripts/measure/verses_ref_fix.py --selftest` | 19/19 | **29/29** (+10 phép của hạn chế mức âm) |
| `scripts/measure/code_facts.py --check` | 18/18 | **18/18** (đã cập nhật `KNOWN_PINS` `book_layout.py:98 → :105`) |
| `book_layout` selftest | 157/0 | **157/0** |
| `align_audit --book all` (vi phạm/bất biến) | 1/8 · 1/8 · 0/8 · 1/12 · 1/12 · 0/9 · 2/12 · 2/12 | **y hệt từng con số** (sau khi sửa align_audit, xem Commit 5) |
| `measure.py --all --report-only` | 170 PASS / 0 FAIL | **170 PASS / 0 FAIL** |
| `ihr_endtoend_eval --book all` (precision nhãn NGƯỜI) | LVT1916 98,04 % (n 11.496) · TK1872 98,61 % (n 18.550) | **98,05 % (n 11.585)** · **98,61 % (n 18.550)** — KHÔNG giảm |
| `merge_datasets --check` | 16/16 PASS | **16/16 PASS** |
| STT `dataset_out/labels.csv` đầy đủ (83.239 ô) dựng lại | — | **BYTE-IDENTICAL với HEAD** |
| `git status -- data dataset_out prepared` | trống | **trống** (`dataset_out/{CHECKSUMS.txt,summary.json}` bị lần chạy ghi thêm đã `git checkout` trả lại — một là nhật ký nối thêm, một là khoá `detector_params_by_book` lệch sẵn từ trước vòng này) |

## 2. Các commit đề nghị

### Commit 1 — `feat(qn): tầng L3 sửa lỗi ký tự OCR quốc ngữ trước align (mặc định TẮT)`
* `pipeline/align_engine/qn_charfix.py` (mới)
* `config/lexicon/qn_charfix.json` (mới)
* `pipeline/align_engine/syllable_normalize.py` — tham số `charfix=`, L3 chạy TRƯỚC tầng dời dấu
* `pipeline/align_engine/book_layout.py` — khoá `books[].qn_charfix`
* `pipeline/align_engine/align_production.py` — truyền cờ + tính `qn_fix_kind` theo vị trí âm
* `pipeline/tools/selftest.py` — 16 phép mới
* `scripts/measure/code_facts.py` — cập nhật `KNOWN_PINS`

Bằng chứng kèm: STT md5 `59e436d7…` không đổi · selftest 1192/9 · bảng ký tự đã LOẠI `ö→ô` và `ø→o`
sau khi đo (giữ chúng thì phá 3 ô GOLD; xem `CHUA_QN` §4).

### Commit 2 — `feat(qn): cột truy vết qn_fix_kind vào labels/trace/parquet`
* `pipeline/align_engine/build_dataset.py` — `COL_QN_FIX_KIND`, ghi có điều kiện
* `pipeline/export_final_dataset.py` — thêm vào `TRACE`
* `pipeline/tools/make_dataset_docs.py` — mô tả cột
* `pipeline/publish/export.py` — `PARQUET_COLUMNS` thêm `syllable_ocr`/`syllable_raw`/`qn_fix_kind`
  (trước đây bộ HF chỉ xuất `syllable` = âm SAU chuẩn hoá ⇒ mất âm bản in)

### Commit 3 — `feat(qn): verses_ref_fix --only-invalid-or-tone (hạn chế mức âm)`
* `scripts/measure/verses_ref_fix.py` — `restrict_tokens()`, cờ CLI, cột `qn_kept_ocr`, 10 phép selftest mới
* `run_pipeline.sh` — đọc `run.verses_ref_fix.only_invalid_or_tone`

### Commit 4 — `feat(config): bật qn_charfix 5 sách + PA2 có hạn chế cho LucVanTien1883`
* `config/pipeline_LucVanTien1883.yaml` — `qn_charfix: true` + khối `verses_ref_fix` (nf1916, 0,9, hạn chế)
* `config/pipeline_KimVanKieu1884_b1.yaml`, `config/pipeline_Chrestomathie1872.yaml`,
  `config/pipeline_LucVanTien1916.yaml`, `config/pipeline_TruyenKieu1872.yaml` — `qn_charfix: true`
* `config/pipeline.yaml` (STT) **KHÔNG đụng**

### Commit 5 — `fix(measure): align_audit miễn ô L3 qn_charfix KHI tái lập được`
* `scripts/measure/align_audit.py` — A3/A4/A5 đếm riêng ô `qn_fix_kind == charfix` **chỉ khi** chạy lại
  đúng luật L3 trên âm thô của `transcriptions/` cho ra đúng `labels.syllable`; khai `charfix` mà không
  tái lập được thì vẫn vi phạm, dưới tên riêng `A3_charfix_khong_tai_lap`.
* Vì sao cần: lần chạy đầu `align_audit` báo FAIL tăng (LVT1883 1→4/12, KVK 1→4/12, Chresto 0→3/9,
  LVT1916 2→5/12) và **100 %** vi phạm mới là ô `charfix` — phép đo so `labels.syllable` với âm THÔ, mà
  L3 (giống hệt tầng dời dấu đã được miễn từ trước) cố ý ghi lại âm đó. Sau khi sửa: **216/216 ô tái lập
  được, 0 ô rơi vào tên mới**, mọi con số bất biến trở về ĐÚNG BẰNG bản chốt 23/09.

### Commit 6 — `feat(measure): hai bộ đo nền cho quyết định (0 API)`
* `scripts/measure/qn_syllable_fix.py` (mới) — đo PA1; nay dùng làm **công cụ CHẨN ĐOÁN** và **CỔNG CHẶN**
  (nhãn `ambiguous` = bằng chứng giữ ô ở REVIEW), KHÔNG làm kênh gán nhãn. `--selftest` 10/10.
* `scripts/measure/qn_orthography.py` (mới) — đo PA3 L0/L1/L2/L2b/L3 + chấm bằng nhãn người.
  `--selftest` 15/15, 48/48 invariants.

### Commit 7 — `docs(qn): báo cáo chữa nút thắt quốc ngữ + kế hoạch commit`
* `docs/CHUA_QN_2026-09-24.md`, `docs/KE_HOACH_COMMIT_QN_2026-09-24.md`, `docs/PIPELINE_FACTS.json` (sinh lại)

### Commit 8 — bộ dựng lại (`dataset/`, nếu vẫn theo lệ commit bộ giao nộp)
`dataset/<Sách>/` của 5 sách + `dataset/_ALL/`; bản trước đã dời vào `archive/dataset/`.

## 3. KHÔNG commit / KHÔNG áp (danh sách dứt khoát)

1. PA1 làm kênh gán nhãn (206 ô) — độ đúng âm 67,5 % / 76,0 %.
2. PA2 **không hạn chế** cho LucVanTien1883 ở bất kỳ ngưỡng nào (0,90 đã phá 37 ô GOLD ĐÚNG).
3. PA2 dưới ngưỡng 0,90 cho LucVanTien1883 (không có tham chiếu độc lập để nghiệm thu).
4. PA3 hiện đại hoá chính tả — cả bảng soạn tay 40 cặp lẫn bảng máy 790 cặp.
5. PA3 tầng L2 máy rút (463 cặp, có cặp sai đang bắn `nụa→nga`).
6. PA3 tầng L1 dời dấu chạy TRƯỚC L3.
7. Ghi bất kỳ chuẩn hoá nào đè lên `syllable_ocr`.
8. Áp hạn chế mức âm vào KimVanKieu1884 (ở đó lớp "âm hợp lệ khác hẳn" LỢI, đã kiểm độc lập bằng 1872).

## 4. Việc còn mở sau vòng này

* **Nút thắt thật chưa chạm tới:** 1.206/2.085 ô REVIEW-do-QN của LVT1883 (57,8 %) có âm tiếng Việt HỢP LỆ
  nhưng ĐỌC SAI. Hướng duy nhất còn lại: nâng chất lượng ảnh trang quốc ngữ / đổi engine OCR ở ĐẦU VÀO.
* **LucVanTien1883 không có tham chiếu độc lập** (nhãn người IHR trùng `nomfoundation_lvt` 97,63 % ký tự
  ⇒ cùng nguồn). Mọi con số PA2 của sách này là CHẶN TRÊN tự khẳng định.
* `auto_precision --steps cross` của LucVanTien1883 nay dùng CÙNG bản 1916 vừa dùng để sửa quốc ngữ
  ⇒ con số "khớp dị bản" của sách này đã **vòng tròn một phần**, phải đọc kèm cảnh báo này.
* Chrestomathie1872 không có phiên âm ⇒ PA2 bất khả; chỉ còn L3.
