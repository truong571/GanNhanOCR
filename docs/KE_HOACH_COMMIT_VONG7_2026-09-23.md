# Kế hoạch commit — vòng 7 (23/09/2026)

Trên HEAD `014a0733fa`. **Chưa commit gì.** Báo cáo: `docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md`.
Cổng chung trước MỌI commit dưới đây:

```bash
PY=.venv/bin/python
$PY -m pipeline.align_engine.book_layout_selftest && $PY -m pipeline.phase1_engine_selftest \
 && $PY -m pipeline.tools.ingest_lithograph_selftest && $PY -m pipeline.tools.ingest_prose_selftest \
 && $PY -m pipeline.tools.ingest_ihr_selftest && $PY scripts/measure/code_facts.py --check
# hồi quy STT: labels.csv md5 phải = 59e436d7641fa849bb6759868ac29259 (556 dòng), diff -rq 0 tệp
```

## C1 — engine: `books[].crop_source` (crop giao nộp giữ ảnh quét gốc)

```
pipeline/align_engine/book_layout.py          (+ CROP_SOURCES, BookLayout.crop_source/.crop_from_original, kiểm giá trị, docstring)
pipeline/align_engine/build_dataset.py        (+ _orig_index / load_original_page / _paper_bg; save_crop(img_orig, bin_path);
                                               PASS1 crop_src_by_book + fail fast; PASS2 nạp ảnh gốc + dọn crops_bin/;
                                               summary.crop_source_by_book chỉ khi có sách khai)
pipeline/tools/enrich_crop_quality.py         (+ --bin-root, tự ưu tiên <src-root>/crops_bin)
pipeline/align_engine/book_layout_selftest.py (+ [10] crop_source, 33 phép)
pipeline/phase1_engine_selftest.py            (+ test_crop_source_original, 17 phép)
scripts/measure/code_facts.py                 (known pin book_layout.py 83 -> 98)
```

Thông điệp đề xuất:

```
feat(engine): books[].crop_source — crop giao nộp cắt từ ảnh quét GỐC, bản đã xử lý sang crops_bin/

Hình học (pad, seam carve, tighten_box) vẫn tính trên ảnh đã xử lý rồi áp nguyên xi sang ảnh gốc:
labels.csv chỉ khác đúng cột image_md5 (đo 126/126 ô, 2 trang LVT1883); crops_bin/ trùng byte với
crop cũ nên enrich_crop_quality giữ nguyên crop_quality_flag/stray_ink/border_ink. Mặc định
'processed' -> STT byte-identical (md5 59e436d7641fa849bb6759868ac29259, diff -rq 0 tệp).
book_layout 133/0 · phase1 270/0 · code_facts 18/18.
```

## C2 — config: bật `crop_source: original` cho 5 sách thạch/mộc bản

```
config/pipeline_LucVanTien1883.yaml  config/pipeline_KimVanKieu1884.yaml
config/pipeline_KimVanKieu1884_b1.yaml  config/pipeline_Chrestomathie1872.yaml
config/pipeline_LucVanTien1916.yaml  config/pipeline_TruyenKieu1872.yaml
```

`config/pipeline.yaml` (3 sách STT) **KHÔNG đụng**. Tách khỏi C1 để hoàn nguyên được bằng 1 lệnh
`git revert` mà không mất cơ chế.

## C3 — bộ đo: ảnh mẫu trước/sau

```
scripts/measure/crop_source_samples.py        (mới; 2 bất biến, 10 ảnh, 0 FAIL)
scripts/measure/README.md                     (thêm 1 dòng bảng cho mô-đun này — CHÚ Ý: tệp này còn 1 dòng
                                               `align_audit.py` của vòng trước, xem mục "KHÔNG thuộc vòng 7")
```

## C4 — lab: guard STT tuỳ chọn + `best_litho.pt`

```
lab/i5_detector_v2/train_kaggle.py            (+ --guard-stt-f1 <số|none>)
lab/i5_detector_v2/i5v2/trainer.py            (guard tuỳ chọn; LUÔN ghi best_litho.pt + warning trong ckpt; report.md)
lab/i5_detector_v2/build_notebook.py          (+ GUARD_STT, HF_REPO mặc định, in cả 2 ckpt, ưu tiên input chỉ-mã)
lab/i5_detector_v2/kaggle_train_i5_v2.ipynb   (sinh lại)
lab/i5_detector_v2/kaggle_run_2026-09-22_ketqua.ipynb  (MỚI — giữ output lần chạy 22/09 làm bằng chứng)
lab/i5_detector_v2/make_bundle.py             (+ i5v2_code.zip 39 KB)
lab/i5_detector_v2/apply_v2.sh                (nhận best_litho.pt, --allow-stt-drop, 6 config, thêm all-ihr)
lab/i5_detector_v2/bundle/{train_kaggle.py,i5v2/trainer.py,kaggle_train_i5_v2.ipynb}  (đồng bộ)
lab/i5_detector_v2/README.md
```

⚠️ `lab/i5_detector_v2/.gitignore` — kiểm trước khi `git add`: `bundle/`, `i5v2_bundle.zip` (243 MB) và
`i5v2_code.zip` **không được** vào git.

## C5 — gỡ PTCL khỏi phạm vi

```
scripts/measure/measure.py            (PTCL ra khỏi ALL_BOOKS/ALL_STEPS)
scripts/measure/auto_precision.py     (bỏ ref Kieu1871_LVD; ref thiếu tệp -> bỏ qua có cảnh báo)
scripts/measure/qn_engine_compare.py  (bỏ REFS[KimVanKieu1884]; lọc theo tệp tồn tại)
scripts/measure/verses_ref_fix.py     (--ref không còn mặc định; báo lỗi chỉ cách khôi phục)
run_pipeline.sh                       (thiếu tệp ref -> cảnh báo + bỏ B1')
config/pipeline_KimVanKieu1884_b1.yaml(comment khối verses_ref_fix)  [gộp vào C2 nếu muốn ít commit]
```

## C6 — tài liệu

```
docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md
docs/KE_HOACH_COMMIT_VONG7_2026-09-23.md
docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md  (§10: hộp cảnh báo PTCL loại khỏi phạm vi)
docs/CHAY_3_BO_CON_LAI_2026-09-23.md          (hộp cảnh báo ở đầu)
docs/CHAY_KVK1884_B1_2026-09-22.md            (lệnh B1' không còn chạy được + cách khôi phục)
docs/PIPELINE_FACTS.json                      (code_facts sinh lại)
```

## KHÔNG commit / KHÔNG đụng

- `data/` (kể cả 5 tệp PTCL đang ` D` — **để nguyên trạng thái xoá trong worktree**, đừng `git rm`:
  tệp phiên âm Kiều 1871 còn cần cho KVK nếu muốn bật lại B1'/nền dị bản).
- `dataset/`, `dataset_out*/`, `prepared*/`, `measure_out/`.
- `config/pipeline.yaml` (STT).

## KHÔNG thuộc vòng 7 (đã có sẵn trong worktree TRƯỚC khi vòng này bắt đầu)

`web/{app.js,index.html,style.css,server.py,sample_data.json}`, `nom-embed` (submodule),
`scripts/measure/align_audit.py` + `docs/RA_SOAT_CAN_CHINH_2026-09-23.md` (chưa theo dõi) và dòng
`align_audit.py` trong `scripts/measure/README.md`. **Đừng gộp vào các commit trên** — tách riêng hoặc
để nguyên. `docs/PIPELINE_FACTS.json` bị `code_facts.py` sinh lại mỗi lần chạy: commit cùng C1.

## Thứ tự

C1 → C2 → C3 → C4 → C5 → C6. Sau C1 và sau C2 chạy lại cổng chung (đặc biệt hồi quy STT).
