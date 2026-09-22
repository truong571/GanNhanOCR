# Kế hoạch commit — diff chưa commit tính đến 2026-09-22 09:30 (HEAD c499c8277c)

Chưa commit gì. Tệp này là kết quả (1) hồi quy STT lần cuối, (2) review toàn bộ diff, (3) đề xuất 3 commit.
Lệnh chạy ở gốc repo, `PY=.venv/bin/python`.

## 1. Bằng chứng hồi quy STT (22/09, worktree sạch `/tmp/gn_base` @ c499c8277c, đã gỡ)

Lệnh (cả hai bên, chỉ khác `--config`/`--out`; `pages3.csv` = `book,page` ∈ {stt2/page_0024, stt4/page_0050, stt11/page_0100}):

```
$PY -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --qd01-cells none --force --pages pages3.csv --out <dir>
```

| Chỉ số | Baseline HEAD | Repo hiện tại |
|---|---|---|
| `labels.csv` md5 | `59e436d7641fa849bb6759868ac29259` (556 dòng) | `59e436d7641fa849bb6759868ac29259` (556 dòng) — `cmp` byte-identical |
| Tệp trong out/ | 346 | 346; `diff -rq`: 344 giống byte |
| `summary.json` | — | chỉ thêm khoá `detector_params_by_book` (3 sách STT = 0,2 / 0,25 toàn cục) + `pass1c.decisions.path` đổi REPO tuyệt đối; KHÔNG có `layout_gate` |
| `decisions_report.json` | — | giống hệt sau khi thay đường dẫn REPO |
| Log | — | 0 dòng `detector theo sách` |

Selftest (repo hiện tại): `book_layout_selftest` **57/57**, `ingest_lithograph_selftest` **35/35**, `phase1_engine_selftest` **253 pass / 0 fail**,
`tools.selftest` **125 pass / 4 fail** — đúng mốc §6.2 `docs/CHAY_LVT1883_2026-09-21.md`. Phân tích 4 fail ở §4.

## 2. Ba commit đề xuất

Trước khi `git add` bất kỳ commit nào: chạy §4 (khôi phục `dataset_out/`). Dùng `git add <tệp>` đích danh, **không** `git add -A` / `commit -a`
(working tree còn `D data/LucVanTien1883/luc_van_tien_quoc_ngu.txt`, submodule `nom-embed` có `best.pt`/`last.pt` sửa, 2 tệp của agent KVK đang chạy).

### Commit 1 — engine: layout / n_columns / det_xmargin / det_thr theo sách (mặc định = hành vi STT cũ, byte-identical)
```
pipeline/align_engine/book_layout.py            (mới)
pipeline/align_engine/book_layout_selftest.py   (mới, 57 phép kiểm)
pipeline/align_engine/align_production.py       (_detect/align_page nhận layout; cổng lithograph; rec["layout_gate"])
pipeline/align_engine/build_dataset.py          (book_layout(b); DETECTOR_* theo sách trong PASS 1 rồi khôi phục; summary detector_params_by_book/layout_gate)
pipeline/step0_setup.py                          (lithograph không cần khoá pdf)
pipeline/step1_extract.py                        (lithograph không có pdf -> lỗi rõ, không chạy)
pipeline/step2_align.py                          (_get_qn_lines(n_columns=…))
config/pipeline.yaml                             (chỉ thêm chú thích khoá tuỳ chọn; 3 sách STT không khai)
```
Thông điệp gợi ý: `feat(engine): tham số bố cục theo sách (layout/n_columns/det_xmargin/det_thr), mặc định = STT; hồi quy 3 trang byte-identical`.
Kèm trong thân: md5 59e436d7…, selftest 57/57 + 253/0.

### Commit 2 — tools: adapter thạch bản + config sách mới + bộ đo + tài liệu
```
pipeline/tools/ingest_lithograph_book.py         (mới)
pipeline/tools/ingest_lithograph_selftest.py     (mới, 35/35)
config/pipeline_LucVanTien1883.yaml              (mới; det_xmargin 0,05 / det_thr 0,15)
config/pipeline_KimVanKieu1884.yaml              (mới; det_xmargin 0,05)
scripts/measure/                                 (mới: measure.py, layout_lithograph.py, qn_print_ocr.py, chresto_map.py, detector_transfer.py, code_facts.py, qn_visual_truth.json, README.md)
scripts/download_chrestomathie1872.py, scripts/extract_kieu1894_pages.py   (mới)
scripts/build_ihr_manifest.py, scripts/download_kieu1884_bilingual.py      (sửa)
CLAUDE.md                                        (mới — quy ước đo)
docs/PIPELINE_FACTS.json                         (SINH LẠI ngay trước commit: `$PY scripts/measure/code_facts.py`; bản hiện tại lệch nhỏ: KVK n_ocr_cache 8→163, 2 tệp untracked mới)
docs/README.md, docs/PIPELINE_SACH_MOI_2026-09-20.md, docs/KET_QUA_DO_CUOI_2026-09-21.md,
docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md, docs/CHAY_LVT1883_2026-09-21.md, docs/KE_HOACH_COMMIT_2026-09-21.md
.gitignore                                       (measure_out/, dataset_out_LucVanTien1883*/, dataset_out_KimVanKieu1884*/, dataset_LucVanTien1883*/, dataset_KimVanKieu1884*/, scripts/measure_wf_*/)
```
Thông điệp gợi ý: `feat(tools): ingest_lithograph_book + config LVT1883/KVK1884 + bộ đo offline scripts/measure + tài liệu 20–22/09`.
**Chờ agent KVK xong** rồi mới quyết: `docs/CHAY_KVK1884_THU_2026-09-21.md`, `docs/KVK1884_TRANG_CAN_XAC_NHAN.csv`,
`pipeline/tools/kiem_nguoi_grid.py`, `pipeline/tools/kiem_nguoi_score.py` (tạo 09:01–09:03 22/09 bởi agent song song — chưa review, chưa có selftest).

### Commit 3 — ocr_api: Guest Mode (token lấy lại mỗi lần thử; 401 not-active -> Guest, nhớ trong tiến trình) + test mock
```
core/ocr/ocr_api.py
pipeline/phase1_engine_selftest.py               (test_ocr_guest_mode, 14 phép kiểm, không mạng)
```
Thông điệp gợi ý: `fix(ocr_api): Guest Mode HCMUS — re-login gửi token mới, 401 not-active rơi về Guest và nhớ; test mock`.
Lưu ý trong thân commit: đổi User-Agent/Origin/Referer + MIME `image/jpeg` cứng (xem R-01) — chỉ ảnh hưởng trang chưa có cache OCR (`pages/` đóng băng).

### Commit riêng (ngoài 3 commit trên) — dữ liệu `data/`
16 tệp `M` (SOURCE.md 12 sách, README, manifest/summary LVT1916 & TruyenKieu1872), `??` `data/Chrestomathie1872/` (82 MB), `data/KimVanKieu1884/` (186 MB),
3 JSON phiên âm Nôm Foundation, `data/KIEM_TRA_KHOP_1-1_2026-09-20.md`. Repo đã track ảnh `data/` (5.101 tệp) nên hợp lệ, nhưng tách commit để lịch sử mã không lẫn 268 MB ảnh.
**Quyết định trước khi commit:** `D data/LucVanTien1883/luc_van_tien_quoc_ngu.txt` (2.059 dòng, tracked) — SOURCE.md không ghi lý do xoá, `scripts/download_quocngu_texts.py:152` vẫn ghi tệp này.
Hoặc khôi phục `git checkout -- data/LucVanTien1883/luc_van_tien_quoc_ngu.txt`, hoặc ghi 1 dòng lý do vào SOURCE.md rồi `git rm`.

## 3. KHÔNG commit (đã thêm .gitignore, trừ mục cuối)
| Đường dẫn | Lý do |
|---|---|
| `dataset_out_LucVanTien1883/`, `dataset_out_LucVanTien1883_xmargin025/` (217 MB ×2) | đầu ra build, tái sinh bằng `build_dataset --config config/pipeline_LucVanTien1883.yaml --out …` |
| `dataset_LucVanTien1883/` (236 MB), `dataset_LucVanTien1883_xmargin025/` (203 MB) | export, tái sinh |
| `dataset_out_KimVanKieu1884/` (48 MB, đang được agent khác ghi), `dataset_KimVanKieu1884/` | đang chạy; tái sinh |
| `prepared/LucVanTien1883`, `prepared/KimVanKieu1884` | đã trong `prepared/` (.gitignore:13); tái sinh bằng `ingest_lithograph_book` |
| `measure_out/` | .gitignore (đã có); `measure.py --all` |
| `scripts/measure_wf_2026-09-21/` (56 tệp, 37 tệp chứa đường dẫn scratchpad) | bản thô của workflow, KHÔNG phải bản sao `scripts/measure/`; bản tái lập là `scripts/measure/`. Không tài liệu nào trỏ tới. Đã ignore `/scripts/measure_wf_*/`; xoá hay lưu trữ là quyết định của chủ repo |
| `nom-embed/best.pt`, `nom-embed/last.pt` (submodule ` m`) | không phải phạm vi đợt này; không `git add nom-embed` |

## 4. `dataset_out/` STT đang bị xoá — khôi phục TRƯỚC commit và TRƯỚC selftest
Trạng thái: `git status` báo `D` 10 tệp tracked (`labels.csv` 23 MB, `labels_final.csv`, `labels_remediated.csv`, `summary.json`, `pair_pages.json`,
`CHECKSUMS.txt`, `confusion_fix_report.json`, `decisions_report.json`, `remediation_report.json`, `self_training_rescue_report.json`); thư mục
không tồn tại trên đĩa (các thư mục crop gold/silver/… vốn gitignored cũng đã mất — không khôi phục được bằng git, phải chạy lại pipeline nếu cần).

```
git checkout -- dataset_out          # khôi phục 10 tệp tracked từ HEAD; không đụng tệp khác
git status --short -- dataset_out    # phải trống
```
Vì sao cần: (a) 10 tệp `D` sẽ bị ghi thành **xoá** nếu ai đó `git add -A`/`commit -a` — mất bộ giao nộp STT 25/08 khỏi lịch sử làm việc;
(b) `pipeline.tools.selftest` đọc `dataset_out/labels_final.csv` (`rebuild_proto_index.split_lech`) → khôi phục xong thì `index.csv không rỗng` PASS và ~14 phép kiểm khác được bật;
(c) `docs/PIPELINE_FACTS.json` ghi `dataset_out_deleted: 10`, `checkpoint_exists 5/8` — sinh lại sau khôi phục để FACTS phản ánh đúng.

Kỳ vọng sau khôi phục (đo 22/09 trên HEAD + `dataset_out/` + `dataset/`): **139 pass / 3 fail**. 3 fail còn lại CÓ SẴN ở HEAD, không do diff này:
1. `image là khoá chính` — `dataset/labels.csv` (gitignored, 20/09 02:19, 71.610 dòng) có **21** khoá `image` trùng (vd `gold/stt11_page_0026_c02_021.png`).
2. `image_md5 giữ đủ 12 hex` — xlsx giao nộp ghi md5 **32** hex; test đòi 12 → test hoặc export lệch schema.
3. `dùng gán rồi mới chữa mã thoát` — `run_pipeline.sh` (0f3aa095ef) không còn `|| n_old=0`/`|| n_new=0`; test `test_run_pipeline_grep_dem` lỗi thời.

## 5. Bảng phát hiện review (file:line → vấn đề → mức → đề xuất)
Mức: **CAO** = sai kết quả/mất dữ liệu; **TB** = hành vi đổi/ngầm sai ở đường ít dùng; **THẤP** = gọn/độ bền. ✅ = đã sửa trực tiếp (không đổi runtime).

| # | Vị trí | Vấn đề | Mức | Đề xuất |
|---|---|---|---|---|
| R-01 | `core/ocr/ocr_api.py:303` | `files={"image_file": (name, f, "image/jpeg")}` — MIME ghim `image/jpeg` trong khi `pages/*.png`; HEAD gửi file thô (không khai MIME). Đổi hành vi cả đường có token | TB | `mimetypes.guess_type(image_path)[0] or "application/octet-stream"`; thêm 1 phép kiểm MIME vào `test_ocr_guest_mode` |
| R-02 | `core/ocr/ocr_api.py:_base_headers` | thêm `Origin`/`Referer` + UA Chrome cho MỌI request (kể cả token) — đổi hành vi mặc định, chưa test chứng minh cần cho Guest | THẤP | ghi rõ trong thân commit 3; nếu server không đòi thì giữ UA cũ cho đường token |
| R-03 | `pipeline/step2_align.py:140,154` | `process_page_structural` vẫn gọi `_get_qn_lines(...)` mặc định 9 và `detect_nom_columns_v3(binary, ocr_columns, 9)`: sách `layout: lithograph` chạy qua `python -m pipeline.step2_align` sẽ ép 10 cột về 9 **im lặng**. run_pipeline.sh không dùng đường này, adapter cũng không | TB | như step1: đầu `process_book` — `if book_cfg.get("layout") == "lithograph": print(ERROR…); return`; hoặc truyền `book_layout(book_cfg).n_columns` xuống 2 chỗ + 1 phép kiểm |
| R-04 | `pipeline/align_engine/build_dataset.py:1177` | `layout_gate_stats` gom theo `lay is not DEFAULT_LAYOUT` → sách STT chỉ khai `det_xmargin` cũng sinh `summary["layout_gate"][book]` với `col_method {"?": N}` (gate chỉ ghi khi lithograph) + dòng log `layout=stt` | THẤP | đổi điều kiện thành `lay.is_lithograph` (cả dòng print 1132); không ảnh hưởng 3 config hiện có |
| R-05 | `scripts/measure/chresto_map.py:50-51` | `TF_ENV_DEFAULT` ghim đường dẫn scratchpad phiên `4224b41e…` (tồn tại hôm nay, sẽ mất) → bước NomNaOCR `skipped` lặng lẽ trên máy khác → SUMMARY khác nhau | TB | `Path(os.environ.get("GN_TF_ENV", REPO / ".tf_env"))`; ghi vào `scripts/measure/README.md`; thêm invariant `nomna.status` vào summary |
| R-06 | `scripts/measure/code_facts.py` (known pins) | invariant `known_pins_all_found` 5/9 FAIL: pin `step2_align.py:70,83` đã trôi về dòng 62/… sau diff | THẤP | cập nhật danh sách pin theo dòng mới (hoặc pin theo regex thay số dòng) rồi sinh lại FACTS |
| R-07 | `pipeline/tools/ingest_lithograph_book.py:736` | `--out` mặc định `"prepared"` tương đối CWD, trong khi `--measure-dir` neo `REPO` → chạy từ thư mục khác ghi sai chỗ | THẤP | `default=str(REPO / "prepared")` |
| R-08 | `data/LucVanTien1883/luc_van_tien_quoc_ngu.txt` | tệp tracked bị `D`, SOURCE.md không ghi lý do; `scripts/download_quocngu_texts.py:152` vẫn sinh nó | TB | quyết định ở §2 (khôi phục hoặc ghi lý do + `git rm`) |
| R-09 | `pipeline/tools/selftest.py:518` | test `n_old/n_new` lỗi thời so với `run_pipeline.sh` — FAIL ngay ở HEAD | THẤP | sửa test theo mã hiện hành (ngoài phạm vi diff này) |
| R-10 | `dataset/labels.csv` (gitignored) | 21 khoá `image` trùng / 71.610; xlsx `image_md5` 32 hex ≠ 12 | TB (dữ liệu giao nộp) | chạy lại export sau khi khôi phục `dataset_out/`; kiểm `pipeline.export_final_dataset` dedup theo `image`; đồng bộ độ dài md5 giữa export và test |
| R-11 | `scripts/measure_wf_2026-09-21/` | 56 tệp scratch, 37 tệp chứa đường dẫn `/private/tmp/claude-501/…` | THẤP | ✅ đã ignore `/scripts/measure_wf_*/`; không commit |
| R-12 | `.gitignore` | thiếu mục cho 5 thư mục build sách mới (≈920 MB) → dễ lọt vào commit | TB | ✅ đã thêm `/dataset_out_LucVanTien1883*/`, `/dataset_out_KimVanKieu1884*/`, `/dataset_LucVanTien1883*/`, `/dataset_KimVanKieu1884*/` |
| R-13 | `config/pipeline.yaml:24` | chú thích trỏ `config/lucvantien1883.yaml` (không tồn tại) | THẤP | ✅ sửa thành 2 tên tệp thật; YAML vẫn load |
| R-14 | `pipeline/align_engine/book_layout.py:59` | docstring `book_layout()` không nói tới `det_xmargin`/`det_thr` | THẤP | ✅ bổ sung docstring |
| R-15 | `scripts/measure/README.md:18` | `cd /Users/truongmdn/…` tuyệt đối | THẤP | ✅ đổi thành `<thư mục gốc repo>` |
| R-16 | `docs/CHAY_LVT1883_2026-09-21.md:26`, `docs/PIPELINE_SACH_MOI_2026-09-20.md:424` | trỏ log ở scratchpad phiên (sẽ mất) | THẤP | ✅ chú "thư mục tạm, không bền" |
| R-17 | `docs/CHAY_LVT1883…§6.2`, `docs/HUONG_DAN…:88` | ghi "4 fail cũ do dataset_out/ bị xoá" — thực tế 1/4 | THẤP | ✅ sửa theo §4 |
| R-18 | `pipeline/align_engine/book_layout_selftest.py:100,293` | test grep chuỗi trong nguồn `build_dataset.py`/`step0`/`step1` — dễ vỡ khi đổi câu chữ | THẤP | chấp nhận (có ghi chú) hoặc kiểm bằng gọi hàm thật với config giả |
| R-19 | `pipeline/tools/kiem_nguoi_grid.py`, `kiem_nguoi_score.py` | tệp mới của agent KVK (09:01–09:03 22/09), chưa review, chưa selftest | — | không đưa vào 3 commit; review riêng khi agent xong |
| R-20 | `nom-embed` (submodule ` m`) | `best.pt`/`last.pt` sửa trong submodule | THẤP | không `git add nom-embed`; xử lý ở đợt ArcFace |

Không phát hiện lỗi mức CAO. Đường STT được chứng minh byte-identical (§1); đường lithograph: `_get_detector` cache theo `thr` nên
`det_thr` theo sách có hiệu lực đúng; PASS 1b/1c dùng `cs["G"]` đã tính ở PASS 1 nên việc khôi phục `DETECTOR_*` toàn cục trước PASS 1b không làm lệch sách.
Tổng: 20 phát hiện — CAO 0 · TB 7 (R-01, R-03, R-05, R-08, R-10, R-12✅) · THẤP 12 (7 đã sửa ✅) · 1 ngoài phạm vi (R-19).

## 6. Thứ tự thực hiện gợi ý
1. `git checkout -- dataset_out` → `$PY -m pipeline.tools.selftest` kỳ vọng 139/3.
2. Quyết định R-08 (txt LVT1883).
3. Sửa (nếu đồng ý) R-01, R-03, R-04, R-05, R-07 → chạy lại `book_layout_selftest`, `phase1_engine_selftest`, hồi quy 3 trang (§1, md5 phải giữ 59e436d7…).
4. `$PY scripts/measure/code_facts.py` (sinh lại FACTS) → commit 1 → commit 2 → commit 3 → commit data.
5. Khi agent KVK xong: review `kiem_nguoi_*.py` + docs KVK, commit riêng.
