# Kế hoạch commit — chuẩn hoá cấu trúc mã nguồn (23/09)

Nền: HEAD `66514d666f`. **Chưa commit gì** — phiên chính commit. Thiết kế: `docs/CAU_TRUC_MA_NGUON_2026-09-23.md`.

## Hồi quy đã chạy (trước và sau)

| Phép kiểm | Mốc (HEAD) | Sau khi gom/dọn |
|---|---|---|
| STT 3 trang `labels.csv` (`build_dataset --config config/pipeline.yaml --qd01-cells none --force --pages pages3.csv`) | md5 `59e436d7641fa849bb6759868ac29259`, 556 dòng | **md5 y hệt, 556 dòng** |
| `./run_pipeline.sh --book all --dry-run` | 359 dòng | **`diff` rỗng** (359 dòng) |
| `bash scripts/run_all_selftests.sh` | 1176 passed, **9 failed** (đã FAIL sẵn ở HEAD: `ground_truth` 2, `remediation` 4, `tools` 3 — do commit `66514d666f` đổi khoá chính bộ gộp sang `cell_uid`) | **1176 passed, 9 failed — bảng từng selftest `diff` rỗng** |
| `book_layout` 157/0 · `ingest_lithograph` 82/82 · `ingest_prose` 33 · `ingest_ihr` 81/81 · `mechanism_gates` 113/0 | như cột này | **không đổi** |
| `scripts/measure/code_facts.py --check` | 18/18 | **18/18** (`docs/PIPELINE_FACTS.json` KHÔNG cần sinh lại: 0 tham chiếu tới đường dẫn đã dời) |
| `git status -- data dataset dataset_out prepared` | trống | **trống** |

## Commit 1 — XOÁ hiện vật dựng lại được (không đụng git index)

Tất cả đều **untracked + đã `.gitignore`** ⇒ commit này **không có tệp nào trong diff**; chỉ là việc dọn đĩa,
ghi lại ở đây để có dấu vết. Đã xoá:

| Đường dẫn | Dung lượng | Dựng lại bằng |
|---|---|---|
| `lab/i5_detector_v2/bundle/` | 246 MB | `.venv/bin/python lab/i5_detector_v2/make_bundle.py` (≈3 phút) |
| `lab/i5_detector_v2/i5v2_bundle.zip` | 231 MB | như trên (`--zip`, mặc định) |
| `ArcFace/data.zip` | 111 MB | `ArcFace/prepare_data.py` |
| `khoiB/KhoiB_kaggle_dataset.zip` · `khoiB/v3/KhoiB_v3_kaggle_dataset.zip` | 153 MB | `khoiB/pack_for_kaggle.py` |
| `lab/enhanced_self_training_v3/{enhanced_ocr_kaggle,enhanced_results}.zip` | 108 MB | `lab/enhanced_self_training_v3/pack_for_kaggle.py` |
| `lab/self_training_v2/self_training_kaggle_dataset.zip` | 78 MB | `lab/self_training_v2/pack_self_training_for_kaggle.py` |
| `logs/*.log` cũ hơn 23/09 (15 tệp) | 19,5 MB | mỗi lần `./run_pipeline.sh` sinh lại |
| `scripts/measure_wf_2026-09-21/**/*.py` (52 tệp) | 0,3 MB | đã GỘP vào `scripts/measure/*` (docstring dòng 4); 37/52 gắn cứng `/private/tmp/claude-501` |

**Tổng ≈ 947 MB** (đĩa 15 G → 14 G). GIỮ lại 3 `*.json` + 1 `*.csv` trong `measure_wf` (docs trích) +
`scripts/measure_wf_2026-09-21/README.md` mới giải thích. KHÔNG đụng `*.pt` nào.
Hoàn nguyên: chạy lại script dựng tương ứng (không mất dữ liệu gốc).

## Commit 2 — GOM: `scripts/_oneoff/` (13 `git mv`)

`git mv` 12 script một lần ở gốc `scripts/` + `docs/ra_soat_2026-09-13_do_lai.py` → `scripts/_oneoff/`
(`build_ihr_manifest`, `build_lucvantien1883_dataset`, `check_ocr_token`, `delete_pua_codepoints`,
`download_{chrestomathie1872,handwritten_manuscripts,kieu1884_bilingual,quocngu_texts}`,
`extract_{kieu1894_pages,stt_quocngu_pages}`, `make_flow_report_html`, `ocr_all_quocngu_pages`).
Thêm `scripts/_oneoff/README.md` (bảng cũ→mới, phân biệt với `_retired/`).
Sửa tham chiếu: `README.md`, `MoTaCode.txt`, `docs/RA_SOAT_TOAN_BO_2026-09-13.md`.
**Không** sửa `data/*.md` (ràng buộc cấm đụng `data/`) và **không** sửa `docs/KE_HOACH_COMMIT_2026-09-21.md`
(hồ sơ lịch sử, giữ nguyên đường dẫn thời điểm đó) — bảng cũ→mới trong `_oneoff/README.md` bù cho việc này.
Hoàn nguyên: `git mv` ngược + `git checkout -- README.md MoTaCode.txt docs/RA_SOAT_TOAN_BO_2026-09-13.md`.

## Commit 3 — GOM: `kaggle_diffusion/` → `lab/kaggle_diffusion/` (16 `git mv`)

Nhánh sinh glyph FontDiffusion chỉ tới được qua đường step-3 đã nghỉ ⇒ là **thí nghiệm**, đưa về `lab/`.
Sửa kèm (bắt buộc, nếu thiếu là hỏng): `parents[1]` → **`parents[2]`** trong 5 tệp
(`build_priority_list`, `build_style_medoid`, `extract_book_chars`, `extract_missing_chars`, `run_local_sanity`)
vì gốc kho lùi thêm một bậc — đã kiểm `REPO → …/GanNhanOCR` và `py_compile` sạch.
Cập nhật đường dẫn (chỉ dòng **chú thích**, không đổi hành vi): 8 `config/pipeline*.yaml`,
`pipeline/align_engine/visual_signal.py:7`, `README.md`, `MoTaCode.txt`. Thêm `lab/README.md`.
Hoàn nguyên: `git mv lab/kaggle_diffusion kaggle_diffusion` + `git checkout -- config pipeline README.md MoTaCode.txt`.

## Commit 4 — Tài liệu

`docs/CAU_TRUC_MA_NGUON_2026-09-23.md` (mới) · `docs/KE_HOACH_COMMIT_CAUTRUC_2026-09-23.md` (tệp này) ·
`lab/README.md` · `scripts/_oneoff/README.md` · trích dẫn trong `README.md` (§Cau truc du an).
`scripts/maintenance/dep_graph.py` (công cụ đo, do nhiệm vụ A tạo) đi kèm commit này hoặc commit riêng.

## KHÔNG làm trong vòng này (có lý do)

- **`khoiB/`**: index git ghi `KhoiB/` (68 tệp **đang track**), đĩa ghi `khoiB/` (macOS không phân biệt hoa/thường)
  ⇒ `git ls-files khoiB` trả 0 và dễ kết luận sai là rác. `run_pipeline.sh:363` dùng `KhoiB/v3/crops_v3.npz`
  làm đầu vào CHÍNH của bước 5, chỉ **âm thầm** rơi xuống `lab/gan_nhan_2026-09-13/crops.npz` nếu vắng. Dời = rủi ro hỏng im lặng.
- **`pipeline/lab/`**: gói chạy bằng `python -m pipeline.lab.*` (`scripts/run_all_selftests.sh:161`), có trong `PIPELINE_FACTS.json`.
- **`evaluation/`**: là thư mục ĐẦU RA đang sống (`pipeline/check_ocr_columns.py:71` + `run_pipeline.sh:285`), không phải vỏ rỗng.
- **`scripts/_retired/`, `pipeline/step3_label.py`, `step4_export.py`, `core/ranking/*`,
  `pipeline/align_engine/to_standard.py`**: đều có chỉ dẫn "giữ có chủ ý" trong mã / config / README.
- **`lab/i5_detector_v2/last.pt`** (330 MB): `best.pt` không còn trên đĩa, HF repo riêng tư 401 ⇒ nguồn duy nhất
  còn lại của `train_crop/detector_r34_v2_litho.pt` (5 config ghim). GIỮ.
- **Submodule** `font_diffusion`, `gannhanocr-fd`, `nom-embed`, `NomNaOCR/ds4v_repo`: gỡ phải `git submodule deinit`.
- **`data/`, `dataset/`, `dataset_out/`, `prepared/`, `nom-embed/`, `web/`**: ràng buộc — không đụng.
- **`CLAUDE.md`**: chưa thêm dòng trỏ tới `docs/CAU_TRUC_MA_NGUON_2026-09-23.md` — việc này do **người dùng/phiên chính**
  quyết (agent không tự sửa CLAUDE.md). Đề nghị thêm 1 gạch đầu dòng ở mục quy ước.

## Việc còn để ngỏ (cần người chốt, KHÔNG tự làm)

`NomNaOCR/finetune_data/` (257 MB, 17.576 tệp đang track, 0 tham chiếu mã) ·
`NomNaOCR/weights/NomNaOCR_SC-CRNNxCTC.h5` (63,6 MB track, 0 tham chiếu) ·
`lab/gan_nhan_2026-09-13/emb.npz` (161 MB, chỉ `lab/tham_dinh_2026-09-16/*` đọc) ·
`font_diffusion/ckpt/PROD` (384 MB) + `gannhanocr-fd/` (351 MB) — vẫn ghim trong 8 config dù đường step-3 đã nghỉ ·
`.git` 2,5 GB (≈1,7 GB là blob của đường dẫn đã biến mất: `Data/`, `Projects/`, `re-dataset/`) — chỉ `git filter-repo`
mới thu hồi, **không nên** làm trước khi nộp luận văn.
