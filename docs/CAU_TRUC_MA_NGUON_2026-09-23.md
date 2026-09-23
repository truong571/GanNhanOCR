# Cấu trúc mã nguồn GanNhanOCR — chuẩn hoá 2026-09-23

Căn cứ: đồ thị phụ thuộc dựng bằng máy (`scripts/maintenance/dep_graph.py --check`, AST + grep,
0 token) + kiểm kê đĩa (`measure_out/maintenance/{inventory.md,by_dir.tsv,duplicates.md,big_files.md}`).
Một tệp được coi là **SỐNG** khi truy ngược được về một ĐIỂM VÀO THẬT, **hoặc** được một tài liệu
trong `docs/` · `README.md` mô tả kèm lệnh chạy (kho này là kho *documentation-driven*: chỉ grep
`*.py`/`*.sh` là chưa đủ, phải grep cả `*.md`, `*.yaml`, `*.ipynb`).

## 0. ĐIỂM VÀO THẬT (mọi thứ sống phải truy về đây)

| # | Điểm vào | Ghi chú |
|---|----------|---------|
| 1 | `./run_pipeline.sh` (mọi nhánh cờ, `--book all`) | gọi trực tiếp 29 tệp `.py`, tất cả đều SỐNG |
| 2 | `.venv/bin/python scripts/measure/measure.py --all` · `scripts/measure/auto_precision.py` | bộ đo tái lập (CLAUDE.md) |
| 3 | `bash scripts/run_all_selftests.sh` + các selftest lẻ (`book_layout`, `ingest_*`, `mechanism_gates`) | 13 selftest trong runner |
| 4 | `lab/i5_detector_v2/{make_bundle,train_kaggle}.py` + notebook Kaggle | quy trình huấn luyện detector v2 |

## 1. Cây thư mục gốc — giữ cái gì để làm gì

| Thư mục | Vai trò | Quy tắc |
|---------|---------|---------|
| `pipeline/` | **Mã sản xuất.** 6 bước của `run_pipeline.sh` + `align_engine/` (engine Step-2 đóng băng) + `remediation/` + `publish/` + `ground_truth/` + `tools/` | Chỉ mã nằm trên đường chạy hoặc selftest của nó. Mã thí nghiệm mới KHÔNG vào đây. |
| `core/` | Thư viện dùng chung: `pdf/`, `ocr/`, `image/`, `align/`, `text/` (+ `ranking/` giữ làm đối chứng luận văn) | Không phụ thuộc ngược vào `pipeline/`. |
| `config/` | `pipeline*.yaml` (8 bộ) + `confusion_fixes.yaml` + `decisions.yaml` | Mỗi sách một tệp `pipeline_<Sách>.yaml`; khoá theo sách khai trong `books[]`. |
| `scripts/` | **Công cụ chạy tay.** `measure/` (bộ đo tái lập), `maintenance/` (bảo trì kho), `_oneoff/` (script một lần: tải/dựng dữ liệu), `_retired/` (đã nghỉ hưu, GIỮ có chủ ý làm bằng chứng luận văn), `run_all_selftests.sh`, `check_*.sh`, `clean_build.sh` | Script một lần → `_oneoff/`; script đã loại bỏ nhưng còn là bằng chứng → `_retired/` kèm README giải thích. |
| `train_crop/` · `nom-embed/` (submodule) · `ArcFace/` | **Mô hình.** detector CenterNet v1/v2 · bộ mã hoá S3 (`best.pt`) · mã huấn luyện lại bộ mã hoá | Checkpoint không commit (`.gitignore`); mã huấn luyện thì commit. |
| `lab/` | **Thí nghiệm.** `i5_detector_v2/`, `gan_nhan_2026-09-13/`, `self_training_v2/`, `enhanced_self_training_v3/`, `ocr_compare/`, `tham_dinh_2026-09-16/`, `kaggle_diffusion/` (dời về 23/09), `configs/` | Một thí nghiệm = một thư mục có ngày hoặc tên riêng + `README.md`. Không có tệp `.py` rời ở gốc `lab/`. |
| `khoiB/` | Khối B (CNN âm tiết 5-fold). **GIỮ Ở GỐC** — là đầu vào SỐNG: `run_pipeline.sh:363` đọc `KhoiB/v3/crops_v3.npz` | Xem §4 (bẫy hoa/thường). |
| `font_diffusion/` · `gannhanocr-fd/` · `NomNaOCR/` | Nhánh sinh glyph tham chiếu + OCR ngoài. `font_diffusion`, `gannhanocr-fd`, `NomNaOCR/ds4v_repo` là **submodule** (`.gitmodules`) | Gỡ = `git submodule deinit`, KHÔNG `rm`. `font_diffusion/fonts/NomNaTong-Regular.ttf` là tệp SỐNG (6 config + `ground_truth/selftest.py`). |
| `evaluation/` | **Thư mục ĐẦU RA đang sống**, không phải vỏ rỗng: `pipeline/check_ocr_columns.py:71` mặc định `--out evaluation/_kinhhannom_debug/_col_check` và `run_pipeline.sh:285` gọi nó | Không xoá; muốn bỏ phải đổi mặc định trong mã (kèm hồi quy md5). |
| `docs/` | Tài liệu + `PIPELINE_FACTS.json` (sinh bằng `scripts/measure/code_facts.py`) | **Không chứa tệp `.py`** (quy tắc mới 23/09). |
| `data/` · `dict/` · `fonts/` | Dữ liệu nguồn, từ điển, phông | Bất khả xâm phạm với mã bảo trì. |
| `web/` | Trình xem kết quả (chạy tay) | Không nằm trong đồ thị 4 điểm vào; coi là điểm vào riêng của người dùng. |

## 2. Thư mục SINH RA KHI CHẠY (không commit)

`dataset/` · `dataset_out/` (trừ vài CSV giao nộp đang track) · `prepared/` · `measure_out/` ·
`logs/` · `archive/` · `lab/i5_detector_v2/bundle/` · `scripts/measure_wf_*/` · mọi `*.zip`, `*.pt`, `*.log`.
Tất cả đã nằm trong `.gitignore` (dòng 121, 132, 136 và các mẫu `*.zip`/`*.log`).
**Quy tắc:** tệp nào dựng lại được bằng một lệnh thì không giữ trên đĩa lâu dài và không commit.

## 3. Quy tắc đặt tên

- Mô-đun chạy tay: `<viec>_<doi_tuong>.py` (`ingest_lithograph_book.py`, `box_ref_eval.py`).
- Selftest: `<mô-đun>_selftest.py` cạnh mô-đun, hoặc `selftest.py` trong gói; phải được
  `scripts/run_all_selftests.sh` gọi thì mới tính là hàng rào.
- Thư mục thí nghiệm: `lab/<ten>_<YYYY-MM-DD>/` khi là mẻ một ngày.
- Tài liệu chốt: `docs/<CHU_DE>_<YYYY-MM-DD>.md`, viết HOA, có ngày.
- Mã bảo trì: `scripts/maintenance/`, chỉ phụ thuộc `.venv`, **không sửa** `pipeline/`, `core/`, `data/`.

## 4. Bẫy đã gặp — bắt buộc đọc trước khi dọn

1. **Hoa/thường (macOS).** Index git ghi `KhoiB/` và `Dict/`, đĩa ghi `khoiB/` và `dict/`.
   `git ls-files khoiB` trả **0** → dễ kết luận sai là "untracked, 387 MB rác". Sự thật:
   `git ls-files "KhoiB" | wc -l` = **68 tệp đang track**, và `run_pipeline.sh:363` dùng
   `KhoiB/v3/crops_v3.npz` làm đầu vào CHÍNH của bước 5 (`self_training_rescue`), chỉ rơi xuống
   `lab/gan_nhan_2026-09-13/crops.npz` khi tệp kia vắng — **hỏng im lặng**. ⇒ KHÔNG dời `khoiB/`.
2. **Grep thiếu phạm vi.** `pipeline/align_engine/to_standard.py` từng bị kết luận chết vì grep
   chỉ quét `*.py`/`*.sh`; nó là điểm vào ghi trong `README.md:177` và `pipeline/align_engine/README.md:16,36`.
3. **Chết theo đồ thị ≠ chết theo luận văn.** `pipeline/step3_label.py`, `step4_export.py`,
   `core/ranking/*` có dòng "KEPT (not deleted) for the thesis comparison" ngay trong mã và trong
   cả 8 `config/pipeline_*.yaml`. `scripts/_retired/README.md` cũng ghi rõ "GIỮ LẠI CÓ CHỦ Ý".
4. **`lab/i5_detector_v2/last.pt` (330 MB)**: `best.pt` KHÔNG còn trên đĩa và HF repo riêng tư trả 401
   ⇒ đây là nguồn duy nhất còn lại để dựng lại `train_crop/detector_r34_v2_litho.pt` (5 config ghim). GIỮ.

## 5. Hồi quy bắt buộc sau mọi thay đổi cấu trúc

```bash
bash -n run_pipeline.sh && ./run_pipeline.sh --book all --dry-run        # đủ 8 bộ
.venv/bin/python -m pipeline.align_engine.build_dataset \
  --config config/pipeline.yaml --qd01-cells none --force --pages pages3.csv --out <tmp>
# pages3.csv = book,page ∈ {stt2/page_0024, stt4/page_0050, stt11/page_0100}
# -> labels.csv md5 59e436d7641fa849bb6759868ac29259, 556 dòng
bash scripts/run_all_selftests.sh                                        # mốc 23/09: 1176 passed, 9 failed
.venv/bin/python scripts/measure/code_facts.py --check                   # 18/18
```
