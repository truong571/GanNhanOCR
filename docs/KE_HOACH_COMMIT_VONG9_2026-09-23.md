# Kế hoạch commit vòng 9 — detector v2 theo sách + gộp `prepared/` (23/09/2026)

Nền: HEAD `1c44d200a9`. **Vòng 9 chưa commit gì.** Tài liệu số liệu:
`docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md`.

> 🔴 **Ba thứ TUYỆT ĐỐI không đưa vào commit này**
> 1. `web/` (6 tệp `M` có từ trước vòng 7, không liên quan) — commit riêng.
> 2. `nom-embed` (submodule `m`) — không đụng.
> 3. **Mọi `*.pt`**, đặc biệt `lab/i5_detector_v2/last.pt` (330 MB) và
>    `train_crop/detector_r34_v2_litho.pt` (86 MB). `.gitignore:37` đã có `*.pt`; kiểm bằng
>    `git check-ignore -v lab/i5_detector_v2/last.pt train_crop/detector_r34_v2_litho.pt`
>    (phải in ra `.gitignore:37:*.pt`). Ckpt lấy lại theo `docs/VONG9…md` §0 bước (1).

---

## 1. Danh mục tệp ĐÍCH DANH (18 tệp `M` + 1 docs `M` + 2 docs mới)

### 1.1 Áp detector v2 theo sách — 4 config (`+12 / −7`)

| Tệp | Δ | Nội dung |
|---|---|---|
| `config/pipeline_LucVanTien1883.yaml` | +2 −1 | `books[].detector_ckpt: train_crop/detector_r34_v2_litho.pt`; `detector_resize: linear → area` |
| `config/pipeline_KimVanKieu1884.yaml` | +4 −2 | như trên (config trỏ `run_config`, khai để 2 đường thống nhất) + sửa chú thích `prepared_b1` |
| `config/pipeline_KimVanKieu1884_b1.yaml` | +5 −4 | như trên + `paths.data_dir`/`run.dataset_out` (§1.2) |
| `config/pipeline_TruyenKieu1872.yaml` | +4 −3 | như trên + `paths.data_dir`/`run.dataset_out` (§1.2) |

**KHÔNG khai v2** (giữ nguyên `detector_resize: linear`, không có `detector_ckpt`):
`config/pipeline_Chrestomathie1872.yaml` (không đổi, **không** nằm trong commit),
`config/pipeline_LucVanTien1916.yaml` (chỉ đổi đường dẫn), `config/pipeline.yaml` (**STT — không đụng**).

### 1.2 Gộp `prepared/` — 10 tệp (`+31 / −31`, thuần đổi đường dẫn)

| Tệp | Δ | Nội dung |
|---|---|---|
| `config/pipeline_KimVanKieu1884_b1.yaml` | *(đã tính ở §1.1)* | `data_dir: prepared_b1 → prepared`; `run.dataset_out: prepared_b1/KimVanKieu1884/dataset_out_b1 → prepared/KimVanKieu1884/dataset_out` |
| `config/pipeline_LucVanTien1916.yaml` | +2 −2 | `data_dir: prepared_ihr → prepared`; `run.dataset_out → prepared/LucVanTien1916/dataset_out` |
| `config/pipeline_TruyenKieu1872.yaml` | *(đã tính ở §1.1)* | như trên |
| `config/pipeline_KimVanKieu1884_b1_boost.yaml` | +2 −0 | 2 dòng đầu: thư mục `prepared_b1_boost/` đã xoá, config giữ làm **hồ sơ ablation** |
| `pipeline/tools/ingest_ihr_book.py` | +3 −3 | `--out` mặc định `prepared_ihr → prepared` + 2 dòng docstring |
| `pipeline/tools/ingest_ihr_selftest.py` | +1 −1 | docstring |
| `scripts/measure/align_audit.py` | +6 −6 | `BOOKS[KimVanKieu1884/LucVanTien1916/TruyenKieu1872]` → `prepared/<Book>` |
| `scripts/measure/ihr_endtoend_eval.py` | +7 −7 | `BOOKS` + docstring |
| `scripts/measure/measure.py` | +1 −1 | cổng bước `ihr_endtoend` |
| `scripts/measure/loss_ledger.py` | +3 −3 | 3 đường `prepared_b1…dataset_out_b1` |
| `scripts/measure/kim_channel_probe.py` | +1 −1 | đường dự phòng dataset KVK |
| `scripts/measure/build_metrics.py` | +1 −1 | ví dụ trong docstring |
| `scripts/measure/auto_precision.py` | +1 −1 | chữ trợ giúp `--trans` |
| `.gitignore` | +3 −6 | bỏ `prepared_ihr/` và `/prepared_b1*/`, thay bằng 3 dòng ghi chú "vòng 9 gộp vào `prepared/`" |

### 1.3 Bộ đo — 1 tệp (`+48 / −11`)

| Tệp | Δ | Nội dung |
|---|---|---|
| `scripts/measure/box_ref_eval.py` | +48 −11 | `PREPARED_BOOKS`/`ALL_BOOKS` + `--book all5`; `book_pages()`; `load_page()` nhận trang `prepared/<Book>/pages/*.png`; nhánh **prose** trong `ref_cells_for_page` (1 tầng/cột, `N = num_syllables`); `col_pitch` lấy từ `detected/*_ocr_cache.json`; 3 sách mới khoá biến thể `prepared` |

### 1.4 Sinh tự động — 1 tệp

| Tệp | Δ | Ghi chú |
|---|---|---|
| `docs/PIPELINE_FACTS.json` | +18 −66 | sinh lại bằng `.venv/bin/python scripts/measure/code_facts.py` (HEAD `1c44d200a9`, `invariants_pass 18/18`). **Không sửa tay.** |

### 1.5 Tài liệu MỚI — 2 tệp

| Tệp | Ghi chú |
|---|---|
| `docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md` | bảng quyết định v2 (4 cấu hình × 5 sách), bảng trước/sau bộ giao nộp, precision nhãn người, bảng gộp `prepared/` |
| `docs/KE_HOACH_COMMIT_VONG9_2026-09-23.md` | tệp này |

### 1.6 Tài liệu SỬA — 1 tệp

| Tệp | Δ | Nội dung |
|---|---|---|
| `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` | +12 −0 | hộp đỏ đầu tệp: mọi đường `prepared_b1`/`prepared_ihr` bên dưới đã lỗi thời + bảng sách nào dùng v2 |

---

## 2. KHÔNG commit (sinh lại được / ngoài phạm vi)

| Mục | Lý do |
|---|---|
| `prepared/` (2,7 GB) | `.gitignore:13`. Dựng lại: `./run_pipeline.sh --book all-new` + `--book all-ihr`, **0 API** (cache `prepared/<Book>/kim_raw/`) |
| `dataset/`, `dataset/<Book>_v8/` | `.gitignore:17`. `_v8` = bản vòng 8, giữ trên đĩa để đối chiếu |
| `measure_out/` (`box_ref_v9_*`, `align_audit`, `crop_source`…) | quy ước CLAUDE.md: không commit |
| `train_crop/detector_r34_v2_litho.pt`, `lab/i5_detector_v2/last.pt` | `*.pt` — xem hộp đỏ đầu tệp |
| `web/` (6 tệp), `nom-embed` | ngoài phạm vi vòng 9 |
| `logs/run_*.log` | sinh mỗi lần chạy |

---

## 3. Lệnh commit đề nghị (KHÔNG chạy tự động — người dùng quyết định)

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
git checkout -b vong9-detector-v2-prepared-gop        # HEAD đang ở main

git add config/pipeline_LucVanTien1883.yaml config/pipeline_KimVanKieu1884.yaml \
        config/pipeline_KimVanKieu1884_b1.yaml config/pipeline_KimVanKieu1884_b1_boost.yaml \
        config/pipeline_LucVanTien1916.yaml config/pipeline_TruyenKieu1872.yaml \
        pipeline/tools/ingest_ihr_book.py pipeline/tools/ingest_ihr_selftest.py \
        scripts/measure/align_audit.py scripts/measure/auto_precision.py \
        scripts/measure/box_ref_eval.py scripts/measure/build_metrics.py \
        scripts/measure/ihr_endtoend_eval.py scripts/measure/kim_channel_probe.py \
        scripts/measure/loss_ledger.py scripts/measure/measure.py \
        .gitignore docs/PIPELINE_FACTS.json \
        docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md docs/KE_HOACH_COMMIT_VONG9_2026-09-23.md \
        docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md

git status --short --cached | grep -E "web/|nom-embed|\.pt$|^A  prepared/|^A  dataset/"   # PHẢI TRỐNG
git commit
```

Thông điệp commit đề nghị:

```
feat(vòng 9): detector v2 theo TỪNG SÁCH (3/5 sách, có kiểm chứng) + gộp mọi prepared_* vào prepared/

Áp `books[].detector_ckpt = train_crop/detector_r34_v2_litho.pt` + `detector_resize: area`
CHỈ cho LucVanTien1883, KimVanKieu1884, TruyenKieu1872 — 3 sách mà box_ref_eval trên ẢNH
PIPELINE THẬT (27 trang/sách, thr 0,15) cho thấy v2 không giảm chỉ số nào:
  cắt vào thân chữ 9,2->0,2 % · 2,5->0,1 % · 63,8->5,9 %
  IoU trung vị    0,75->0,87 · 0,74->0,87 · 0,63->0,82
  I5 thô (cột n_det==N) 70,7->96,3 % · 62,3->97,0 % · 11,9->15,5 %
GIỮ v1 cho Chrestomathie1872 (I5 67,6->50,6, ok50 -5,2) và LucVanTien1916 (I5 21,5->1,5;
390 hộp ngoài tầng ở 85,9 % số cột). STT không đụng: config/pipeline.yaml không khai khoá
này, labels.csv 3 trang giữ md5 59e436d7641fa849bb6759868ac29259 (556 dòng, 42 cột).

Bộ giao nộp: ảnh export 11.780->11.936, 19.632->20.545, 13.482->19.605 (5 bộ: 60.469->67.661).
crop_quality bleed 2.223->2, 2.179->16, 3.623->112; GOLD_text_only 143->2, 720->1, 6.670->135.
Precision nhãn NGƯỜI TruyenKieu1872 98,64 % (n 12.706) -> 98,61 % (n 18.550, CI 98,44-98,77);
LucVanTien1916 97,96 % không đổi. M==N và trôi (7/16/91/32) không đổi — detector chỉ đổi HỘP.

Gộp prepared_b1/ + prepared_ihr/ vào prepared/<Book>/ (bỏ hậu tố _b1/_ihr), xoá ablation
prepared_b1_080/ + prepared_b1_boost/; cache kim tái dùng 100 % (verify_cache_image ok
105/163/65/99/161) nên 0 lượt gọi API. box_ref_eval mở rộng cho cả 5 sách (--book all5,
nhánh prose 1 tầng/cột).

selftest: book_layout 157/0, phase1 288/0, ingest_ihr 81/81, mechanism_gates 113/0,
ingest_lithograph 82/82, ingest_prose 33, pitch_decode 22/0, verses_ref_fix 19/19,
align_audit OK, ihr_endtoend 24/24, code_facts 18/18.
Chi tiết + chỗ đi ngược nghĩa đen luật nghiệm thu (TK1872, -0,03 điểm precision đổi lấy
+6.120 ô GOLD có ảnh): docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md §4.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

---

## 4. Cổng kiểm TRƯỚC khi commit

```bash
PY=.venv/bin/python
git check-ignore -v lab/i5_detector_v2/last.pt train_crop/detector_r34_v2_litho.pt   # PHẢI in .gitignore:37
git status --short -- dataset_out prepared/SachThanhTruyen\*                          # PHẢI TRỐNG
$PY -m pipeline.align_engine.book_layout_selftest        # 157 / 0
$PY -m pipeline.phase1_engine_selftest                   # 288 / 0
$PY -m pipeline.tools.ingest_ihr_selftest                # 81 / 81
$PY -m pipeline.remediation.mechanism_gates_selftest     # 113 / 0
$PY -m pipeline.tools.ingest_lithograph_selftest         # 82 / 82
$PY -m pipeline.tools.ingest_prose_selftest              # 33
$PY -m pipeline.align_engine.char_detector.pitch_decode --selftest   # 22 / 0
$PY scripts/measure/verses_ref_fix.py --selftest         # 19 / 19
$PY scripts/measure/align_audit.py --selftest            # SELFTEST OK
$PY scripts/measure/ihr_endtoend_eval.py --selftest      # 24 / 24
$PY scripts/measure/code_facts.py --check                # 18 / 18
# hồi quy STT (bắt buộc, ~2 phút)
printf 'book,page\nstt2,page_0024\nstt4,page_0050\nstt11,page_0100\n' > /tmp/pages3.csv
$PY -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --qd01-cells none \
    --force --pages /tmp/pages3.csv --out /tmp/stt_v9 && md5 /tmp/stt_v9/labels.csv
#   -> 59e436d7641fa849bb6759868ac29259
```

**Tất cả 12 cổng trên đã chạy và ĐẠT ngày 23/09/2026** (`docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md` §6).

---

## 5. Hoàn nguyên

| Muốn gì | Lệnh |
|---|---|
| Bỏ v2 ở **một** sách | `$PY lab/i5_detector_v2/set_book_keys.py config/pipeline_<X>.yaml --book <X> --remove detector_ckpt --set detector_resize=linear` rồi `./run_pipeline.sh --book <X>` |
| Bỏ v2 ở **cả ba** sách | lặp lệnh trên cho `LucVanTien1883`, `KimVanKieu1884` (cả `_b1`), `TruyenKieu1872` |
| Lấy lại bộ giao nộp vòng 8 | `dataset/<Book>_v8/` còn nguyên trên đĩa (5 sách) |
| Tách lại `prepared_*` | **không nên** — nhưng ingest lại là đủ: `ingest_lithograph_book --out prepared_b1 …`, `ingest_ihr_book --out prepared_ihr …` (0 API, cache `kim_raw/` dùng chung) |
