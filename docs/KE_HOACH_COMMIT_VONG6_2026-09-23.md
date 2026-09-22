# Kế hoạch commit vòng 6 (23/09) — 3 bộ còn lại của `data/`: adapter IHR, 3 bộ đo mới, 2 config

Trên HEAD **12a00ede3e**, nhánh `main`. **Chưa commit gì** — tệp này là danh mục đích danh để người ký.
Nội dung vòng 6: chạy hai bộ mộc bản IHR-NomDB (LucVanTien1916, TruyenKieu1872) làm **TẬP ĐÁNH GIÁ** và đo
**độ đúng end-to-end bằng nhãn người**; đo bố cục bản chép tay TruyenKieuPhongTinhCoLuc (chưa chạy).
Báo cáo: `docs/CHAY_3_BO_CON_LAI_2026-09-23.md`; tổng hợp: `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` §10.

## 0. Trạng thái nghiệm thu lúc chốt

| Cổng | Kết quả |
|---|---|
| **Hồi quy STT 3 trang** (`build_dataset --config config/pipeline.yaml --qd01-cells none --force --pages pages3.csv`, stt2/0024 · stt4/0050 · stt11/0100) | `labels.csv` md5 **`59e436d7641fa849bb6759868ac29259`**, **556 dòng** — TRÙNG mốc vòng 5 |
| Vì sao chỉ cần md5 (không cần worktree đối chứng) | Vòng 6 **không sửa một dòng nào** trong `pipeline/align_engine/`, `pipeline/remediation/`, `core/`, `config/pipeline.yaml`. `git diff` chỉ có `run_pipeline.sh` = **18 dòng THÊM**, toàn bộ nằm trong nhánh `--book <sách mới>` (`ingest: ihr`, `--book all-ihr`); đường STT (`./run_pipeline.sh` không tham số) không đổi |
| selftest MỚI | `ingest_ihr` **64/64** · `mark_eval_dataset` **15/15** · `ihr_layout` **22/22** · `ptcl_layout` **15/15** · `ihr_endtoend` **24/24** |
| selftest CŨ (không hồi quy) | book_layout **115/115** · ingest_lithograph **82/82** · ingest_prose **33/33** · mechanism_gates **94/94** · pitch_decode **22/22** · verses_ref_fix **19/19** · phase1_engine **253/0** |
| `scripts/measure/code_facts.py` | **18/18** invariants |
| `measure.py --all --report-only` | **13 bước, 177 PASS / 0 FAIL / 2 mềm**, mã thoát 0 |
| `./run_pipeline.sh --dry-run` | in đúng 6 bước STT |
| `git status --short -- dataset_out prepared/SachThanhTruyen*` | **trống** |
| `git status --short -- data` | Không trống nhưng **KHÔNG do vòng 6**: cùng danh sách `D data/{CacThanhTruyen1646/SOURCE.md, IHR-NomDB_nlp/*, KimVanKieu1894/SOURCE.md, LyHangCaDao/*, TamTuKinhDienAm/*}` đã ghi ở `KE_HOACH_COMMIT_VONG5 §0` (có từ 20–21/09). Vòng 6 **chỉ đọc** `data/` |
| Ngân sách kim | LVT1916 ingest 94 + TK1872 ingest 161; đo bố cục A/B 40; đo C 6 → **301 / 450** |

## 1. Tệp cần commit

### 1.1 Mã mới (chưa tracked)

| Tệp | Dòng | Nội dung | Kiểm chứng |
|---|---|---|---|
| `pipeline/tools/ingest_ihr_book.py` | ~420 | Adapter ingest 2 bộ IHR-NomDB → `prepared_ihr/<book>/` đúng hợp đồng engine thạch bản. Ô cột từ `pages/bboxes.json` (VoTT), QN từ `pages/annotation.json` → `translation`; cổng cứng `số ô cột == ceil(số câu/2)`; ranh giới tầng theo từng cột (6/14 chiều cao) + cân lại theo luật 6/8; `--scale N` (mặc định 1, đã đo ×3 không hơn); `evaluation_only: true` trong manifest | `ingest_ihr_selftest` 64/64 |
| `pipeline/tools/ingest_ihr_selftest.py` | ~190 | 64 phép: hình học cột/tầng, gán hộp kim, cân tầng, cổng trang, ghép câu, **3 phép soi MÃ THẬT bằng AST** (không đọc `hn_text` / `nom_text`, `manifest.tsv` chỉ ở metadata), đọc dữ liệu thật 2 sách, ghi thật 1 trang ra thư mục tạm ở `--scale` 1 và 2 | chạy 6 s, không gọi API |
| `pipeline/tools/mark_eval_dataset.py` | ~150 | Đóng dấu `dataset/<Book>/` là TẬP ĐÁNH GIÁ: ghi `TAP_DANH_GIA.md` (3 điều cấm) + `evaluation_only.json` (cờ máy đọc được + `is_marked()` cho cổng CI). Không sửa `labels.csv`, không đọc `data/`. `run_pipeline.sh` gọi tự động khi `ingest: ihr` | `--selftest` **15/15** |
| `scripts/measure/ihr_layout.py` | ~440 | Đo bố cục A/B: cột/trang từ bbox và từ số câu, cột = cặp hay câu, px/chữ, bước cột, đối chứng chiếu mực 2 lượt; **thăm dò kim ×1 vs ×3 so NHÃN NGƯỜI**; cache kim theo md5 ảnh đã gửi | selftest 22/22; invariants 7 (2 xếp **mềm**, lý do trong `measure.py`) |
| `scripts/measure/ptcl_layout.py` | ~430 | Đo bố cục bản chép tay R.987: khung, bước cột (2 lượt, khoá cửa sổ quanh trung vị sách vì 35/120 tờ khoá nhầm hài), **ranh giới tầng**, bước chữ tầng dưới (trung vị khoảng cách tâm dải mực — tự tương quan hỏng trên chép tay), số chữ/cột; kim 6 tờ đối chứng; nền dị bản 1871↔1872 | selftest 15/15; invariants **7/7 PASS** |
| `scripts/measure/ihr_endtoend_eval.py` | ~340 | **Độ đúng end-to-end thật**: `labels_gated.csv` ↔ `data/<book>/manifest.tsv` theo (trang, cột, vị trí chữ); precision từng tầng + Wilson CI, coverage, phân loại lỗi (dị thể/gần hình/GT PUA/khác hẳn), **kim THÔ so GT**, so mốc patch 22/09 | selftest 24/24; invariants 7/7 PASS mỗi sách |
| `config/pipeline_LucVanTien1916.yaml` | ~130 | Config bộ đánh giá A. Đầu tệp ghi rõ **ĐÂY LÀ TẬP ĐÁNH GIÁ, KHÔNG PHẢI TẬP HUẤN LUYỆN** | `step0_setup` "Validation passed" |
| `config/pipeline_TruyenKieu1872.yaml` | ~130 | Config bộ đánh giá B, cùng khuôn | `step0_setup` "Validation passed" |

### 1.2 Tệp tracked đã sửa

| Tệp | Thay đổi | Rủi ro với đường STT / 3 sách giao nộp |
|---|---|---|
| `run_pipeline.sh` | **+18 dòng**: `EVAL_BOOKS_IHR`, `--book all-ihr`, nhánh `BK_INGEST == "ihr"` ở bước 0 (cổng bộ đo), bước 1 (lệnh ingest) và bước 5 (đóng dấu tập đánh giá), 2 dòng help/chú thích | **0** — mọi dòng thêm nằm trong nhánh `--book`; `./run_pipeline.sh` không tham số đi đúng nhánh cũ (`--dry-run` in đúng 6 bước) |
| `scripts/measure/measure.py` | Thêm `IHR_BOOKS`, `PTCL_BOOK` vào `ALL_BOOKS`; 3 bước mới `ihr_layout`, `ihr_endtoend`, `ptcl_layout` vào `ALL_STEPS` + `KEY_METRICS` + `plan_steps`; 2 cờ `--ihr-kim-pages`, `--ptcl-kim-pages`; 2 mục `SOFT_INVARIANTS` (kèm lý do đo được) | **0** cho các bước cũ (chỉ thêm nhánh `elif`); `measure.py --all --dry-run` in 12 bước, 6 bước cũ nguyên văn |
| `scripts/measure/README.md` | 3 dòng bảng mô-đun + cập nhật danh sách `--steps` | tài liệu |
| `.gitignore` | thêm `prepared_ihr/` (206 MB: ảnh trang + cache kim + `dataset_out` trung gian của 2 bộ đánh giá) | **0** |
| `docs/PIPELINE_FACTS.json` | sinh lại bởi `code_facts.py` trên HEAD 12a00ede3e (số dòng ghim trôi theo commit vòng 5; `git_dirty_pipeline_core` 10 → 3 = 3 tệp mới trong `pipeline/tools/`) | **0** (tệp sinh ra) |
| `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` | §1 sửa "13 thư mục" → 9 bộ còn trên đĩa + trạng thái nhóm 2/3; **§10 mới** (bảng đầy đủ 9 bộ, bài học mức-ảnh-gửi-kim, vì sao chưa chạy C); §9 thêm 2 dòng chỉ mục | tài liệu |

### 1.3 Tài liệu mới

| Tệp | Nội dung |
|---|---|
| `docs/CHAY_3_BO_CON_LAI_2026-09-23.md` | Báo cáo chính vòng 6: §1 đo bố cục A/B (+ hệ số phóng kim), §2 adapter + config + kết quả chạy, §3 đo bố cục C và 4 lý do chưa chạy + đường chạy đã định lượng, §4 **độ đúng end-to-end thật** + 3 giới hạn phải đọc kèm, §5 bảng rủi ro R1–R7 + khuyến nghị, §6 hồi quy |
| `docs/KE_HOACH_COMMIT_VONG6_2026-09-23.md` | tệp này |

### 1.4 KHÔNG commit (sinh ra khi chạy, đã/nên nằm trong `.gitignore`)

`prepared_ihr/` (ảnh + cache kim 2 sách), `dataset/LucVanTien1916/`, `dataset/TruyenKieu1872/`,
`measure_out/**` (gồm `ihr_layout/kim_cache/`, `ptcl_layout/kim_cache/`), `logs/run_*.log`.
**Kiểm trước khi commit**: `git status --short | grep -E "prepared_ihr|measure_out|dataset/"` phải trống
(nếu không, thêm vào `.gitignore` chứ đừng `git add -A`).

## 2. Thứ tự commit đề nghị

| # | Commit | Tệp | Vì sao tách |
|---|---|---|---|
| 1 | `feat(measure): bộ đo bố cục 3 bộ còn lại của data/ (ihr_layout, ptcl_layout) + gắn vào measure.py` | `scripts/measure/{ihr_layout,ptcl_layout}.py`, `scripts/measure/measure.py`, `scripts/measure/README.md` | Đo trước, code sau — commit này một mình đã tái lập được mọi số ở §1/§3 của báo cáo, kể cả khi từ chối adapter |
| 2 | `feat(ingest): adapter ingest 2 bộ IHR-NomDB làm TẬP ĐÁNH GIÁ (không đọc nhãn người)` | `pipeline/tools/{ingest_ihr_book,ingest_ihr_selftest,mark_eval_dataset}.py`, `config/pipeline_{LucVanTien1916,TruyenKieu1872}.yaml`, `run_pipeline.sh` | Khối mã chạy được, tự kiểm bằng selftest; hồi quy STT nằm ở đây |
| 3 | `feat(measure): ihr_endtoend_eval — độ đúng end-to-end so nhãn người` | `scripts/measure/ihr_endtoend_eval.py` (+ đăng ký bước trong `measure.py` nếu tách khỏi commit 1) | Phép đo, không phải pipeline; tách để review riêng phần "có vòng tròn không" |
| 4 | `docs: ba bộ còn lại của data/ — số liệu, rủi ro, khuyến nghị (vòng 6)` | `docs/CHAY_3_BO_CON_LAI_2026-09-23.md`, `docs/KE_HOACH_COMMIT_VONG6_2026-09-23.md`, `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` | Tài liệu |

## 3. Điểm người ký phải quyết

1. **R1 — rò rỉ tập đánh giá.** Hai bộ mới có nhãn người. Vòng 6 chặn ở **3 lớp**: `evaluation_only: true` trong
   `prepared_ihr/<Book>/manifest.json`; **đóng dấu bằng máy** vào `dataset/<Book>/` (`TAP_DANH_GIA.md` +
   `evaluation_only.json`, hàm `is_marked()`) do `run_pipeline.sh` gọi tự động; và hai sách nằm ngoài
   `--book all-new`. **Còn thiếu mắt xích cuối**: `pipeline/publish/` chưa gọi `is_marked()` để từ chối bộ đã
   đóng dấu khi dựng split huấn luyện — đề nghị làm ở vòng 7 (≈20 dòng + 1 test).
2. **Cách đọc con số 97–98 %.** Đây là độ đúng khi **QN là phiên âm của người** và **ô cột do người vẽ** ⇒ là
   **cận trên của phương pháp**, không phải độ đúng dự kiến của 3 sách giao nộp. Mọi trích dẫn phải kèm điều kiện này
   (`CHAY_3_BO_CON_LAI §4.4`). Nếu người ký thấy nguy cơ bị đọc nhầm cao hơn giá trị, có thể yêu cầu đổi nhãn chỉ số
   thành "precision của luật gán nhãn khi đầu vào QN đúng".
3. **Gỡ mốc cũ.** "kim thô đúng 42–49 % trên mộc bản" (`CHAN_DOAN_3_SACH_MOI §3.2` cuối, `BAO_CAO_TONG_HOP §4` bảng (1))
   là **chỉ số của cách đo bằng patch từng câu**, không phải của kênh OCR. Đề nghị sửa tại chỗ hai tài liệu ấy —
   **vòng 6 chỉ ghi chú, chưa sửa** để không đụng số đã ký.
4. **C chưa chạy.** Quyết định: (a) làm tiếp theo `CHAY_3_BO_CON_LAI §3.4` ở vòng 7 (~1 ngày công, ~130 lượt kim),
   hay (b) để ngoài luận văn và ghi lý do đo được. Vòng 6 **không** chạy C vì adapter hiện có sẽ gán âm QN của câu lục
   cho **lời bình chữ Hán** ở tầng trên — sai hệ thống 100 % ở tầng ấy.
5. **`det_thr = 0.15` cho 2 bộ mới là mượn từ thạch bản**, chưa quét riêng. Lập luận (detector letterbox về 1024 px nên
   chữ ~50 px ở cả hai loại sách) có trong config; nếu muốn chắc thì quét 0,3/0,2/0,15/0,1 trên 8 trang trước khi ký.

## 4. Câu lệnh kiểm lại toàn bộ trước khi ký

```bash
PY=.venv/bin/python
$PY -m pipeline.tools.ingest_ihr_selftest            # 64/64
$PY -m pipeline.tools.mark_eval_dataset --selftest   # 15/15
$PY scripts/measure/ihr_layout.py --selftest         # 22/22
$PY scripts/measure/ptcl_layout.py --selftest        # 15/15
$PY scripts/measure/ihr_endtoend_eval.py --selftest  # 24/24
$PY pipeline/align_engine/book_layout_selftest.py    # 115/115
$PY -m pipeline.phase1_engine_selftest               # 253/0
$PY scripts/measure/code_facts.py --summary measure_out/code_facts/summary.json   # 18/18
printf 'book,page\\nstt2,page_0024\\nstt4,page_0050\\nstt11,page_0100\\n' > /tmp/pages3.csv
$PY -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --qd01-cells none \\
    --force --pages /tmp/pages3.csv --out /tmp/stt_regress
md5 -q /tmp/stt_regress/labels.csv                   # 59e436d7641fa849bb6759868ac29259
./run_pipeline.sh --dry-run                          # 6 bước STT
git status --short -- dataset_out prepared/SachThanhTruyen*   # trống
```
