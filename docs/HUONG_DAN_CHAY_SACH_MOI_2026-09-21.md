# Hướng dẫn chạy sách mới (LVT1883 / KVK1884 thạch bản; Chrestomathie1872 văn xuôi) — 2026-09-21, cập nhật vòng 2 22/09

> 🔴 **CẬP NHẬT VÒNG 9 (23/09/2026) — hai thay đổi làm SAI mọi đường dẫn viết bên dưới**
> 1. **Chỉ còn `prepared/`.** `prepared_b1/`, `prepared_ihr/` đã dời vào `prepared/<Book>/`;
>    `prepared_b1_080/` + `prepared_b1_boost/` đã xoá. Mọi `data_dir` / `dataset_out` trong 6 config
>    nay là `prepared/<Book>/dataset_out` (bỏ hậu tố `_b1` / `_ihr`).
> 2. **Detector v2 khai THEO SÁCH**: `LucVanTien1883`, `KimVanKieu1884`, `TruyenKieu1872` dùng
>    `detector_ckpt: train_crop/detector_r34_v2_litho.pt` + `detector_resize: area`;
>    `Chrestomathie1872` và `LucVanTien1916` **giữ v1**; STT không đụng.
>
> Số liệu + lý do từng sách: `docs/VONG9_DETECTOR_V2_APDUNG_2026-09-23.md`.
> Kế hoạch commit: `docs/KE_HOACH_COMMIT_VONG9_2026-09-23.md`.


Tổng hợp NV1–NV5 (21/09), NV-B/NV-D (22/09 sáng) và **vòng 2** 22/09: (A) cổng cơ chế B4' + tầng `GOLD_text_only`; (B) B1' KVK
(QN đầu vào = phiên âm 1871, `verses_ref_fix.py`); (C) `layout: prose` + adapter văn xuôi (Chrestomathie1872). Chi tiết từng lần chạy:
`docs/CHAY_LVT1883_2026-09-21.md` §6, `docs/CHAY_KVK1884_2026-09-21.md` (không B1'), `docs/CHAY_KVK1884_B1_2026-09-22.md` (**chính thức**),
`docs/CHAY_CHRESTO1872_2026-09-22.md`; hợp đồng dữ liệu/cổng: `docs/PIPELINE_SACH_MOI_2026-09-20.md` §2–§7; tổng thể:
`docs/BAO_CAO_TONG_THE_SACH_MOI_2026-09-22.md` §9; kế hoạch commit vòng 2: `docs/KE_HOACH_COMMIT_VONG2_2026-09-22.md`. **Chưa commit.**

> **Vòng 3 (22/09 chiều) — chốt cuối = `box_decoder: pitch` + cổng (a')** trong cả 3 config sách mới (`pipeline/align_engine/char_detector/pitch_decode.py`,
> `mechanism_gates.py` luật (a'): hạ `GOLD_text_only` theo Ô `box_source ∈ {ink_cut, detector_low}`, cột `n_det ≠ n_qn` chỉ ghi cờ `n_det_mismatch`). Số chốt mới
> (GOLD ảnh LVT 8.650 / KVK 13.908 / Chresto 5.359; ảnh export 11.013 / 17.752 / 6.645), so sánh với legacy, lý do và giới hạn: **`docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md`**
> (báo cáo gộp duy nhất, §3/§6). Bản legacy vòng 2 (số trong §1/§3 dưới đây) giữ ở `dataset_<BOOK>_v3_legacy/`, `dataset_out_<BOOK>[_b1]_v3_legacy/`.
> Kế hoạch commit vòng 3: `docs/KE_HOACH_COMMIT_VONG3_2026-09-22.md`. Selftest mechanism_gates nay **94/94**; pitch_decode 22/22.
>
> **Vòng 4 (22/09 tối, chưa commit)**: khoá theo sách `detector_ckpt` / `detector_resize` (linear|area) + `lab/i5_detector_v2/` (gói Kaggle v2). Thử INTER_AREA trên 3 sách (`_area`): I5 thô 77,7 / 77,9 / 69,3 % nhưng LVT bleed +3,7 điểm → **chốt vẫn pitch-linear** (§6.1, BAO_CAO_TONG_HOP §3.3). STT md5 59e436d7… (worktree 8c08591e9a); book_layout 100/100. Kế hoạch commit: `docs/KE_HOACH_COMMIT_VONG4_2026-09-22.md`.

## 1. Trạng thái — cái gì đã chạy được

| Thành phần | Trạng thái | Bằng chứng |
|---|---|---|
| API kim (`core/ocr/ocr_api.py`) | token qua SN_OCR_USERNAME/PASSWORD; Guest Mode đã commit 65f7ca9 (chưa từng kích hoạt) | 10/10 HTTP 200; STT2 page_0024 206/206 chữ trùng cache, bbox lệch 0 |
| kim trên thạch bản nền xám 128 | KHÔNG mù: raw và stretch cùng 22 hộp/143 chữ | LVT trang 10: chars/cột [14,14,14,16,14,14,14,14,15,14] |
| Engine `layout: lithograph` / **`prose`** (`pipeline/align_engine/book_layout.py`) | mọi tham số có mặc định = STT; khoá theo sách `n_columns` (prose: `auto` = số dòng QN của trang), `det_xmargin`/`det_thr`; prose: cổng `prose_gate` (số cột Nôm == số dòng QN) | book_layout_selftest **79/79**; phase1_engine 253/0; hồi quy STT 3 trang byte-identical (§3) |
| Adapter thạch bản `pipeline/tools/ingest_lithograph_book.py` | `--verse-map formula\|anchor\|content`, `--plan-only`, `--contrast otsu`, **`--verses PATH`** (B1'), `--dict-boost` (KHÔNG dùng — tự khẳng định) | ingest_lithograph_selftest **61/61** (35 cũ + 26 content/verses_b1/dict_boost) |
| **B1' `scripts/measure/verses_ref_fix.py`** | thay dòng QN OCR bằng câu phiên âm dị bản (1871 LVĐ) khi exact / khớp mờ ≥ 0,9 cùng parity, offset ≤ 3 → `verses_b1.tsv` + `matches.csv` | `--selftest` 19/19; KVK: exact 465 · fuzzy 1.548 · giữ ocr 1.238 / 3.251 dòng |
| **Adapter văn xuôi `pipeline/tools/ingest_prose_book.py`** | ô cột từ `chresto_map.analyze_nom`, kim 1 hộp/cột, DP đơn điệu chuỗi chữ kim cả truyện ↔ âm tiết (`bang_truyen_trang.csv`), cột 1 = phải nhất | ingest_prose_selftest **33/33**; Chresto 65/65 trang, 404/417 cột ghép |
| Hồi quy STT (chạy lại 22/09 sau vòng 2, worktree HEAD 853cadfd9c) | 3 trang stt2/0024, stt4/0050, stt11/0100 | labels.csv md5 **59e436d7641fa849bb6759868ac29259** (556 dòng) cả hai bên; 344/346 tệp giống byte (2 tệp khác đường dẫn REPO); không `layout_gate` |
| Cổng cơ chế B4' `pipeline/remediation/mechanism_gates.py` + tầng `GOLD_text_only` (`export_final_dataset.py`/`make_dataset_docs.py`) | tự bật khi `books[].layout == lithograph`; prose PHẢI khai `books[].mechanism_gates: true` (config Chresto đã khai); STT tắt = sao byte | selftest 62/62; STT export HEAD vs mới `diff -rq` rỗng (71.592 tệp) |
| Bộ mẫu người kiểm mù `kiem_nguoi_grid.py`/`kiem_nguoi_score.py` | 300 GOLD + 100 SYLLABLE/sách (bản v1) | chưa có selftest, chưa review, không có người kiểm → để ngoài commit |

### Số liệu chốt vòng 2 (labels_final → B4' → export; chi tiết §3)
- **LVT1883** (105 trang, `formula`, 0,05/0,15; `dataset_out_LucVanTien1883/` → `dataset_LucVanTien1883/`): 14.476 ô; labels_final GOLD 9.666 / SYL 1.683 / REVIEW 3.111 / QUAR 16;
  sau B4' **GOLD ảnh 5.793 / GOLD_text_only 2.985** / SYL 2.363 / REVIEW 3.319 → export **8.156 ảnh** (11.141 dòng). Bản trước cổng: `dataset_LucVanTien1883_v1/` (11.349 ảnh).
- **KVK1884 — chính thức B1'** (163 trang, `content --contrast otsu`, QN = `verses_b1.tsv`, 0,05/0,15; `prepared_b1/` → `dataset_out_KimVanKieu1884_b1/` → `dataset_KimVanKieu1884/`):
  22.704 ô; labels_final GOLD **15.938** (+882 so với không B1') / SYL 3.209 / REVIEW 3.465 / QUAR 92; sau B4' **GOLD ảnh 8.593 / GOLD_text_only 5.784** / SYL 3.844 / REVIEW 4.391
  → export **12.437 ảnh** (18.221 dòng). Bản không B1' + cổng: `dataset_KimVanKieu1884_v2_gates_noB1/` (8.124 + 5.451; 12.344 ảnh); bản v1 không cổng: `dataset_KimVanKieu1884_v1/` (18.560 ảnh).
- **Chrestomathie1872** (65 trang, prose `n_columns: auto`, 0,05/0,15; `dataset_out_Chrestomathie1872/` → `dataset_Chrestomathie1872/`): 8.303 ô; page_ok 65/65; labels_final GOLD 5.892 / SYL 998 /
  REVIEW 1.413 / QUAR 0; sau B4' **GOLD ảnh 3.922 / GOLD_text_only 1.557** / SYL 1.287 / REVIEW 1.537 → export **5.209 ảnh** (6.766 dòng); không có dị bản → không cổng (d).

## 2. Lệnh chạy từng bước cho một sách `<BOOK>`

### 2.0 Đường tắt `run_pipeline.sh --book …` (22/09 chiều — gói trọn B0→B6 dưới đây, 1 lệnh)

```bash
./run_pipeline.sh --book LucVanTien1883          # thạch bản, formula; kim_raw/ có sẵn -> 0 gọi API; ~6 phút
./run_pipeline.sh --book KimVanKieu1884          # config chính có `run_config:` -> pipeline_KimVanKieu1884_b1.yaml (B1' chính thức)
./run_pipeline.sh --book Chrestomathie1872       # văn xuôi -> ingest_prose_book
./run_pipeline.sh --book all-new                 # cả 3 sách, lần lượt
./run_pipeline.sh --book LucVanTien1883 --dry-run            # chỉ in đủ lệnh B0→B6, không chạy, không ghi gì
./run_pipeline.sh --book KimVanKieu1884 --suffix _rp         # ra dataset_out_KimVanKieu1884_b1_rp + dataset_KimVanKieu1884_rp (so với bản chốt, không ghi đè)
# cờ khác: --skip-ingest (dùng prepared*/<BOOK> sẵn có) · --no-api (ingest --ocr none; content -> formula, KHÁC bản chốt)
#          --no-auto-precision (bỏ auto_precision -> mechanism_gates KHÔNG --cross, cổng (d) tắt, không B6)
./run_pipeline.sh            # KHÔNG tham số = đường STT cũ, không đổi; ./run_pipeline.sh --dry-run = in chuỗi 6 bước STT
```
Chuỗi thật (mỗi lệnh + toàn bộ output ghi `logs/run_<BOOK>_<thời điểm>.log`; sha256 + thời gian vào `dataset_out_<BOOK>/CHECKSUMS.txt`):
0 `step0_setup` → (`scripts/measure/measure.py --book <BOOK> --steps layout,qn_ocr | chresto_map` **chỉ khi thiếu** `measure_out/<BOOK>`) → (B1' `verses_ref_fix` nếu config khai)
→ 1 ingest (`--ocr kim`) → 2 build (`--use-s3 --two-pass --box-rule syl_index`, `--out dataset_out_<BOOK>`) + `enrich_crop_quality`
→ 3 `remediation census` / `apply --tau 0.62 --out dataset_out_<BOOK>` / `confusion_fix` → 4 `auto_precision --steps cross,gates` trên `labels_final`
(`dataset_out_<BOOK>/auto_precision/`, chỉ sách trong `CROSS_BOOKS`) → `mechanism_gates [--cross …/cross/<BOOK>/cells.csv]` → 5 `export_final_dataset --n-columns N`
+ `make_dataset_docs` + `make_xlsx` → 6 `auto_precision --steps cross` trên `labels_gated` (`dataset_out_<BOOK>/auto_precision_gated/` = B6).
Hồ sơ mỗi sách đọc từ `config/pipeline_<BOOK>.yaml`: khoá `run_config:` (chuyển sang config chính thức) và khối `run:` {`ingest` lithograph|prose,
`ingest_args` chuỗi cờ adapter, `verses_ref_fix{ref,ref_name,fuzzy_min}`, `dataset_out`, `n_columns`, `cross`, `measure_steps`} — chỉ `run_pipeline.sh` đọc,
engine/adapter/step0/mechanism_gates không đọc; vắng khối → mặc định theo `books[].layout`. Sách thứ tư: thêm config có `run:` (§7) là chạy được.
Kiểm tái lập 22/09 (`--suffix _rp` rồi so với bản chốt): xem §2.1 cuối mục này. Lệnh chi tiết từng bước (để chạy tay/khảo sát) giữ nguyên dưới đây.

Tiền đề: `data/<BOOK>/pages/`, `measure_out/<BOOK>/layout/layout_pages.csv` (+ `qn_ocr/verses.tsv` thạch bản; văn xuôi: `chresto/{qn_lines,qn_stories,bang_truyen_trang}.csv`)
sinh bởi `scripts/measure/measure.py`; `.env` có SN_OCR_USERNAME/PASSWORD.

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
PY=.venv/bin/python
# B0. Config riêng (copy config/pipeline_LucVanTien1883.yaml; KHÔNG khai pdf). books.<BOOK>:
#     thạch bản: layout: lithograph, n_columns: 10, qn_syllables_per_column: 14 | văn xuôi: layout: prose, n_columns: auto, (không khai qn_syllables_per_column), mechanism_gates: true
#     det_xmargin: 0.05 (mặc định), det_thr: 0.15 (ĐO THEO SÁCH — quét 0,3/0,2/0,15/0,1, CHAY_KVK1884 §3); paths.data_dir/output_dir riêng.
$PY -m pipeline.step0_setup config/pipeline_<BOOK>.yaml            # "Validation passed"; BookLayout(lithograph,10,14,…) | BookLayout(prose,auto,0,…)
# B1. Thạch bản: so cách ghép câu trước khi gọi API. KVK: anchor lệch +4/+5 từ trang 54 → content
$PY -m pipeline.tools.ingest_lithograph_book --book <BOOK> --ocr none --plan-only --out <scratch>/plan
# B1'. (tuỳ chọn, KHUYẾN NGHỊ khi có phiên âm dị bản gần) sửa QN OCR bằng câu tham chiếu — KVK dùng 1871 LVĐ, 1872 DMT để ĐỐI CHỨNG độc lập
$PY scripts/measure/verses_ref_fix.py --book <BOOK> --ref data/<ref>/<phienam>.json --ref-name nf1871    # --fuzzy-min 0.9; → measure_out/<BOOK>/qn_ref_fix/verses_b1.tsv
# B2. Ingest có kim (thử --limit 5 rồi bỏ); kim_raw/ đã có → 0 gọi API (--force để gọi lại). LVT: formula; KVK: --contrast otsu --verse-map content --verses …
$PY -m pipeline.tools.ingest_lithograph_book --book <BOOK> --ocr kim --out prepared \
   [--contrast otsu] [--verse-map formula|anchor|content] [--verses measure_out/<BOOK>/qn_ref_fix/verses_b1.tsv]   # KHÔNG --dict-boost
#     Văn xuôi (Chrestomathie): 1 lượt kim/trang trên JPG gốc, truyện↔cột theo bang_truyen_trang.csv
$PY -m pipeline.tools.ingest_prose_book --book <BOOK> --limit 5 --ocr kim ; $PY -m pipeline.tools.ingest_prose_book --book <BOOK> --ocr kim
# B3. Build (thử --limit 8 rồi toàn bộ; det_thr/det_xmargin từ config theo sách — log phải có "detector theo sách")
$PY -m pipeline.align_engine.build_dataset --config config/pipeline_<BOOK>.yaml --reseg detector \
   --qd01-cells none --decisions none --use-s3 --two-pass --box-rule syl_index --force --out dataset_out_<BOOK>
$PY -m pipeline.tools.enrich_crop_quality --labels dataset_out_<BOOK>/labels.csv --src-root dataset_out_<BOOK>
# B4. Remediation. **CẢNH BÁO: `pipeline.remediation apply` KHÔNG có `--out` sẽ ghi đè `dataset_out/` của STT (đang tracked trong git) —
#     đã xảy ra 2 lần 22/09, khôi phục bằng `git checkout -- dataset_out`. LUÔN truyền `--out dataset_out_<BOOK>`.**
$PY -m pipeline.remediation --labels dataset_out_<BOOK>/labels.csv --out dataset_out_<BOOK> census
$PY -m pipeline.remediation --labels dataset_out_<BOOK>/labels.csv --out dataset_out_<BOOK> apply --tau 0.62
$PY -m pipeline.remediation.confusion_fix --in dataset_out_<BOOK>/labels_remediated.csv \
   --out dataset_out_<BOOK>/labels_final.csv --fixes config/confusion_fixes.yaml --measure
git status --short -- dataset_out            # PHẢI trống
# B4'. Cổng theo CƠ CHẾ (PHUONG_AN_TU_DONG §4). Tự bật khi books[].layout == lithograph; prose PHẢI khai books[].mechanism_gates: true;
#      STT không khai → TẮT, --out = sao BYTE của --in. (a) n_det≠n_qn | box_source midpoint/split → GOLD_text_only (giữ nhãn, KHÔNG export ảnh);
#      (a') khi books[].box_decoder: pitch (tự nhận: --box-decoder auto = config > summary.json cạnh --in > labels): box_source ink_cut|detector_low → GOLD_text_only,
#      midpoint/split vẫn hạ, n_det≠n_qn CHỈ ghi cờ n_det_mismatch=1 (labels_gated + labels_trace) — muốn bộ chặt hơn thì lọc cờ này;
#      (b) s1_inter_s2_similar (CHAR_B) | direct_am_sua_dau → SYLLABLE; (c) blank/truncated → REVIEW; (d) chỉ khi có --cross (cells.csv của
#      auto_precision bước cross chạy TRƯỚC trên labels_final): GOLD bất đồng dị bản gần hình → REVIEW, không gần hình → cờ di_ban_khac.
#      Ưu tiên (c)>(d)>(b)>(a); gate_reason ghi cổng quyết; báo cáo JSON. Số đo SAU (d) là tự khẳng định (đọc `after_abc_only` để so công bằng).
$PY scripts/measure/auto_precision.py --steps cross --books <BOOK> [--labels dataset_out_<BOOK>/labels_final.csv --trans prepared[_b1]/<BOOK>/transcriptions --out <dir>]
$PY -m pipeline.remediation.mechanism_gates --in dataset_out_<BOOK>/labels_final.csv --out dataset_out_<BOOK>/labels_gated.csv \
   --config config/pipeline_<BOOK>.yaml --book <BOOK> [--cross <dir>/cross/<BOOK>/cells.csv] --report dataset_out_<BOOK>/mechanism_gates_report.json
# B5. Export + docs + xlsx — đầu vào labels_gated.csv; --n-columns chỉ sửa LOG/README "trang không đủ N cột" (STT 9; thạch bản 10; văn xuôi 7 = số cột phổ biến)
$PY pipeline/export_final_dataset.py --labels dataset_out_<BOOK>/labels_gated.csv --src-root dataset_out_<BOOK> --out dataset_<BOOK> --n-columns 10
$PY -m pipeline.tools.make_dataset_docs --dataset dataset_<BOOK> --n-columns 10 ; $PY -m pipeline.tools.make_xlsx --labels dataset_<BOOK>/labels.csv
# B6. Khớp dị bản của tầng GOLD-ảnh SAU cổng (thư mục riêng để measure_out/auto_precision giữ số "trước")
$PY scripts/measure/auto_precision.py --steps cross --books <BOOK> --labels dataset_out_<BOOK>/labels_gated.csv --trans prepared[_b1]/<BOOK>/transcriptions --out <dir>_gated
# Test trước/sau mỗi lần sửa mã (kỳ vọng: 79/79, 61/61, 33/33, 62/62, 19/19, 253/0, tools 139 pass/3 fail có sẵn)
$PY pipeline/align_engine/book_layout_selftest.py ; $PY pipeline/tools/ingest_lithograph_selftest.py ; $PY pipeline/tools/ingest_prose_selftest.py
$PY -m pipeline.remediation.mechanism_gates_selftest ; $PY scripts/measure/verses_ref_fix.py --selftest ; $PY -m pipeline.phase1_engine_selftest ; $PY -m pipeline.tools.selftest
```
Lệnh đúng như đã chạy cho KVK B1': `docs/CHAY_KVK1884_B1_2026-09-22.md` §1 (config `config/pipeline_KimVanKieu1884_b1.yaml`, `data_dir: prepared_b1`);
Chrestomathie: `docs/CHAY_CHRESTO1872_2026-09-22.md` §3. Tham chiếu CLI thật: `docs/PIPELINE_FACTS.json`. Không chạm `data/`, `prepared/SachThanhTruyen*`, `dataset_out`.

### 2.1 Kiểm tái lập đường tắt (22/09 chiều, HEAD 4e0a314bca + mã chưa commit, `--suffix _rp`, so với bản chốt §1)

| Sách | Lệnh | Thời gian | API kim | `labels.csv` / `labels_trace.csv` / `columns.csv` export | `dataset_out` labels / labels_final / labels_gated | ảnh crop (`diff -rq`) |
|---|---|---|---|---|---|---|
| LucVanTien1883 | `--book LucVanTien1883 --suffix _rp` | 161 s (build 147) | 0 | md5 **993a9950…** khớp / khớp / khớp | khớp / khớp / khớp | 0 tệp khác (chỉ `labels.xlsx` — dấu thời gian) |
| KimVanKieu1884 (B1') | `--book KimVanKieu1884 --suffix _rp` | 227 s (build 193) | 0 | md5 **6c703f9d…** khớp / khớp / khớp | khớp / khớp / khớp; `verses_b1.tsv` sinh lại md5 97c96d08… = cũ | 0 tệp khác |
| Chrestomathie1872 | `--book Chrestomathie1872 --suffix _rp` | 135 s (ingest 57, build 74) | 0 | md5 **01ebaf99…** khớp / khớp / khớp | khớp / khớp / khớp | 0 tệp khác; README/DATASHEET khác vì bản chốt sinh docs với `--n-columns` 9 (mặc định), đường tắt truyền 7 (đúng §2 B5) |

`prepared/LucVanTien1883`, `prepared_b1/KimVanKieu1884`, `prepared/Chrestomathie1872` ingest lại: `.txt` + `detected/` cache byte-identical (LVT: `transcriptions/*.json` đổi md5 vì mã 22/09 ghi thêm
`qn_source/ref_*`, không đổi nội dung cột). `mechanism_gates_report.json` chỉ khác đường dẫn `in/out/cross`. STT: `./run_pipeline.sh --dry-run` in đúng 6 bước cũ; `git status -- dataset_out` trống;
diff `run_pipeline.sh` cũ↔mới chỉ 2 hunk **thêm** (5 dòng ghi chú đầu tệp + khối SÁCH MỚI/THAM SỐ trước MAIN), mọi hàm STT và khối MAIN không đổi byte. Log: `logs/run_<BOOK>_20260922_*.log`.
Thư mục `_rp` giữ lại để đối chiếu (giống bản chốt) — xoá hoặc chạy lại không `--suffix` để bản chốt là sản phẩm trực tiếp của `run_pipeline.sh` (có `CHECKSUMS.txt` + log).

## 3. Cổng nghiệm thu từng bước — 3 sách sau vòng 2

Giá trị thật của lần chạy chốt: LVT1883 105 trang (`formula`, 0,05/0,15; CHAY_LVT §6.4); KVK1884 163 trang **B1'** (`content`, otsu, `verses_b1.tsv`, 0,05/0,15; CHAY_KVK1884_B1);
Chrestomathie1872 65 trang (prose, `auto`, 0,05/0,15; CHAY_CHRESTO §4).

| Bước | Cổng | Ngưỡng / kỳ vọng | LVT1883 | KVK1884 (B1') | Chrestomathie1872 |
|---|---|---|---|---|---|
| B0 | step0 Validation passed; `book_layout(b)` đúng layout/n_columns/det_* | bắt buộc | đạt (lithograph,10,14) | đạt (lithograph,10,14) | đạt (prose,auto,0) |
| B1/B1' | cách ghép câu chọn xong (`--plan-only`); phủ câu; B1': dòng exact/fuzzy/ocr | 0 trang thiếu câu; cột không khớp → chỉ REVIEW | formula 1.044 cặp, 0 thiếu; không B1' (LVT1916 xa, 29 % câu khớp) | content offset 0/−4/−5; B1': 465/1.548/1.238 dòng, 0 đổi số âm; **6 cột không khớp** → 84 ô REVIEW (trước 9/126) | truyện↔cột theo bảng đo: 404/417 cột ghép (96,9 %), 13 giữ chỗ; LTR thắng 20/20 truyện |
| B2 | cache_ok = n_pages; kim 6⧺8 (thạch bản); QN đủ 14 âm; unassigned; lượt kim | cache 100 %; 6⧺8 ≥ ~96 % | 105/105; 1.024/1.044; 893/1.044 (85,5 %); 12; 100 lượt | 163/163 (**0 lượt**, kim_raw chép); 1.560/1.628; 1.511/1.628 (B1' ≥ 0,9 không sửa dòng lệch số âm); 27 | 65/65; kim 1 hộp/cột, 8.500 chữ (bộ đo ước 8.692); khớp dict 0,645; 60 lượt |
| B3 | page_ok; M==N; ocr_char; p_register≥0,8; **I5 n_det==N ≥ 75 %** | page_ok ≥ ~98 %; ocr_char 100 % | 103/105; 83,8 %; 100 %; 99,0 %; **65,2 % ❌** | 162/163; 88,8 %; 100 %; 99,7 %; **59,2 % ❌** (B1' không đổi hộp) | **65/65**; 69,5 %; 100 %; —; **70,7 % ❌** |
| B4 | census F1 cross-col → quarantine; apply demoted; confusion_fix demoted; qd01 | ghi số; apply demoted 0 | 16/8 → quar 16; 0; 0; 0 | 94/47 → quar 92; 0; **5** (㝵/người); 0 | 0 → 0; 0; 0; 0 |
| labels_final | GOLD / SYLLABLE / REVIEW / QUARANTINE | — | 9.666 / 1.683 / 3.111 / 16 | **15.938** / 3.209 / 3.465 / 92 | 5.892 / 998 / 1.413 / 0 |
| B4' | GOLD ảnh còn / GOLD_text_only / hạ SYL (b: cầu + sửa dấu) / hạ REVIEW (c blank-cụt + d dị bản gần hình) | bật (lithograph/prose) | **5.793** (59,9 %) / **2.985** / 569 + 112 / 14 + 194 | **8.593** (53,9 %) / **5.784** / 601 + 58 / 230 + 696 | **3.922** (66,6 %) / **1.557** / 238 + 71 / 124 + — |
| B5 | ảnh copy = GOLD + SYLLABLE, 0 thiếu; text_only trong labels.csv không ảnh | 0 thiếu | **8.156** (v1 11.349) | **12.437** (v2_gates_noB1 12.344; v1 18.560) | **5.209** (trước cổng 6.890) |
| B6 | khớp dị bản ô GOLD-ảnh trong câu khớp, bỏ ref PUA (proxy văn bản, MÙ lỗi hộp; +(d) = tự khẳng định) | ≥ trước, trong CI | LVT1916: 72,6 % (n 2.796) → (a)(b)(c) 73,4 % (n 1.866) → +(d) 79,5 %; text_only 77,9 % | 1871: 80,6 % (n 11.467) → (a)(b)(c) **81,2 %** [80,2–82,1] (n 6.772) → +(d) 85,7 %; **1872 (độc lập với B1')**: 73,0 % → **74,1 %** [73,0–75,1] (n 6.848) → +(d) 78,4 %; text_only (abc) 80,2 / 72,7 % | không có dị bản số hoá |
| STT | labels.csv 3 trang byte-identical HEAD vs bản sửa; không `layout_gate`; export/gates STT sao byte | bắt buộc trước commit | md5 **59e436d7641fa849bb6759868ac29259** hai bên (22/09 sau vòng 2); mechanism_gates với config/pipeline.yaml → TẮT, labels_gated == labels_final; export `diff -rq` rỗng | (chung mã) | (chung mã) |

**I5 chưa đạt ở cả ba sách (65,2 / 59,2 / 70,7 % < 75 %)** và KHÔNG sửa được bằng tham số (det_xmargin chỉ chữa F1 cross-col 393 → 16 ô LVT; det_thr đã ở đỉnh 0,15).
Gốc: (a) detector CenterNet (học chữ STT) đếm lệch ±1 trên thạch bản (KVK ở cột 14 âm: −1: 188 / +1: 242) → cần NMS theo bước dọc hoặc fine-tune (không có hộp GT);
(b) cột QN lệch số âm (LVT 151/1.044, KVK 117/1.628 — B1' ≥ 0,9 không sửa được vì 5/6, 7/8 < 0,9). B4'(a) chặn hậu quả ở export (`GOLD_text_only`), không chữa gốc.

## 4. Kiểm người (không có người kiểm trong đề tài → thay bằng đo tự động §3 B6 + `docs/PHUONG_AN_TU_DONG_2026-09-22.md`)

Bộ mẫu mù v1 (seed 20260921, 300 GOLD + 100 SYLLABLE/sách) nằm ở `dataset_LucVanTien1883_v1/kiem_nguoi/`, `dataset_KimVanKieu1884_v1/kiem_nguoi/` (tạo bằng
`kiem_nguoi_grid.py --dataset dataset_<BOOK>_v1 --pages prepared/<BOOK>/pages --seed 20260921`; chấm bằng `kiem_nguoi_score.py`). Bản sau cổng chưa tạo lại;
hai công cụ chưa có selftest/review → không commit vòng 2. Nếu có người: ưu tiên KVK 6 cột không khớp (CHAY_KVK1884_B1 §3), 230 ô blank/truncated (đã hạ REVIEW), LVT 2 trang biên.

## 5. Kế hoạch commit

`docs/KE_HOACH_COMMIT_VONG2_2026-09-22.md` (3 commit: engine prose + gates/export; tools adapter B1'/prose + config + measure; docs). Trước commit: `git status --short -- dataset_out data` trống,
selftest như §2, hồi quy STT md5 59e436d7…, `scripts/measure/code_facts.py` 18/18. `git add` đích danh, không `-A`; `dataset_*`, `dataset_out_*`, `prepared*`, `measure_out/` không commit.

## 6. Việc còn mở (sau vòng 2)

1. **I5 chưa đạt cả ba sách** (§3, 65,2 / 59,2 / 70,7 % — n_det giữ nghĩa hộp thô nên số này KHÔNG đổi khi bật pitch) — gốc detector ±1 chưa chữa ở mức mô hình;
   vòng 3 chữa phần hoà giải bằng `box_decoder: pitch` (hộp IoU ≥ 0,5 với ô tham chiếu 98,1 / 97,0 %, trong cột n_det≠N 95,3 / 95,3 %) và cổng (a') theo ô →
   text_only chỉ còn 130 / 510 / 126; phương án B (huấn luyện v2) chưa chạy: `docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md` §3.
   **22/09 tối**: gốc I5 = răng cưa khi thu ảnh 3.200 → 1.024 (INTER_LINEAR); khoá `books[].detector_resize: area` (INTER_AREA, cùng ckpt v1) cho I5 thô
   **77,7 / 77,9** / 69,3 % và GOLD ảnh +83 / +257 (`--suffix _area`, 0 API) nhưng LVT bleed ảnh export 18,0 → 21,7 % → **chưa đổi chốt**, config ghi `linear` kèm số
   (BAO_CAO_TONG_HOP §3.3, HUONG_DAN_HUAN_LUYEN_I5 §0); v2 trên Kaggle: `lab/i5_detector_v2/README.md`, mốc so sánh = v1+area.
2. **B1' KVK**: thay cả dòng theo 1871 làm cột `syllable` mang chính tả Bắc ở 676 dòng (sanh→sinh, nhơn→nhân…; `qn_source` có trong `transcriptions/*.json`, CHƯA vào labels/export);
   50 âm (2,0 %) nghi xoá dị bản QN thật của 1884; 125 dòng lệch số âm không sửa được ở 0,9 (0,8 thêm 240 âm nghi xoá → không hạ ngưỡng). `--dict-boost` LOẠI (tự khẳng định 100 % trên 1871).
3. **Blank KVK 230 ô** hạ REVIEW bởi (c): ngưỡng `enrich_crop_quality` hiệu chuẩn trên STT, 40 ô đo có mực trên ảnh gốc → có thể hạ oan.
4. **`run_pipeline.sh` (STT) chưa gọi B4'** (đường STT giữ nguyên có chủ đích); sách mới đã có đường tắt `--book` (§2.0, luôn truyền `--out dataset_out_<BOOK>`).
   `pipeline.remediation apply` chưa chặn trong mã việc ghi `dataset_out/` khi thiếu `--out` (chỉ cảnh báo ở §2). `--no-api` với KVK đổi `content` → `formula` (khác bản chốt);
   `book_profile` nhập `auto_precision.CROSS_BOOKS` để biết sách có dị bản (sách thứ tư có dị bản phải thêm vào đó mới có cổng (d)/B6).
5. **Chrestomathie**: QN tesseract lỗi âm ≈ 13 % + mất dòng → 1.239 ô `no_context` REVIEW, 12 cột giữ chỗ; chưa quét `det_thr` riêng; kim 8.500 vs ước 8.692 chữ (−2,2 %); 2 ô trang 65 không có chữ kim.
6. `--verse-map content`/prose DP: hằng `CONTENT_WINDOW 60 / SKIP_PEN 2 / MIN_COL 4`, `n_match < 3 | ratio < 0,25` cứng; cổng (b) hạ SYLLABLE nhưng ảnh vẫn nằm `gold/…` (đường `image` là khoá bền).
7. `docs/KVK1884_TRANG_CAN_XAC_NHAN.csv` lỗi thời (27 trang formula≠anchor) — `content` quyết từng cột; `CHAY_KVK1884_THU_2026-09-21.md` chỉ còn giá trị lịch sử.
8. `--use-s3` dựng 0 lớp nguyên mẫu crop (index.csv trỏ `dataset_out/` STT — lỗi sẵn có); README các sách vẫn tiêu đề khuôn STT; NGUON_THU_TICH 7 mục ⬜ chờ người giữ bản quét.
9. LVT page_0001/page_0105 `projection_fallback` → REVIEW 89 %/82 %; tools.selftest 3 fail có sẵn ở HEAD (KE_HOACH_COMMIT_2026-09-21 §4).

## 7. Thêm sách thứ tư

1. Dữ liệu: `data/<BOOK>/pages/` (+ `SOURCE.md`), QN cùng nguồn theo trang (thạch bản: số câu in; văn xuôi: theo truyện/đoạn).
2. Bộ đo: thêm sách vào `BOOKS` của `scripts/measure/` (README ở đó), `measure.py --all` → `measure_out/<BOOK>/layout/` + `qn_ocr/verses.tsv` (thạch bản) hoặc `chresto/*.csv` (văn xuôi); invariants PASS.
3. **Chọn layout**: cột đều, mỗi cột = cặp lục bát 6⧺8 → `lithograph` + `ingest_lithograph_book` (so `--plan-only` 3 cách; verses.tsv mất dòng → `content`; có phiên âm dị bản gần → B1' `verses_ref_fix`).
   Cột số chữ biến thiên, QN theo truyện/đoạn không đánh số → `prose` + `ingest_prose_book` (cần bảng truyện↔trang như `bang_truyen_trang.csv`, `n_columns: auto`).
   Đánh số trang khác (KVK page = 167 − canvas; Chresto page = canvas − 105) → thêm map vào adapter + phép kiểm selftest.
4. Config `config/pipeline_<BOOK>.yaml` (§2 B0): quét `det_thr` trên toàn sách trước khi khoá; `det_xmargin` mặc định 0,05; STT `config/pipeline.yaml` KHÔNG khai layout.
   Thêm khối `run:` (§2.0: `ingest`, `ingest_args`, `verses_ref_fix`, `n_columns`, `cross`; mẫu: 3 config hiện có) để `./run_pipeline.sh --book <BOOK>` chạy được; sách có
   dị bản số hoá → thêm vào `CROSS_BOOKS` của `scripts/measure/auto_precision.py` (đường tắt tự phát hiện); thêm tên vào `NEW_BOOKS_ALL` trong `run_pipeline.sh` nếu muốn `all-new` gồm sách này.
5. Chạy B0→B6 (`./run_pipeline.sh --book <BOOK>` hoặc tay §2), điền bảng §3, viết `docs/CHAY_<BOOK>_<ngày>.md`; nếu sách có dị bản số hoá thì thêm vào `CROSS_BOOKS` của `auto_precision.py` để có B6 và cổng (d).
   Hồi quy STT 3 trang chỉ cần chạy lại khi sửa `pipeline/`, `core/` hoặc `config/pipeline.yaml`.
