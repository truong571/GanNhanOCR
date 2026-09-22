# Hướng dẫn chạy sách thạch bản mới (LVT1883 / KVK1884) — 2026-09-21

Tài liệu tổng hợp các nhiệm vụ 21–22/09 (NV1 engine → NV5 KVK thử; NV-B det_xmargin/det_thr LVT; NV-D KVK 163 trang). Chi tiết từng lần chạy:
`docs/CHAY_LVT1883_2026-09-21.md` (§6 = lần chạy cuối), `docs/CHAY_KVK1884_2026-09-21.md` (thay bản THU 8 trang); hợp đồng dữ liệu/cổng:
`docs/PIPELINE_SACH_MOI_2026-09-20.md` §2–§7; kế hoạch commit: `docs/KE_HOACH_COMMIT_2026-09-21.md`.
Trạng thái tiến trình: lúc 09:07 22/09 không còn tiến trình pipeline nào chạy (mọi bước đã kết thúc). **Chưa commit.**

## 1. Trạng thái — cái gì đã chạy được

| Thành phần | Trạng thái | Bằng chứng |
|---|---|---|
| API kim (`core/ocr/ocr_api.py`) | token qua SN_OCR_USERNAME/PASSWORD (60 phút); Guest Mode (diff chưa commit) KHÔNG kích hoạt | 10/10 request HTTP 200; STT2 page_0024 9 cột, 206/206 chữ trùng cache 20/09, bbox lệch 0 px |
| kim trên thạch bản nền xám 128 | KHÔNG mù: raw và stretch cùng 22 hộp/143 chữ, 8/10 cột đúng 14 = 6⧺8 | LVT trang 10: chars/cột [14,14,14,16,14,14,14,14,15,14] |
| Engine `layout: lithograph` (`pipeline/align_engine/book_layout.py`) | mọi tham số mới có mặc định = hành vi cũ (n_columns 9, DEFAULT_LAYOUT); 22/09 thêm khoá theo sách `books[].det_xmargin` / `det_thr` (None = `step2.*` toàn cục; lithograph vắng khoá → det_xmargin 0,05) | book_layout_selftest 57/57; phase1_engine_selftest 253/0; tools.selftest 125 pass/4 fail (1 fail do `dataset_out/` bị xoá, 3 fail có sẵn ở HEAD — `docs/KE_HOACH_COMMIT_2026-09-21.md` §4) |
| Adapter `pipeline/tools/ingest_lithograph_book.py` | `--verse-map formula\|anchor\|content` (`content` = ghép cột Nôm ↔ cặp dòng QN theo NỘI DUNG chữ kim tra `dict/QuocNgu_SinoNom.csv`, DP đơn điệu; cột không khớp → 14 token `khongkhop` → chỉ REVIEW), `--plan-only`, `--contrast otsu` | ingest_lithograph_selftest 35/35 (bộ cũ — `content` CHƯA có selftest) |
| Hồi quy STT (NV4, chạy lại 22/09 sau NV-B) | worktree sạch @ c499c8277c vs bản sửa, 3 trang stt2/0024, stt4/0050, stt11/0100 | labels.csv byte-identical, md5 59e436d7641fa849bb6759868ac29259 (556 dòng); 344/346 tệp giống byte (summary thêm `detector_params_by_book`, decisions khác đường dẫn tuyệt đối); 448/448 trang `_detect` byte-identical (NV1) |
| Bộ mẫu người kiểm mù `pipeline/tools/kiem_nguoi_grid.py` / `kiem_nguoi_score.py` | 300 GOLD + 100 SYLLABLE mỗi sách, phân tầng theo trang, người kiểm không thấy nhãn máy (mục 4) | LVT 400 ô / KVK 400 ô, 400/400 có ngữ cảnh, nhãn máy nằm trong ứng viên 400/400; chưa có selftest, chưa review |

### Số liệu cuối LVT1883 (105/105 trang, `det_xmargin 0,05 / det_thr 0,15`, `dataset_out_LucVanTien1883/`; CHAY_LVT §6.4)
- Ingest (`--verse-map formula`): 8 ph 15 s, 100 gọi kim, cache_ok 105/105; 1.044 cột = 2.088 câu; kim 6⧺8 đúng 1.024/1.044; QN đủ 14 âm 893/1.044 (85,5 %); hộp không gán 12; số câu in lọc 169.
- Build (2 ph 22 s): 14.476 ô — GOLD 9.680 / SILVER 0 / SYLLABLE 1.685 / REVIEW 3.111 (tier KHÔNG đổi so với ±0,25w — tier là luật văn bản); **page_ok 103/105** (2 trang biên page_0001 cột tựa, page_0105 projection_fallback); **M==N 83,8 %** (872/1.041); ocr_char 100 %; p_register≥0,8 99,0 %; n_anchor_pairs 2.292; **n_det==N 65,2 %** (679/1.041; trước 60,6 %) — I5 <75 % KHÔNG đạt.
- Remediation: F1 cross-col **16 ô/8 nhóm** (trước 393/196) → quarantine 16 (conflict 16); demoted 0; confusion_fix n_fixes 1, demoted 0; qd01 0.
- labels_final: GOLD **9.666** / SYLLABLE 1.683 / REVIEW 3.111 / QUARANTINE 16 (trước 9.344/1.634/3.111/387) → export `dataset_LucVanTien1883/` **11.349 ảnh** (0 thiếu; 14 ô blank/truncated: 8 + 6). Bản cũ ±0,25w dời sang `dataset_LucVanTien1883_xmargin025/`.

### Số liệu cuối KVK1884 (163/163 trang, `--verse-map content --contrast otsu`, `det_xmargin 0,05 / det_thr 0,15`; CHAY_KVK1884 §2–§4)
- Vì sao `content`: `anchor` neo theo số câu in, nhưng số in Nôm = seq QN + 4 từ trang 54 cột 7 (verses.tsv mất 4 dòng quanh 1069–1072) và + 5 từ trang 153 (mất 1 dòng quanh ~3045) → build anchor 162 trang chỉ GOLD 5.108/22.558 (22,6 %). `content` offset 0: 534 cột · −4: 983 · −5: 102 · không khớp **9 cột** (126 ô `khongkhop` → REVIEW); trang 163 ghép đủ 8 cột (offset −5), không bỏ.
- Ingest lại (3 s, **0 gọi API** — đọc `kim_raw/` từ lượt anchor): cache_ok 163/163; 162 trang 10 cột (page_0163 = 8 cặp); 1.628 cột = 3.256 câu; kim 6⧺8 1.560/1.628 (95,8 %); QN đủ 14 âm 1.511/1.628 (92,8 %); unassigned 27; số in lọc 210.
- Chọn det_thr (quét NATIVE 1.630 cột): 0,3 → n_det==N 2,7 %; 0,2 → 45,5 %; **0,15 → 59,2 %**; 0,1 → 59,0 % (thừa hộp +1 gấp đôi) → 0,15, trùng LVT.
- Build (3 ph 11 s): 22.704 ô — GOLD 15.133 / SILVER 0 / SYLLABLE 3.520 / REVIEW 4.051; **page_ok 162/163** (99,4 %; chỉ page_0163 n_qn_cols 8 ≠ 10 nhưng nội dung đúng); **M==N 88,8 %** (1.445/1.628); ocr_char 100 %; p_register≥0,8 99,7 %; n_anchor_pairs 4.244; **n_det==N 59,2 %** (964/1.628) — I5 KHÔNG đạt; 0 trang REVIEW > 50 %.
- Remediation: F1 cross-col 90 ô/45 nhóm → quarantine 88 (conflict 86, dup 2); demoted 0; confusion_fix **total_demoted 5** (5 ô GOLD 㝵/người → REVIEW theo luật `confusion_fixes.yaml`, hợp lệ; LVT 0 vì không có ô này); qd01 0.
- labels_final: GOLD **15.056** / SYLLABLE 3.504 / REVIEW 4.056 / QUARANTINE 88 → export `dataset_KimVanKieu1884/` **18.560 ảnh** (0 thiếu); crop blank 216 (1,16 %; đo 40 ô: có mực trên ảnh gốc, không phải otsu xoá nét) + truncated 12 → 226 ô có cờ `crop_quality_flag` trong `labels_trace.csv`, vẫn export.

## 2. Lệnh chạy từng bước cho một sách thạch bản `<BOOK>`

Tiền đề: `data/<BOOK>/pages/`, `measure_out/<BOOK>/layout/layout_pages.csv`, `measure_out/<BOOK>/qn_ocr/verses.tsv` (sinh bởi `scripts/measure/measure.py`), `.env` có SN_OCR_USERNAME/PASSWORD.

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
PY=.venv/bin/python
# B0. Config riêng (copy config/pipeline_LucVanTien1883.yaml, đổi tên sách/output_dir; KHÔNG khai pdf)
#     books.<BOOK>: layout: lithograph, n_columns: 10, qn_syllables_per_column: 14,
#                   det_xmargin: 0.05 (mặc định lithograph), det_thr: 0.15 (ĐO THEO SÁCH — quét 0,3/0,2/0,15/0,1 như CHAY_KVK1884 §3;
#                   LVT1883 và KVK1884 cùng ra 0,15) ; paths.output_dir: dataset_<BOOK>. STT config/pipeline.yaml KHÔNG khai 2 khoá này.
$PY -m pipeline.step0_setup config/pipeline_<BOOK>.yaml            # kỳ vọng: "Validation passed", BookLayout(lithograph,10,14, det_xmargin 0.05, det_thr 0.15)
# B1. So cách ghép câu trước khi gọi API (formula / anchor / content). KVK: anchor lệch +4/+5 từ trang 54 → dùng content
$PY -m pipeline.tools.ingest_lithograph_book --book <BOOK> --ocr none --plan-only --out <scratch>/plan
# B2. Ingest có kim (thử 5 trang trước, rồi bỏ --limit); kim_raw/ đã có → 0 gọi API, --force để gọi lại
$PY -m pipeline.tools.ingest_lithograph_book --book <BOOK> --limit 5 --ocr kim --out prepared
$PY -m pipeline.tools.ingest_lithograph_book --book <BOOK> --ocr kim --out prepared \
   [--contrast otsu] [--verse-map formula|anchor|content]           # LVT: formula (mặc định); KVK: --contrast otsu --verse-map content
# B3. Build (thử --limit 8 rồi toàn bộ; det_thr/det_xmargin lấy từ config theo sách — log phải có dòng "detector theo sách")
$PY -m pipeline.align_engine.build_dataset --config config/pipeline_<BOOK>.yaml --reseg detector \
   --qd01-cells none --decisions none --use-s3 --two-pass --box-rule syl_index --force --limit 8 --out <scratch>/build8
$PY -m pipeline.align_engine.build_dataset --config config/pipeline_<BOOK>.yaml --reseg detector \
   --qd01-cells none --decisions none --use-s3 --two-pass --box-rule syl_index --force --out dataset_out_<BOOK>
$PY -m pipeline.tools.enrich_crop_quality --labels dataset_out_<BOOK>/labels.csv --src-root dataset_out_<BOOK>   # + dòng build vào CHECKSUMS.txt
# B4. Remediation
$PY -m pipeline.remediation --labels dataset_out_<BOOK>/labels.csv census
$PY -m pipeline.remediation --labels dataset_out_<BOOK>/labels.csv apply --tau 0.62
$PY -m pipeline.remediation.confusion_fix --in dataset_out_<BOOK>/labels_remediated.csv \
   --out dataset_out_<BOOK>/labels_final.csv --fixes config/confusion_fixes.yaml --measure
# B5. Export + docs + xlsx
$PY pipeline/export_final_dataset.py --labels dataset_out_<BOOK>/labels_final.csv --src-root dataset_out_<BOOK> --out dataset_<BOOK>
#     make_dataset_docs ; make_xlsx (như LVT)
# Test trước/sau mỗi lần sửa mã
$PY pipeline/align_engine/book_layout_selftest.py ; $PY pipeline/tools/ingest_lithograph_selftest.py
$PY -m pipeline.phase1_engine_selftest ; $PY -m pipeline.tools.selftest
```
Tham chiếu CLI thật: `docs/PIPELINE_FACTS.json`. Không chạm `data/`, `prepared/SachThanhTruyen*`, `dataset_out`.

## 3. Cổng nghiệm thu từng bước

Giá trị thật của lần chạy cuối mỗi sách (LVT1883 105 trang, det_xmargin 0,05 / det_thr 0,15 — CHAY_LVT §6.4; KVK1884 163 trang, `content`, 0,05 / 0,15 — CHAY_KVK1884 §2–§4).

| Bước | Cổng | Ngưỡng / kỳ vọng | LVT1883 (105 trang) | KVK1884 (163 trang) |
|---|---|---|---|---|
| B0 | step0 Validation passed; `book_layout(b)` = BookLayout(lithograph,10,14, det_xmargin, det_thr) | bắt buộc | đạt (0,05 / 0,15) | đạt (0,05 / 0,15) |
| B1 | cách ghép câu chọn xong bằng `--plan-only`; cổng phủ câu verses.tsv | 0 trang thiếu câu; cột không khớp → chỉ REVIEW | formula: 1.044 cặp, 0 trang thiếu | content: offset 0/−4/−5 = 534/983/102 cột; **9 cột không khớp** → 126 ô REVIEW (anchor bị loại: lệch +4/+5 từ trang 54) |
| B2 | n_cache_ok = n_pages (`verify_cache_image`='ok'); n_pages_10cols; kim 6⧺8; QN đủ 14 âm; unassigned | cache_ok 100 %; kim 6⧺8 ≥ ~96 %; QN 14 âm là cờ (không ép) | 105/105; 103; 1.024/1.044 (98,1 %); 893/1.044 (85,5 %); 12 | 163/163 (0 gọi API); 162; 1.560/1.628 (95,8 %); 1.511/1.628 (92,8 %); 27 |
| B3 | page_ok (cột Nôm == n_columns & cột QN == n_columns & QN đủ 14 & col_method ≠ hybrid_no_image); M==N; ocr_char; p_register≥0,8; SPEC §7 I5 n_det==N | page_ok ≥ ~98 %; ocr_char 100 %; **I5 ≥ 75 %** | 103/105 (98,1 %); 83,8 %; 100 %; 99,0 %; **65,2 % KHÔNG đạt** | 162/163 (99,4 %); 88,8 %; 100 %; 99,7 %; **59,2 % KHÔNG đạt** |
| B4 | census F1 cross-col → quarantine; demoted (apply) / confusion_fix demoted / qd01 | ghi số; demoted == 0 kỳ vọng SPEC §6 | 16 ô/8 nhóm → quar 16; 0; 0; 0 | 90/45 → quar 88 (conflict 86, dup 2); 0; **5** (㝵/người theo luật — hợp lệ); 0 |
| B5 | ảnh copy = usable, 0 thiếu; crop blank/truncated chỉ cờ | 0 thiếu | **11.349**, 0 thiếu; 14 ô cờ | **18.560**, 0 thiếu; 226 ô cờ (blank 216 + truncated) |
| STT | labels.csv 3 trang byte-identical HEAD vs bản sửa; summary không có `layout_gate` | bắt buộc trước commit | md5 59e436d7… cả hai bên (chạy lại 22/09 sau NV-B, KE_HOACH_COMMIT §1) | không cần chạy lại: chỉ đổi `config/pipeline_KimVanKieu1884.yaml`, không đụng mã/STT config |

**I5 chưa đạt ở cả hai sách (65,2 % / 59,2 % < 75 %)** và KHÔNG sửa được bằng tham số theo sách (det_xmargin chỉ chữa F1 cross-col: 393 → 16 ô LVT; det_thr đã ở đỉnh 0,15 cho cả hai). Hai gốc còn lại (đo trên 1.050 cột LVT / 1.630 cột KVK):
(a) **detector CenterNet (học chữ STT) đếm lệch ±1 trên thạch bản** — LVT ở cột n_ocr = 14 (thr 0,2): n_det − 14 = −1: 142 / +1: 135, ở thr 0,15 còn chủ yếu +1: 215 cột; KVK ở cột n_qn = 14: −1: 188 / +1: 242 (+1 = hộp trùng lệch nửa chữ trong cột) → cần NMS theo bước dọc hoặc fine-tune detector;
(b) **cột QN lệch số âm** (OCR QN in): LVT 151/1.044, KVK 117/1.628 → N sai → thay `verses.tsv` bằng phiên âm chuẩn (Nôm Foundation). Chưa có chặn `n_det ≠ N → REVIEW` như SPEC §7 I5 đòi; ô ở cột lệch vẫn vào GOLD qua midpoint/syl_index.

## 4. Bước kiểm người (bắt buộc trước khi công bố tier GOLD — hiện CHƯA có ground truth người cho sách mới)

Bộ mẫu **mù** đã tạo (seed 20260921, 300 GOLD + 100 SYLLABLE, phân tầng theo trang, phân bổ tỷ lệ → tỷ lệ đúng trên mẫu = ước lượng
không chệch cho tier; người kiểm KHÔNG thấy nhãn máy, chọn trong 3–5 ứng viên Nôm cùng âm / "không có" / "crop sai" / "không chắc"):

| Sách | Lưới cho người kiểm | Tổng thể có crop | Phủ trang GOLD / SYLLABLE | Khoá (KHÔNG đưa người kiểm) |
|---|---|---|---|---|
| LVT1883 | `dataset_LucVanTien1883/kiem_nguoi/index.html` (400 ô) | GOLD 9.666 / SYLLABLE 1.683 | 103 / 98 | `dataset_LucVanTien1883/kiem_nguoi/items.csv`, `summary.json` |
| KVK1884 | `dataset_KimVanKieu1884/kiem_nguoi/index.html` (400 ô) | GOLD 15.056 / SYLLABLE 3.504 | 163 / 100 | `dataset_KimVanKieu1884/kiem_nguoi/items.csv`, `summary.json` |

```bash
# Tạo lại (idempotent theo seed): --n-gold 300 --n-second 100; --second-tier auto = SYLLABLE (REVIEW không có crop trong export)
$PY pipeline/tools/kiem_nguoi_grid.py --dataset dataset_<BOOK> --pages prepared/<BOOK>/pages --seed 20260921 --n-gold 300 --n-second 100
# Người kiểm: mở index.html (không cần server), chấm, bấm "Xuất CSV" → verdicts_<tên>_<ngày>.csv (item_id, verdict, choice_index, reviewer, ts)
# Chấm điểm: precision GOLD + Wilson 95 % CI (bảo thủ: "Không chắc" = sai), tách lý do sai; SYLLABLE: tỷ lệ xác nhận âm; ≥2 người trùng ≥10 ô → % đồng thuận + κ
$PY pipeline/tools/kiem_nguoi_score.py --items dataset_<BOOK>/kiem_nguoi/items.csv --verdicts <csv|thư mục> [--out dataset_<BOOK>/kiem_nguoi/score.json]
```
Giới hạn: tối đa 5 ứng viên nên 375/400 (LVT) và 376/400 (KVK) ô bị cắt bớt danh sách — "Không có trong danh sách" vẫn tính nhãn máy SAI
(đúng cho precision) nhưng không định vị được chữ đúng. 10 ô GOLD mẫu đọc nhanh: CHAY_LVT §4, CHAY_KVK1884 §5. Ngoài mẫu, người cần xem
thêm: KVK 9 cột không khớp (CHAY_KVK1884 §2, 126 ô REVIEW) + 226 ô blank/truncated (`labels_trace.csv` cột `crop_quality_flag`); LVT 2 trang biên page_0001/page_0105.
Hai công cụ `kiem_nguoi_*.py` chưa có selftest, chưa review (KE_HOACH_COMMIT R-19) — commit riêng sau khi review.

## 5. Kế hoạch commit

Theo `docs/KE_HOACH_COMMIT_2026-09-21.md`: (1) `git checkout -- dataset_out` trước (10 tệp STT đang `D`; tools.selftest kỳ vọng 139/3 sau khôi phục);
(2) quyết R-08 (`data/LucVanTien1883/luc_van_tien_quoc_ngu.txt` bị D); (3) sửa R-01/R-03/R-04/R-05/R-07 nếu đồng ý → chạy lại selftest + hồi quy 3 trang
(md5 phải giữ 59e436d7…); (4) `$PY scripts/measure/code_facts.py` sinh lại FACTS → **commit 1** engine (book_layout, det_* theo sách) → **commit 2** tools
(adapter, config 2 sách, scripts/measure, docs) → **commit 3** ocr_api Guest Mode → **commit data** riêng (`data/` 268 MB). `git add` đích danh, không `-A`.
Chưa quyết: docs KVK (`CHAY_KVK1884_2026-09-21.md`, `CHAY_KVK1884_THU…`, `KVK1884_TRANG_CAN_XAC_NHAN.csv` lỗi thời), `kiem_nguoi_*.py` — review rồi commit riêng.
Diff Guest Mode `core/ocr/ocr_api.py`: chưa kích hoạt trong mọi lần chạy (token OK; KVK content 0 gọi API) → giữ hay hoàn nguyên do người quyết.

## 6. Việc còn mở

1. **I5 chưa đạt cả hai sách** (65,2 % / 59,2 % < 75 %) — gốc (a) detector ±1, (b) QN lệch số âm (mục 3); chưa có chặn `n_det ≠ N → REVIEW`. Ngoài phạm vi "tham số theo sách".
2. **`--verse-map content` chưa có selftest** (`ingest_lithograph_selftest` 35/35 = bộ cũ); `CONTENT_WINDOW 60 / SKIP_PEN 2 / MIN_COL 4` là hằng cứng; nên đối chiếu với phiên âm chuẩn (Nôm Foundation) khi có.
3. **KVK 9 cột không khớp** (54 c5–c6, 153 c5 = chỗ verses.tsv mất 4 + 1 dòng; 76 c10, 80 c7, 143 c7, 145 c6, 150 c2, 158 c6 = dòng QN có nhưng điểm < 4 chữ) → người đối chiếu 2 dòng QN tương ứng.
4. **KVK 226 ô blank/truncated** trong bộ export (blank 216 = 1,16 %, GOLD 186 / SYLLABLE 30; truncated 12) — chỉ cờ; ngưỡng `enrich_crop_quality` hiệu chuẩn trên STT.
5. Người kiểm ≥ 300 GOLD mỗi sách (mục 4) chưa chấm → chưa công bố tier GOLD; `make_dataset_docs` sinh NGUON_THU_TICH.md theo khuôn STT, 7 mục ⬜ chờ người giữ bản quét (cả 2 sách).
6. **`docs/KVK1884_TRANG_CAN_XAC_NHAN.csv` LỖI THỜI** (27 trang formula≠anchor): `content` quyết từng cột bằng chữ; danh sách người cần xem = 9 cột ở điểm 3 trên — đã ghi chú ở đầu `docs/CHAY_KVK1884_2026-09-21.md`.
7. Hồi quy STT: đã chạy lại 22/09 sau NV-B (md5 59e436d7…); đổi `config/pipeline_KimVanKieu1884.yaml` (det_thr 0,15) KHÔNG cần chạy lại vì chỉ là config riêng — chạy lại chỉ khi sửa mã `pipeline/`, `core/` hoặc `config/pipeline.yaml`.
8. `--use-s3` dựng 0 lớp nguyên mẫu crop (index.csv trỏ `dataset_out/` STT đã xoá — lỗi sẵn có, ~6.500 dòng WARN imread); tier không phụ thuộc S3 ở đường `--two-pass`.
9. LVT: page_0001 (cột tựa) / page_0105 (5 cặp) `projection_fallback` → REVIEW 89 %/82 %, cần bỏ 2 trang biên hoặc adapter ghi n_columns theo trang; 151 cột QN lệch số âm.

## 7. Thêm sách thứ tư

1. Dữ liệu: `data/<BOOK>/pages/` (+ `SOURCE.md`), QN theo trang nếu có.
2. Bộ đo: thêm sách vào `BOOKS` trong `scripts/measure/` (xem `scripts/measure/README.md`), chạy `measure.py --all` → `measure_out/<BOOK>/layout/layout_pages.csv` + `qn_ocr/verses.tsv`; kiểm invariants PASS trong `measure_out/REPORT.md`.
3. Adapter: đọc trực tiếp `measure_out/<BOOK>/`; nếu đánh số trang khác (như KVK page = 167 − canvas) thì bổ sung map vào `ingest_lithograph_book.py` và thêm phép kiểm vào `ingest_lithograph_selftest.py`; nếu số câu in không theo công thức verse_no = 20(page−1)+2k−1,2k thì so `--plan-only` cả 3 cách; **nếu verses.tsv mất dòng (số in ≠ seq QN) thì `anchor` cũng sai → dùng `--verse-map content`** (bài học KVK).
4. Config: `config/pipeline_<BOOK>.yaml` (mục 2 B0): quét `det_thr` trên toàn sách trước khi khoá (KVK §3); `det_xmargin` để mặc định 0,05. STT `config/pipeline.yaml` KHÔNG khai layout → hành vi cũ.
5. Chạy B0→B5, điền bảng cổng mục 3, viết `docs/CHAY_<BOOK>_<ngày>.md`, tạo bộ mẫu người kiểm (mục 4). Hồi quy STT 3 trang chỉ cần chạy lại khi sửa mã `pipeline/`/`core/` hoặc `config/pipeline.yaml` (config riêng của sách không đòi).
