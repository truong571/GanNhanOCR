# BÁO CÁO TỔNG THỂ — Pipeline gán nhãn tự động cho dữ liệu mới (ngoài Sách Thánh Truyện)

> **Đã gộp vào `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` (vòng 3, 22/09 chiều: + I5/pitch_decode, cổng (a'), run_pipeline --book); bản này giữ làm lịch sử** — số chốt hiện hành đọc ở báo cáo gộp §3.

Ngày 2026-09-22 · Trạng thái: **hoàn tất vòng 1 và vòng 2** cho 2 sách thạch bản (LucVanTien1883, KimVanKieu1884) và
sách văn xuôi Chrestomathie1872 (vòng 2, §9); 5 sách chép tay chưa có phiên âm cùng nguồn không chạy được.
Mọi con số dưới đây đều tái lập được bằng script (0 token LLM), nguồn ghi ở cột cuối. Không có người kiểm (ràng buộc đề tài).

---

## 1. Câu trả lời ngắn

| Câu hỏi | Trả lời |
|---|---|
| Pipeline STT có chạy được trên sách mới không? | **Có**, sau 2 thay đổi: engine nhận `layout (lithograph\|prose)/n_columns/det_xmargin/det_thr` theo sách (mặc định = STT, hồi quy byte-identical) và adapter `ingest_lithograph_book` / `ingest_prose_book` dựng `prepared/<sách>/` từ ảnh JPG + kim. |
| Đã chạy đủ chưa? | LVT1883 **105/105**, KVK1884 **163/163** (B1'), Chrestomathie1872 **65/65** trang; 3 dataset sau cổng B4': **8.156 + 12.437 + 5.209 ảnh** (§9). |
| Có biết nhãn đúng bao nhiêu không (không người kiểm)? | **Có, bằng đối chứng độc lập**: precision GOLD 89,3 % / 84,6 % trên GT IHR-NomDB (mộc bản); khớp dị bản KVK1884 GOLD-ảnh sau B1'+cổng **81,2 %** (1871, bỏ PUA) / **74,1 %** (1872, độc lập với B1') khi nền dị bản 1871↔1872 = 83,4 %. |
| Đã "tốt nhất" chưa? | Tốt nhất **trong ràng buộc hiện tại** (kim + từ điển + căn chỉnh, không người). Vòng 2 đã làm A (cổng B4') và B (B1') của §6; giới hạn thật còn lại: §9.5. |

---

## 2. Dữ liệu đã thẩm định (data/, 13 thư mục)

| Nhóm | Sách | Loại | QN cùng nguồn | Trạng thái |
|---|---|---|---|---|
| 1 — cặp cùng nguồn (đầu vào gán nhãn) | SachThanhTruyen2/4/11 | viết tay thật | ✅ trang đối diện | kho chính, đã chạy từ trước |
| | **LucVanTien1883** | thạch bản, 10 cột × 2 tầng (cột = cặp lục bát 6/8), 2.088 câu | ✅ 139 trang QN in cùng sách | **đã chạy đủ** |
| | **KimVanKieu1884** | thạch bản, canvas đảo (page = 167 − canvas), 3.256 câu | ✅ 295 trang QN xen dịch Pháp | **đã chạy đủ** |
| | **Chrestomathie1872** | thạch bản, văn xuôi 20 truyện, cột 3–7 chữ biến thiên | ✅ 28 trang QN (mức truyện) | **đã chạy đủ (vòng 2, `layout: prose`)** |
| 2 — đối chứng ngoài có phiên âm độc lập | LucVanTien1916, TruyenKieu1872 (IHR-NomDB / Nôm Foundation) | mộc bản, 35 px/chữ | ✅ câu-với-câu, đã đối chiếu 96,6 % / 99,97 % | **dùng làm GT độc lập** để đo precision |
| 3 — chép tay chưa phiên âm | TruyenKieuPhongTinhCoLuc, KimVanKieu1894, TamTuKinhDienAm, CacThanhTruyen1646 | viết tay | ❌ | chỉ chạy được ở chế độ "QN dị bản" (sai ≥ 6 % âm tiết), không đạt đề tài |
| Loại | LyHangCaDao | viết tay, 1000 px | ❌ | loại |

Nguồn: `data/README.md`, `data/KIEM_TRA_KHOP_1-1_2026-09-20.md`, `docs/KET_QUA_DO_CUOI_2026-09-21.md`.

---

## 3. Những gì đã xây (đã commit trên main, 5 commit fb345a29b1 → 853cadfd9c; vòng 2 chưa commit — §9)

| Thành phần | Tệp | Kiểm chứng |
|---|---|---|
| Engine tham số theo sách | `pipeline/align_engine/book_layout.py`; sửa `align_production.py`, `build_dataset.py`, `step0/1/2` | STT 3 trang **byte-identical** (md5 59e436d7…); selftest 57/57, 253/0, tools 139/3 (3 fail có sẵn ở HEAD) |
| Adapter ingest thạch bản | `pipeline/tools/ingest_lithograph_book.py` (`--ocr kim`, `--contrast otsu`, `--verse-map formula\|anchor\|content`) | selftest 35/35; LVT 105/105, KVK 163/163 cache hợp lệ |
| Config riêng từng sách | `config/pipeline_LucVanTien1883.yaml`, `config/pipeline_KimVanKieu1884.yaml` (n_columns 10, det_xmargin 0,05, det_thr 0,15) | step0 validation pass |
| Bộ đo offline (0 token) | `scripts/measure/measure.py --all` (layout, OCR QN, detector, Chrestomathie, code_facts) — 118 invariants | idempotent (md5 giống 2 lần chạy) |
| Đo độ đúng tự động | `scripts/measure/auto_precision.py --all` (GT độc lập IHR + dị bản + cổng) | cache kim → chạy lại 0 lượt |
| Sửa ocr_api Guest Mode | `core/ocr/ocr_api.py` + test mock | STT cache 206/206 chữ trùng, bbox lệch 0 |
| Quy ước | `CLAUDE.md`: không đo lại bằng LLM, đọc SUMMARY/REPORT, đọc mã qua `PIPELINE_FACTS.json` | — |

Đã commit thêm 853cadfd9c (auto_precision, PHUONG_AN, báo cáo này). **Chưa commit: toàn bộ vòng 2** (§9) — kế hoạch `docs/KE_HOACH_COMMIT_VONG2_2026-09-22.md`; `kiem_nguoi_*.py` để ngoài (không có người kiểm).

---

## 4. Kết quả chạy thật — vòng 1 (trước B1' và cổng B4'; số chốt vòng 2 ở §9.1)

| Chỉ số | LucVanTien1883 | KimVanKieu1884 | STT (mốc) |
|---|---|---|---|
| Trang / page_ok | 105 / 103 | 163 / 162 | 448 |
| Ô sinh ra | 14.476 | 22.704 | ~82k |
| GOLD thô → cuối | 9.680 → **9.666** | 15.133 → **15.056** | — |
| SILVER | 0 (S3 học chữ thảo, không hiệu chuẩn thạch bản) | 0 | có |
| SYLLABLE / REVIEW / QUARANTINE | 1.683 / 3.111 / 16 | 3.504 / 4.056 / 88 | — |
| Cột M==N (kim = QN) | 83,8 % | **88,8 %** | 93,2 % |
| Cổng I5 detector n_det==N (≥ 75 %) | **65,2 % ❌** | **59,2 % ❌** | 90,1 % |
| F1 cross-col (sau det_xmargin 0,05) | 10 ô (từ 87) | 90 ô | — |
| Export | 11.349 ảnh | 18.560 ảnh | — |
| Thời gian | ingest 8 ph (100 lượt kim) + build 2,4 ph | ingest 3 s (cache) + build 3,2 ph | — |

Điểm kỹ thuật quyết định ở KVK: ghép câu theo số in (`anchor`) sai từ trang 54 (lệch +4/+5 do OCR QN mất 5 dòng) → GOLD 22,6 %;
chuyển `--verse-map content` (ghép chữ kim ↔ dòng QN qua từ điển, quy hoạch động) → GOLD 66,7 %. Bài học đã ghi vào hướng dẫn.

Nguồn: `docs/CHAY_LVT1883_2026-09-21.md`, `docs/CHAY_KVK1884_2026-09-21.md`.

---

## 5. Độ đúng đo tự động (thay người kiểm)

| Phép đo | Kết quả | Ý nghĩa / caveat |
|---|---|---|
| (1) Precision GOLD trên **GT độc lập** IHR-NomDB (200 cột/sách, kim ×3, luật GOLD của pipeline) | LVT1916 **89,3 % [86,9–91,3]**, Kiều1872 **84,6 % [81,7–87,1]**; bỏ GT PUA 90,5 / 86,6 %; coverage 55 / 50 % | Đo *văn bản* trên mộc bản 35 px (kim thô chỉ 42–49 % → từ điển cộng ~40 điểm). Khác miền thạch bản. |
| (2) Khớp dị bản trên **chính thạch bản** | KVK1884↔1871: **76,5 %** (80,4 % bỏ PUA) với **nền 1871↔1872 = 82,9 %** → lỗi văn bản GOLD ≈ 3 %; LVT1883↔1916: 71,3 % | Cận dưới; đo trên 1.898 / 609 câu QN giống hệt |
| (3) Bản chất lỗi | **100 % lỗi GOLD là đồng âm dị thể hoặc GT PUA; 0 ca ngoài từ điển**; ~23 % bất đồng là cặp gần hình → trần lỗi kim ≈ 4,5–6,5 % | Lỗi còn lại là chọn dị thể cùng âm, không phải sai âm |
| (4) Cổng máy | mọi cổng dịch proxy ≤ +1,2 điểm (trong CI) → chọn cổng theo **cơ chế**, không theo proxy | proxy văn bản mù với lỗi hộp |

Nguồn: `measure_out/auto_precision/REPORT.md`, `docs/PHUONG_AN_TU_DONG_2026-09-22.md`.

---

## 6. Ba giới hạn còn lại — và có đáng làm tiếp không

| Giới hạn | Bằng chứng | Hướng | Lợi dự kiến | Đề nghị |
|---|---|---|---|---|
| **A. Hộp chữ (crop) trên thạch bản** — I5 65 / 59 %, không đo tự động được crop đúng chữ | detector lệch ±1 chữ (LVT −1: 142, +1: 135 cột); nền xám làm CenterNet mù (KVK raw 23,5 %) | Cổng cơ chế: `n_det≠N`/midpoint/split → **không export ảnh** GOLD (giữ nhãn văn bản); fine-tune detector cần hộp GT — không có người → chỉ có thể dùng hộp kim làm nhãn yếu | Loại ~35–40 % ô khỏi tầng ảnh, giữ precision văn bản | ✅ **Đã làm vòng 2** (B4', §9.1: loại 40–46 %), ghi rõ trong luận văn là điểm yếu |
| **B. Đồng âm dị thể** — lỗi GOLD còn lại 10–15 % (mộc bản), ≈ 3–6 % (thạch bản) | (3) ở §5 | S3/ArcFace hiện **mù** (AUC 0,57) nên không phân xử được; dùng phiên âm chuẩn dị bản (B1') để chọn dị thể theo tần suất cùng nguồn | +2–4 điểm precision văn bản trên câu có dị bản khớp (≈58 % KVK, 30 % LVT) | ✅ **Đã làm vòng 2** (B1', §9.2): GOLD +5,9 %, khớp dị bản không đổi (+0,2–0,7 điểm, trong CI) — lợi nằm ở PHỦ, không ở precision |
| **C. Không có GT trên chính miền thạch bản** | (1) đo trên mộc bản, (2) là cận dưới | Không có cách nào không cần người; cách trung thực nhất là **trình bày đúng tên** hai số (§5 PHUONG_AN) | — | Chấp nhận, viết rõ |

Ngoài ra: SILVER = 0 trên thạch bản là **kết quả đúng** (S3 học chữ thảo), không "sửa" bằng self-training (vòng tròn — đã bác trước đây).

---

## 7. Việc đề nghị cho vòng 2 — trạng thái sau 22/09

1. ✅ Cổng cơ chế A + tầng `GOLD_text_only` — `pipeline/remediation/mechanism_gates.py` (§9.1).
2. ✅ B1' KVK — `scripts/measure/verses_ref_fix.py` + `--verses`; **chính thức** (§9.2); `--dict-boost` bị loại.
3. ✅ Chrestomathie — `layout: prose` + `ingest_prose_book` (§9.3).
4. Sách thứ tư trở đi: `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` §7 (chọn layout lithograph | prose).
5. Hạn chế biết trước còn nguyên: `step2_align.py` (CLI riêng) vẫn ghim 9 cột; ocr_api MIME jpeg cứng (R-01); `run_pipeline.sh` STT chưa gọi B4'; việc mở thật sự: §9.5.

---

## 8. Chỉ mục tài liệu

| Nội dung | Tệp |
|---|---|
| Hướng dẫn chạy sách mới (lệnh, cổng 3 sách, thêm sách) | `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` |
| Vòng 2: B1' KVK · Chrestomathie · kế hoạch commit | `docs/CHAY_KVK1884_B1_2026-09-22.md`, `docs/CHAY_CHRESTO1872_2026-09-22.md`, `docs/KE_HOACH_COMMIT_VONG2_2026-09-22.md` |
| Phương án tự động không người + độ đúng | `docs/PHUONG_AN_TU_DONG_2026-09-22.md` |
| Kết quả chạy LVT1883 / KVK1884 | `docs/CHAY_LVT1883_2026-09-21.md`, `docs/CHAY_KVK1884_2026-09-21.md` |
| Đặc tả pipeline + đối chiếu số đo | `docs/PIPELINE_SACH_MOI_2026-09-20.md` (§11), `docs/KET_QUA_DO_CUOI_2026-09-21.md` |
| Sự kiện mã | `docs/PIPELINE_FACTS.json` |
| Kế hoạch commit / review diff | `docs/KE_HOACH_COMMIT_2026-09-21.md` |
| Dữ liệu | `data/README.md`, `data/*/SOURCE.md`, `data/KIEM_TRA_KHOP_1-1_2026-09-20.md` |
| Bộ đo | `scripts/measure/` (`measure.py --all`, `auto_precision.py --all`) → `measure_out/` |

---

## 9. Vòng 2 (22/09) — cổng cơ chế B4', B1' KVK, văn xuôi Chrestomathie

Ba luồng chạy song song trên HEAD 853cadfd9c, **chưa commit**. Số đọc từ `dataset_out_<sách>/mechanism_gates_report.json`, `dataset_<sách>/labels.csv`,
`measure_out/auto_precision_b1/{b1,b1_gated}/`, `measure_out/auto_precision/`. Hồi quy STT sau cả ba luồng: 3 trang `labels.csv` byte-identical HEAD vs bản sửa,
md5 **59e436d7641fa849bb6759868ac29259** (556 dòng), 344/346 tệp giống byte (2 tệp chỉ khác đường dẫn repo). Selftest: book_layout 79/79 · ingest_lithograph 61/61 ·
ingest_prose 33/33 · mechanism_gates 62/62 · verses_ref_fix 19/19 · phase1_engine 253/0 · tools 139 pass/3 fail (3 có sẵn ở HEAD) · `code_facts.py` 18/18.

### 9.1 Ba sách sau vòng 2 (labels_final → B4' → export)

| Chỉ số | LucVanTien1883 | KimVanKieu1884 (**B1'**) | Chrestomathie1872 (prose) |
|---|---|---|---|
| Trang / page_ok | 105 / 103 | 163 / 162 | 65 / **65** |
| Ô sinh ra · M==N · **I5 n_det==N** | 14.476 · 83,8 % · **65,2 % ❌** | 22.704 · 88,8 % · **59,2 % ❌** | 8.303 · 69,5 % · **70,7 % ❌** |
| labels_final GOLD / SYL / REVIEW / QUAR | 9.666 / 1.683 / 3.111 / 16 | **15.938** / 3.209 / 3.465 / 92 (không B1': 15.056 / 3.504 / 4.056 / 88) | 5.892 / 998 / 1.413 / 0 |
| B4' hạ: (a) n_det≠N \| midpoint/split → text_only · (b) cầu + sửa dấu → SYL · (c) blank/cụt → REVIEW · (d) dị bản gần hình → REVIEW | 2.985 · 681 · 14 · 194 | 5.784 · 659 · 230 · 696 | 1.557 · 309 · 124 · — |
| **Sau B4': GOLD ảnh / GOLD_text_only / SYLLABLE / REVIEW** | **5.793 / 2.985** / 2.363 / 3.319 | **8.593 / 5.784** / 3.844 / 4.391 | **3.922 / 1.557** / 1.287 / 1.537 |
| Export: ảnh (GOLD + SYL) · dòng labels.csv | **8.156** · 11.141 | **12.437** · 18.221 | **5.209** · 6.766 |
| Khớp dị bản GOLD-ảnh, bỏ ref PUA: trước cổng → (a)(b)(c) → +(d) *(tự khẳng định)* | LVT1916: 72,6 % (n 2.796) → **73,4 %** (n 1.866) → 79,5 % | 1871: 80,6 % (n 11.467) → **81,2 %** [80,2–82,1] (n 6.772) → 85,7 %; **1872 độc lập**: 73,0 % → **74,1 %** [73,0–75,1] (n 6.848) → 78,4 % | không có dị bản số hoá |
| GOLD_text_only khớp dị bản (sau (a)(b)(c)) | 77,9 % (n 709) | 1871 80,2 % (n 4.122) / 1872 72,7 % (n 4.205) | — |
| Thư mục | `dataset_LucVanTien1883/` (v1 không cổng: `_v1/` 11.349 ảnh) | `dataset_KimVanKieu1884/` (không B1' + cổng: `_v2_gates_noB1/` 8.124 + 5.451, 12.344 ảnh; v1: 18.560 ảnh) | `dataset_Chrestomathie1872/` |

Đọc: cổng B4' loại 40–46 % ô GOLD khỏi tầng ảnh (hộp nghi lệch — thứ proxy văn bản mù) mà giữ nhãn văn bản ở `GOLD_text_only`; proxy dị bản sau (a)(b)(c) chỉ nhích
+0,8–1,1 điểm (trong CI) — đúng kỳ vọng, KHÔNG đọc là bằng chứng cổng đúng; số sau (d) là tự khẳng định (cổng dùng chính bất đồng dị bản).

### 9.2 B1' KVK — QN đầu vào từ phiên âm 1871 (chi tiết `docs/CHAY_KVK1884_B1_2026-09-22.md`)

- `verses_ref_fix.py --fuzzy-min 0.9`: 3.251 dòng QN OCR → exact 465 · fuzzy 1.548 (0 dòng đổi số âm) · giữ OCR 1.238; offset ref−seq 0 / +2 / +3 khớp chẩn đoán "1884 thay 4 câu 1069–1072, QN mất 1 dòng ~3045".
- Kết quả: GOLD +882 (+5,9 %, toàn bộ ở dòng fuzzy 66,7 → 74,7 % GOLD), cột không khớp 9 → 6; tỉ lệ khớp dị bản **không đổi** (1871: 80,4 → 80,6 %; **1872 độc lập: 72,3 → 73,0 %**, đều trong CI) → sửa QN không kéo nhãn sai.
- **Caveat**: thay cả dòng theo 1871 làm cột `syllable` mang chính tả **Bắc** thay Nam ở **676 dòng** (sanh→sinh, nhơn→nhân, chơn→chân, đàng→đường…; `qn_source` ghi trong `transcriptions/*.json`, chưa vào export);
  50 âm (2,0 % âm đổi) nghi xoá dị bản QN thật của 1884; 125 dòng lệch số âm không sửa được ở 0,9 (0,8 thêm +376 GOLD nhưng 240 âm nghi xoá → không hạ).
- **`--dict-boost` LOẠI**: đổi chữ nhãn 556 ô GOLD thành chữ 1871 → khớp 1871 = 100 % theo định nghĩa (tự khẳng định); trên 1872: 404 ô thành khớp nhưng 59 ô kim đúng 1872 bị làm sai; nhiều cặp là dị thể mã Unicode (別/别, 內/内).
- Chọn: **B1' fuzzy 0,9, không boost** = đường chính thức (`config/pipeline_KimVanKieu1884_b1.yaml`, khoá ghi nhận `verses_b1` trong config chính); 1872 giữ làm đối chứng độc lập.

### 9.3 Chrestomathie1872 chạy được (chi tiết `docs/CHAY_CHRESTO1872_2026-09-22.md`)

`layout: prose` + `n_columns: auto` (số cột kỳ vọng = số dòng QN của trang, cổng `prose_gate`), adapter `ingest_prose_book` (ô cột từ bộ đo `chresto_map`, kim 1 hộp/cột trên JPG gốc,
DP đơn điệu chuỗi chữ kim cả truyện ↔ âm tiết theo `bang_truyen_trang.csv`): 65/65 trang page_ok, 404/417 cột ghép (LTR thắng 20/20 truyện), 8.303 ô, GOLD 71,0 % trước cổng;
60 lượt kim. Giới hạn: QN tesseract lỗi âm ≈ 13 % + mất dòng → 1.239 ô `no_context` REVIEW; không có dị bản số hoá nên chỉ có cổng (a)(b)(c).

### 9.4 Sự cố

`pipeline.remediation apply` **không có `--out` thì ghi đè `dataset_out/` của STT** (tracked trong git) — xảy ra ở hai luồng B và C, khôi phục ngay bằng `git checkout -- dataset_out`
(`git status -- dataset_out data` trống lúc chốt); HUONG_DAN §2 B4 đã thêm `--out` + cảnh báo; `code_facts.py` thêm invariant `dataset_out_tracked_clean`. Chưa chặn trong mã.

### 9.5 Việc còn mở thật sự

1. **I5 detector ±1 chưa chữa** (65,2 / 59,2 / 70,7 % < 75 %): B4'(a) chỉ chặn ở export (LVT 2.985, KVK 5.784, Chresto 1.557 ô text_only); chữa gốc cần NMS theo bước dọc / fine-tune detector (không có hộp GT).
2. **Blank KVK 230 ô** hạ REVIEW bởi (c) có thể hạ oan (ngưỡng `enrich_crop_quality` hiệu chuẩn trên STT; 40 ô đo có mực trên ảnh gốc).
3. **`run_pipeline.sh` (STT) chưa gọi B4'**; sách mới chạy tay; `remediation apply` chưa chặn thiếu `--out` trong mã.
4. **QN tesseract Chrestomathie lỗi ≈ 13 %** + mất dòng; không có phiên âm chuẩn để B1'.
5. B1': chính tả Bắc ở 676 dòng chưa ghi `qn_source` vào export/DATASHEET; luật theo từng âm với kim trọng tài chưa làm.
6. Không có người kiểm → precision ảnh crop vẫn không đo được; số đo (2) là cận dưới; bộ mẫu mù `kiem_nguoi_*` để ngoài commit.
