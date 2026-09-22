# BÁO CÁO TỔNG THỂ — Pipeline gán nhãn tự động cho dữ liệu mới (ngoài Sách Thánh Truyện)

Ngày 2026-09-22 · Trạng thái: **hoàn tất vòng nghiên cứu 1** cho 2 sách thạch bản (LucVanTien1883, KimVanKieu1884);
Chrestomathie1872 ở mức "đã đo, chờ mô-đun căn chỉnh văn xuôi"; 5 sách chép tay chưa có phiên âm cùng nguồn không chạy được.
Mọi con số dưới đây đều tái lập được bằng script (0 token LLM), nguồn ghi ở cột cuối. Không có người kiểm (ràng buộc đề tài).

---

## 1. Câu trả lời ngắn

| Câu hỏi | Trả lời |
|---|---|
| Pipeline STT có chạy được trên sách mới không? | **Có**, sau 2 thay đổi: engine nhận `layout/n_columns/det_xmargin/det_thr` theo sách (mặc định = STT, hồi quy byte-identical) và adapter `ingest_lithograph_book` dựng `prepared/<sách>/` từ ảnh JPG + kim. |
| Đã chạy đủ chưa? | LVT1883 **105/105** trang, KVK1884 **163/163** trang, xuất 2 dataset (11.349 + 18.560 ảnh). |
| Có biết nhãn đúng bao nhiêu không (không người kiểm)? | **Có, bằng đối chứng độc lập**: precision GOLD 89,3 % / 84,6 % trên GT IHR-NomDB (mộc bản); khớp dị bản 76,5 % trên chính KVK1884 khi nền dị bản chỉ 82,9 %. |
| Đã "tốt nhất" chưa? | Tốt nhất **trong ràng buộc hiện tại** (kim + từ điển + căn chỉnh, không người). Ba giới hạn còn lại được nêu ở §6, kèm hướng và mức lợi dự kiến. |

---

## 2. Dữ liệu đã thẩm định (data/, 13 thư mục)

| Nhóm | Sách | Loại | QN cùng nguồn | Trạng thái |
|---|---|---|---|---|
| 1 — cặp cùng nguồn (đầu vào gán nhãn) | SachThanhTruyen2/4/11 | viết tay thật | ✅ trang đối diện | kho chính, đã chạy từ trước |
| | **LucVanTien1883** | thạch bản, 10 cột × 2 tầng (cột = cặp lục bát 6/8), 2.088 câu | ✅ 139 trang QN in cùng sách | **đã chạy đủ** |
| | **KimVanKieu1884** | thạch bản, canvas đảo (page = 167 − canvas), 3.256 câu | ✅ 295 trang QN xen dịch Pháp | **đã chạy đủ** |
| | Chrestomathie1872 | thạch bản, văn xuôi 20 truyện | ✅ 28 trang QN (mức truyện) | đã đo bố cục + bảng truyện↔trang; chưa có mô-đun căn chỉnh văn xuôi |
| 2 — đối chứng ngoài có phiên âm độc lập | LucVanTien1916, TruyenKieu1872 (IHR-NomDB / Nôm Foundation) | mộc bản, 35 px/chữ | ✅ câu-với-câu, đã đối chiếu 96,6 % / 99,97 % | **dùng làm GT độc lập** để đo precision |
| 3 — chép tay chưa phiên âm | TruyenKieuPhongTinhCoLuc, KimVanKieu1894, TamTuKinhDienAm, CacThanhTruyen1646 | viết tay | ❌ | chỉ chạy được ở chế độ "QN dị bản" (sai ≥ 6 % âm tiết), không đạt đề tài |
| Loại | LyHangCaDao | viết tay, 1000 px | ❌ | loại |

Nguồn: `data/README.md`, `data/KIEM_TRA_KHOP_1-1_2026-09-20.md`, `docs/KET_QUA_DO_CUOI_2026-09-21.md`.

---

## 3. Những gì đã xây (đã commit trên main, 4 commit fb345a29b1 → bd9280e682)

| Thành phần | Tệp | Kiểm chứng |
|---|---|---|
| Engine tham số theo sách | `pipeline/align_engine/book_layout.py`; sửa `align_production.py`, `build_dataset.py`, `step0/1/2` | STT 3 trang **byte-identical** (md5 59e436d7…); selftest 57/57, 253/0, tools 139/3 (3 fail có sẵn ở HEAD) |
| Adapter ingest thạch bản | `pipeline/tools/ingest_lithograph_book.py` (`--ocr kim`, `--contrast otsu`, `--verse-map formula\|anchor\|content`) | selftest 35/35; LVT 105/105, KVK 163/163 cache hợp lệ |
| Config riêng từng sách | `config/pipeline_LucVanTien1883.yaml`, `config/pipeline_KimVanKieu1884.yaml` (n_columns 10, det_xmargin 0,05, det_thr 0,15) | step0 validation pass |
| Bộ đo offline (0 token) | `scripts/measure/measure.py --all` (layout, OCR QN, detector, Chrestomathie, code_facts) — 118 invariants | idempotent (md5 giống 2 lần chạy) |
| Đo độ đúng tự động | `scripts/measure/auto_precision.py --all` (GT độc lập IHR + dị bản + cổng) | cache kim → chạy lại 0 lượt |
| Sửa ocr_api Guest Mode | `core/ocr/ocr_api.py` + test mock | STT cache 206/206 chữ trùng, bbox lệch 0 |
| Quy ước | `CLAUDE.md`: không đo lại bằng LLM, đọc SUMMARY/REPORT, đọc mã qua `PIPELINE_FACTS.json` | — |

Chưa commit: `scripts/measure/auto_precision.py`, `docs/PHUONG_AN_TU_DONG_2026-09-22.md`, báo cáo này; `kiem_nguoi_*.py` (để ngoài — không có người kiểm).

---

## 4. Kết quả chạy thật

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
| **A. Hộp chữ (crop) trên thạch bản** — I5 65 / 59 %, không đo tự động được crop đúng chữ | detector lệch ±1 chữ (LVT −1: 142, +1: 135 cột); nền xám làm CenterNet mù (KVK raw 23,5 %) | Cổng cơ chế: `n_det≠N`/midpoint/split → **không export ảnh** GOLD (giữ nhãn văn bản); fine-tune detector cần hộp GT — không có người → chỉ có thể dùng hộp kim làm nhãn yếu | Loại ~35–40 % ô khỏi tầng ảnh, giữ precision văn bản | **Làm ngay** (chỉ export), ghi rõ trong luận văn là điểm yếu |
| **B. Đồng âm dị thể** — lỗi GOLD còn lại 10–15 % (mộc bản), ≈ 3–6 % (thạch bản) | (3) ở §5 | S3/ArcFace hiện **mù** (AUC 0,57) nên không phân xử được; dùng phiên âm chuẩn dị bản (B1') để chọn dị thể theo tần suất cùng nguồn | +2–4 điểm precision văn bản trên câu có dị bản khớp (≈58 % KVK, 30 % LVT) | Làm ở vòng 2, chi phí thấp (0 API) |
| **C. Không có GT trên chính miền thạch bản** | (1) đo trên mộc bản, (2) là cận dưới | Không có cách nào không cần người; cách trung thực nhất là **trình bày đúng tên** hai số (§5 PHUONG_AN) | — | Chấp nhận, viết rõ |

Ngoài ra: SILVER = 0 trên thạch bản là **kết quả đúng** (S3 học chữ thảo), không "sửa" bằng self-training (vòng tròn — đã bác trước đây).

---

## 7. Việc đề nghị cho vòng 2 (theo thứ tự, ước công)

1. Cổng cơ chế A + export tầng `GOLD_text_only` (1 ngày, không đụng engine lõi).
2. B1' cho KVK: dùng phiên âm 1871 Nôm Foundation làm `verses.tsv` ở 1.898 câu khớp; đo lại (2) (0,5 ngày, 0 API).
3. Chrestomathie: mô-đun căn chỉnh chuỗi QN cả truyện vào cột Nôm (dùng lại lõi `--verse-map content`), rồi chạy B0→B5 (2–3 ngày).
4. Sách thứ tư trở đi theo `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` §7: đo (measure.py) → adapter → config → B0→B6.
5. Hạn chế biết trước: `step2_align.py` (đường CLI riêng) vẫn ghim 9 cột; ocr_api MIME jpeg cứng (R-01); `content` chưa có selftest riêng.

---

## 8. Chỉ mục tài liệu

| Nội dung | Tệp |
|---|---|
| Hướng dẫn chạy sách mới (lệnh, cổng, thêm sách) | `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` |
| Phương án tự động không người + độ đúng | `docs/PHUONG_AN_TU_DONG_2026-09-22.md` |
| Kết quả chạy LVT1883 / KVK1884 | `docs/CHAY_LVT1883_2026-09-21.md`, `docs/CHAY_KVK1884_2026-09-21.md` |
| Đặc tả pipeline + đối chiếu số đo | `docs/PIPELINE_SACH_MOI_2026-09-20.md` (§11), `docs/KET_QUA_DO_CUOI_2026-09-21.md` |
| Sự kiện mã | `docs/PIPELINE_FACTS.json` |
| Kế hoạch commit / review diff | `docs/KE_HOACH_COMMIT_2026-09-21.md` |
| Dữ liệu | `data/README.md`, `data/*/SOURCE.md`, `data/KIEM_TRA_KHOP_1-1_2026-09-20.md` |
| Bộ đo | `scripts/measure/` (`measure.py --all`, `auto_precision.py --all`) → `measure_out/` |
