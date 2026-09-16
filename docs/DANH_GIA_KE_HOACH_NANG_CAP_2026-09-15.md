# ĐÁNH GIÁ KẾ HOẠCH `KE_HOACH_NANG_CAP_GAN_NHAN_FINAL.md` (v3.0) — VÀ HƯỚNG CHO KẾT QUẢ TỐT NHẤT

**2026-09-15** · Đối tượng: `docs/KE_HOACH_NANG_CAP_GAN_NHAN_FINAL.md` (lập 14/09, tự nhận là "tài liệu thiết
kế tối hậu, tích hợp toàn bộ nghiên cứu"). Phương pháp: 11 lượt kiểm chứng độc lập (mỗi bước của kế hoạch
một lượt), mỗi phát hiện mức chặn/lớn bị 2 lượt phản biện theo hai lăng kính (sự thật mã/số; tác động
thật), rồi 3 đề xuất hướng đi từ 3 góc nhìn (rủi ro / giá trị / luận văn) và một lượt phê bình hoàn chỉnh —
tổng 181 lượt, 3.386 lần mở mã hoặc chạy đo. Mọi con số dưới đây hoặc trích `file:line`, hoặc đo lại được
bằng lệnh ghi kèm. Số mới không có trong `thuc_nghiem.py` tái sinh bằng:

```
.venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem_ke_hoach.py all   # ~30 s: luật tier của kế hoạch · T posterior · ma trận 5,1/4,5
.venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem.py calib|recursive|v3|geo
```

---

## 0. KẾT LUẬN NGẮN

**Kế hoạch đúng ở "lõi văn bản" (Bước 3 + 5.1 về ý tưởng, 1.2, 6.2-3) nhưng sai ở số, sai ở luật, và ba
trong sáu bước (2.1, 2.2, 4.2) sẽ làm dữ liệu xấu đi hoặc không làm gì.** Nếu thực thi nguyên văn:

| # | Điều sẽ xảy ra | Số đo |
|---|---|---|
| 1 | Luật tier §5.1 **không phải** luật đã mô phỏng ra "70.108 ô": nó cho **82.206 ô "usable" (99,3% bộ)**, REVIEW còn 885, trong đó 12.202 ô SYLLABLE không có bằng chứng ngữ cảnh, ghép sai 0,7–2,7% (gấp 2–3 lần SYL v3) | `thuc_nghiem_ke_hoach.py tier` |
| 2 | Ngưỡng P<0,5 / P≥0,8 **vô nghĩa** với ma trận hiệu chuẩn: 99,1–99,8% cặp có p≥0,99 ở cả T=0,35 lẫn T=1,0 | `thuc_nghiem_ke_hoach.py T` |
| 3 | **QĐ-01 (2.014 ô GOLD duy nhất có người ký) mất một phần theo 4 đường**: khoá (người,㝵) chỉ phủ 1.183 ô; tắt `--use-s3` mất tới 552 ô (crop chỉ tồn tại nhờ tier SILVER lúc build); thay `glyph_fix` bằng luật lớp mà không giữ `confusion_fix` → 1.178 ô giữ nhãn 㝵 ở CHAR_A; đổi hộp → 675 crop người chưa nhìn | §2.6, §5 |
| 4 | Bước 2.2 "khối tâm mực" **mù** với đúng lỗi nó muốn sửa (0/300 ô lệch >0,25 pitch; tương quan với độ lệch thật 0,025); hướng C chưa bao giờ thử COM — nó chọn F3g (hình chiếu mực) | §2.3 |
| 5 | Bước 2.1 luật "bổ đôi chỉ khi cao >1,6 pitch" kích hoạt ở **0,2% hộp** → 1.443 cột M<N (35,8% cột, 23.138 ô usable) rơi **100% hộp midpoint** (hiện 30–58%) | §2.2 |
| 6 | Bước 4.1 luật gán hộp viết sai: nguyên văn cho **47,5%** đúng hộp (< production 59,1%); luật đã đo 91% là `hộp[j] ↔ âm j` | §2.5 |
| 7 | Bước 4.2 gần như **thừa** (carve/tighten/measure/cờ đã chạy sản xuất từ `d2d442f8ba`, `labels.csv` đã có 3 cột) hoặc **có hại** (`resolve_overlap` đã đo xấu đi, T4-A) | §2.5 |
| 8 | Bảng nghiệm thu: **1/8 dòng đúng**; 3 dòng suy từ cấu hình khác cấu hình đặc tả; 2 dòng không đo được; lệnh selftest chạy **0 test** | §3, §4 |
| 9 | Bỏ toàn bộ tài liệu 14/09 (H1 ảnh làm trọng tài thanh ghi AUC 0,946; H2 gom cụm theo (sách, âm); H3 nhãn hai tầng) và 4/5 P0 của rà soát 13/09 | §6 |

**Cái đúng và nên giữ (đã tái lập):** ma trận chi phí hiệu chuẩn (khe giả 345→54 cột), posterior
forward-backward, luật tier_v3 *có ngữ cảnh* (64.525→70.108 ô, CHAR_A ghép sai 0,5%), decisions.yaml
tập trung, BUG-1, tự động hoá chuỗi bằng chứng. Hướng tối ưu ở §7.

---

## 1. NHỮNG GÌ TÔI KIỂM VÀ CÁCH KIỂM

| Lượt | Mục kế hoạch | Cách kiểm chính |
|---|---|---|
| B1 | 1.1 loanword, 1.2 BUG-1 | đọc `core/text/loanword.py`, `config/lexicon/*.json`, `step2_align.py`; đếm cột OCR>QN có token OOV trên `cols.pkl`; mô phỏng fix BUG-1 trên `stt4/page_0146` |
| B2a | 2.1 enforce_count | chạy detector sản xuất (`detector_r34.best.pt`, epoch 38) trên 448 trang; quét ngưỡng điểm; giả lập tiêm lỗi 2.030 cột |
| B2b | 2.2 tái định tâm | đọc `hoan_tat_4_huong.json[0]`; đo COM vs tâm hộp vs hình chiếu trên 300 ô; mô phỏng 1D |
| B3 | 3.1–3.2 ma trận, posterior | `thuc_nghiem.py calib/recursive`; phân bố p theo (ma trận × T); benchmark AUC |
| B4a | 4.1 hộp theo khe | `thuc_nghiem.py geo` + kịch bản INS/DEL với 3 luật gán hộp |
| B4b | 4.2 crop_quality | chạy `resolve_column` thật trên 40 trang; tái lập md5 crop; đọc `lab/t4a_grid.csv` |
| B5a | 5.1 tier v3 | monkeypatch `tier_v3` → luật kế hoạch; crosstab + benchmark 600 cột × 5 loại hỏng × 3 cấu hình |
| B5b | 5.2 chốt L1 | join 755 ô L1 với thế hệ `2f0b9dc116`; phân loại bằng 2-gram ngoài ô; mô phỏng chốt |
| B6 | 6.1–6.2 decisions/runner | đọc `run_pipeline.sh`, `remediation/*`; đếm QĐ-01 theo rule gốc; đo thời gian S3 |
| ACC | bảng nghiệm thu + verification | chạy từng lệnh nghiệm thu của kế hoạch; đọc `re-dataset/check/labels.xlsx` |
| OMIT | bỏ sót, trình tự | đối chiếu 3 tài liệu 13–14/09 |

Phản biện: 83 phát hiện chặn/lớn được 166 lượt bác bỏ thử — **0 bị bác**, 5 giữ mức chặn, 26 mức lớn,
còn lại hạ mức vì tác động nhỏ hoặc có cách vá một dòng. Lượt phê bình cuối tự đo thêm 7 điểm không ai
kiểm (§5).

---

## 2. CHẤM TỪNG BƯỚC CỦA KẾ HOẠCH

### 2.1 Bước 1 — tiền xử lý văn bản

| Mục | Kế hoạch nói | Kiểm | Bằng chứng |
|---|---|---|---|
| 1.1 | `loanword.py` + `loan_phrases.json` bóc "ôliva" → [ô, li, va] | **sai công cụ** | `core/text/loanword.py` chỉ có `find_loan_spans` (đánh dấu chỉ số các âm *đã tách rời* khớp cụm) và `should_demote_loan_syllable` (hạ cấp) — **không có hàm tách**; `loan_phrases.json` 15 cụm dạng "ma ri a" bỏ dấu, không có "ôliva"; module chỉ được import bởi `pipeline/step3_label.py` (đã nghỉ) |
| 1.1 | từ mượn làm "toàn bộ cột lệch số" | **quá tay** | "ôliva" xuất hiện **1** lần trong toàn ngữ liệu, "mari-a" 1, "an-ti-ô-ki-a" 0. Lớp OCR>QN 352 cột: chỉ **45** có token ≥5 ký tự ngoài từ điển (9 là BUG-1, 36 từ dính thật — phần lớn VietOCR dính hai từ Việt thường: "Songle", "mẹnghe"); 307 cột còn lại không liên quan từ mượn |
| 1.1 | (thay thế) | cách P1 của rà soát 13/09 (tách token OOV theo âm từ điển, số mảnh = gap+1) **có đáp án kiểm được**: 17/36 cột tách duy nhất (348 ô), 7 mơ hồ, 12 không bù đủ. Lợi ích ≈100–200 ô usable (≤0,3%) — giá trị là đưa cột về lớp khớp số để dùng hộp detector, không phải tăng số |
| 1.2 | BUG-1 fallback trả nguyên câu | **đúng** | `step2_align.py:78-88`; chỉ **1/448** trang rơi fallback (`stt4/page_0146`); fix → 8/9 cột khớp đếm (185 âm / 186 chữ), ≈145 ô usable. Phải sửa **cả hai** `return` (dòng 84 và 88) hoặc gọi `core/pdf/pdf_parser.build_transcription_columns:178` đã có sẵn |

### 2.2 Bước 2.1 — enforce_count "Pitch DP"

Mô tả mã đúng (`train_crop/infer_centernet.py:159-179`: M>N giữ top-N điểm, M<N bổ đôi hộp cao nhất;
N = `len(syllables)` từ `align_production.py:398`). Nhưng nhắm **sai nhánh** và bỏ **nguyên nhân gốc**:

- Đo lại 4.029 cột bằng đúng checkpoint sản xuất: nhánh **M>N chỉ 312 cột (7,7%)**; nhánh **M<N 1.443 cột
  (35,8%; 23.138 ô usable, 19.375 GOLD)**. M<N gần như hoàn toàn do **ngưỡng điểm `thr=0.3`**
  (`align_production.py:252`, đặt từ `a6e1affe47`): trên 583 cột OCR=QN, thr 0,3 → M==N 69,1% / M<N
  26,9%; **thr 0,2 → 92,1% / 0,9%**; hộp điểm 0,2–0,3 nằm đúng lưới 99,2% (khe lệch >0,25 pitch 0,8% so
  với 0,2% đối chứng) — tức glyph thật bị vứt *trước* khi `enforce_count` chạy.
- Pitch DP có cơ sở (lưới in rất đều: 0,3% khe lệch >0,25 pitch trên 40.506 khe; giả lập "hộp giả điểm cao
  đè hai chữ + hộp thật mờ": top-score đúng 0%, Pitch DP 100%) nhưng: (i) **mù với hộp cột bên cùng y**
  (65,6%; thêm phạt điểm λ=0,5 tụt còn 11%); (ii) với **N sai** (lớp OCR>QN: detector đếm = OCR ở 179/216
  cột, 92% hộp bị top-N bỏ là glyph thật) DP chỉ đổi *chỗ* bỏ hộp (đầu/cuối thay vì giữa), không sửa được;
  (iii) luật "**bổ đôi chỉ khi cao >1,6 pitch**" kích hoạt ở **0,2% hộp** (>1,3 pitch: 3,5%) → cột M<N trả
  <N hộp → `_pick_reseg:401` rơi **toàn cột về midpoint**: ~28,8 nghìn ô đang 30–58% midpoint sẽ thành 100%.
- Không hoà giải ba nguồn đếm (P1 rà soát 13/09; RA_SOAT_GAN_NHAN §7). Trên cột lệch, detector đứng về
  phía OCR 144 lần vs QN 59, "không ai" 8,5% cột (thr 0,2).
- "Nguồn đo kiểm `thuc_nghiem.py geo`" **không đo enforce_count**: `cmd_geo` gọi thẳng `_nms_vertical`,
  bỏ mọi cột `len(col) != len(chars)` (37% cột bị loại), giả lập rụng chữ OCR khi M_det = N.

Ước lượng hộp nội suy (x1,x2 trùng trong cột) trên `labels.csv`: OCR=QN&M=N **15,3%**, OCR=QN&M<N **58,1%**,
OCR>QN **99,9%**. Cột ví dụ `stt11/page_0136 c9` của rà soát: khe/pitch 0,91–1,02 trừ *một* khe 2,84 (2 chữ
thiếu), điểm giảm dần 0,53→0,31 xuống cuối cột — "lệch 40%" là **hộp thiếu do ngưỡng**, không phải chữ cao
thấp không đều.

### 2.3 Bước 2.2 — "tái định tâm theo khối tâm mực"

| Kế hoạch | Sự thật |
|---|---|
| Lấy số "tâm hộp lệch tâm mực >0,25 pitch ở 34–38% ô" của hướng C | đúng số |
| Sửa bằng khối tâm mực COM `y_c = Σy·(255−gray)/Σ(255−gray)` | **hướng C không hề thử COM** ("khối tâm/trọng tâm/centroid": 0 lần trong JSON). Hướng C chọn **F3g**: cặp ranh giới theo *hình chiếu mực*, h ∈ [0,95;1,10] pitch, tiên nghiệm tâm, **bảo hiểm** \|lệch\|>0,40p → giữ cửa sổ cũ, seam cong ±0,10p, tighten |
| — | **Đo mới, 300 ô GOLD-direct / 60 trang**: \|COM − tâm hộp\|/pitch trong cửa sổ sản xuất 1,49p: trung vị **0,044**, p90 0,117, **0/300 ô > 0,25p**; tương quan với độ lệch theo hình chiếu **0,025**. Máy ước lượng hình chiếu (dựng lại theo `chi_tiet[3]`) cho >0,25p ở 26,3% ô — tức COM **mù với đúng lỗi 2.2 muốn sửa**. Mô phỏng 1D: mực hàng xóm hai phía triệt tiêu nhau, \|COM − tâm hộp\| ≤ 0,037p dù hộp lệch tới 0,40p; COM lặp thì trôi về điểm giả (50% >0,25p, tương quan 0,02) |
| Đặt ở `align_production.py` trước khi chốt bbox | **sai tầng**: đó là PASS 1 → phải chạy lại detector + DP + mất QĐ-01; bbox trong `labels.csv` tái lập `image_md5` **40/40** (và 854/868 ở lượt khác) → chỉ cần PASS 2 (`save_crop`), đúng khuyến cáo (d) của hướng C |
| Dùng `crop_quality_flag` làm cổng REVIEW (5.1-5) | hướng C đã đo: sau tái định tâm cờ "ok" 90,8→98,2%, stray p95 0,188→0,001 — "ba ngưỡng gần như hết phân biệt". Hơn nữa cờ được tính **sau build** bởi `pipeline/tools/enrich_crop_quality` → `decide_label` không đọc được |
| Không nói hệ quả | hướng C: **cứu 0 ô**, đổi **~92% md5** → CHECKSUMS, labels.xlsx, dedup-by-md5 của remediation, mẻ chấm, tất cả đổi theo. Mã F3g gốc (`/tmp/huong_c`) đã mất, phải dựng lại |

### 2.4 Bước 3 — ma trận chi phí + posterior

| Mục | Kế hoạch | Kiểm |
|---|---|---|
| 3.1 hằng số | DICTMISS **5,1**, NODICT **4,5**, khe 8,6 | CALIB đã đo (`thuc_nghiem.py:55`, RA_SOAT_GAN_NHAN §2 N1) là **DICTMISS 6,7 / NODICT 5,1**: 5,1 là chi phí NODICT bị gán nhầm cho DICTMISS, 4,5 **không có nguồn**. Thứ tự vẫn đúng nên không hỏng — đo được: khe giả PROD 346 / CALIB 54 / kế hoạch **41** cột; lệch chéo 1,63% / 0,82% / 0,68%; benchmark drop_char 4,67 / 3,82 / 3,96%; đọc nhầm 2,09 / 1,15 / 0,75%. Nhưng số nghiệm thu "≤55" không tái lập bằng cùng cấu hình, và ma trận này *không phải* "theo log-likelihood thực nghiệm" như kế hoạch ghi |
| 3.2 T | `T=0.35` với ma trận hiệu chuẩn | **tổ hợp `thuc_nghiem.py` không bao giờ dùng** (PROD→T=0,35 dòng 128; CALIB→T=1,0 dòng 583). Chi phí CALIB lớn gấp 6–12× nên posterior **bão hoà**: 300 cột thật, CALIB T=0,35 → p≥0,99 ở **99,8%** cặp; T=1,0 → 99,1%; T=3,0 mới còn 19,1%. Ngưỡng 0,5/0,8 của §5.1 chỉ còn bắt cặp hoà. Benchmark drop_char: CALIB T=1,0 AUC 0,954 nhưng p<0,5 bắt **57%** cặp sai (oan 0,7%) so với PROD T=0,35 85% (oan 18,8%) — điểm vận hành khác hẳn bảng §4.1 mà kế hoạch mượn ngưỡng. Bin hiệu chuẩn tốt nhất là CALIB T=1,0 ([0,5;0,8) đúng 62%, [0,8;0,95) 94%, ≥0,95 100%) |
| 3.3 đệ quy | chỉ có trong sơ đồ, **không có đặc tả** | mọi số kỳ vọng (khe đúng 85%, ghép sai 1,42%, 70.108 ô) đo **với** đệ quy. Không đệ quy: 64% / 3,14% / 69.566. Đệ quy đóng góp nhiều hơn cả ma trận (production+neo 82%/1,70% > hiệu chuẩn không neo 64%/3,14%). Chi phí neo 2,0 nat thay vì 0 tốt hơn một chút (khe 86%, ghép sai 1,36%, không hỏng 0,52%) |
| nguồn đo | `thuc_nghiem.py posterior` | **không đo được**: hard-code PROD (dòng 213) và đọc ops cache `cols.pkl` dựng với PROD (dòng 75) → phải `rebuild` sau khi sửa `anchor_align.py` |
| hồi quy | không nêu | đổi ma trận đổi tập cặp: **439 GOLD, 506 REVIEW, 54 SILVER, 42 SYLLABLE đổi âm ghép** → đổi `anchored`/`anchors`/độ thuần L5/bể QĐ-01. `phase1_engine_selftest.py` có **0 test** cho `realign_column` |

### 2.5 Bước 4 — hộp theo khe + crop_quality

**4.1** — luật gán hộp viết **sai**. Production ép detector về N = len(syllables) hộp
(`align_production.py:398-400` → `column_boxes` → `enforce_count`), rồi gán theo chỉ số **chữ OCR i** qua
`_monotone_assign` (`:461`); m > N thì cả cột midpoint (`:339-340, :405`). Đo trên 171 cột detector đếm đúng,
rụng 1 chữ OCR nhân tạo (3.343 ô):

| luật | đúng hộp |
|---|---|
| production `_monotone_assign` | 59,1% (11,2% hộp chữ khác, 29,7% midpoint) |
| **luật kế hoạch nguyên văn** ("ins không tiêu thụ hộp") | **47,5%** |
| `hộp[j] ↔ âm j` (chỉ số ÂM QN) — luật `cmd_geo` đo 91% | 90,8% (PROD) · **96,9–97,2%** (CALIB / ma trận kế hoạch) |

Kịch bản ngược (VietOCR rụng 1 âm, m > N, 3.330 ô): production 100% midpoint; `hộp[j]` trên hộp ép N 65,3%;
luật kế hoạch trên hộp ép N 61,5%; luật kế hoạch trên hộp **thô** (|G| = m, không ép) **100%** → hộp phải là
detector *không ràng buộc* + hoà giải đếm, đúng như P1 rà soát 13/09; 2.1 (ép về |Q|) và 4.1 (duyệt theo chữ
OCR) mâu thuẫn nhau. "≥91% khi M≠N" trong bảng nghiệm thu suy rộng từ tập 171/216 cột (37% cột bị loại).
Kế hoạch không ghi `box_source/n_ocr/n_qn/n_det`, không có `no_glyph_box` (grep = 0).

**4.2** — phần lớn **thừa hoặc sai nguyên nhân**:
- `carve_neighbor_ink` + `tighten_box` + `measure` → `stray_ink/border_ink/crop_quality_flag` **đã chạy sản
  xuất** (cột 13, 23–25 của `labels.csv`; từ `d2d442f8ba`). Trên 64.525 ô usable: ok 92,9% · bleed 6,34% ·
  truncated 0,68% · blank 0,06% — "~11% dính / ~6% cụt" của kế hoạch **không có nguồn**.
- `resolve_overlap`/`resolve_column`: mã chết. Chạy thật 40 trang / 5.811 ô: 567 hộp đổi (9,8%); cờ ok
  90,47→**90,21%**, truncated 2,70→2,74% (xấu đi nhẹ). `lab/t4a_grid.csv`: resolve=True flag_ok 0,9157 <
  0,9201; `lab/t4a_decision.json`: "GIỮ MỐC, candidate resolve=false".
- "84,5% cặp đè nhau": thật ra 96,3% cặp overlap>0, trung vị overlap/min(h) 0,175 — **do thiết kế** cửa sổ
  1,49×pitch (`BOX_OVERLAP_FRAC`, `align_production.py:137-147`); Spearman overlap↔stray_ink **−0,009**;
  bleed ở ô overlap>0,25: 5,92% vs ≤0,25: 6,49% → chồng lấn không phải nguyên nhân dính.
- "carve bị chặn ở [oy1,oy2]": đúng theo mã nhưng là chủ ý (chỉ dọn phạm vi hàng xóm).
- Chỉ tiêu "≤1,5% đo bằng `crop_quality.measure`": cờ này **không bắt lớp lỗi crop sai glyph** (RA_SOAT_TOAN_BO
  §3.1) và hướng C đã đo nó hết phân biệt sau tái định tâm → chỉ tiêu **tự thoả**.

### 2.6 Bước 5 — phân tầng v3 và chốt L1

**5.1** — luật của kế hoạch **khác 4 chỗ** với `tier_v3` (`thuc_nghiem.py:541-551`) đã sinh ra mọi số kỳ vọng:

| | tier_v3 đã mô phỏng | kế hoạch §5.1 |
|---|---|---|
| direct ∧ p<0,8 | CHAR_B | **bỏ ngỏ** (0,5≤p<0,8 không định nghĩa; p<0,5 → REVIEW) |
| CHAR_B cầu | sim_unique ∧ (bigram ∨ corpus2 ∨ ~thanh) ∧ p≥0,8 | bỏ ~thanh |
| SYL | p≥0,5 ∧ **(bigram ∨ corpus4 ∨ ~thanh)** | p≥0,8 ∧ ¬direct, **không cần ngữ cảnh** |
| posterior | PROD T=0,35 hoặc CALIB+neo T=1,0 | CALIB T=0,35 (§3.2) |

Kết quả trên dữ liệu thật (cùng cấu hình CALIB + neo ≥2 trang khác, T=1,0 — cấu hình sinh ra 70.108):

| tier | tier_v3 (tài liệu) | **kế hoạch §5.1** |
|---|---|---|
| CHAR_A | 47.482 | 47.482 |
| CHAR_B | 3.338 | 3.216 |
| SYL | 19.288 | **31.508** (REVIEW→SYL 11.097 thay vì 4.242; SILVER→SYL 6.218 thay vì 2.415) |
| REVIEW | 12.983 | **885** |
| dùng được | 70.108 | **82.206** (99,3% bộ) |
| GOLD hiện hành → REVIEW | 1.304 | 136 |

Trong 31.508 SYL của kế hoạch, **12.202 ô không có bằng chứng ngữ cảnh**; benchmark 600 cột (CALIB+neo T=1,0,
none/drop_char/drop_syl): SYL-không-ngữ-cảnh ghép sai **0,70 / 2,68 / 1,40%** so với SYL v3 0,88 / 0,86 /
0,90%; REVIEW của kế hoạch (n≈30–200) sai 50–87% — tức kế hoạch chỉ giữ lại ở REVIEW những ca hiển nhiên
và thăng mọi thứ còn lại. Dưới PROD T=0,35 thì ngược lại: chỉ 50.890 ô dùng được (p≥0,8 chỉ 62% cặp) → luật
**hoàn toàn phụ thuộc T chưa hiệu chuẩn**, từ 50,9 đến 82,3 nghìn ô. Con số "68.500–71.000" và "CHAR_A ≥99%"
chỉ đúng với tier_v3 *có ngữ cảnh* (tái lập chính xác 70.108).

Thêm: CHAR_A của kế hoạch có |R|>30 ở **12.657/47.482** ô (26,7%); 1.066/1.453 ô GOLD "thiểu số nhìn giống đa
số" là `s1_inter_s2_direct` → vào CHAR_A "≥99%" nguyên vẹn. `corpus2` phải loại-một-trang (kế hoạch không
ghi): không LOO thì +3.292 cặp, neo +12%. `decide_label` chạy **trong PASS 1 theo trang**
(`build_dataset.py:433`), còn 2-gram/cặp lặp cần thống kê toàn ngữ liệu (`:450` đã ghi lý do L1/L3/L5 đặt sau
PASS 1) → §5.1 viết sai tầng, phải là PASS 1b.

**5.2** — chốt L1 **sai về bản chất và có hại**:
- 755 ô L1 (`rule = s1_inter_s2_direct_am_sua_dau`), 729 có âm gốc trong từ điển — đúng. Nhưng phân loại bằng
  **2-gram ngoài ô** (đếm trên thế hệ `2f0b9dc116` chưa bị L1 sửa): **498 ô ủng hộ âm mới (sửa đúng)**, **161
  ủng hộ âm gốc (ghi đè: vồ 68, nhiều 61)**, 70 hoà. "729 ô sửa sai" thật ra ≈161 (22%).
- Chốt "âm gốc ∈ từ điển → giữ nguyên" chặn **cả 498 ca sửa đúng** (chứa/lê/quý/lăm… đều là từ có thật). Mô
  phỏng với mã hiện hành: 729 ô → 381 SYLLABLE mang âm gốc (**218 trong đó là âm VietOCR đọc sai**), 348 rớt;
  usable **−430**. Chỉ tiêu "0 ô sửa sai" đạt được bằng cách đưa 218 ô âm sai vào tầng "chỉ nhãn âm".
- "vít vồ → viết vô" là **hai cơ chế**: "vít→viết" do bảng tên thánh `core/text/text_utils.py:138`
  (`"vít vồ": "viết vồ"`; 400 lần "vít-vồ" trong transcriptions → 0 ô "vít", 158 ô "viết"), chỉ "vồ→vô" (無,
  68–70 ô) mới là L1 — chốt L1 không chữa được nửa đầu.
- `check_consistency.sh` (46 dòng, 4 phép) **không có phép nào** về L1/âm tiết → "nguồn đo kiểm" sai.
- Đúng cách: cột `syllable_raw` giữ nguyên bản in; L1 chỉ đổi khi 2-gram ngoài ô ủng hộ; 161 ca ghi đè →
  `corpus_readings` (僥→nhiều, 無→vồ…) có người duyệt.

### 2.7 Bước 6 — decisions.yaml + run_pipeline.sh

- **Mâu thuẫn nội tại**: 6.1 nói yaml thay `confusion_fix + glyph_fix + s3_unwind`; 6.2 chỉ bỏ `s3_unwind` +
  `--use-s3`. Nếu bỏ `glyph_fix` mà `decide_label` chưa nạp yaml → QĐ-01 mất; nếu giữ cả hai → áp hai lần.
- Khoá QĐ-01 của kế hoạch `(người, 㝵)` chỉ phủ **1.183/2.014** ô (còn 早 223, 昇 193, 身 126, 𭔿 79, 旱 53,
  尋 20, 子 17). Phán quyết người là **theo ô đã nhìn** (`docs/NGUOI_CHAM_QUYET_DINH.md`: "toàn bộ 2.014 ảnh
  crop", không áp cho ô không ảnh) → khoá đúng ngữ nghĩa là theo ô, hoặc luật *âm="người" ∧ có ảnh ∧
  label≠𠊚 ∧ ∉12 ô `khong_dung_cho`* (đếm ra đúng 2.014).
- **S3 không phải "0 ô giao nộp" theo nghĩa gián tiếp**: 552 ô QĐ-01 ở tier SILVER lúc build (rule
  `s2_inter_s3_corrected` 547 + `s3_head_bank_consensus` 5); crop chỉ cắt cho {GOLD, SILVER, SYLLABLE}
  (`build_dataset.py:553`), `glyph_fix.py:91-95` bỏ ô không ảnh → tắt `--use-s3` mà không `--crop-review` thì
  QĐ-01 ≤ 1.462.
- "Tiết kiệm ~35%, 45–60 → 25–35 phút": **không có nguồn**. Log: build→build ngắn nhất 10,3 phút; S3 đo mới
  16 ms/ô ấm → 33.309 ô ≈ **5,5–9 phút**; `s3_unwind` ≈ 1 s.
- `'德 = 德'` là lỗi gõ (cùng U+5FB7); phải là 徳 U+5FB3 = 德 U+5FB7. Bốn cặp dị thể đều ∈ R(âm); 爲/為 không có
  trong SinoNom_Similar.
- yaml **toàn cục sai đơn vị**: (thì) stt2 寺 261/時 17, stt4 時 375/寺 136, stt11 時 609/寺 14 — đúng như E3.
  `(cùng, 其) → REVIEW`: E3 đo (共/其) là **một hình** 158 ô → phải gộp về 共, không phải hạ REVIEW.
- Không nhắc `remediate` (quarantine trùng crop, ép md5 về 1 split) — vẫn bắt buộc.
- Chuỗi bằng chứng: `check_consistency.sh` hôm nay **2/4 hỏng**; `labels_final.csv` sha `2de85198…`
  (`bff2a3bd28`, 64.525 ô) **chưa bao giờ được `run_pipeline.sh` sinh ra và ghi bằng chứng** — mục cuối
  `EVIDENCE_INDEX.md:826` và chú thích `run_pipeline.sh:667` đều nói về `550a9726…` = thế hệ 59.371.

---

## 3. BẢNG NGHIỆM THU CỦA KẾ HOẠCH — TỪNG DÒNG

| Dòng | Kế hoạch | Thực tế |
|---|---|---|
| khe giả 345 → ≤55, đo bằng `posterior` | một phần | số in ở `calib`, không phải `posterior`; CALIB 54, ma trận kế hoạch 41; cần `rebuild` cols.pkl sau khi sửa engine |
| AUC ≥0,88 | một phần | 0,88 đo với PROD/T=0,35; CALIB T=1,0 cho 0,954 nhưng ngưỡng tuyệt đối khác hẳn (bắt 57% ở p<0,5) |
| hộp đúng 59,1% → ≥91% khi M≠N | một phần | 91% chỉ trên cột detector đếm đúng (171/216), luật `hộp[j]`; luật kế hoạch cho 47,5% |
| L1 sửa sai 729 → 0, đo bằng `check_consistency.sh` | sai | 729 ≠ số sai (≈161); script không có phép này; chốt làm mất 430 ô và giữ 218 âm sai |
| crop dính/cụt ~11%/~6% → ≤1,5% | sai | hiện 6,3%/0,7%; `resolve_overlap` xấu đi; cờ không đo cái cần đo |
| usable 64.525 → 68.500–71.000 | sai | luật kế hoạch cho 82.206 (hoặc 50.890 tuỳ T); 70.108 là của luật khác |
| GOLD 93–96% → ≥98% bằng "bộ kiểm độc lập" | **không đo được** | `re-dataset/check` là heuristic từ điển ngoài, không đọc ảnh; GOLD TRUE 92,1% / FALSE 6,0%; chấm 𠊚/người **2.051/2.051 TRUE** vì khớp từ điển → mù với chính lớp lỗi cần kiểm. Không có mẻ chấm người thì không có precision |
| 45–60 → 25–35 phút | không nguồn | xem §2.7 |

Chỉ dòng "64.525 hiện hành" là đúng tuyệt đối — và chính bộ ấy chưa có chuỗi bằng chứng (§2.7).

---

## 4. PHẦN "VERIFICATION PLAN"

- `.venv/bin/python -m unittest pipeline/phase1_engine_selftest.py` → **Ran 0 tests, exit 5**. Lệnh đúng:
  `-m pipeline.phase1_engine_selftest` (62 pass) hoặc `bash scripts/run_all_selftests.sh` (722 pass, ~6 phút).
  Cả hai đều **không có test** cho `realign_column`, `apply_am_sua_dau`, `apply_cot_lech_cau_xuoi`, `_pair_new`,
  `enforce_count` (BUG-8) → selftest xanh không chứng minh gì về thay đổi của kế hoạch.
- `thuc_nghiem.py posterior/geo`: đọc cache `cols.pkl` dựng với engine cũ; `posterior` hard-code PROD.
- `build_dataset` **không có cờ chọn sách** (`--limit` là số trang) → "chạy thử SachThanhTruyen11" cần yaml tạm.
- `labels.csv` chưa có cột `p_posterior`; kế hoạch không nói tier mới đi vào `USABLE_TIERS` (định nghĩa
  **5 lần**, 11 mô-đun ghi cứng "GOLD") như thế nào.
- Thiếu hẳn: crosstab thế hệ trước/sau theo (tier × rule), assert QĐ-01 = 2.014, kiểm split không rò, đếm
  md5 crop đổi, mốc thời gian thật.

---

## 5. BẢY ĐIỀU KHÔNG AI KIỂM — LƯỢT PHÊ BÌNH CUỐI ĐO THÊM

1. **Không có mốc để so.** Bộ 64.525 ô (sha `2de85198…`) chưa từng được `run_pipeline.sh` tái lập tại HEAD
   (`git diff bff2a3bd28 HEAD -- run_pipeline.sh pipeline/ core/` rỗng nhưng không có lần chạy nào chứng
   minh). Trước khi đổi bất kỳ hằng số nào phải build lại vào thư mục riêng và so theo khoá
   `(book,page,column,bbox) → (tier,rule,label)`; `run_pipeline.sh` ghi cứng `dataset_out` ở ≥12 chỗ
   (`:55-57, :67, :69, :373, :397-398, :432-433, :451, :507`).
2. **"Toàn pipeline ≤5–10 phút" cũng sai**: hai mục EVIDENCE cách nhau 4,7 phút có sha giống hệt 10/10 tệp →
   chỉ là gọi lại `evidence()`. Mốc an toàn: ≤30 phút, chưa ai `time`.
3. **552 ô QĐ-01 sống nhờ S3** (§2.7).
4. **Luật QĐ-01 theo lớp phải giữ `confusion_fix`** hoặc viết đúng: dưới tier_v3, 2.014 ô QĐ-01 rơi CHAR_A
   1.178 / CHAR_B 281 / SYL 512 / REVIEW 41; `CO_THE_GAN` của `glyph_fix.py:49` không gồm SYLLABLE.
5. **Chất lượng hộp ảnh phẳng theo tier v3**: ước lượng hộp nội suy CHAR_A 36,1% · CHAR_B 34,9% · SYL 37,3% ·
   REVIEW 39,5%; **4.242 ô REVIEW→SYL (nguồn +usable chính của v3) 40,8% là hộp nội suy và 0/4.242 có ảnh hôm
   nay**. "Làn văn bản" thăng ~7.000 ô mà crop là hộp chia đều — phải ghi `box_source` để mẻ chấm phân tầng.
6. **QĐ-01 là phán quyết theo ô**: 675/2.014 ô (33,5%) đang trên hộp kiểu midpoint → mọi thay đổi hộp làm
   crop đổi, người chưa nhìn → phải xem lại (~15 phút) và ghi phụ lục QĐ-01a.
7. **`decide_label` ở PASS 1 theo trang** — ngữ cảnh toàn ngữ liệu (cặp lặp, 2-gram, posterior có neo) phải ở
   PASS 1b/1c; cả kế hoạch lẫn ba đề xuất ban đầu đều viết sai tầng.

---

## 6. BỎ SÓT SO VỚI NGHIÊN CỨU 13–14/09

| Nghiên cứu | Trong kế hoạch? | Mất gì |
|---|---|---|
| H1 · CNN âm tiết làm trọng tài thanh ghi (E2a định vị khe **72%** vs văn bản 60–64%; E2b AUC **0,946**, độc lập với từ điển) | **không** | nguồn bằng chứng thanh ghi duy nhất *không vòng tròn* với văn bản; `thi_giac_am_tiet.py` + `model.pt` đã có, thiếu wrapper ~10 dòng; LP hiện rò (p50 0,88 toàn bộ vs 0,71 test) → phải huấn luyện out-of-fold 5 fold theo trang |
| H2 · gom cụm (sách, âm): 63 lớp/1.536 ô GOLD một-hình, 38 lớp/508 ô dị thể thật, cụm "người" theo sách | thay bằng yaml **tĩnh 4 cặp** toàn cục | đơn vị (sách, âm) mất; `e3_all_classes.csv` (112 lớp) đã có sẵn |
| H3 · hai cột `glyph_cluster_id` + `label` | không | vẫn "accuracy đo tính nhất quán" |
| Ba hàng đợi phân xử theo lớp (128 + 286 + 1.150 lớp) + mẻ chấm 600 ô | không — "tự động 100%" | mâu thuẫn với chỉ tiêu "GOLD ≥98%" khi chưa có GT người |
| Cột `dict_support`, `context_evidence`, `p_register`, `box_source`, `n_ocr/n_qn/n_det`, `flank_gold`, `syllable_raw` | chỉ `p_posterior` | không phân tầng được mẻ chấm |
| BUG-5 (symlink `dict/dict`), BUG-6 (`USABLE_TIERS` ×5), BUG-8 (0 test) | không | |
| H5 "không làm": cửa sổ hẹp quanh tâm hộp | 4.2 `resolve_overlap` seam ở trung điểm ≈ "B_seam" hướng C đã bác (mất >10% nét ở 59% ô) | |
| Trình tự | theo luồng pipeline 1→6 | lộ trình nghiên cứu xếp theo tác động/rủi ro; crop-v2 hoãn sau luận văn (04/09) |

---

## 7. HƯỚNG CHO KẾT QUẢ TỐT NHẤT

Ba đề xuất độc lập (rủi ro / giá trị / luận văn) **đồng thuận** ở lõi: CALIB 0/2,5/6,7/5,1/8,6; T=1,0;
tier_v3 nguyên văn; đệ quy loại-một-trang; L1 theo 2-gram + `syllable_raw` + `corpus_readings`; bỏ COM,
Pitch-DP-bổ-đôi, `resolve_overlap`, loanword; F3g sau luận văn. Bất đồng đã phân xử bằng số (S3, khoá QĐ-01,
hộp ảnh, tên tier — ghi trong từng bước). Nguyên tắc: **văn bản trước – ảnh làm trọng tài – người ký theo lớp**;
mọi bước có benchmark đáp án, xây ở thư mục song song, không đổi md5 crop cho tới khi có mẻ chấm.

### Khối A — làn văn bản (≈1 tuần, tự động, hoàn nguyên bằng git)

| # | Việc | Tệp | Đo bằng | Kỳ vọng |
|---|---|---|---|---|
| A0 | **Hàng rào**: `OUT_DIR` cho `run_pipeline.sh` → build vào `dataset_out_v3/`; tái lập HEAD và so khoá với `dataset_out/` (§5-1); xuất `config/qd01_cells.csv` (2.014 ô theo `book,page,column,bbox,image_md5`) và cho `glyph_fix` chế độ khoá theo ô; chụp crosstab tier×rule; **≥15 assertion** cho `realign_column`/`decide_label`/`apply_am_sua_dau`/`apply_cot_lech_cau_xuoi`/`_pair_new` | `run_pipeline.sh`, `glyph_fix.py`, `phase1_engine_selftest.py` | 3 tệp labels trùng khoá; QĐ-01 = 2.014/2.014; selftest 722+15 | 0,5–1 ngày |
| A1 | Ma trận **CALIB đúng** + `posterior_matches(nom_chars, syllables, qn_to_nom, similar_dict, T=1.0, cost_fn)` vào `anchor_align.py`; sửa `thuc_nghiem.py` đọc hằng số từ engine và có `rebuild` | `anchor_align.py:33-38` | `thuc_nghiem.py rebuild calib recursive` | khe giả 345→54; lệch chéo 1,63→0,82%; 439 GOLD đổi âm về đường chéo |
| A2 | **Đệ quy hai lượt** (mục 3.3 còn thiếu): sau PASS 1 dựng `pair_pages[(chữ, âm)]`; PASS 1b căn chỉnh lại với chi phí `min(base, 2.0)` cho cặp thấy ở ≥2 trang **khác**; lưu `pair_pages.json` | `build_dataset.py` (~:404) | benchmark 600 cột | khe đúng 64→85–86%; ghép sai khi rụng chữ 3,14→1,36% |
| A3 | **Hộp theo chỉ số âm** (không đổi detector, không Pitch DP): `thr 0,3→0,2` (`align_production.py:252`), lọc x ±10%·w **trước** `enforce_count`, khi \|G\|=\|Q\| dùng `hộp[j]`; ghi `n_ocr/n_qn/n_det/box_source/flank_gold` — **chỉ gắn cờ, không hạ tier**; xem lại ~675 crop QĐ-01 đổi hộp (QĐ-01a) | `align_production.py`, `build_dataset.py` | `thuc_nghiem.py geo` + kịch bản DEL | M==N ở cột OCR=QN 69→92%; đúng hộp 59→91–97% ở cột rụng chữ. *Nếu còn <2 tuần: chỉ ghi cột, giữ hộp cũ* |
| A4 | **tier_v3 nguyên văn** ở PASS 1b/1c (không phải `decide_label` PASS 1); cột `p_register`, `dict_support=\|R\|`, `context_evidence`, `tier_v3`; **giữ tên GOLD/SYLLABLE** cho `USABLE_TIERS` (ánh xạ CHAR_A/CHAR_B→GOLD, SYL→SYLLABLE) thay vì đổi tên ở 11 mô-đun; gom `USABLE_TIERS` về `pipeline/tiers.py` | `build_dataset.py`, `consensus.py` | crosstab + benchmark | 70.108 ô; CHAR_A ghép sai 0,5%; 1.304 GOLD yếu → REVIEW |
| A5 | `config/decisions.yaml` với 4 loại mục, **bắt buộc `xuat_xu`**, kiểm unicode, trường `book` tuỳ chọn: gộp dị thể (徳→德, 别→別, 為→爲, 廪→廩), QĐ-01 khoá theo ô, lớp nhầm hệ thống (其→共 "cùng" theo E3), `corpus_readings` (僥→nhiều 61, 無→vồ 68, 生→chẳng 41…) có người duyệt; L1 chỉ đổi âm khi 2-gram ngoài ô ủng hộ, luôn giữ `syllable_raw`; BUG-1 (cả hai return), BUG-5, BUG-6; nếu tắt `--use-s3` thì **bắt buộc** `--crop-review` + assert QĐ-01 = 2.014 | `decisions.yaml`, `build_dataset.py`, `step2_align.py`, `run_pipeline.sh` | assert QĐ-01; đếm L1 theo nhóm | 0 ô ghi đè bản in; 498 ca sửa đúng giữ được |
| A6 | Rebuild 3 sách vào `dataset_out_v3`, `time` thật (có/không S3), bảng đối chiếu thế hệ (tier×rule, md5 đổi, QĐ-01), `update_bang_so_lieu` + `evidence()` tự động cuối `step_export` | `run_pipeline.sh` | `check_consistency.sh` 4/4 | chuỗi bằng chứng liền |

### Khối B — kênh ảnh (đóng góp mới, chỉ ĐO, không gác cổng; 5–8 ngày)

| # | Việc | Đo bằng | Kỳ vọng |
|---|---|---|---|
| B1 | CNN âm tiết **out-of-fold** 5 fold theo trang (~15 phút/fold MPS); wrapper `p_visual_register = exp(LP[row, cid[âm]])`; cột này vào `labels_v3.csv`; ở cột lệch, DP hộp↔âm bằng phát xạ ảnh và so khe với DP văn bản → bất đồng = cờ | E2a/E2b trên fold giữ lại | định vị khe ≈72%, AUC ≈0,9 (số out-of-fold sẽ thấp hơn 0,946 một chút) |
| B2 | H2-lite: dùng ngay `e3_all_classes.csv`/`e3_clusters.csv` (112 lớp) + `thuc_nghiem.py classes` (128 lớp/1.453 ô) + `Cap_sai` (286 lớp/3.267 ô) → giao diện 8–12 crop/lớp → người quyết (đúng / nhầm hệ thống → g / dị thể gộp / hỗn hợp); ghi vào `decisions.yaml` với `book`; H3: hai cột `glyph_cluster_id` + `label` | precision theo lớp có người ký | ~500 quyết định lớp thay ~5.000 ô |

### Khối C — người (điều kiện để có số công bố)

Mẻ chấm **~600 ô mù**, phân tầng theo (tier_v3 × lớp cột × `box_source` × `dict_support` × có/không `p_vis`
đồng thuận), oversample ~30 ca hình học nặng; **vá KHỐI 6** của `audit_grid` trước (ô mồi có đáp án ~10%,
lặp ẩn ~8%, dwell, làm mù tuyệt đối — bài học κ=0,13 ngày 04/08); hai câu tách bạch (crop đúng glyph? mã
đúng?). Không có mẻ này, mọi số precision trong luận văn là số đếm.

### Khối D — sau luận văn (ghi rõ "hạn chế / việc sau", có số)

`crops_v2` F3g theo **hình chiếu mực** tại `save_crop` PASS 2 (cắt lại từ cột bbox, 5 ms/crop ≈ 5 phút; crop→glyph
top-1 1,06→5,87%, crop→crop 1-NN 34,5→57,2%; cứu 0 ô; đổi ~92% md5); Pitch DP + lấp khe ≈k·pitch cho M<N;
tách từ dính theo từ điển (≈100–200 ô); hoà giải ba nguồn đếm với `REVIEW_count_conflict`.

### Bảng chỉ tiêu nghiệm thu mới (đo được, không tự thoả)

| Chỉ tiêu | Cách đo | Mốc |
|---|---|---|
| Tái lập HEAD | so khoá `dataset_out` vs `dataset_out_v3` trước khi đổi gì | 0 lệch, QĐ-01 2.014 |
| Khe giả (cột khớp số ≥10 chữ) | `thuc_nghiem.py rebuild calib` | 54 ± vài cột |
| Khe đúng chỗ / ghép sai khi rụng 1 chữ | benchmark 600 cột, CALIB+neo | ≥85% / ≤1,5% |
| Ghép sai theo tier v3 (benchmark) | CHAR_A / SYL / CHAR_B | ≤0,6% / ≤1,0% / ≤6% (có cờ) |
| Dùng được | crosstab | 70.108 ± 300 (tier_v3 nguyên văn) |
| Hộp detector đúng ở cột rụng chữ | `geo` với `hộp[j]`, thr 0,2 | ≥91% trên **mọi** cột \|G\|=\|Q\| |
| Cột `box_source` | phân bố theo tier | báo cáo, không mốc |
| L1 | 0 ô đổi âm mà 2-gram ngoài ô ủng hộ âm gốc; `syllable_raw` 100% giữ | 0 |
| QĐ-01 | assert sau mỗi bước hậu xử lý | 2.014 |
| Thời gian | `time ./run_pipeline.sh` | ghi số thật |
| Precision mã chữ / thanh ghi | mẻ chấm 600 ô mù, CI Wilson | công bố kèm CI |
| `p_visual_register` | E2a/E2b out-of-fold | ≥65% / AUC ≥0,90 |

---

## 8. GIỚI HẠN CỦA CHÍNH BẢN NÀY

- Benchmark "có đáp án" vẫn lấy đường chéo cột khớp số làm sự thật (≈99,5%); mọi tỷ lệ "ghép sai" là cận trên.
- Phân loại 729 ô L1 bằng 2-gram ngoài ô là heuristic; 70 ô hoà chưa quyết.
- Ước lượng "hộp nội suy" theo x1,x2 trùng trong cột là gần đúng (khớp ±2 điểm với số §3.1 rà soát 13/09);
  phân bố `box_source` thật cần A3.
- Đo COM trên 300 ô GOLD-direct; F3g được dựng lại từ mô tả, không phải mã gốc.
- Không có phán quyết người mới nào trong bản này — mọi kết luận về "đúng/sai nhãn" là suy từ số gián tiếp,
  như chính các tài liệu 13–14/09.
