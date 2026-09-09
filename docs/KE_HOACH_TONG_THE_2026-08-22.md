# KẾ HOẠCH TỔNG THỂ HOÀN THIỆN ĐỀ TÀI — GanNhanOCR
> **⚠️ ĐÍNH CHÍNH 2026-09-09 — KHÔNG SỬA THÂN TỆP (đây là nhật ký có ngày).**
> Ngữ liệu được xác nhận là **chữ Nôm CHÉP TAY bút lông, thể hành-thảo** trên giấy hiện đại,
> KHÔNG phải văn bản khắc gỗ. Vì vậy nhan đề đề xuất ở mục cuối tệp này ("văn bản Hán Nôm
> khắc gỗ") là SAI và không được dùng. Xem `docs/KE_HOACH_CAP_NHAT_2026-09-09.html`.

**Ngày lập**: 2026-08-22 · **Căn cứ**: đợt kiểm định 199 khẳng định (v2) + tuyên bố của chủ nhiệm
đề tài rằng **toàn bộ phần "chấm tay" hiện có thực chất do máy chấm**.

---

# PHẦN 0 — RESET: XÁC LẬP LẠI ĐIỀU GÌ CÒN ĐÚNG

## 0.1 Bằng chứng: vì sao mọi số precision hiện tại đều phải huỷ

Truy nguyên `dataset_out/human_audit/verdicts_reanchored.csv` — tệp mà **mọi** con số precision
trong luận văn neo vào:

| kiểm chứng | kết quả |
|---|---|
| Số phán quyết | 846 |
| Trùng `item_id` với `audit_gold/audit_gold.jsonl` (tệp **máy** chấm, có cột `decision`) | **846/846 — trùng hoàn toàn** |
| Khớp giá trị verdict với tệp máy đó | **47/846 (5,6%)** |
| Trùng với `audit_gold_human/verdicts{1,2}.jsonl` | **1/846** |
| Verdict thô gốc (`verdicts_001–006.jsonl`) | **ĐÃ MẤT** — `docs/EVIDENCE_INDEX.md:18` tự khai |
| Giãn cách `ts` ở các tệp còn `ts` | trung vị **1.275 giây/ô** — không phải nhịp người chấm (5–10 s/ô), cũng không phải nhịp máy |

**Đọc ra**: bộ 846 phán quyết dùng **đúng mẫu** của một mẻ máy chấm, nhưng **giá trị verdict đến từ
một nguồn không còn tồn tại trong repo**. Không thể chứng minh xuất xứ, không thể tái kiểm.
Cộng với tuyên bố của chủ nhiệm đề tài, kết luận bắt buộc:

> **Đề tài hiện có 0 (không) phán quyết người nào dùng được. Mọi con số precision phải được đánh
> dấu "CHƯA ĐO" cho tới khi có mẻ chấm mới.**

## 0.2 Danh sách số liệu phải huỷ ngay

| số liệu | nơi đang được trích | trạng thái mới |
|---|---|---|
| Precision GOLD 97,98% [96,7–98,8] (777/793) | `BANG_SO_LIEU:39`, datasheet | **HUỶ** |
| Precision GOLD 98,00% / 97,08% (trước-sau confusion_fix) | `BANG_SO_LIEU:90` | **HUỶ** |
| `rule_precision_direct` 737/752 = 98,0% | `s3_unwind_report.json` | **HUỶ** |
| `rule_precision_similar` 40/41 = 97,6% | `s3_unwind_report.json` | **HUỶ** |
| Fisher p = 5,4×10⁻⁸ cho lớp 㝵/"người" (8/23) | `confusion_fixes.yaml:17` | **HUỶ tư cách bằng chứng** (xem 0.3) |
| Error-AUC S3 = 0,566 và ArcFace = 0,577 | `s3_unwind_report.json`, cả hai bản góp ý | **HUỶ** — cả hai tính trên chính bộ 846 này |
| κ = 0,13 (độ tin cậy người chấm) | `FLOW_CAP_NHAT` B1 | **HUỶ** — không có người chấm để mà đo κ |
| Mọi phán quyết trên SILVER (750 ô, AUC 0,600) | `audit_SILVER/verdicts_ai.jsonl` | **Giữ nhưng đổi tên**: đây là `source: ai_vision`, vốn đã khai báo đúng là máy. Chỉ dùng làm **xếp hạng ưu tiên**, không làm bằng chứng |

## 0.3 Hệ quả dây chuyền — hai quyết định kỹ thuật mất căn cứ

1. **Bước 5 `confusion_fix`** hạ 1.924 ô 㝵/"người" xuống REVIEW dựa trên Fisher p tính từ 23 ô của
   mẻ đã huỷ. Quyết định **có thể vẫn đúng** (lớp nhiễu này rất có khả năng là thật), nhưng **hiện
   không có bằng chứng**. → Phải nằm trong mẻ chấm mới, với cỡ mẫu đủ.
2. **Bước 6 `s3_unwind`** gỡ S3 dựa trên error-AUC tính từ cùng mẻ đó. Quyết định **vẫn giữ** vì lý
   do quản trị (không cho tín hiệu *chưa chứng minh* quyết định nhãn), nhưng lý do phải viết lại
   thành "chưa chứng minh", tuyệt đối không viết "đã bác bỏ".

## 0.4 Điều gì CÒN NGUYÊN GIÁ TRỊ (không phụ thuộc phán quyết)

Đây là vốn liếng thật của đề tài, không bị ảnh hưởng:

- **Toàn bộ pipeline tất định** 7 bước, chạy lại ra byte giống nhau khi còn cache OCR.
- **82.269 ô đã căn chỉnh** trên 445 trang, 3 sách — dữ liệu thô là thật.
- **Cache OCR 445+445 tệp** — primary data, tái lập 0 đồng.
- **Mọi đại lượng hình học và thống kê mô tả**: phân bố tier, số lớp ký tự (1.582), đuôi dài
  (466 lớp singleton), rò rỉ split (0), trùng md5, `ink_pct`, bbox.
- **Phát hiện DPI**: STT4 ~201 DPI vs STT2/STT11 ~302 DPI — đo trực tiếp từ PDF, không qua phán quyết.
- **Cấu trúc 9 cột**: 439/445 trang đủ 9 cột — đếm từ nhãn, không qua phán quyết.
- **Toàn bộ hạ tầng đo lường đã viết**: `ground_truth/` (rank→plan→sample→grid→estimate với
  Clopper-Pearson / Horvitz-Thompson / PPI), `publish/` (Frictionless + Croissant + datasheet + HF).
  Công cụ **đã có và đã test**; cái thiếu là **dữ liệu đầu vào thật**.

**Đây là tin tốt**: phần đắt nhất (hạ tầng) đã xong. Cái thiếu là 4–5 giờ chấm tay thật.

---

# PHẦN 1 — BỨC TRANH CÔNG NGHỆ TOÀN CẢNH

## 1.1 Bản đồ theo tầng

| # | Tầng | Công nghệ đang dùng | Trạng thái | Thiếu / cần sửa |
|---|---|---|---|---|
| 1 | Đọc PDF | PyMuPDF (`fitz`) | ✅ chạy | ⚠️ Ảnh Nôm **bóc ảnh nhúng**, không render → DPI 200–460. **STT4 toàn bộ ~201 DPI** |
| 2 | Nhị phân hoá | **Otsu** (`image_processing.py:42`) + ngưỡng cứng 128 (`bbox_fix.py:63`) | ✅ chạy | ⚠️ Sauvola (k=0,2/w=25) đã viết nhưng **nằm ở đường cũ, không được gọi**. Chưa từng so sánh Otsu vs Sauvola trên chính dữ liệu này |
| 3 | OCR Nôm (S1) | API SinoNom HCMUS `kimhannom` + cache 445 tệp | ✅ chạy | ⚠️ Một nguồn duy nhất, không có phiếu thứ hai độc lập |
| 4 | OCR Quốc ngữ | VietOCR 2-pass (beam→text, greedy→confidence) | ✅ chạy | ⚠️ Chưa đo độ chính xác âm tiết QN — mà **toàn bộ nhãn neo vào nó** |
| 5 | Tách dòng QN | `projection_deskew` (ghim cứng) · có sẵn `dbnet`, `projection` | ✅ chạy | ⚠️ Mặc định **trong mã** vẫn là `auto`; `config/pipeline_today.yaml:29` còn `auto` |
| 6 | Bóc 9 dòng QN | `parser_v5` (9 pattern chống lỗi marker) | ✅ chạy | — |
| 7 | Dò cột Nôm | `nom_detect_v3` (hybrid→merge→projection) | ✅ chạy | 439/445 trang đủ 9 cột; 6 trang lệch, chưa xử lý |
| 8 | Căn chỉnh | **Needleman-Wunsch có băng** `\|i−j\| ≤ \|m−n\|+2`, ma trận chi phí từ điển (0,0/0,3/0,9/1,0/0,7) | ✅ chạy | ⚠️ **0 selftest phủ ma trận chi phí**. Hằng số chưa từng được quét thử |
| 9 | Tách ký tự | **CenterNet ResNet34+FPN+seam** (`train_crop`) | ✅ chạy | 🔴 **Rơi ngầm về midpoint** nếu thiếu checkpoint, không ném lỗi, không ghi vết backend |
| 10 | Tinh chỉnh crop | `tighten_box` + `carve_neighbor_ink` | ✅ chạy | 🔴 `crop_quality.py` (208 dòng: `stray_ink`, `border_ink`, `resolve_overlap`) **chưa từng được import** |
| 11 | Tín hiệu thị giác S3 | ResNet18 + **đầu ArcFace 1.591 lớp** (`nom-embed/best.pt`) + ngân hàng tham chiếu 3 tier (crop thật / simfont / FontDiffusion 89.898 glyph) + hiệu chuẩn isotonic | ⏸️ đã gỡ ở Bước 6 | ⚠️ Đầu ArcFace thiếu **18 lớp** so với bộ nhãn (1.591 vs 1.609). Điểm MLS/Energy/margin **chưa từng đo trên phán quyết** |
| 12 | Retrain S3 | Sub-center ArcFace K=3 + SAM + split page-disjoint/LOBO (`ArcFace/`, Kaggle ~1h GPU) | ✅ đã chạy 1 lần | ⚠️ `manifest.csv` là thế hệ cũ (61.387 dòng, 1.564 lớp) |
| 13 | Hợp nhất tier | `consensus.decide_label(S1,S2,S3)` → GOLD/SILVER/SYLLABLE/REVIEW | ✅ chạy | ⚠️ `consensus.py:92-93` gán GOLD **vô điều kiện** khi `ocr_char ∈ qn_to_nom[syllable]` |
| 14 | Vá lỗi | quarantine · dedup-md5-split · `confusion_fix` (Fisher) · `s3_unwind` | ✅ chạy | 🔴 48 ô 㝵 lọt về GOLD · `--measure` trả `null` · `s3_demote` mặc định tắt gây lệch 56.776/56.824 |
| 15 | Chuẩn hoá âm | `normalize_tone_marks` hai phía · `is_plausible_qn_syllable` · `fix_tone` (nhóm A) | ⚠️ một phần | 🔴 `fix_tone.py` **untracked**, 73 ô sửa **mồ côi**, không vào bản công bố |
| 16 | Từ điển | `QuocNgu_SinoNom.csv` 104.053 cặp · `SinoNom_Similar.csv` 33.396 · Unihan `kDefinition` 23.285 | ✅ chạy | ⚠️ Tên thư mục lệch `Dict/` (git) vs `dict/` (đĩa) ở 7 tệp `.py` + 2 config |
| 17 | **Đo lường** | `ground_truth/`: rank→plan→sample→grid(HTML mù)→estimate (Clopper-Pearson, Horvitz-Thompson, PPI) | ✅ code xong, 60/60 test | 🔴 **KHÔNG CÓ DỮ LIỆU ĐẦU VÀO THẬT** — đây là lỗ hổng số 1 |
| 18 | Hợp phiếu | `consensus_fusion/`: Kish n_eff, stacking IRLS+PAV, gate bất đối xứng | ✅ code xong, 44/44 test | ⏸️ chưa dùng — thiếu kênh phiếu độc lập thật |
| 19 | Đóng gói | `publish/`: Frictionless + Croissant + datasheet Gebru + HF Parquet, CI gate 11/11 | ✅ code xong, 56/56 test | ⏸️ chạy ngoài `run_pipeline.sh` |
| 20 | Bằng chứng | `evidence()` ghi SHA256 vào `EVIDENCE_INDEX.md` | 🔴 hỏng | `CHECKSUMS.txt` **chưa từng sinh**; hash hiện hành không có trong index; toàn bộ cây làm việc **chưa commit** |

## 1.2 Công nghệ **chưa có**, cần quyết định có đưa vào không

| công nghệ | trạng thái | khuyến nghị |
|---|---|---|
| Giao diện chấm tay có kiểm soát chất lượng (ô mồi, lặp ẩn, khoá phiên) | `audit_grid` sinh HTML mù nhưng **không có ô mồi** | **BẮT BUỘC BỔ SUNG** — xem GĐ 1 |
| Phiếu OCR Nôm thứ hai độc lập (Kraken / NomNaOCR / VLM) | không có trong luồng | Tuỳ chọn, chỉ để **xếp hạng ưu tiên chấm**, không làm nhãn |
| Dữ liệu IDS (phân rã bộ thủ) | không có | **Bỏ** — xem PHẦN 3 |
| SVTR backbone | không có | **Bỏ** |
| Chuẩn hoá DPI | không có | **BẮT BUỘC BỔ SUNG** — xem GĐ 3 |

---

# PHẦN 2 — KẾ HOẠCH THEO GIAI ĐOẠN

**Nguyên tắc xuyên suốt**: mỗi giai đoạn có (a) một **thước đo** định trước, (b) một **lưới cấu
hình** để thử nhiều phương án, (c) một **luật chọn** viết trước khi chạy, (d) **tiêu chí hoàn
thành** kiểm được. Không giai đoạn nào được dùng tập `test` để chọn cấu hình.

**Giả định** (nếu sai thì báo lại, kế hoạch sẽ đổi): người chấm là chính bạn, chấm được ~4–6 giờ
tổng, chia nhiều buổi; chưa có chuyên gia Hán-Nôm ngoài.

---

## GIAI ĐOẠN 0 — ĐÓNG BĂNG & VÁ (2–3 ngày) 🔴 chặn mọi thứ

**Mục tiêu**: có một commit mà mọi con số về sau neo vào được.

### Việc

| # | việc | tệp |
|---|---|---|
| 0.1 | Commit toàn bộ `dataset_out/`, `docs/`, `pipeline/tools/fix_tone.py`, `pipeline/ground_truth/make_lookalike_page.py` | — |
| 0.2 | Chốt chặn blacklist trong `s3_unwind`: sau readmit, **không ô nào** thuộc lớp trong `confusion_fixes.yaml` được về GOLD | `s3_unwind.py` |
| 0.3 | Chuẩn hoá tiền tố `yen*`→`stt*` trước khi join | `confusion_fix.py:53-60` |
| 0.4 | `raise FileNotFoundError` thay vì rơi ngầm về midpoint; ghi cột `seg_backend` vào labels | `align_production.py:183` |
| 0.5 | Ép `checkpoint()` ghi `CHECKSUMS.txt` sau mỗi bước; `evidence()` **thêm** bảng mới thay vì chỉ append log | `run_pipeline.sh:331,410` |
| 0.6 | Sửa đánh số bước (`banner()` in "BƯỚC 7/6") | `run_pipeline.sh:83` |
| 0.7 | Thống nhất tên thư mục từ điển: `git mv Dict dict_tmp && git mv dict_tmp dict`, sửa 7 tệp `.py` + 2 config | — |
| 0.8 | Xoá `config/pipeline_today.yaml` (bẫy copy-paste `auto`) | — |
| 0.9 | Xoá bảng số cũ trong `BANG_SO_LIEU_CHINH_THUC.md` (dòng 59-70), sửa `EVIDENCE_INDEX.md:71` (66.589→66.529) | — |
| 0.10 | **Đánh dấu "CHƯA ĐO" mọi số ở §0.2** của tài liệu này trong mọi tệp docs | — |
| 0.11 | Sinh lại `index.csv` cho crop-proto từ thế hệ `stt*` (bản hiện tại 100% trỏ `yen*`) | — |

### Hoàn thành khi

- `./run_pipeline.sh` từ bước 4 chạy hết, ra **56.776 dòng**, `CHECKSUMS.txt` tồn tại, hash của
  `labels_final.csv` xuất hiện trong `EVIDENCE_INDEX.md`;
- `python -c "…"` đếm ô 㝵/"người" ở GOLD ra **0**;
- 414+ assertion selftest xanh;
- một commit SHA duy nhất để trích dẫn.

---

## GIAI ĐOẠN 1 — XÂY GROUND TRUTH THẬT (1,5–2 tuần) 🔴 trái tim của đề tài

Đây là giai đoạn quyết định. Không có nó, không giai đoạn nào sau đo được gì.

### 1.1 Tách đôi nhiệm vụ — sửa lỗi thiết kế gốc

| chiều | ai làm | cách đo | vì sao |
|---|---|---|---|
| **NHÃN** — chữ trong ô có đúng là chữ được gán không | **người** | Clopper-Pearson từng tier | Đây là việc của mắt người, có đáp án đúng/sai rõ ràng |
| **CROP** — ô có cắt đúng một chữ không | **máy** | `crop_quality.py`: `border_ink`, `stray_ink`, IoU với detector, aspect-ratio | Là đại lượng **hình học có ngưỡng**, giao cho mắt người là nguồn nhiễu lớn nhất của mẻ cũ |

Trộn hai chiều vào một nút bấm là lỗi thiết kế của mẻ trước. Tách ra thì chiều NHÃN trở thành một
nhiệm vụ đơn giản, nhanh, và tái lập được.

### 1.2 Thiết kế mẻ chấm — bốn cơ chế chống lại chính vấn đề đã xảy ra

1. **Ô mồi (planted controls) — cơ chế quan trọng nhất.** Trộn ngẫu nhiên **10%** ô có đáp án
   biết trước: một nửa là ô chắc chắn đúng (chữ phổ biến, dict-confirmed, crop sạch), một nửa là ô
   **cố ý gán sai** (đổi nhãn sang chữ khác trong cùng danh sách đọc âm). Người chấm không biết ô
   nào là mồi.
   → Nếu độ chính xác trên ô mồi < 90%, **huỷ cả buổi chấm**. Đây là thứ sẽ phát hiện ngay việc
   một mẻ bị chấm ẩu hoặc bị đưa cho máy.
2. **Ô lặp ẩn 8%** — cùng ô xuất hiện lại ở buổi khác, để đo κ nội tại thật.
3. **Làm mù**: HTML không hiển thị tier, rule, `s3_cosine`, hay bất cứ thứ gì gợi ý hệ thống nghĩ gì.
4. **Ghi vết phiên**: mỗi phán quyết ghi `item_id`, `verdict`, `ts` **thật**, `session_id`,
   `dwell_ms`. Ô nào `dwell_ms < 800ms` bị gắn cờ để xem lại — nhịp chấm là bằng chứng phụ về việc
   ai đã chấm.

### 1.3 Chia mẻ: dev / test khoá

> **Đây là điều kiện sống còn của mọi con số cuối cùng.** Nếu bạn dùng cùng một mẻ vừa để chỉnh
> tham số vừa để báo cáo, con số cuối cùng không còn ý nghĩa thống kê.

| mẻ | mục đích | cỡ mẫu | dùng lúc nào |
|---|---|---|---|
| **DEV** | chỉnh tham số ở GĐ 3–5, thử bao nhiêu lần cũng được | GOLD 300 · SILVER 200 · SYLLABLE 150 = **650** | GĐ 3–5 |
| **TEST (khoá)** | **chỉ mở một lần duy nhất**, ở GĐ 6, để lấy số công bố | GOLD 600 · SILVER 250 · SYLLABLE 200 = **1.050** | GĐ 6 |
| + ô mồi 10% + lặp ẩn 8% | kiểm soát chất lượng | ~300 | mọi buổi |

**Tổng ~2.000 lượt chấm ≈ 4–6 giờ**, chia 4–6 buổi ≤ 45 phút/buổi (mẻ cũ trôi 4,2%→16%→35% qua ba
buổi dài — chia nhỏ là chống mệt).

### 1.4 Chọn mẫu — tránh post-hoc

- Rút **SRS phân tầng theo tier**, `design_weight` ghi sẵn (code `plan.py`/`sample.py` đã hỗ trợ).
- **Loại trừ mọi trang từng dùng để phát hiện lớp lỗi 㝵** khỏi mẻ TEST → con số cuối cùng không
  còn là post-hoc.
- Mẻ TEST rút từ **đúng tệp** mà Bước 7 xuất ra, so bằng SHA256.

### 1.5 Cỡ mẫu nói lên điều gì (Clopper-Pearson một phía 95%)

| muốn công bố | số lỗi quan sát | n tối thiểu |
|---|---|---|
| precision ≥ 99,0% | 0 | 299 |
| precision ≥ 99,5% | 0 | 598 |
| precision ≥ 99,0% | 1 | 473 |
| precision ≥ 98,0% | 12 (tỉ lệ lỗi thực ~2%) | ~600 |

→ Với n=600 GOLD và tỉ lệ lỗi thực ~2%, kỳ vọng ~12 lỗi → công bố được **~98%**. Muốn chạm 99,5%
thì **phải diệt lớp lỗi hệ thống trước** (đó là vai trò của GĐ 2), không phải tăng cỡ mẫu.

### Hoàn thành khi

- ≥ 1.700 phán quyết người có `session_id` + `dwell_ms`;
- độ chính xác trên ô mồi ≥ 95%;
- κ nội tại từ ô lặp ẩn ≥ 0,6 ở chiều NHÃN (nếu < 0,4: nhiệm vụ vẫn còn mơ hồ, phải sửa hướng dẫn
  chấm rồi làm lại — **đừng đi tiếp**);
- mẻ TEST được niêm phong (hash ghi vào `EVIDENCE_INDEX.md`, không mở tới GĐ 6).

---

## GIAI ĐOẠN 2 — ĐO LẠI TOÀN BỘ + KHAI THÁC LỚP NHIỄU (1 tuần)

**Mục tiêu**: dựng lại mọi con số đã huỷ ở §0.2, bằng mẻ DEV.

### Việc

1. **Precision từng tier** với Clopper-Pearson + Horvitz-Thompson (có `design_weight`).
2. **Precision từng rule** — đây là số quan trọng nhất: `s1_inter_s2_direct` (46.188 ô) vs
   `s1_inter_s2_similar` (3.875 ô). Nếu rule `similar` kém hơn rõ rệt, có căn cứ tách nó thành tier
   riêng — quyết định này trước đây dựa trên 41 ô của mẻ đã huỷ.
3. **Precision phân tầng theo sách** (bắt buộc vì STT4 lệch DPI).
4. **Khai thác lại lớp nhiễu hệ thống** — chạy `mine_confusions.py` trên mẻ DEV mới: với mỗi cặp
   (âm, chữ) có ≥1 lỗi, kiểm định Fisher so với nền, hiệu chỉnh **Benjamini-Hochberg** cho đa so
   sánh (mẻ cũ **không** hiệu chỉnh — đó là một điểm yếu nữa). Chỉ cặp nào qua FDR 5% mới vào
   `confusion_fixes.yaml`.
5. **Xác nhận hoặc bác lớp 㝵/"người"** bằng mẻ mới. Nếu không qua FDR: **hoàn 1.924 ô về GOLD**.

### Test nhiều phương án

| phương án ước lượng | khi nào dùng |
|---|---|
| SRS thuần | mặc định, dễ bảo vệ nhất |
| Phân tầng + Horvitz-Thompson | khi `design_weight` lệch nhiều |
| PPI (dùng điểm máy làm biến phụ) | chỉ khi độ phủ tín hiệu ≥ 90% — hiện chỉ SILVER đạt (100%) |

Báo cáo **cả ba**, chọn cái hẹp nhất mà giả định còn giữ. Chênh lệch giữa ba ước lượng tự nó là một
kết quả đáng đưa vào luận văn.

### Hoàn thành khi

`BANG_SO_LIEU_CHINH_THUC.md` được viết lại hoàn toàn, mỗi số có: giá trị · CI · cỡ mẫu · lệnh tái
sinh · commit SHA · nguồn kiểm định (👤 người / 🤖 máy / ⚪ chưa đo).

---

## GIAI ĐOẠN 3 — TỐI ƯU CROP (1 tuần) · quét đa cấu hình

**Mục tiêu**: giảm lớp lỗi `wrong_image`. Ở mẻ cũ lớp này chiếm ~1/3 tổng lỗi GOLD — nếu mẻ mới xác
nhận tỉ lệ tương tự, đây là đòn bẩy lớn nhất để đẩy precision lên.

### 3.1 Lưới cấu hình

| yếu tố | các mức thử |
|---|---|
| `pad_frac` | 0,12 · 0,15 · **0,18** (hiện tại) · 0,22 |
| Nhị phân hoá khi siết hộp | ngưỡng cứng 128 (hiện tại) · **Otsu per-crop** · **Sauvola k=0,2 w=25** |
| `carve_neighbor_ink` | bật (hiện tại) · tắt |
| `crop_quality.resolve_overlap` | bật · **tắt** (hiện tại) |
| Chuẩn hoá DPI | không (hiện tại) · **resample về 300 DPI** |

4 × 3 × 2 × 2 × 2 = **96 cấu hình**.

> ⚠️ **Không dùng công thức `pad_px = pad_frac × pitch × (300/DPI)`** mà bản góp ý đề xuất.
> `pitch` đo bằng pixel *của chính trang đó*, nên `pad_frac × pitch` **vốn đã tự chuẩn hoá** theo
> độ phân giải; nhân thêm 300/201 sẽ làm pad của STT4 rộng hơn 1,5 lần tương đối. Cách đúng là
> **resample ảnh crop về một kích thước chuẩn** rồi giữ nguyên `pad_frac`.

### 3.2 Quy trình hai vòng (tránh 96 lần chấm tay)

**Vòng A — sàng lọc bằng thước đo không cần nhãn** (tự động, chạy hết 96 cấu hình):

| thước đo | hướng tốt |
|---|---|
| `border_ink` (% mực chạm biên) | thấp |
| `stray_ink` (% mực rời không thuộc thân chữ) | thấp |
| tỉ lệ crop trắng (`ink_pct` < 2%) | thấp |
| IoU chồng lấn giữa hai hộp kề trong cột | thấp |
| tỉ lệ aspect-ratio ngoại lai (> p99 hoặc < p1) | thấp |
| tỉ lệ crop bị cắt cụt (thành phần liên thông chạm biên và bị cắt) | thấp |

Gộp thành một điểm tổng hợp (chuẩn hoá z rồi cộng có trọng số bằng nhau), lấy **top 5 cấu hình**.

**Vòng B — xác nhận trên mẻ DEV** (chỉ 5 cấu hình): với mỗi cấu hình, sinh lại crop cho đúng 650 ô
DEV, chấm **chỉ chiều CROP bằng máy** + rà mắt nhanh 100 ô ngẫu nhiên. Chọn cấu hình có tỉ lệ
`wrong_image` thấp nhất; hoà thì chọn cái ít thay đổi so với hiện trạng nhất.

### 3.3 Cạm bẫy phải tránh

- **Đổi crop = đổi ảnh dưới chân bộ nhãn**. Mọi phán quyết chiều NHÃN đã chấm trên ảnh cũ **vẫn
  dùng được** (nhãn không đổi), nhưng phán quyết chiều CROP thì phải đo lại — đó là lý do tách đôi
  hai chiều ở GĐ 1.
- Chốt cấu hình **một lần**, rồi mới sinh mẻ TEST. Không đổi crop sau khi mở mẻ TEST.

### Hoàn thành khi

Một cấu hình được chốt, có bảng so sánh 96 cấu hình theo 6 thước đo, và `crop_quality.py` được nối
vào PASS 2 **ở chế độ ghi cờ** (`crop_quality_flag` ∈ `ok/bleed/truncated/blank` + `stray_ink` +
`border_ink` vào `labels.csv`).

---

## GIAI ĐOẠN 4 — TỐI ƯU CĂN CHỈNH (1 tuần) · quét đa cấu hình

### 4.1 Lưới cấu hình

| tham số | mức thử | hiện tại |
|---|---|---|
| `BAND_SLACK` | 2 · 3 · 4 · **thích nghi** (+3 khi \|m−n\| > 2) | 2 |
| `COST_SIMILAR` | 0,20 · 0,30 · 0,45 · 0,60 | 0,30 |
| `COST_NODICT` | 0,80 · 0,90 · 1,00 | 0,90 |
| `COST_DEL` = `COST_INS` | 0,60 · 0,70 · 0,85 | 0,70 |

4 × 4 × 3 × 3 = **144 cấu hình**. Chạy lại `build_dataset` mỗi cấu hình — tất định, không gọi API
(cache OCR còn) → chi phí là CPU, không tiền.

### 4.2 Thước đo — và cái bẫy vòng tròn

> ⚠️ **Tuyệt đối không tối ưu theo "số ô GOLD" hay "tỉ lệ dict-confirm".** Đó chính là luật gán
> nhãn; tối ưu theo nó là tự chấm điểm cho mình. Cấu hình nào nới lỏng nhất sẽ luôn "thắng" trong
> khi tạo ra nhiều nhãn sai nhất.

Thước đo đúng là **đường cong sản lượng–độ chính xác**:

- trục X: **số ô GOLD** sinh ra (sản lượng);
- trục Y: **precision chiều NHÃN trên mẻ DEV** ở đúng những ô đó;
- chọn cấu hình nằm trên **biên Pareto**, ưu tiên precision ở mức sản lượng ≥ 48.000.

Thước đo phụ (không cần nhãn, dùng để loại sớm):
- số thao tác `del`/`ins` trên mỗi cột (cao = align đang vật lộn);
- tỉ lệ cột có số ô ≠ số âm tiết;
- độ ổn định: đổi seed / đổi thứ tự sách phải ra kết quả **byte-identical** (nếu không, có phi tất định ẩn).

### 4.3 Bổ sung bắt buộc

Thêm **assertion cho `anchor_align`** vào bộ selftest (hiện **0** test phủ ma trận chi phí) — ít
nhất: chi phí đối xứng, `COST_CONFIRM` là sàn tuyệt đối, băng không bao giờ cho ghép ngoài giới hạn,
và một ca hồi quy cho mỗi hằng số.

### Hoàn thành khi

Có biểu đồ Pareto 144 cấu hình, một cấu hình được chốt kèm lý do, selftest tăng thêm ≥ 10 assertion.

---

## GIAI ĐOẠN 5 — TÍN HIỆU THỊ GIÁC: MỘT THÍ NGHIỆM QUYẾT ĐỊNH (5 ngày)

**Không SVTR. Không IDS.** Lý do ở PHẦN 3. Thay vào đó là một thí nghiệm rẻ mà chưa ai chạy.

### 5.1 Vì sao đáng chạy

`ArcFace/evaluate.py:96-97` tự in ra: *"proxy uses auto-labels as truth (upper bound). Rerun with
human verdicts for the real error-AUC"*. Nghĩa là **các điểm MLS / Energy / margin / head-logit
chưa từng được đo trên phán quyết nào** — chỉ có cosine thô từng được đo, và đo trên mẻ đã huỷ với
74,7% rò rỉ train.

### 5.2 Lưới cấu hình

| yếu tố | mức thử |
|---|---|
| Loại điểm | `bank_cosine` · `head_logit` · **MLS** · **Energy** · **margin** (top1−top2) · fusion(head+bank) |
| Ngân hàng tham chiếu | chỉ crop thật · crop+simfont · crop+simfont+fd (hiện tại) |
| Chuẩn hoá | thô · hiệu chuẩn isotonic · **temperature scaling** |

6 × 3 × 3 = **54 cấu hình**, mỗi cấu hình chỉ là một lần forward — không train lại.

### 5.3 Điều kiện bắt buộc

- Đánh giá trên **holdout page-disjoint tuyệt đối**: mọi trang trong mẻ đánh giá phải bị loại 100%
  khỏi tập train của checkpoint đang dùng. Nếu checkpoint hiện tại không đảm bảo được, **train lại
  1 giờ GPU** với split đúng — rẻ hơn nhiều so với việc công bố một con số rò rỉ.
- Bổ sung 18 lớp còn thiếu vào đầu ArcFace (1.591 → 1.609) hoặc khai báo rõ giới hạn từ vựng.

### 5.4 Luật quyết định — viết trước khi chạy

| kết quả trên mẻ TEST | hành động |
|---|---|
| Cận dưới CI95 của error-AUC **> 0,5** và precision@recall50 ≥ 95% | S3 được cấp vai trò **cổng demote-only**, không bao giờ phong cấp |
| Cận dưới CI95 **> 0,5** nhưng precision thấp | S3 chỉ được dùng để **xếp hạng ưu tiên chấm tay** |
| Cận dưới CI95 **≤ 0,5** | Giữ nguyên Bước 6. **Đây là một kết quả âm sạch — và là đóng góp khoa học thật** |

Cả ba nhánh đều cho ra nội dung viết được. Đó là lý do thí nghiệm này đáng làm còn SVTR thì không.

### 5.5 Cỡ mẫu cần để kết luận

Với tỉ lệ lỗi GOLD ~2–3%, để phân biệt AUC 0,60 với 0,50 ở lực 80%, α=0,05 hai phía (Hanley–McNeil)
cần cỡ **60–180 ca lỗi** tuỳ giả định — tức **2.000–6.000 ô chấm** nếu đo trên GOLD. Điều này
**vượt ngân sách chấm tay**.

→ **Cách đi vòng, rẻ hơn nhiều**: đo trên **SILVER**, nơi tỉ lệ lỗi ~14–28%. Với n=300 ô SILVER
chấm tay ta có ~50–80 ca lỗi — đủ để phân biệt AUC 0,60 với 0,50. Mẻ SILVER 250 ô trong TEST cộng
200 ô DEV vừa đúng cỡ này. **Đây là lý do mẻ chấm phải bao gồm SILVER, dù SILVER không nằm trong
bộ giao nộp.**

---

## GIAI ĐOẠN 6 — MỞ MẺ TEST, ĐÓNG GÓI, VIẾT (1,5 tuần)

1. **Mở mẻ TEST một lần duy nhất.** Chạy `estimate.py`, lấy precision từng tier + từng rule + từng
   sách, kèm CI. Con số này là con số của luận văn. Không chỉnh gì sau khi mở.
2. Chạy Bước 7, `publish/` (Frictionless + Croissant + datasheet Gebru + HF Parquet), CI gate.
3. **Datasheet phải khai đủ 5 giới hạn**:
   - ngữ liệu 1 nét chữ, 3 sách, Công giáo thế kỷ XIX → không tổng quát hoá ngoài phạm vi;
   - **STT4 ở ~201 DPI** vs hai sách kia ~302 DPI, kèm precision tách theo sách;
   - tier SYLLABLE là nhãn **cấp âm tiết**, 316 cặp không nguồn nào xác nhận;
   - SILVER công bố riêng, gắn nhãn "uncalibrated", kèm precision đo được;
   - κ người chấm đo được, và **lịch sử**: mẻ phán quyết trước 2026-08-22 đã bị huỷ vì không truy
     nguyên được xuất xứ (khai thẳng, đây là điểm cộng về liêm chính khoa học, không phải điểm trừ).
4. Chốt chặn Pre-Export: Bước 7 **từ chối chạy** nếu SHA256 của `labels_final.csv` khác với hash mà
   **mẻ audit đã được rút ra từ đó**.

> ⚠️ Bản góp ý thiết kế chốt chặn so hash Bước 6 với Bước 7 — đó chỉ là toàn vẹn *trong một lần
> chạy*, không phải "bộ đem đo = bộ đem nộp". Hash phải so với hash **ghi trong manifest của mẻ
> audit**.

---

# PHẦN 3 — NHỮNG GÌ DỨT KHOÁT CẮT, VÀ VÌ SAO

| hạng mục | lý do cắt |
|---|---|
| **SVTR** | Là mô hình nhận dạng **chuỗi text-line**; tác vụ ở đây là phân lớp **một ký tự đã cắt sẵn**. Lợi ích còn lại chỉ là "đổi backbone sang ViT" — không chữa được nguyên nhân, vì bộ xếp hạng thật hiện đã là **đầu ArcFace**, không phải backbone |
| **IDS** | Repo không có dữ liệu IDS; tiền đề của báo cáo ("chữ chưa có mã Unicode") **sai** — 1.582/1.582 lớp đều có codepoint; ví dụ ⿰口巴 xuất hiện **0 lần**. Đuôi dài 466 lớp singleton chỉ chiếm **0,93%** số ảnh |
| **Mở rộng từ điển lên 120k** | `kDefinition` sinh 0 cặp; `kVietnamese` chỉ thêm ròng +109. Đổ `dict_gap` vào từ điển sẽ tự động đẩy 20.144 ô lên GOLD (+40,2%) mà không ô nào được xác nhận |
| **Sửa dấu thanh nhóm B** | Chạm **0 ô** bộ nhãn công bố; áp dụng sẽ **phá 170 cặp đang đúng** |
| **Hợp phiếu có VLM** | Vi phạm ràng buộc gốc: phán đoán của AI không bao giờ trở thành nhãn |
| **Mốc Error-AUC "0,850" hoặc "0,720–0,750"** | Cả hai đều là số tự chế, không dựa trên công trình nào. Không đặt KPI cho một đại lượng chưa ai đo trên loại dữ liệu này |

---

# PHẦN 4 — LỊCH TRÌNH 9 TUẦN

| tuần | giai đoạn | sản phẩm |
|---|---|---|
| 1 | GĐ 0 | Commit SHA · 56.776 dòng · `CHECKSUMS.txt` · 0 ô 㝵 ở GOLD |
| 1–2 | GĐ 1 (dựng công cụ) | `audit_grid` có ô mồi + lặp ẩn + `dwell_ms` · mẻ DEV/TEST rút xong, TEST niêm phong |
| 2–3 | GĐ 1 (chấm) | ~650 phán quyết DEV, ô mồi ≥ 95%, κ ≥ 0,6 |
| 3–4 | GĐ 2 | `BANG_SO_LIEU` viết lại · lớp nhiễu khai thác lại có hiệu chỉnh FDR |
| 4–5 | GĐ 3 | Bảng 96 cấu hình crop · chốt 1 · `crop_quality` nối vào PASS 2 |
| 5–6 | GĐ 4 | Pareto 144 cấu hình align · chốt 1 · +10 assertion |
| 6–7 | GĐ 5 | Bảng 54 cấu hình điểm thị giác trên holdout sạch · quyết định theo luật 5.4 |
| 7–8 | GĐ 1 (chấm TEST) | ~1.050 phán quyết TEST |
| 8–9 | GĐ 6 | Mở TEST · publish · datasheet · viết luận văn |

**Đường găng**: GĐ 1. Nếu mẻ chấm trượt (ô mồi < 90% hoặc κ < 0,4), mọi thứ sau đó dừng. Vì vậy
hãy chấm **50 ô thử nghiệm** ngay tuần 1 để kiểm tra hướng dẫn chấm trước khi cam kết 2.000 ô.

---

# PHẦN 5 — LUẬN ĐỀ CUỐI CÙNG

Sau khi mất toàn bộ ground truth cũ, luận đề mạnh nhất **không phải** "tôi xây được mô hình tốt
hơn" — mà là:

> **Một phương pháp gán nhãn cấp ký tự cho văn bản Hán Nôm khắc gỗ bằng cầu song ngữ, kèm một
> quy trình kiểm định tự phát hiện được sai sót của chính nó.**

Ba đóng góp:

1. **Phương pháp**: NW có băng + ma trận chi phí từ điển + tier hoá theo giao tín hiệu — và đặc
   biệt là **quy trình khai thác lớp nhiễu hệ thống bằng kiểm định Fisher có hiệu chỉnh FDR**. Đây
   là phần độc đáo nhất, chuyển giao được sang bất kỳ corpus nào gán nhãn bằng cầu từ điển, và
   đang bị bán rẻ trong các bản thảo hiện tại.
2. **Kết quả âm có đo đạc về tín hiệu thị giác**: cộng đồng vẫn mặc định so khớp glyph hoạt động
   trên chữ Nôm. Bạn sẽ có bằng chứng định lượng trên holdout sạch — kèm phân tích lực thống kê
   nói rõ mẫu cần bao nhiêu để kết luận. Cả kết quả dương lẫn âm đều viết được.
3. **Bộ dữ liệu ~57.000 ô có precision đo được kèm CI**, chuẩn quốc tế, chuỗi bằng chứng SHA256
   khép kín — và một **datasheet khai cả những gì đã hỏng**, kể cả việc mẻ phán quyết cũ bị huỷ.

Điều thứ ba là thứ phân biệt một luận văn tốt với một luận văn trung bình trong lĩnh vực này. Phần
lớn bộ dữ liệu di sản được công bố kèm những con số chất lượng không ai tái lập được. Bạn vừa tự
phát hiện ra mình đang ở trong tình trạng đó và sửa nó — **đó chính là chương hay nhất của luận
văn**, nếu viết thẳng.
