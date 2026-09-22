# ĐẶC TẢ PIPELINE CHUẨN CHO 3 SÁCH NHÓM 1 MỚI (LVT1883, KVK1884, Chrestomathie1872)

Ngày: 2026-09-20 (phê bình & đính chính 2026-09-21, xem §10; số liệu chốt bằng bộ đo offline, xem §11) · Trạng thái: **đặc tả (chưa thực thi)** · Phạm vi: chỉ đọc mã + đo; không sửa mã, không chạm `data/`, không gọi API trả phí.

Quy ước: mọi con số ghi kèm **[phương pháp · n · caveat]**; trích mã dạng `file:line`. Nguồn số liệu: 3 báo cáo đọc mã (step1_ocr, align_engine, run_config) + 5 phép đo có phản biện độc lập (lvt1883_layout, kvk1884_layout, qn_ocr_feasibility, detector_transfer, chresto_mapping). Khi phản biện đính chính, tài liệu này dùng **giá trị đã đính chính**. Script/kết quả trung gian nằm ở `scratchpad/wf/<tên-phép-đo>/` (ngoài repo, xem §8).

---

## 0. Tóm tắt 10 dòng

1. **Chạy được ngay (0 sửa mã):** tách 2 tầng × 10 cột bằng chiếu mực cho LVT1883 (105/105 trang tách tầng; 103/105 đủ 10 cột — trang 1 và 105 là đầu/cuối) và KVK1884 (163/163 tách tầng; **162/163** đủ 10+10 cột khi lề trái ≤ 2 % — `kvk1884_canvas_map.csv`; bộ `layout_bands.py` cắt 6 % lề chỉ còn 132/163, `detector_review/layout_all_pages.csv`); đổi tên canvas KVK bằng quy tắc `page = 167 − k` (đúng 163/163); OCR QN bằng tesseract psm 4 (LVT **2.088/2.088** dòng sau lọc cước chú; KVK **3.251** dòng vật lý, chuỗi số in 3.253 — [measure_out 21/09, §11]); detector CenterNet chạy trên thạch bản nếu kéo tương phản nền 128→255 (thr 0,2: **LVT stretch 77,6 %, KVK otsu 77,6 %** cột đúng số hộp trên 27 trang/sách = 540 cột, CI ±3,5 [measure_out 21/09]; mẫu 9 trang cũ 82,2 / 80,3 tái lập đúng bằng `--page-ids` nhưng lạc quan ~4 điểm).
2. **Bố cục thật (đính chính so với đề bài):** mỗi cột vật lý = **1 cặp lục bát** xuyên 2 tầng (tầng trên = câu lẻ 6 chữ, tầng dưới = câu chẵn 8 chữ), thứ tự đọc cột k (phải→trái): trên rồi dưới → "20 cột" phải xếp (c1-trên, c1-dưới, c2-trên, …), **không** phải 10 trên rồi 10 dưới.
3. **Số câu thật:** LVT1883 = 2.088 (18 + 20×103 + 10; 0 trang mâu thuẫn số in); KVK1884 Nôm = **3.256** (số in cuối 3245/3255/3250 trên canvas 4; 162×20+16 cột), QN cùng ấn bản = số in cuối **3253** nhưng chỉ **3.251 dòng vật lý** (4 bất thường của chuỗi số in: nhảy 1581→1583, 2220→2222, 3053→3055 và 2232 lặp — [measure_out 21/09, §11 #22]) → hai phía **lệch 5 câu**, chưa định vị được câu dôi.
4. **Bị chặn bởi API kim (401):** không có S1 thì (a) `columns: []` → `realign_column` (`align_production.py:567`) không có chữ để ghép → 0 bản ghi; (b) `char: null` nhưng có bbox → **vẫn ghép được** (`substitution_cost`, `anchor_align.py:61-72`: char None → COST_DICTMISS 6,7 < DEL+INS 17,2; mô phỏng 6 None ↔ 6 âm = 6 match, p_register ≈1,0) nhưng 100 % ô là REVIEW/no_context (`build_dataset.py:485`). **GOLD bất khả** ở chế độ ngoại tuyến; mọi tier khác REVIEW cần luật mới (§5). *Đính chính 21/09:* `core/ocr/ocr_api.py` trong working tree (chưa commit, `git diff` +40/−13) đã thêm nhánh **Guest Mode** tự chuyển khi token 401 (`upload_image`/`recognize`); nhánh này **chưa được thử** — kết luận "không có S1" giữ nguyên cho tới khi người dùng cho phép 1 lời gọi thử.
5. **Phải sửa mã trước khi build:** số 9 ghim ở ≥8 chỗ (`align_production.py:81,87`; `step2_align.py:70,83`; `parser_v5.py:136`; `pdf_parser.py:128,179,243`; `column_detector.py:11`); đo: cache 20 cột đúng vẫn bị ép về 9 cột gộp, lỗi **im lặng** (page_ok không chặn).
6. **Neo liên sách:** PASS 1b/1c dùng pair_pages của **mọi** sách trong config (`build_dataset.py:137`), không có công tắc → chạy **config riêng từng sách** (0 sửa mã) hoặc thêm `--anchor-scope book`.
7. **Tắt cho sách mới:** self-training rescue (pseudo CSV/npz 100 % stt*, model học chữ thảo STT), `--decisions none` (22/23 mục `book: null` học từ STT), confusion_fix (người,㝵).
8. **OCR QN:** tesseract psm 4 ≫ VietOCR-projection (mất trắng 2,7–4,6 % dòng có số câu); số câu neo phải dùng **vị trí mod 5** (nhất quán 97,7–99,1 %) chứ không tin giá trị (đúng 71–76 %); cổng "đủ N câu" chỉ khả thi **bán tự động** (với 2 phương pháp neo đầu dòng + lề [measure_out 21/09]: LVT 0 trang trôi, 5 xung đột giá trị + 1 không neo; KVK 4 trôi = 4 bất thường bản in + 22 mơ hồ + 7 xung đột + 14 không neo).
9. **Chrestomathie1872:** bảng truyện↔trang 20/20 đã dựng (cần 30 phút người biết Nôm rà), nhưng đơn vị chung duy nhất là **truyện** (120–870 âm) → không dùng engine cột↔cột; để Nhóm 1b.
10. **Thứ tự việc:** (1) config sách riêng + adapter ingest LVT1883 (B1: 10 cột × 14 chữ) → (2) sửa 2 dòng ghim 9 + tham số n_columns → (3) chạy thử 5 trang, kiểm cổng → (4) KVK1884 sau khi định vị câu dôi → (5) người kiểm ≥300 chữ/sách → (6) Chrestomathie mô-đun riêng.

---

## 1. Hiện trạng đầu vào từng sách (đã đo)

### 1.1 Bảng so sánh nhanh

| Mục | LVT1883 | KVK1884 | Chrestomathie1872 | STT (tham chiếu) |
|---|---|---|---|---|
| Ảnh Nôm | 105 JPG mode L, 1896–1910 × 3196–3212, nền xám 128 | 163 canvas chữ (4–166) 1500 × 2456–2863, nền xám 128 | 65 trang (canvas 106–170), 400 dpi | PNG 1647 × 2717 nền trắng (p50 = 255) |
| Bố cục | 2 tầng × 10 cột; cột = cặp lục bát (6 trên / 8 dưới) | như LVT | văn xuôi, lưới 7 ô cột/trang, 13,7–27,8 chữ/cột | 9 cột × ~22 chữ |
| Thứ tự trang | tên tệp = thứ tự đọc | **đảo**: canvas 166 = câu 1–20; `page = 167 − k` | tăng dần, cột đọc **trái→phải** | PDF |
| Số câu | 2.088 | Nôm 3.256 / QN số in 3253, **3.251 dòng vật lý** [measure_out 21/09] | 20 truyện, 419 cột ≈ 8.692 chữ Nôm, 8.151 âm QN [measure_out 21/09] | — |
| QN nguồn | 139 trang in, ~15 câu/trang, số mỗi 5 câu ở lề trái, cước chú Pháp | 295 canvas lẻ (vol1 27–317, vol2 7–303), ~11 câu/trang, xen trang Pháp | 25 trang (canvas 29–53), dòng in, không số câu | trang text PDF 9 dòng đánh số |
| Detector (thr 0,2, ảnh gốc) | 66,7 % cột đúng số hộp (n=6 trang; 9 trang: 64,4 %) | 19,2 % (9 trang: 25,3 %) | chưa đo cột | 93,2 % (cột OCR=QN, FLOW v3.1 N3f; N = kim, không phải GT người) |
| Detector (stretch/otsu, thr 0,2) | 82,2 % (n=180 cột, 9 trang); **77,6 % stretch / 76,3 % otsu (n=540, 27 trang)** [measure_out 21/09] | 80,3 % otsu (n=178); **77,6 % otsu / 68,9 % stretch (n=540)** [measure_out 21/09] | — | không đổi (90,1 % mọi cột n=81; 92,7 % cột N kim = chiếu, n=55 [measure_out 21/09]) |

### 1.2 LVT1883 — Nôm

| Chỉ số | Giá trị | Phương pháp · n · caveat |
|---|---|---|
| Tách 2 tầng | 105/105 | Otsu → chiếu ngang → gom hàng đầy; khe tầng 239–279 px (median 260); tầng trên 5,84–6,22 hàng, dưới 7,88–8,29 (trang 1: 9,63 do dấu BnF) · n=105 · phản biện đã sửa dải 8,05→8,29 |
| Số cột/tầng | 104 trang = 10; trang 105 = 5; trang 1: 10 trên (cột 1 = tựa 陸雲僊歌演) / 9 dưới | pitch cột median 156 px (149–168) · 210 tầng · tổng 2.089 đoạn = 2.088 câu + 1 tựa |
| Đoạn cột đúng 6/8 chữ | 2.075/2.089 = 99,3 % tự động; 2.088/2.088 sau kiểm mắt | chiếu ngang trong cột neo pitch · n=2.089 · **bằng chứng phụ thuộc** tách tầng (chỉ bắt chữ thiếu, không bắt thừa); 1 trường hợp nhãn số in lọt crop (p4 t0 c9) → crop cột phải cắt bỏ dải số |
| Số in (mỗi 5 câu) | 397/417 đọc nhất quán, 0 mâu thuẫn, 1 nhãn nhà in lệch cột (trang 71 '1410') | tesseract ensemble + kNN mẫu chữ số · n=417 · kiểm mắt của agent; kNN 99,5 % chỉ trên chữ số dễ, 20/111 số khó không cứu được |
| Hình học | pitch hàng 157 (153–159); rộng cột 146 (p5 130, p95 163); cao chữ 119; số in cách đỉnh tầng 6–49 px (median 27); nghiêng median 0,47°, max 2,3° | n=2.089 cột |
| Khe mực cột kề | median 9 px; 740/1.879 ≤ 5 px; **596/1.879 = 0 px** | n=1.879 khe · crop cột sẽ dính đuôi nét cột kề như STT |
| Xử lý tay | 1/105 (trang 1: bỏ cột tựa, cắt dấu BnF) | |

### 1.3 LVT1883 — QN (`quocngu_pages/`, 139 trang) và TSV hiện có

| Chỉ số | Giá trị | Phương pháp · n · caveat |
|---|---|---|
| tesseract psm 4, số dòng | 2.092 thô / 2.088; chuỗi theo vị trí neo khớp 2.088; trôi 7/139 trang + 2 mơ hồ. **[measure_out 21/09]: 2.088/2.088 dòng (897 dòng cước chú lọc theo vùng), 0 trôi, 0 mơ hồ; 5 trang xung đột giá trị + 1 không neo (`pages_review.json`)** | chain_analysis (s₁=1, sửa theo mod 5) · n=139 · tham chiếu tự quy chiếu, không GT người |
| Dòng đúng 6/8 xen kẽ | tess4 281/298 = 94,3 % (toàn sách 93,5 %); VietOCR nguyên trạng 84,3 % | 24 trang mẫu (12 LVT + 12 KVK) · 13/17 dòng lệch ±1 là tên riêng in nghiêng, 4/17 là token neo `ö_` chưa tách |
| Neo số câu (tess4) | tìm 362/419 = 86,4 %, giá trị đúng 275/419; **vị trí mod 5 nhất quán 339/347 = 97,7 %**. [measure_out 21/09]: đầu dòng 345 neo (giá trị 76,2 %, vị trí 99,1 %) + lề `--psm 7` 369 neo → 714 neo, phủ 407/417 dòng có số = 97,6 %, vị trí 99,6 %, giá trị 80,3 %; hai phương pháp đồng ý 207/304, khi đồng ý đúng 206/207 | đã đính chính cho phép `.`/`_` sau số · "đúng" so chuỗi tự sửa |
| Neo số câu (VietOCR) | giá trị đúng 371/388 = 95,6 % nhưng mất phần thơ 96/2.069 dòng (4,6 %) | toàn LVT · backend projection_deskew (`line_detector.py:114`) |
| CER thơ | tess4 7,1 % (thường 3,7 %, in nghiêng 16,8 %); phản biện 3 trang khác 5,6 %; VietOCR 14,0 %/13,6 % | 44 + 29 dòng · tham chiếu = mô hình đọc ảnh, không người |
| Dòng không-thơ lọt | ≥1/2.092 (p046 cước chú) | scan_junk |
| Âm tiết trong dict | 93,0 % | proxy; chính tả Nam Bộ (nhơn, chơn) vắng trong dict → không dùng làm cổng cứng |
| `luc_van_tien_quoc_ngu.tsv` | 2.088 dòng, verse_id đơn điệu, **963/2.088 dòng đảo hẳn 6↔8 + 101 dòng ngoài {6,8}** (đếm lại 21/09; phản biện đếm 1.007 với luật gộp ±1 → ≈46–48 %) ở 5 đoạn liên tục trong 2 vùng 371–1164 và 1380–1602; confidence = 1,00 toàn bộ 2.088 dòng (kiểm 21/09) | đếm âm tiết · **không dùng verse_id TSV để ghép cột** cho tới khi tái neo (đã tách khỏi git: `D data/LucVanTien1883/luc_van_tien_quoc_ngu.txt`) |
| Thời gian | tess 0,47 s/trang; VietOCR 9,5–14 s/trang CPU (139 trang: 22 phút) | Apple Silicon CPU |

### 1.4 KVK1884 — Nôm

| Chỉ số | Giá trị | Phương pháp · n · caveat |
|---|---|---|
| Canvas chữ | 4–166 (163 trang); loại 1 (bìa Pháp), 2–3, 169–171 (trắng), 167 (thơ Hán 詩云…華堂范先生撰), 168 (trang tên 金雲翹新傳) | mật độ mực · n=171 |
| Tách tầng / hàng | 163/163 hai tầng; 6+8 hàng 163/163; khe tầng 299 px [274–315] (mực–mực; hộp tầng đệm ±15 px = 260 [228–283], cùng định nghĩa LVT [measure_out 21/09]); pitch hàng 127; glyph cao 85 (P5 75, P95 96); cột đúng 6/8 chữ 3.192/3.257 = 98,0 % [measure_out 21/09] | chiếu ngang ≥90 px mực/hàng · n=163, 2.282 hàng |
| Số cột | 162/163 = 10+10; canvas 4 = 8+1 tiêu đề (金雲翹傳卷完) / 8 | pitch cột 124 [114–142]; std max 48,7 (canvas 4), 28,7 nếu loại; **canvas 113 tầng dưới cột 5–6 nén ~70 px** → engine pitch cố định sẽ hỏng ở trang này |
| Lề trái | canvas LẺ x_min ≈78 px [33–242], CHẴN ≈190 [116–257] | **không cắt lề trái > 2 % (30 px)**, nếu không mất cột 10 (đã xảy ra canvas 17, 19) |
| Số in | 651/652 có in (thiếu 3260 canvas 4); detector bắt 649/651 (sót canvas 111 t0c3, 113 t1c5) | dải 170 px trên đỉnh tầng, cao 15–23 px, cách đỉnh 25–100 px → **phải loại khỏi bước dò glyph** |
| Đọc số | tesseract **thất bại** (≈1/4 đúng); kNN tự dựng: đọc tự do 538/649 = 82,9 %, giải mã theo trang 162/163 tự động, 163/163 sau kiểm mắt canvas 149 (3↔8). [measure_out 21/09, không nhãn tay]: kNN tự huấn từ tesseract nhất trí (1.169 chữ số, CV 97,4 %) đọc đúng 472/651 = 72,5 %; giải mã theo trang + Viterbi: first_seq đúng **163/163** hoàn toàn tự động | nhãn cụm 2.070 glyph do agent gán; 5 canvas biên < 3 đều 3↔8 |
| Ánh xạ | `page_index = 167 − k` (k = 4…166); câu trang p = 20(p−1)+1 … min(20p, 3256) | vị trí số → first ≡ 1 (mod 10); pha mod 20 suy từ 20 cột/trang + giá trị đọc mắt 5 canvas · bảng `kvk1884_canvas_map.csv` |
| Số trang in góc | 126/163 khớp `169 − k`; [measure_out 21/09]: 143/162 tự do, **161/163 Viterbi** (lệch canvas 8, 9) | bằng chứng phụ; canvas lẻ số ở góc phải, chẵn góc trái |
| **Tổng câu** | **3.256** | số in canvas 4: 3245/3255 (trên) 3250 (dưới); 162×20+16 |

### 1.5 KVK1884 — QN (`quocngu_pages/vol1`, `vol2`)

| Chỉ số | Giá trị | Phương pháp · n · caveat |
|---|---|---|
| Phân loại 631 canvas | QN 295 (vol1 lẻ 27–317 = 146; vol2 lẻ 7–303 = 149); FR 313; khác 23 | tesseract vie psm 4 + vị trí số trang + tỉ lệ ký tự Việt · phản biện đã sửa FR 251→313 |
| Dòng thơ | 3.261 thô; chuỗi số in tới **3253**; trôi 11/295 trang + 28 neo mơ hồ. **[measure_out 21/09]: 3.251 dòng vật lý (vol1 1.497 + vol2 1.754), chuỗi 3.253, 4 trôi + 22 mơ hồ + 7 xung đột + 14 không neo** | 8 dòng trôi = trích Hán văn + phiên âm lọt lưới ở vol2 canvas 301, không phải "phần kết" |
| Đánh số | nhảy 1581→1585 (vol2 canvas 21/23) → từ 1585 câu lẻ có 8 âm; số dòng thật ≈ 3253 − 1 = **3.252** (28 trang mơ hồ chưa soát). [measure_out 21/09, `verses.tsv`]: **4 bất thường**: 1581→1583 (vol2 c23), 2220→2222 (c135), 2232 lặp (c137), 3053→3055 (c269); parity đảo tại 1583 và 2225, đảo lại tại 2235; kiểm ảnh c135: 11/11 dòng OCR đủ, số in 2225/2230 thật → dòng vật lý = **3.251** | kiểm ảnh |
| Đúng 6/8 | toàn sách tess4 96,6 % | luật xen kẽ đảo pha tại 1585 |
| Neo (tess4) | tìm 599/651 = 92,0 %, giá trị đúng 417/651; vị trí mod 5 nhất quán 546/551 = 99,1 % | đã cho phép `.`/`_` |
| vol1/vol2 | vol1 kết 1497; vol2 canvas 7 = 1498–1503 | |
| Dòng không-thơ lọt | ≥11/3.261 (Hán văn 8, Pháp 1, rác 2) | scan_junk · cần lọc thêm theo tỉ lệ âm trong dict < 0,5 |
| **Khớp Nôm↔QN** | Nôm 3.256 vs QN **3.251 dòng** → **lệch 5 câu, chưa định vị** [measure_out 21/09]; giả thuyết: bản QN bỏ câu có trong Nôm tại các chỗ nhảy số (mỗi lần bỏ 1 câu làm đảo parity, đúng như quan sát ở 1583/2225) → ghép theo đoạn giữa các neo, `verses.tsv` có `verse_no` + `seq_no` + `page_flag` | **CHẶN** ghép KVK cho tới khi định vị |

### 1.6 Chrestomathie1872

| Chỉ số | Giá trị | Phương pháp · n · caveat |
|---|---|---|
| Phạm vi | QN văn bản canvas 29–53 (25 trang; 54/56 trắng, 55 tựa 'Traduction française', 57–101 Pháp); Nôm 106–170 (104 tựa, 105/171 trắng) | SOURCE.md lệch 1–3 canvas → cần đính chính |
| Truyện | 20 (I–XX), 610 dòng thân, 8.177 âm OCR (608 / 8.151, OOV 4,76 % [measure_out 21/09]); tiêu đề số La Mã tesseract đúng chỉ 6/20 → dò tiêu đề theo bố cục (tự động khớp REF 17/20 [measure_out 21/09]) | |
| Bố cục Nôm | 7 ô cột/trang (pitch 133–143 px), ≈416 cột chữ thật (420 − 4 nhiễu), ≈8,7k chữ; đọc cột **trái→phải**. [measure_out 21/09]: 419 cột, 8.692 chữ, 21,3 chữ/cột đầy, pitch↔run đồng ý 94,7 % | n_est ±10–15 % |
| Bảng truyện↔trang | 20/20 (I 106–109 … XX 167–170) | đọc mắt số đếm Nôm 傳次 + tỉ lệ chữ/âm 0,96–1,24 · chưa người biết Nôm rà |
| Ranh giới tự động | precision 18/18, recall 18/20 (quy tắc cài lại + loại cột nhiễu); [measure_out 21/09] `chresto_map.py`: precision 18/18, recall 18/19 (bỏ đầu sách; sót 137/0) sau khi sửa luật "thiếu cột cuối trang" | chỉnh trên chính 20 ranh giới, chưa kiểm chéo |
| OCR QN tesseract | OOV 4,83 % (cận dưới); **lỗi âm tiết thật ≈13 %** (14/107) | 8 dòng 1 trang · phần lớn lỗi tạo âm hợp lệ (chồng→chống) |
| NomNaOCR cục bộ | định vị đúng truyện 7/7 (trọng số gốc), 6/7 (fine-tuned STT); CER 0,61–0,74 | đủ để định vị trang→truyện, **không đủ gán nhãn** |
| Khuyên son | scan xám, không có kênh đỏ; recall bộ dò 25–85 % (2 trang) | chưa dùng làm neo |

### 1.7 Detector CenterNet trên thạch bản (`train_crop/detector_r34.best.pt`)

| Chỉ số | LVT1883 | KVK1884 | Phương pháp · n · caveat |
|---|---|---|---|
| Hộp/trang ảnh gốc, thr 0,1/0,2/0,3/0,4 (kỳ vọng 140) | 142,8 / 132,5 / 76,8 / 11,8 | 137,0 / 101,7 / 39,5 / 4,0 | n=6 trang/sách; điểm trung vị 0,30 / 0,25 (STT 0,44–0,45) |
| % cột M==N, ảnh gốc | thr 0,1: 86,7 (mở rộng 9 trang: 83,9); thr 0,2: 66,7 (9 trang: 64,4) | 75,8 (78,7); 19,2 (9 trang: 25,3) | gán kiểu pipeline x_range ±0,25w · N = 6/8 giả định lục bát, không GT hộp |
| % cột M==N, **stretch** (p2→0, p90→255), thr 0,2 | 85,0 → gộp 9 trang **82,2 % (148/180)** | 70,0 → 74,2 % | CI 95 % ≈ ±6 điểm, cột cùng trang tương quan |
| % cột M==N, **otsu**, thr 0,2 | 84,2 | 77,5 → gộp **80,3 % (143/178)** | |
| % cột M==N, **27 trang/sách = 540 cột** [measure_out 21/09] | raw 61,7 / stretch **77,6** / otsu 76,3 | raw 23,5 / stretch 68,9 / otsu **77,6** | mẫu 9 trang cũ tái lập đúng (64,4 / 82,2 / 81,7; 25,3 / 74,2 / 80,3) bằng `--page-ids`; CI ±3,5; `find_blocks` lề trái 6 → 2 % (mất cột 10 KVK 4/27 trang) |
| Phân bố M−N (stretch, thr 0,2) | {−2:1, −1:10, 0:102, +1:6, +2:1} | {−3:5, −2:2, −1:21, 0:84, +1:8} | n=120 cột |
| Hộp ngoài khối | gốc: 2 @0,1, 0 @≥0,2; stretch 10 @0,1, 1 @0,2 | otsu 3 @0,1, 0 @0,2 | ở thr 0,1 sau stretch bắt nhãm số câu → phải lọc theo y |
| Kích thước hộp | w 139 / h 174; w/pitch 0,88; h/pitch_y 1,12; ink recall 0,988 (p10 0,871) | w 108 / h 136; 0,86; 1,08; 0,992 (p10 0,805) | dy_cv 0,065–0,075 (STT 0,055) |
| Phóng to | letterbox 1536 hoặc dò riêng khối → M==N rơi 13–26 % | | **KHÔNG phóng**; model đặc thù thang chữ ~38 px @1024 |
| Thời gian | 0,7 s/trang CPU; CPU ≡ MPS | | |
| Tách chữ chiếu ngang (dự phòng) | lục 72 %, bát 63 % | 83 % / 60 % | kém detector+stretch; [measure_out 21/09] gom cụm neo pitch hàng (thuật toán layout) == N: LVT 100 %, KVK 99,4 % — **không độc lập với giả định 6/8** (dùng pitch), chỉ là kiểm chéo |

Tham chiếu STT: 70,4 % trên mọi cột theo N kim (n=54, kim đếm sai N) và **93,2 %** trên cột OCR=QN (FLOW v3.1 N3f) — thạch bản sau stretch vẫn kém mốc sạch 8–13 điểm. Ckpt VAL = 44 trang chẵn STT11 (0010–0102); STT2 nằm trong TRAIN.

---

## 2. Hợp đồng dữ liệu đầu vào cho engine (`prepared/<Book>/`)

Copy từ báo cáo đọc mã (step1_ocr + align_engine), đã đối chiếu 3 nguồn. `Book` = khoá `name` trong `books:` của config; mã sách trong labels/crop = `_book_code(name)` = `name.lower()` cho sách không phải STT (`build_dataset.py:76-87`), ví dụ `lucvantien1883`, `kimvankieu1884`.

### 2.1 Tệp bắt buộc

| # | Tệp | Ai đọc (file:line) | Yêu cầu |
|---|---|---|---|
| 1 | `pages/<page>.png` | `align_production.py:47`, `build_dataset.py:1152`, `step2_align.py:118` | **PNG** (đuôi ghim cứng); chuyển từ JPG bằng `PIL Image.save(..., 'PNG')` giữ nguyên kích thước (0,06 s/trang, ~877 KB); tên bắt đầu `page_`, sắp **lexical = thứ tự đọc** (`page_0001…`) |
| 2 | `detected/<page>_ocr_cache.json` | `align_production.py:53-57` (thiếu → trang bị **bỏ im lặng**), `nom_detect_v3.py:27,92`, `anchor_align.py:56` | xem §2.2 |
| 3 | `transcriptions/<page>.txt` | `step2_align.py:61-92` → `load_v1_transcription` (`parser_v2.py:133`) | mỗi dòng = 1 cột QN theo đúng thứ tự `columns[i]`; **không** đánh số, **không** dấu câu/«»/cước chú; âm tiết đã qua `clean_line_text` + `split_to_syllables` (+ `normalize_syllables`) rồi `' '.join` — vì đường .txt **không** clean (đo: 'Vân-Tiên', "d'oiseau" giữ nguyên nếu không clean trước); số dòng **không** giới hạn (đo 20 dòng → 20 khoá) |
| 4 | `transcriptions/<page>.json` | `build_dataset.py:1123` (glob `page_*.json`, loại `*_qn_ocr_cache.json`) — **chỉ liệt kê trang, không đọc nội dung** | phải tồn tại; khuyến nghị schema như step1: `{book_page:int, columns:[{column, raw_text, cleaned_text, syllables[], num_syllables}], qn_line_confidences, qn_page_confidence}` |

### 2.2 Khoá JSON của `detected/<page>_ocr_cache.json`

```
{
  "image": "prepared/<Book>/pages/<page>.png",   // chỉ backfill_pixel_hash đọc
  "image_hash": "<md5 BYTE của CHÍNH tệp pages/<page>.png đã ghi>",   // ocr_api.py:533 _file_md5 (working tree 21/09)
  "pixel_hash": "<md5(f'{mode}|{size}|' + im.tobytes())>",           // ocr_api.py:541 _pixel_hash
  "framed": false,            // BẮT BUỘC false
  "frame_pad": 0,
  "coords_space": "fullpage", // BẮT BUỘC; nếu thiếu → load_columns_fullpage dịch bbox theo frame (ocr_api.py:656; đo [10,10,20,20]→[225,433,235,443])
  "n_columns": <int>,         // tuỳ chọn, không ai đọc
  "columns": [                // THỨ TỰ ĐỌC; index i ghép với dòng i+1 của .txt (iter_pairs, align_production.py:86)
    [ {"char": "<1 ký tự Nôm | null>", "y_center": <float, tuỳ chọn>, "bbox": [x1,y1,x2,y2]}, ... ],  // chữ TRÊN→DƯỚI, pixel toàn trang PNG
    ...
  ],
  "boxes_raw": []             // tuỳ chọn, không ai đọc
}
```

Khoá thật sự được đọc: `char` (`anchor_align.py:56` `_char_of`; None → `substitution_cost` `:61-72` trả COST_DICTMISS 6,7 nên **vẫn ghép được**, chỉ không lên tier — 0 bản ghi chỉ khi `columns` rỗng), `bbox` (`run_full.py:23,51`; `nom_detect_v3.py`; `align_production.py:604`). x-range cột = min/max x của bbox chữ → **phải là cột đơn**, không phải cả khối. Mỗi cột ≥ 4 chữ để `nom_cols_hybrid(min_len=4)` coi là cột thân.

Hash: engine build **không** verify hash (`align_production.py:53-57` đọc JSON thẳng); `verify_cache_image` (`ocr_api.py:562`) chỉ chạy ở step1: image_hash khớp → 'ok'; byte lệch nhưng pixel_hash khớp → 'healed' (tự vá); thiếu image_hash → 'skipped'. Ghi cả hai từ chính PNG đã lưu để 'ok'.

### 2.3 Tệp KHÔNG ĐƯỢC ghi / tuỳ chọn

| Tệp | Quy định | Lý do (đo) |
|---|---|---|
| `transcriptions/<page>_qn_ocr_cache.json` với `text` ≠ "" | **CẤM** cho sách 10/20 cột | `step2_align.py:61-90` ưu tiên nó → `parse_v5(max_lines=9)` (`parser_v5.py:136`) ép về 9 dòng; đo: dòng 9 nuốt câu 9..20 (95–107 âm tiết), trả 'v5' hợp lệ, **không cảnh báo**. Muốn lưu vết OCR: tên khác hoặc `text=""` |
| `pages_denoised/<page>.png` | tuỳ chọn | thiếu → binarize từ `pages/<page>.png` (`align_production.py:76`); ảnh hưởng lên carve/tighten cho thạch bản **chưa đo** |
| `manifest.json` | tuỳ chọn | chỉ `step4_export.py:129` đọc, không trong run_pipeline.sh |

### 2.4 Đầu ra engine

`$out/labels.csv` (42 cột từ engine; **45** sau `enrich_crop_quality` thêm crop_quality_flag/stray_ink/border_ink — kiểm header `git show HEAD:dataset_out/labels.csv`; gồm image, book, page, column=line_id, ocr_char, syllable, label, tier, rule, s3_cosine, bbox, nom_idx, syl_idx, p_register, n_ocr/n_qn/n_det, box_source, tier_v3, …), `summary.json`, `pair_pages.json`, crop `$out/{gold,silver,syllable[,review]}/<book>_<page>_cNN_III.png` (tên đủ cho 20 cột). Export chỉ **LOG** trang ≠ 9 cột (`export_final_dataset.py:127,184`), không loại.

---

## 3. Adapter ingest cho sách JPG (đặc tả, không viết mã)

Vị trí đề xuất: `pipeline/tools/ingest_offline_book.py` — **hiện KHÔNG tồn tại** (docs/README.md:102 mô tả sai). Bước 1 hiện tại chỉ nhận PDF (`step1_extract.py:43,61`) → adapter **thay thế toàn bộ Bước 1** cho sách mới.

### 3.1 CLI đề xuất

```
python -m pipeline.tools.ingest_offline_book \
    --book LucVanTien1883 \
    --nom-dir data/LucVanTien1883/pages \
    --qn-dir  data/LucVanTien1883/quocngu_pages \
    --layout  couplet10            # 10 cột × (6 trên + 8 dưới); hoặc verse20
    --page-map identity|reverse:167   # KVK: page = 167 − k
    --qn-ocr tess4                 # tesseract psm 4 (mặc định), vietocr-split
    --expected-verses 2088
    --contrast stretch|otsu|none   # chỉ ảnh hưởng bước dò hộp, KHÔNG ghi đè pages/*.png
    --out prepared/LucVanTien1883
    [--pages 1-5] [--dry-run] [--report ingest_report.json]
```

### 3.2 Hàm và cổng theo giai đoạn

| Giai đoạn | Hàm (đề xuất) | Vào | Ra | Cổng kiểm định (fail → dừng trang, ghi report) |
|---|---|---|---|---|
| I1 Đổi tên & PNG | `map_pages(nom_dir, page_map)` | JPG canvas | `pages/page_NNNN.png` (PIL, giữ pixel) | KVK: `page = 167 − k`, k = 4…166; loại 1–3, 167–171; LVT: identity |
| I2 Tách tầng | `split_tiers(png)` | PNG | 2 dải y (trên/dưới) | đúng 2 tầng; khe LVT 239–279 / KVK 274–315 px; tầng trên ≈6 hàng, dưới ≈8 (LVT 5,84–6,22 / 7,88–8,29) |
| I3 Tách cột | `split_columns(tier, n_expected=10)` | dải tầng | 10 x-range/tầng | 10/10 (trừ LVT p1 = 10/9 + tựa, p105 = 5/5; KVK canvas 4 = 8+tiêu đề/8); **lề trái ≤ 30 px** (KVK); pitch LVT 149–168 / KVK 114–142; cảnh báo std pitch > 30 (KVK canvas 113) |
| I4 Lọc số in | `mask_verse_numbers(tier_top_strip)` | dải 0,9 pitch (LVT) / 170 px (KVK) trên đỉnh tầng | mask | số in cách đỉnh tầng LVT 6–49 px, KVK 25–100 px, cao 15–25 px → **cắt bỏ khỏi crop cột và khỏi dò hộp**; đọc số bằng kNN mẫu (tesseract KVK ≈1/4 đúng) chỉ để kiểm mức trang |
| I5 Hộp chữ | `detect_boxes(png, contrast, thr=0.2)` | PNG (bản kéo tương phản tạm) | hộp toàn trang | `DetectorInfer` letterbox 1024, **không phóng**; lọc theo x-range cột **và y-range tầng** + NMS dọc 0,45; **KHÔNG ép về N** (ép top-N làm `n_ocr==n_qn` luôn đúng → luật G §5 thành tautology và che lỗi detector); ghi hộp thật, cờ `n_det ≠ N` vào report |
| I6 Ghi cache | `write_ocr_cache(page, columns)` | hộp đã gán | `detected/page_NNNN_ocr_cache.json` | schema §2.2; **B1:** `columns[k]` = hộp tầng trên (≈6) ⧺ hộp tầng dưới (≈8) cùng x-range, sắp theo y (≈14 chữ); ghi `y_range` từng tầng vào report (engine không đọc); `char` theo chế độ §5 (kim / null / nguồn khác + khoá `char_source`); hash từ PNG đã lưu |
| I7 OCR QN | `ocr_qn_pages(qn_dir, engine=tess4)` | JPG QN | dòng + toạ độ + conf | quy tắc `extract_verses` 8 luật (§3.4); loại dòng Pháp/cước chú; giữ token số đầu dòng kể cả `NNN.`/`NNN_` |
| I8 Chuỗi câu toàn sách | `build_verse_chain(lines, expected)` | dòng QN mọi trang | `verses[verse_id] = syllables` | s₁=1, s_{k+1}=s_k+n_k; sửa theo **vị trí neo mod 5** trước, theo giá trị chỉ khi ≥2 neo cùng giá trị và \|Δ\| ≤ 25; cờ trang trôi/mơ hồ cho người |
| I9 Ghép câu→cột | `assign_verses_to_columns(page, chain)` | page p | .txt + .json | trang p LVT: câu 20(p−1)+1…; cột k = câu 2k−1 (trên) ⧺ 2k (dưới) (B1) hoặc 2 dòng (B2); **kiểm số âm tiết ∈ {6,8} (KVK đảo pha từ 1585)** |
| I10 Báo cáo | `write_report()` | | `ingest_report.json` | tổng cột = tổng câu kỳ vọng; số dòng .txt == len(columns) từng trang; liệt kê trang cần tay |

### 3.3 Đặc thù từng sách

**LVT1883**
- Trang 1: bỏ cột tựa 陸雲僊歌演 (cột 1 tầng trên), cắt dấu BnF dưới tầng dưới; 18 câu (9 cột). Trang 105: 5 cột, 10 câu. Còn lại 20 câu/trang → tổng 18 + 20×103 + 10 = 2.088.
- QN: 139 trang, ~15 câu/trang, **không 1:1 với trang Nôm** → bắt buộc I8 chuỗi toàn sách. Tesseract đọc số lề kém (giá trị đúng ~76 %) → dùng vị trí; nếu chạy thêm VietOCR thì phải **tách hộp dòng thành [cột số | thơ] tại x_text** (đo 56 dòng neo: mất text 8→0, số đúng 52/56) và lấy đồng thuận (31/31 đúng khi 2 bộ đồng ý; bất đồng ưu tiên VietOCR 11/13).
- **Không dùng** `luc_van_tien_quoc_ngu.tsv` làm QN (≈48 % verse_id lệch parity).

**KVK1884**
- Đổi tên `page = 167 − k`; canvas 4 là page_0163 (16 câu: 8 cột/tầng + cột tiêu đề 金雲翹傳卷完 phải loại).
- QN: 295 canvas lẻ; vol1 kết 1497, vol2 canvas 7 = 1498–1503; số in nhảy 1581→1585 → luật xen kẽ 6/8 đảo pha từ 1585; ≥11 dòng không-thơ lọt (Hán văn vol2 canvas 301) → thêm luật lọc tỉ lệ âm trong dict < 0,5.
- **CHẶN:** Nôm 3.256 câu vs QN ≈3.252 dòng (số in 3253). Phải định vị các câu dôi (so văn bản Nôm↔QN theo cửa sổ, hoặc soát 28 trang neo mơ hồ + 2 vùng nghi) **trước** I9; nếu không, mọi cột sau điểm lệch ghép sai 1–4 câu.
- Canvas 113 tầng dưới: cột 5–6 nén → I3 phải dùng cực tiểu chiếu dọc, không pitch cố định.

**Chrestomathie1872 — trạng thái: CHỜ (Nhóm 1b)**
- Lý do đo được: Nôm chảy liên tục qua cột/trang (≈416 cột, 13,7–27,8 chữ/cột), QN là 610 dòng in **không số câu**; đơn vị chung duy nhất = truyện (120–870 âm) → engine cột↔dòng đếm bằng nhau không áp dụng; cần mô-đun "căn chỉnh chuỗi cả truyện" (ước 1–2 ngày mã) + tách ô chữ trong cột (1 ngày) + bộ dò khuyên (recall hiện 25–85 %).
- Đã có: bảng truyện↔trang 20/20 (`scratchpad/wf/chresto/bang_truyen_trang.tsv`) cần 30 phút người biết Nôm rà trước khi chép vào `data/Chrestomathie1872/`; đính chính SOURCE.md (QN 29–53, Nôm 106–170).
- Không đưa vào lượt chạy đầu.

### 3.4 Quy tắc lọc QN (đã đo trên 24 trang + toàn sách tess4, `extract_verses.py`)

1. Bỏ dòng ở 3 % mép trên; dòng rác (<3 chữ cái / ≤2 token không dấu Việt).
2. Tiêu đề chạy = dòng ở 40 % trên, ≥2 từ của tựa sách và ≥60 % chữ hoa, hoặc chỉ số.
3. Gộp token số lề tách rời vào dòng chồng lấn dọc ≥50 % bên phải.
4. Khối cước chú bắt đầu ở dòng đầu có ≥2/5 tín hiệu: cao chữ <0,78 H (tess)/0,85 H (VietOCR); bước dòng <0,6 bước thơ; hư từ Pháp ≥25 % hoặc `1)`/`1.`/`Litt`/`»`; x1 lệch trái >8 % W; ≥10 từ.
5. Trong vùng thơ loại dòng Pháp và dòng <3 âm tiết; **thêm (đính chính):** loại dòng có tỉ lệ âm trong dict < 0,5 hoặc ≥3 ký hiệu lạ.
6. Số câu = token đầu dòng gồm chữ số/ký tự nhầm cố định (ö õ ð→5, O o Q D→0, I l ï→1), **cho phép** `.`/`_`/`,` sau số; **không** ánh xạ S/B/Z/G.
7. Âm tiết = token có chữ cái sau tách gạch nối, bỏ dấu câu.
8. Chuỗi toàn sách: sửa theo vị trí mod 5 trước, theo giá trị khi ≥2 neo cùng giá trị và \|Δ\| ≤ 25.

Caveat chung: luật tinh chỉnh trên chính 2 sách này; không có GT người cho QN; tham chiếu số dòng là chuỗi tự sửa.

---

## 4. Thay đổi mã TỐI THIỂU trong engine

Nguyên tắc: mặc định giữ nguyên hành vi STT (n=9, 1 tầng); kiểm hồi quy bằng `scripts/run_all_selftests.sh` (BASELINE_PASS=1140) + so byte `labels.csv` STT trước/sau.

| # | file:line | Hiện tại | Thay đổi | Rủi ro với STT |
|---|---|---|---|---|
| A1 | `config/pipeline.yaml:11-20` | `books: [{name, pdf, reocr}]` | thêm tuỳ chọn `layout: {n_columns: 10, tiers: 2, qn_per_column: couplet}`, `det_thr`, `contrast`, `source: images`; **khoá `pdf` vẫn phải có** (`step0_setup.py:49` `book["pdf"]` và `step1_extract.py:40` `book_cfg["pdf"]` truy cập trực tiếp → KeyError nếu vắng); chỉ WARN/return sớm khi tệp không tồn tại (`step0_setup.py:50-52`, `step1_extract.py:43-45`) → ghi `pdf: data/LucVanTien1883/LucVanTien_1883_TranNguyenHanh.pdf` (có thật nhưng không dùng) | không (khoá mới, mặc định vắng) |
| A2 | `build_dataset.py:1135` | `align_page(page, data_dir, …)` | truyền `layout` từ `b.get('layout')` | không |
| A3 | `align_production.py:81` | `detect_nom_columns_v3(binary, ocr_columns, 9)` | `n_exp = layout.n_columns if layout else 9` | không khi layout vắng; đo: n_expected=10 cho 10 hộp đều (LVT 140–176, KVK 106–142 px), n=9 luôn gộp 2 cột (10/10 trang) |
| A4 | `align_production.py:87` | `qn_parse_ok = (len(qn_lines) == 9)` | `== n_exp` | không |
| A5 | `step2_align.py:61-92` | `parse_v5(text, qn_dict)` max_lines mặc định 9 | thêm tham số `max_lines`; **hoặc** không sửa và adapter không ghi `_qn_ocr_cache.json` (đường .txt v1) — khuyến nghị cách 2 | cách 2: 0 rủi ro |
| A6 | `parser_v5.py:88,136` | P7 rogue-1 nhầm '12.' với '2.' khi max_lines>9 | chỉ cần nếu A5 cách 1: tắt P7 khi max_lines>9 | không |
| B1 | (không sửa) | 1 cột = 1 cặp lục bát 14 chữ; .txt dòng k = câu 2k−1 ⧺ 2k | **khuyến nghị**: 0 mã tách tầng; ghi sidecar `verse_id` từ syl_idx (< len(câu lẻ) → 2k−1) trong `build_dataset._record` — ghi **độ dài câu lẻ thật** vào sidecar, không giả định 6 (câu 5/7/9/11 âm 1,7 % trong TSV) | `_reseg_column` (`align_production.py:226`) hộp midpoint chữ 6/7 kéo qua khe tầng ~100 px → tighten_box cắt về mực, chấp nhận được; detector: \|G\|≈14 == n_qn |
| B2 | `align_production.py:81` (sau), `detector_infer.py:76-89`, `infer_centernet.py:243`, `align_production.py:442,580` | 20 cột thật | nếu `layout.tiers==2`: tìm khe ngang lớn nhất trong text_box (113–244 px, 10/10 trang), tách chars mỗi cluster theo y thành 2 cluster con có `y_range`; `raw_column_boxes`/`column_boxes` nhận `y_range` tuỳ chọn | trung bình: `nom_cols_hybrid` (`nom_detect_v3.py:27`) gom chỉ theo x nên 20 rec cùng x bị `_close_pair_merge` gộp lại → phải bypass; **chưa đo** |
| C | `build_dataset.py:1010`, `align_production.py:196` | `DETECTOR_THR` toàn cục 0,2 | `books[i].det_thr` ghi đè trong vòng PASS 1; `_get_detector` đã cache theo thr | không; đo: thr 0,1 gốc ≈ thr 0,2 stretch |
| C' | `align_production.py:72` `load_and_binarize` / `DetectorInfer` | đọc ảnh gốc | tiền xử lý theo sách `contrast: stretch\|otsu` **chỉ cho detector**, STT `none` | không khi none; đo LVT 66,7→82,2 %, KVK 19,2→80,3 % |
| D | `build_dataset.py:137-149`, `tier_v3.py:41-53,68-83,156` | pair_pages/corpus2/4/bigram từ mọi sách, LOO chỉ trừ (book,page) | cờ `--anchor-scope book\|all` lọc `pg[0]==book`; **hoặc** config riêng 1 sách (0 mã) — khuyến nghị cách 2 | cách 2: 0 rủi ro; STT ANCHOR_MIN_OTHER_PAGES=2 chỉnh trên 448 trang, LVT 105/KVK 163 trang ít cặp đủ ngưỡng hơn (chưa đo) |
| E | `run_pipeline.sh:82-107,236-254,311-348` | menu ghim 3 STT; `check_ocr_columns.py:23-24` EXPECTED=9; rescue pseudo CSV/npz 100 % stt* | **không gọi run_pipeline.sh** cho sách mới; gọi từng bước tay (§6); bỏ rescue | không |
| F | `config/decisions.yaml:18`, `confusion_fix.py:30` | 22/23 mục `book: null`; fix (người,㝵) không lọc sách | chạy `--decisions none`; bỏ confusion_fix cho sách mới; hoặc thêm `book:` vào từng mục | không nếu chỉ đổi lệnh gọi |
| G | `apply_tier_v3` `build_dataset.py:485` | không ocr_char → REVIEW/no_context | **chính sách (cần người dùng duyệt)**: rule mới `positional_equal_count` → SYLLABLE khi n_ocr==n_qn==N kỳ vọng, ops2 không ins/del, box_source detector — đặt **trước** nhánh `not ch` | không cho STT (luôn có ch); nhưng đây là nhãn vị trí, không phải nhãn đồng thuận |
| H | `ocr_api.py:389,406,678` (working tree 21/09) `boxes_to_columns`, `expected_cols=9` | gộp \|Δx\|<15 px, chia đều cao hộp | chỉ khi kim sống lại: expected_cols theo sách, OCR **từng tầng** riêng để 2 tầng cùng x không thành 1 cột 14–20 chữ | không |
| I | `export_final_dataset.py:127,184`, `pipeline/tools/selftest.py:443-467`, `audit_grid.py:68`, `eval_crops_v2.py:58`, `recrop_v2.py:35` | 'trang không đủ 9 cột'; `book_to_scan_dir` regex số → `SachThanhTruyen1883` | tham số theo sách hoặc bỏ qua cho sách mới; audit_grid cần bảng book→scan_dir | không nếu để nguyên và chấp nhận log đỏ |

Ghi chú: chỉ **A1–A4 + C/C'** là bắt buộc để build đúng cột; A5 cách 2, D cách 2, E, F là thay đổi **lệnh gọi**, không phải mã.

---

## 5. Chế độ chạy và tier khả thi

Điều kiện tier trích từ mã (đường `--two-pass` mặc định):

- `consensus.py:128` `decide_label`: `ocr_char ∈ readings(syllable)` → GOLD `s1_inter_s2_direct`; `:135` similar-bridge; `:158` ngược; SILVER `:171,:193` đòi `s3 ≠ None`; REVIEW `diverged_column | below_visual_threshold | unconfirmed_no_s3`.
- `build_dataset.py:485` `apply_tier_v3`: không plausible → REVIEW/not_plausible; **không ocr_char → REVIEW/no_context** trước khi vào `tier_v3`.
- `tier_v3.py:92-99`: CHAR_A = direct ∧ p≥0,8; CHAR_B = direct hoặc sim_unique ∧ ctx ∧ p≥0,8; SYL = p≥0,5 ∧ (bigram|corpus4|tone); `direct(None)=False`, `sim(None)=∅`. `TIER_OF` **không có SILVER** → đường two-pass không bao giờ sinh SILVER.
- `build_dataset.py:809` `maybe_s3` trả None khi không ocr_char → S3 không chạy cho ô không có S1.
- `run_pipeline.sh:257` không truyền `--use-s3`, `--visual-emission` → SILVER = 0 ngay cả với STT (đo dataset_out: GOLD 49.946 / SYLLABLE 18.925 / REVIEW 14.368).

| | (A) Có kim (API mở) | (B) Không kim — ngoại tuyến |
|---|---|---|
| `columns[].char` | từ kim, OCR **từng tầng** (H) | `null` hoặc từ NomNaOCR/CenterNet **kèm khoá `char_source`** (engine không đọc khoá này — `consensus.py:128` không phân biệt nguồn) |
| GOLD | có (`s1_inter_s2_direct`, CHAR_A/B) | **KHÔNG** — nếu điền char từ NomNaOCR thì engine sẽ báo GOLD như kim → **cấm** để char không rỗng trừ khi chặn tier theo sách sau build |
| SILVER | 0 (như STT, --use-s3 tắt) | 0; "SILVER-S3" chỉ có nếu bật `--use-s3` **và** có ocr_char → không |
| SYLLABLE | có (SYL: p≥0,5 ∧ ngữ cảnh) | chỉ qua luật mới G `positional_equal_count` (chưa có trong mã); **điều kiện tiên quyết**: adapter không được ép số hộp về N (§3.2 I5), nếu không `n_ocr==n_qn` luôn đúng và luật G gán SYLLABLE cho cả cột detector đếm sai |
| REVIEW | phần còn lại | **100 %** với mã hiện tại (đo mô phỏng: 6 char=None ↔ 6 âm → 6 match, p_register 0,999–1,0, tier REVIEW/no_context) |
| Ghi nhãn | như STT | **phải ghi rõ**: "nhãn vị trí, không đồng thuận S1∩S2"; cột `tier_goc`/`rule_goc` giữ `positional_equal_count`; không được gộp với GOLD STT trong số liệu luận văn |
| NomNaOCR làm S1 | không cần | **không khuyến nghị** cho LVT/KVK: pretrain gồm Lục Vân Tiên 104 trang + Kiều 1866/1871/1872 (`NomNaOCR/ds4v_repo/README.md:81`) = rò rỉ văn bản; CER trên thạch bản 0,61–0,74 (Chrestomathie, n=61–98) |
| Self-training rescue | tắt (pseudo CSV/npz 100 % stt*; `self_training_rescue.py:128-160` — nhánh `elif` :160 chỉ chạy model khi KHÔNG có pseudo CSV; nếu ép chạy: tự phong GOLD tau 0,70 trên miền lạ, `:254-278` ghi `image=gold/<tên>.png` kể cả khi crop không có trong npz) | tắt |

---

## 6. Lệnh chạy từng bước

Ký hiệu: **[CÓ]** = lệnh tồn tại trong repo; **[ĐỀ XUẤT]** = chưa có, phải viết.

```bash
# 0. Config riêng (không chạm config/pipeline.yaml)            [ĐỀ XUẤT: tệp mới]
#    config/lucvantien1883.yaml : books: [{name: LucVanTien1883, source: images,
#                                  pdf: data/LucVanTien1883/LucVanTien_1883_TranNguyenHanh.pdf,  # BẮT BUỘC có khoá (step0_setup.py:49 KeyError nếu vắng)
#                                  reocr: false,
#                                  layout: {n_columns: 10, tiers: 2, qn_per_column: couplet},
#                                  det_thr: 0.2, contrast: stretch}]
#    paths.* và step2.* giữ như pipeline.yaml
export CONFIG=config/lucvantien1883.yaml
export DS_OUT=dataset_out_lvt1883          # ≠ dataset_out → không đụng CHECKSUMS/EVIDENCE/BANG_SO_LIEU/.FROZEN của STT

# 0b. Preflight thủ công (run_pipeline.sh:165-227 không chạy được vì menu ghim STT, :82-107):
test -f train_crop/detector_r34.best.pt || { echo thiếu detector; exit 1; }   # --reseg detector fail-fast
test ! -f $DS_OUT/.FROZEN || { echo "$DS_OUT đóng băng"; exit 1; }            # run_pipeline.sh:412

# 1. Tạo thư mục                                                 [CÓ] step0_setup chỉ WARN khi tệp pdf không tồn tại (khoá pdf phải có)
.venv/bin/python -m pipeline.step0_setup $CONFIG

# 2. Ingest (thay thế step1)                                     [ĐỀ XUẤT] pipeline/tools/ingest_offline_book.py (§3)
.venv/bin/python -m pipeline.tools.ingest_offline_book --book LucVanTien1883 \
   --nom-dir data/LucVanTien1883/pages --qn-dir data/LucVanTien1883/quocngu_pages \
   --layout couplet10 --qn-ocr tess4 --expected-verses 2088 --contrast stretch \
   --out prepared/LucVanTien1883 --report $DS_OUT/ingest_report.json
#    KHÔNG chạy: pipeline.step1_extract (cần PDF), check_ocr_columns.py (EXPECTED=9)

# 3. Build (sau khi sửa A1–A4, C/C' ở §4)                        [CÓ] lệnh; [ĐỀ XUẤT] tham số layout
.venv/bin/python -m pipeline.align_engine.build_dataset --config $CONFIG \
   --reseg detector --qd01-cells none --decisions none \
   --force --out $DS_OUT
#    (khớp run_pipeline.sh:259-260 trừ --decisions none; --box-rule mặc định syl_index (build_dataset.py:937);
#     không --use-s3; không --visual-emission)
.venv/bin/python -m pipeline.tools.enrich_crop_quality --labels $DS_OUT/labels.csv --src-root $DS_OUT   # [CÓ] run_pipeline.sh:263 (CLI thật, không có --out)

# 4. Checkpoint sha256                                           [CÓ] checkpoint() run_pipeline.sh:267-276 (ghi dòng "<ISO-time>  <tag>  <sha>  <file>" vào $DS_OUT/CHECKSUMS.txt sau MỖI bước); gọi tay tương đương:
printf '%s  build  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$(shasum -a 256 $DS_OUT/labels.csv | awk '{print $1"  "$2}')" >> $DS_OUT/CHECKSUMS.txt

# 5. Remediate                                                    [CÓ] run_pipeline.sh:298-308 (CLI thật: mô-đun pipeline.remediation + sub-command)
.venv/bin/python -m pipeline.remediation --labels $DS_OUT/labels.csv --out $DS_OUT census
.venv/bin/python -m pipeline.remediation --labels $DS_OUT/labels.csv --out $DS_OUT apply --tau 0.62
#    → $DS_OUT/labels_remediated.csv
#    confusion_fix là bước DUY NHẤT sinh labels_final.csv (run_pipeline.sh:304-306) → KHÔNG được bỏ hẳn, nếu không bước 6 thiếu đầu vào.
#    Ở chế độ B nó là no-op (chỉ hạ GOLD/SILVER — confusion_fix.py:26 DEMOTABLE; sách mới 100 % REVIEW) → chạy giữ nguyên để có tệp:
.venv/bin/python -m pipeline.remediation.confusion_fix --in $DS_OUT/labels_remediated.csv \
   --out $DS_OUT/labels_final.csv --fixes config/confusion_fixes.yaml --measure
#    (kiểm: confusion_fix_report.json phải ghi demoted = 0 cho sách mới; nếu ≠ 0 là fix STT đã chạm sách mới → xem lại)
#    Cổng run_pipeline.sh:286-295 (0 ô rule 'quyet_dinh_nguoi:*'):
.venv/bin/python -c 'import csv,sys; n=sum(1 for r in csv.DictReader(open(sys.argv[1],encoding="utf-8")) if (r.get("rule") or "").startswith("quyet_dinh_nguoi:")); print("qd01",n); sys.exit(n!=0)' $DS_OUT/labels_final.csv
#    BỎ step_rescue (run_pipeline.sh:311-348): pseudo CSV/npz chỉ có stt*

# 6. Export                                                        [CÓ] run_pipeline.sh:351-367 (FINAL_DIR=$DS_OUT/dataset khi DS_OUT≠dataset_out, :43)
.venv/bin/python pipeline/export_final_dataset.py --labels $DS_OUT/labels_final.csv --src-root $DS_OUT --out $DS_OUT/dataset
#    Sẽ LOG đỏ "trang không đủ 9 cột" cho mọi trang (export_final_dataset.py:127,184) — chấp nhận, không loại
.venv/bin/python -m pipeline.tools.make_dataset_docs --dataset $DS_OUT/dataset     # [CÓ] run_pipeline.sh:361
.venv/bin/python -m pipeline.tools.make_xlsx --labels $DS_OUT/dataset/labels.csv    # [CÓ] run_pipeline.sh:362
#    KHÔNG chạy update_bang_so_lieu / evidence (chỉ khi DS_OUT==dataset_out)

# 7. Selftest hồi quy STT (bắt buộc sau mỗi thay đổi §4)          [CÓ]
#    LƯU Ý 21/09: dataset_out/ đang bị XOÁ trong working tree (git status: D dataset_out/*.csv|json) — phải khôi phục
#    bản đã commit trước khi selftest/so byte:  git checkout -- dataset_out   (hoặc so với git show HEAD:dataset_out/labels.csv)
bash scripts/run_all_selftests.sh        # kỳ vọng BASELINE_PASS=1140/0 với DS_OUT=dataset_out (scripts/run_all_selftests.sh:147)
git diff --stat -- dataset_out/labels.csv   # phải rỗng sau khi build lại STT với engine đã sửa (byte-identical)
```

Chạy thử lần đầu: `--pages lucvantien1883,page_0002,…,page_0006` (`build_dataset.py:949`; book = mã `_book_code`) để kiểm 5 trang trước khi chạy 105.

---

## 7. Cổng kiểm định & tiêu chí nghiệm thu

| Bước | Cổng | Tiêu chí PASS | Căn cứ đo |
|---|---|---|---|
| I1 | ánh xạ trang | LVT 105 PNG; KVK 163 PNG tên `page_0001…0163`, `page_0067` = canvas 100 (câu 1321–1340) | kvk1884_canvas_map.csv 163/163 |
| I2–I3 | tầng/cột | 2 tầng 100 %; 10 cột ≥ 98 % LVT (103/105 + 2 trang biên xử lý riêng), ≥ 99 % KVK (162/163, `kvk1884_canvas_map.csv`) khi lề trái ≤ 30 px (cắt 6 % lề → 132/163); cột lệch → cờ tay | layout_all_pages.csv |
| I4 | số in | ≥1 số đúng/trang (LVT 105/105); giá trị trang khớp công thức; số in **không** lọt vào crop cột (0 hộp chữ trong dải số) | LVT 92/105 đủ 4 số, KVK 162/163 |
| I5 | hộp | % cột n_det == N ≥ **75 %** trên ≥ 27 trang (sửa 21/09: mẫu 9 trang 82,2/80,3 lạc quan; measure_out 27 trang: LVT stretch 77,6, KVK otsu 77,6, CI ±3,5 — mốc 80 % **không đạt**); cột n_det ≠ N phải có cờ và **không** vào tier khác REVIEW | measure_out/detector_transfer; N = 6/8 giả định |
| I7–I8 | QN | tổng câu chuỗi = 2.088 (LVT) / số in cuối 3253 với 3.251 dòng vật lý và đúng 4 bất thường đã biết (KVK); trang trôi: LVT 0, KVK 4 (= 4 bất thường) — mỗi trang trong `pages_review.json` phải có người xem (LVT 6, KVK 40); 6/8 xen kẽ ≥ 93 % (đo 93,4 / 96,1); dòng không-thơ lọt = 0 sau lọc dict<0,5 (đo 0 / 1 đã loại) | measure_out/<book>/qn_ocr (invariants 20 + 24 PASS) |
| I9 | ghép | số dòng .txt == len(columns) từng trang (100 %); tổng cột = tổng câu; **KVK: chặn cho tới khi lệch 3.256 vs ≈3.252 được định vị** | |
| I6 | cache | (thêm 21/09) 100 % trang `verify_cache_image(cache, pages/<page>.png)` = 'ok' (engine build **không** tự kiểm — `align_production.py:54-57`); mỗi cột ≥ 4 chữ (`nom_cols_hybrid min_len=4`); **không tồn tại** `transcriptions/*_qn_ocr_cache.json` có `text` ≠ "" (`step2_align.py:61-90`) | đo hash §2.2 |
| I9 | phủ câu | (thêm 21/09) mỗi verse_id 1…N xuất hiện **đúng 1 lần** trong toàn bộ .txt/sidecar; sidecar `verse_id` + `len(câu lẻ)` thật cho từng cột (B1) | |
| Build | engine | `method` cột = hybrid (không projection_fallback) ≥ 95 % trang; iter_pairs = 10 (B1) / 20 (B2) cặp/trang; page_ok True; 0 trang bị bỏ vì thiếu cache; `column` trong labels.csv ∈ 1…10 (B1) và mỗi (page, column) có n_qn = 14 ± 2 | dry-run n_expected=20 → 20 cột hybrid_9; **cache 10 cột × 14 chữ với n_expected=10 chưa dry-run** (chỉ đo projection với kim rỗng → fallback) |
| Build | neo | (thêm 21/09) `summary.json.n_anchor_pairs` và tỉ lệ `band_touched`/`p_register ≥ 0,8` báo riêng cho sách mới (ANCHOR_MIN_OTHER_PAGES=2, `build_dataset.py:120`, chỉnh trên 448 trang STT) | STT: n_anchor_pairs 24.466 (git HEAD summary.json) |
| Build | nhãn | chế độ B: 100 % REVIEW với mã hiện tại (đúng như mong đợi); nếu bật luật G: báo riêng tỉ lệ `positional_equal_count`, không gộp GOLD | §5 |
| Người kiểm | GT đầu tiên | **≥ 300 chữ/sách** SRS theo trang (phân tầng lục/bát, tầng trên/dưới), 2 người, κ ≥ 0,6 trên nhãn; đo precision hộp (IoU ≥ 0,5) và đúng chữ↔âm; thêm **≥ 20 cột vẽ hộp tay** để đo precision detector | hiện chưa có GT hộp/nhãn nào cho thạch bản |
| Người kiểm | QN | ≥ 100 dòng ngẫu nhiên phiên tay/sách để công bố CER (hiện tham chiếu = mô hình đọc ảnh 73 dòng) | |
| Hồi quy STT | selftest | 1140/0; labels.csv STT byte-identical (45 cột sau enrich) — **khôi phục `dataset_out/` từ git HEAD trước** (working tree đang xoá) | `git show HEAD:dataset_out/summary.json`: 83.239 ô, GOLD 49.946 / SYLLABLE 18.925 / REVIEW 14.368 (kiểm 21/09) |

---

## 8. Rủi ro & việc chưa đo được

1. **API kim 401** — không có S1 → không GOLD; chưa biết kế hoạch kích hoạt lại. Mọi nhãn ngoại tuyến là nhãn vị trí.
2. **KVK Nôm 3.256 vs QN ≈3.252**: chưa định vị câu dôi; 28 trang neo mơ hồ chưa soát → **chặn KVK**.
3. **Không GT người** cho hộp/nhãn/QN thạch bản: mọi % cột M==N dựa N=6/8 giả định; CER QN dựa mô hình đọc ảnh; số in đọc bằng kNN học từ nhãn bán vòng tròn.
4. **Detector**: chỉ 9 trang/sách đo (CI ±6 điểm); precision hộp chưa đo; thr 0,1 sau stretch bắt nhãn số câu (10 hộp/6 trang LVT); tiền xử lý stretch/otsu chưa có trong mã.
5. **Tách tầng B2 chưa đo** trong engine (`nom_cols_hybrid` gộp theo x; `raw_column_boxes` không lọc y → \|G\|=11–15 cho 1 tầng → enforce_count top-6 trộn tầng). B1 tránh được nhưng verse_id suy từ syl_idx sai ở câu 5/7/9/11 âm (1,7 % TSV; chưa đo trên QN thật).
6. **Âm QN Nam Bộ TK19 ngoài dict** (nhơn, sanh, chẵng…): chưa đo tỉ lệ trên OCR thật (proxy 93–94 % âm trong dict) → ảnh hưởng COST_NODICT/dict_support/CHAR_A-B nếu có kim.
7. **Neo LOO liên sách** (`build_dataset.py:137`): chạy chung 6 sách trộn miền TK17 chữ thảo ↔ TK19 thạch bản; tránh bằng config riêng; ngưỡng ANCHOR_MIN_* chỉnh trên 448 trang STT, sách 105–163 trang có thể thiếu cặp đủ ngưỡng (chưa đo).
8. **`normalize_syllables`** bung SAINT_NAMES/TOPONYMS STT cho mọi dòng QN: đo 14.595 âm LVT 0 va chạm — rủi ro thấp nhưng vẫn là luật STT.
9. **Khe cột 0 px** ở 31,7 % khe LVT → crop cột dính đuôi nét cột kề (như STT); ~7 % cột có chữ dính dọc.
10. **pages_denoised**: bilateral + close 51×51 + Otsu (`image_processing.py:27`) chỉnh cho STT; ảnh hưởng carve/tighten trên nền xám thạch bản chưa đo.
11. **Tài liệu hiện có sai**: `docs/README.md` (chưa commit) mô tả `ingest_offline_book.py` không tồn tại, `_detect` dùng `len(qn_keys)` (sai, ghim 9), menu run_pipeline cho sách mới (sai), `--use-s3` trong run (sai), KVK 121 hộp/trang (đo 106,1); `data/*/SOURCE.md` Chrestomathie lệch 1–3 canvas; LVT SOURCE.md:13 tự nhận TSV "100 % Ground Truth" trong khi ≈46–48 % lệch parity **và** tự mâu thuẫn với SOURCE.md:44 ("TSV chỉ có 2.059 câu, bản 1916" — tệp thật 2.088 dòng); KVK SOURCE.md:16,26,38-39 ghi 3.254 câu/vol2 = 1501–3254 (đo: Nôm 3.256, vol1 kết 1497); `docs/CODE_FREEZE.md:45` xếp "thêm sách mới" vào đóng băng.
12. **VietOCR DBNet** không đo (thiếu PaddleOCR); luật tách cột số của VietOCR mới đo 56 dòng.
13. **Chrestomathie**: bảng truyện↔trang chưa người rà; CER NomNaOCR n=61–98 tham chiếu agent; bộ dò khuyên recall 25–85 %.
14. Mọi "kiểm mắt" trong các phép đo là của agent, **không phải chuyên gia Nôm**.
15. (thêm 21/09) **`core/ocr/ocr_api.py` đã bị sửa trong working tree, chưa commit** (Guest Mode khi 401, +40/−13 dòng): mọi `file:line` của ocr_api trong tài liệu này là theo bản working tree; nhánh Guest chưa thử; nếu commit/hoàn tác thì số dòng đổi.
16. (thêm 21/09) **`dataset_out/` bị xoá trong working tree** (git status D 10 tệp) — cổng hồi quy STT và selftest (`pipeline/tools/selftest.py` đọc `$DS_OUT`) không chạy được cho tới khi khôi phục; số liệu STT trong tài liệu này lấy từ `git show HEAD:dataset_out/*`.

---

## 9. Thứ tự việc làm đề xuất

| Ưu tiên | Việc | Điều kiện | Ước công | Sản phẩm |
|---|---|---|---|---|
| P0 | Quyết định chính sách chế độ B (luật G `positional_equal_count` → SYLLABLE hay giữ 100 % REVIEW) và có/không kích hoạt kim | người dùng | 0,5 ngày | ghi vào tài liệu này §5 |
| P0 | Đính chính tài liệu: `docs/README.md`, `data/*/SOURCE.md` (Chrestomathie 29–53/106–170; LVT TSV không phải GT; KVK 3.256), `EVIDENCE_INDEX.md` chỉ ghi sha256 đầu ra thật | — | 0,5 ngày | commit docs |
| P1 | Config riêng `config/lucvantien1883.yaml` + sửa A1–A4, C/C' (§4) + selftest 1140 + cmp labels STT | — | 1 ngày | PR nhỏ, hành vi STT byte-identical |
| P1 | Adapter ingest LVT1883 (I1–I10, B1: 10 cột × 14 chữ, tess4 + chuỗi mod 5) | scratchpad scripts `layout_lvt1883.py`, `extract_verses.py`, `chain_analysis.py`, `detector/best_variant.py` làm mẫu | 2–3 ngày | `prepared/LucVanTien1883/` + `ingest_report.json` |
| P1 | Chạy thử 5 trang → 105 trang; kiểm cổng §7; DS_OUT riêng | P1 trên | 0,5 ngày | `dataset_out_lvt1883/` |
| P2 | Mẫu người kiểm LVT: ≥300 chữ + ≥20 cột vẽ hộp + ≥100 dòng QN phiên tay | 2 người biết Nôm | 2–3 giờ/người | GT đầu tiên cho thạch bản; precision detector; CER thật |
| P2 | KVK1884: định vị câu dôi (so cửa sổ Nôm-OCR↔QN hoặc soát 28 trang mơ hồ + 2 vùng), rồi ingest với `--page-map reverse:167`, đảo pha 6/8 từ 1585, lọc Hán văn | P1 xong | 2 ngày + 1 ngày soát | `prepared/KimVanKieu1884/` |
| P3 | Tách tầng B2 (20 cột thật) chỉ nếu B1 cho crop chữ 6/7 quá xấu sau kiểm P2 | P2 | 1–2 ngày | y_range trong cluster/raw_column_boxes |
| P3 | Nếu kim mở: thêm H (OCR từng tầng, expected_cols theo sách), chạy chế độ A, so GOLD với mẫu P2 | API | 1 ngày | GOLD thạch bản |
| P4 | Chrestomathie: người rà bảng 20 ô (30 phút) → tách ô chữ trong cột (1 ngày) → mô-đun căn chỉnh chuỗi truyện (1–2 ngày) → kiểm ≥300 chữ | P1–P2 | 3–4 ngày | Nhóm 1b |

---

## Phụ lục — Tệp đo (scratchpad, ngoài repo)

Gốc (thư mục tạm của phiên làm việc, KHÔNG bền; bản tái lập được nằm ở `scripts/measure/`): `/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/4224b41e-fbbe-477e-9f2e-69af7d5afeca/scratchpad/wf/`

| Thư mục | Nội dung chính |
|---|---|
| `adapter-contract/DO_DAC_2026-09-20.txt` | nhật ký đo hợp đồng Bước 1, prepared giả, dry-run engine 20 cột |
| `pipeline-reader/measure_frontend.py`, `frontend_measure.json` | 10 trang: projection n9/n10/n20, khe tầng, detector per_col |
| `pipeline-audit/det_probe.json`, `det_probe2.json` | detector 21 LVT + 20 KVK + 3 Chresto |
| `lvt1883_layout/` (+`_review/`) | `layout_lvt1883.py`, `pages.csv`, `columns.csv` (2.089), `numbers*.csv`, `continuity_union.csv`, montage kiểm mắt |
| `kvk1884_layout/` (+`_review/`) | `kvk1884_canvas_map.csv` (171 dòng), `kvk_layout.py`, `digits_decode.py`, `layout_raw.json`, crop số 6× |
| `qn_ocr/` (+`_critic/`) | `extract_verses.py`, `chain_analysis.py`, `tess_all/` (770 trang), `vietocr_all/` (139 LVT), `kvk_classification.tsv`, `visual_truth*.json` |
| `detector/` (+`_review/`) | `measure_detector.py`, `layout_bands.py`, `best_variant.py`, `detector_*.csv`, `layout_all_pages.csv` (268 trang), overlay |
| `chresto/` (+`_review/`) | `bang_truyen_trang.tsv`, `nom_layout.json`, `match_window*.json`, `segs2_ocr.json` |

Các tệp này **chưa** được đưa vào repo; nếu cần bằng chứng lâu dài, sao chép có chọn lọc vào `evaluation/sach_moi_2026-09-20/` kèm sha256 (việc P0).

---

## 10. Phê bình & đính chính (2026-09-21)

Lượt rà độc lập: đọc toàn bộ tài liệu, mở từng `file:line` được trích, đối chiếu với 3 báo cáo đọc mã + 5 phép đo có phản biện, và tái lập một số con số từ scratchpad/git. Quy tắc: chỉ đọc và đo, không sửa mã, không chạm `data/`, không gọi API. Script đối chiếu: `scratchpad/wf/doc-critic/edit{1,2,3,4}.py`, bản trước sửa: `scratchpad/wf/doc-critic/PIPELINE_SACH_MOI_2026-09-20.before.md`.

### 10.1 Con số không có nguồn / mâu thuẫn với phản biện — đã sửa

| # | Vị trí | Trước | Sau | Căn cứ |
|---|---|---|---|---|
| 1 | §0.1, §7 I2–I3 | KVK "158/163 đủ 10 cột khi không cắt lề" | **162/163** (lề ≤ 2 %); 132/163 khi cắt 6 % lề | `kvk1884_canvas_map.csv` (đếm lại: 162 × (10,10) + 1 × (9,8)); `detector_review/layout_all_pages.csv` (132 × 10;10, 16 × 9;9, 10 × 10;9, 4 × 9;10, 1 × 8;8). Số 158 không có ở tệp nào |
| 2 | §1.3 | TSV "1.007–1.031 (≈48–49 %) đảo parity" | 963 dòng đảo hẳn 6↔8 + 101 dòng ngoài {6,8}; phản biện 1.007 (luật gộp ±1) → ≈46–48 % | đếm lại 21/09 trên 2.088 dòng; cận trên 1.031 không có nguồn |
| 3 | §1.1, §1.7 | detector ảnh gốc thr 0,2: 66,7 % / 19,2 % (n=6) đặt cạnh stretch/otsu 82,2 / 80,3 (n=9) | ghi thêm cùng cơ sở 9 trang: 64,4 % (116/180) / 25,3 % (45/178) | `detector_review/verify_rows.csv` (tổng hợp lại: 148/180 = 82,2; 143/178 = 80,3; 151/180 = 83,9; 140/178 = 78,7 — tái lập đúng) |
| 4 | §2.4 | labels.csv "42 cột" | 42 từ engine, **45** sau enrich_crop_quality | header `git show HEAD:dataset_out/labels.csv` |
| 5 | §5 | GOLD 49.946 / SYLLABLE 18.925 / REVIEW 14.368 (không ghi nguồn) | giữ, ghi nguồn `git show HEAD:dataset_out/summary.json` (83.239 ô, 448 trang) | tái lập từ git HEAD vì working tree đã xoá dataset_out/ |
| 6 | §1.7 | "Ckpt VAL = 44 trang chẵn STT11 (0010–0102)" | giữ (xác nhận) | `detector_r34.best.pt['val_images']` = 44 tệp yen11_page_0010…0102, epoch 38, img 1024 |
| 7 | §1.1 | STT 93,2 % "(cột OCR=QN, FLOW v3.1 N3f)" | thêm "N = kim, không phải GT người" | `docs/FLOW_PIPELINE_v3.1_2026-09-15.md:98` |

Chưa đối chiếu được trong lượt này (đầu vào bị cắt): các số của §1.6 Chrestomathie (7 ô cột/trang, 416 cột, 8.177 âm, 7/7 định vị, CER 0,61–0,74) và hàng "Kích thước hộp / Phóng to / Tách chữ chiếu ngang" của §1.7 — giữ nguyên, đánh dấu là chưa qua phản biện độc lập lần 2.

### 10.2 Khẳng định về mã sai hoặc lệch dòng — đã sửa

| # | Vị trí | Sai | Đúng (đã mở tệp) |
|---|---|---|---|
| 8 | §0.4, §2.2 | "`char` rỗng/None → 0 cặp (`anchor_align.py:56`)" | `_char_of` :56 trả None → `substitution_cost` :61-72 trả COST_DICTMISS 6,7 (< DEL+INS 17,2) → DP **vẫn ghép** (mô phỏng align_engine: 6 None ↔ 6 âm = 6 match); 0 bản ghi chỉ khi `columns` rỗng. Hệ quả: chế độ B không phải "0 nhãn" mà "100 % REVIEW" |
| 9 | §2.2, §4 H | `ocr_api.py:378/512/534/632/651` | tệp đang bị sửa trong working tree (chưa commit): `boxes_to_columns` :389 (Δx<15 :406), `_file_md5` :533, `_pixel_hash` :541, `verify_cache_image` :562, `load_columns_fullpage` :656, `ocr_page` :672 / `expected_cols` :678. Diff còn thêm nhánh **Guest Mode khi 401** — chưa thử, ghi vào §0.4 và §8.15 |
| 10 | §4 A1 | "step0_setup chỉ WARN thiếu pdf; step1 return sớm → thêm sách không cần pdf" | `step0_setup.py:49` `book["pdf"]` và `step1_extract.py:40` `book_cfg["pdf"]` → **KeyError nếu vắng khoá**; chỉ WARN/return khi tệp không tồn tại → config sách mới phải có `pdf:` (§6 bước 0 đã thêm) |
| 11 | §5 | `self_training_rescue.py:126` (model không chạy khi có pseudo), `:253` (ghi image) | `elif` :160; ghi `image=gold/…` :254-278 |
| 12 | §4 I | `tools/selftest.py:467,484` | tệp là `pipeline/tools/selftest.py`; phép kiểm 9 cột ở :443-467 (:484 là dòng trống) |
| 13 | §6 bước 3 | `enrich_crop_quality --out $DS_OUT` | CLI thật `--labels … --src-root …` (`run_pipeline.sh:263`, `enrich_crop_quality.py:94-95`); `--box-rule syl_index` là mặc định (`build_dataset.py:937`), run_pipeline không truyền |
| 14 | §6 bước 5 | `python -m pipeline.remediation.census …` / `.apply …` | CLI thật `python -m pipeline.remediation --labels … --out … census\|apply --tau` (`pipeline/remediation/cli.py:68-79`, `run_pipeline.sh:300-301`) |
| 15 | §6 bước 6 | `python -m pipeline.export_final_dataset --labels … --out …` | script `pipeline/export_final_dataset.py --labels --src-root --out` (`:210-212`, `run_pipeline.sh:354-355`) |
| 16 | §6 bước 7 | `cmp dataset_out/labels.csv <bản trước sửa>` | `dataset_out/` đang bị xoá trong working tree (git status D) → phải `git checkout -- dataset_out` trước; so bằng `git diff --stat` |

Các trích dẫn khác đã mở và **đúng** (không sửa): `align_production.py:47,54-57,72,76,81,86,87,196,226,442,567,580,604`; `step2_align.py:61,70,83,92,118`; `parser_v5.py:88,136`; `pdf_parser.py:128,179,243`; `column_detector.py:8,11,83`; `build_dataset.py:76-87,120,137,485,809,937,949,953,1010,1123,1135,1152`; `consensus.py:128,135,158,171,193`; `tier_v3.py:41-53,68-83,92-99,156`; `parser_v2.py:133`; `nom_detect_v3.py:27,92`; `detector_infer.py:76-89`; `infer_centernet.py:243`; `run_full.py:23,51`; `step1_extract.py:43,61`; `image_processing.py:27`; `line_detector.py:114`; `step4_export.py:129`; `export_final_dataset.py:127,184`; `audit_grid.py:68`; `eval_crops_v2.py:58`; `recrop_v2.py:35`; `check_ocr_columns.py:23-24`; `config/pipeline.yaml:11-20`; `config/decisions.yaml` (22/23 mục `book: null`); `confusion_fix.py:26` DEMOTABLE = {GOLD, SILVER}; `docs/README.md:102` (ingest_offline_book.py không tồn tại — xác nhận); `docs/CODE_FREEZE.md:45`; `NomNaOCR/ds4v_repo/README.md:81-84`; `scripts/run_all_selftests.sh:147` BASELINE_PASS=1140; `run_pipeline.sh` các mốc 82, 165, 230, 236, 257, 263, 267, 286, 298, 311, 351, 370, 412, 426-438.

### 10.3 Bước pipeline bị bỏ sót so với `run_pipeline.sh` — đã bổ sung vào §6

| # | Bước trong run_pipeline.sh | Tình trạng trước | Sửa |
|---|---|---|---|
| 17 | `preflight` (:165-227): kiểm venv, detector ckpt, `.FROZEN` (:412) | không có | bước 0b thủ công (menu `ask_book_choice` :82-107 ghim STT nên không gọi được script) |
| 18 | `confusion_fix` (:304-306) — **bước duy nhất sinh `labels_final.csv`** | tài liệu bảo "BỎ" rồi bước 6 lại đọc `labels_final.csv` → gãy | giữ chạy; ở chế độ B là no-op (chỉ hạ GOLD/SILVER); kiểm `demoted = 0` |
| 19 | `assert_qd01` trước export (:352, :433, :436) | không có | thêm lệnh kiểm 0 ô `quyet_dinh_nguoi:*` |
| 20 | `make_dataset_docs`, `make_xlsx` (:361-362) | không có | thêm |
| 21 | `checkpoint` sau MỖI bước, định dạng `<time> <tag> <sha> <file>` (:267-276) | 1 lệnh shasum khác định dạng | ghi đúng định dạng |
| 22 | `FINAL_DIR=$DS_OUT/dataset` khi DS_OUT ≠ dataset_out (:43) | đúng nhưng không dẫn | dẫn dòng |

### 10.4 Cổng kiểm định thiếu — đã thêm vào §7

23. **Cache hash**: engine build không kiểm hash (`align_production.py:54-57`) → adapter phải tự chạy `verify_cache_image` = 'ok' 100 % trang, và tài liệu trước không có cổng "không tồn tại `_qn_ocr_cache.json` có text" dù §2.3 cấm.
24. **Phủ câu**: mỗi verse_id xuất hiện đúng 1 lần toàn sách (trước chỉ có "tổng cột = tổng câu", không bắt được hoán vị/trùng).
25. **Không ép số hộp về N** (§3.2 I5 trước ghi "ép về N = 6/8 top-N"): nếu ép, `n_ocr == n_qn` luôn đúng → luật G §5 thành tautology; cổng §7 I5 "cột n_det ≠ N phải có cờ" vô nghĩa. Đã đổi I5 thành "ghi hộp thật, chỉ gắn cờ".
26. **Neo LOO cho sách nhỏ**: báo riêng `n_anchor_pairs`, `band_touched`, `p_register` cho sách mới (ANCHOR_MIN_OTHER_PAGES=2 chỉnh trên 448 trang).
27. **Hồi quy STT**: bổ sung điều kiện khôi phục `dataset_out/` từ git và mốc 83.239/49.946/18.925/14.368.
28. Ghi nhận hạn chế: dry-run engine mới làm với cache **20 cột** (n_expected=20 → hybrid_9); tổ hợp được khuyến nghị (B1: cache 10 cột × 14 chữ, n_expected=10) **chưa dry-run** — đã ghi vào §7 hàng Build/engine.

### 10.5 Cách gọi "Ground Truth"

Rà toàn văn: tài liệu **không** gọi nhãn máy là GT (các chỗ "GT" đều là "không GT người", "GT đầu tiên" = mẫu người kiểm §7/§9, hoặc trích dẫn để bác `SOURCE.md`). Đã siết thêm 2 chỗ dễ hiểu nhầm: §1.1 mốc STT 93,2 % ghi rõ N = kim; §8.11 nêu `data/LucVanTien1883/SOURCE.md:13` ("100 % Ground Truth") mâu thuẫn với chính `SOURCE.md:44` và với TSV thật (2.088 dòng, ≈46–48 % lệch parity), `data/KimVanKieu1884/SOURCE.md:16,26,38-39` (3.254 câu, vol2 = 1501–3254) lệch với số đo (3.256; vol1 kết 1497). Việc sửa SOURCE.md nằm ở P0 §9, chưa làm (quy tắc không chạm `data/`).

### 10.6 Không sửa, chỉ ghi nhận

- Toàn bộ "kiểm mắt" (số in, câu dôi, montage) là của agent; tài liệu đã nói rõ ở §8.14.
- Số liệu Chrestomathie (§1.6) và một phần §1.7 chưa được rà lại lần 2 (10.1).
- Kết luận chính của tài liệu (bố cục cột = cặp lục bát; 2.088 / 3.256 vs ≈3.252; ghim 9 phải tham số hoá; không S1 → không GOLD; chạy config riêng; tắt rescue/decisions cho sách mới) **đứng vững** sau rà; các sửa ở trên là số liệu phụ, dòng mã, lệnh gọi và cổng.


---

## 11. Số liệu chốt bằng bộ đo offline (2026-09-21)

Nguồn: `measure_out/SUMMARY.json`, `measure_out/REPORT.md` (sinh bởi `scripts/measure/measure.py --all`, 0 token LLM, ~8 phút CPU). Cột "số cũ" lấy từ §0/§1/§7 (scratchpad 20/09) và `scripts/measure_wf_2026-09-21/*/summary*.json`. Quy ước sai số hợp lý: đếm rời rạc = 0; tỉ lệ trên mẫu ≤ 200 cột = ±6 điểm (CI 95 %); tỉ lệ toàn sách = ±1 điểm. Lệch vượt ngưỡng đã mở script tìm nguyên nhân (cột "kết luận"). Mọi số "mới" đều tái lập được bằng lệnh ở §11.3.

### 11.1 Bảng đối chiếu

| # | Đại lượng | Số cũ (nguồn) | Số mới (measure_out) | Lệch | Kết luận |
|---|---|---|---|---|---|
| 1 | LVT1883 trang tách 2 tầng | 105/105 (§1.2) | 105/105 (`LucVanTien1883/layout`) | 0 | khớp |
| 2 | LVT1883 cột/tầng | t0: 104×10 + 1×5; t1: 103×10, 1×9, 1×5 (§1.2) | t0 {10:104, 5:1}; t1 {10:103, 9:1, 5:1}; A==B 210/210 | 0 | khớp; 2 phương pháp (đỉnh vs run) đồng ý 100 % |
| 3 | LVT1883 tổng câu từ bố cục | 2.088 = 18+20×103+10 (§0.3) | 1.044 cặp → 2.088; 0 trang lệch | 0 | khớp |
| 4 | LVT1883 cột đúng 6/8 chữ | 2.075/2.089 = 99,3 % (§1.2) | 2.075/2.089 = 99,33 % | 0 | khớp |
| 5 | LVT1883 số in đọc đúng | 397/417 = 95,2 % (summary.json cũ, union tess+template) | 397/417 (final = kNN tự huấn 1.107 chữ số, CV 99,82 %); tess↔kNN đồng ý 77,6 % | 0 | khớp; trang đủ mọi số 92 → 90 vì "final" không lấy hợp của 2 bộ đọc (khác định nghĩa, không phải lỗi) |
| 6 | LVT1883 số câu đơn điệu / first_seq | 0 trang mâu thuẫn (§0.3) | monotonic True, Viterbi lệch 0/105 | 0 | khớp |
| 7 | LVT1883 hình học | pitch hàng 157 [153–159], pitch cột 156 [149–168], khe tầng 260 [239–279] (§1.2) | 157 [153–159]; 156 [149–161]; 263 [239–281] | ≤ 3 px | khớp; khe tầng mới = hộp tầng đã đệm ±15 px |
| 8 | LVT1883 QN dòng thơ | 2.092 thô / 2.088 kỳ vọng, ≥1 cước chú lọt (§1.3) | 2.088/2.088 (897 dòng cước chú loại, 0 lọt), chuỗi 2.088 | −4 dòng thô | mới đúng hơn: lọc cước chú theo vùng, không còn dòng thừa |
| 9 | LVT1883 QN trang trôi | 7 trôi + 2 mơ hồ /139 (§1.3, neo đầu dòng) | 0 trôi + 0 mơ hồ; 5 xung đột giá trị + 1 không neo (2 phương pháp neo: đầu dòng 345 + lề 369 = 714) | −7 | mới đúng hơn: thêm bộ đọc số ở lề và ưu tiên vị trí mod 5; 4 dòng thô thừa trước đây là nguồn "trôi" |
| 10 | LVT1883 QN neo | tìm 362/419 = 86,4 %, giá trị đúng 275/419; vị trí mod 5 339/347 = 97,7 % (§1.3) | đầu dòng: 345 tìm, giá trị 263 (76,2 %), vị trí 342 (99,1 %); gộp 2 phương pháp: vị trí 99,6 %, giá trị 80,3 %, phủ 407/417 = 97,6 % | ≈0 | khớp (cùng bộ đọc đầu dòng); phương pháp lề nâng phủ 86 → 98 % |
| 11 | LVT1883 QN parity 6/8 | 93,5 % toàn sách (§1.3) | 1.950/2.088 = 93,4 % | −0,1 | khớp |
| 12 | CER QN (đọc mắt) | 7,1 % trên 44 dòng (§1.3) | LVT 7,2 % (16 dòng, 471 ký tự) + KVK 7,05 % (28 dòng, 808) = 91/1.279 = 7,1 % | 0 | khớp (cùng 44 dòng, `qn_visual_truth.json`) |
| 13 | KVK1884 canvas chữ / ánh xạ | 4–166 (163), page = 167 − k, 163/163 (162 tự động + 1 mắt) (§1.4) | 163 canvas; non-text {1,2,3,167–171}; first_seq = 20·(page−1)+1 đúng 163/163 bằng Viterbi (tự do 155/162) | 0 | khớp, nay hoàn toàn tự động |
| 14 | KVK1884 cột/tầng | 162/163 = 10+10; canvas 4 = 9 (8+tựa)/8 (§1.4) | t0 {10:162, 9:1}; t1 {10:162, 8:1}; A==B 326/326 | 0 | khớp |
| 15 | KVK1884 tổng câu Nôm | 3.256 (§1.4) | 1.628 cặp → 3.256; 0 canvas lệch lưới 20 | 0 | khớp |
| 16 | KVK1884 cột đúng 6/8 | 163/163 tầng đúng 6+8 hàng (§1.4, mức tầng) | 3.192/3.257 cột = 98,0 % (mức cột; t0 38 cột 7, t1 26 cột 9) | — | đại lượng mới, mịn hơn |
| 17 | KVK1884 khe tầng | 299 px [274–315] (§1.4, khoảng mực–mực) | 260 [228–283] (hộp tầng đệm ±15/16 px, cùng định nghĩa với LVT) | −39 px | khác định nghĩa (299 − 31 đệm ≈ 268); không phải lỗi; dùng số mới cho cả 2 sách |
| 18 | KVK1884 số in đọc đúng | 538/649 = 82,9 % (kNN với 2.070 nhãn do agent gán, §1.4) | 472/651 = 72,5 % (kNN tự huấn 1.169 chữ số từ tesseract nhất trí, CV 97,4 %); tess↔kNN 52,3 % | −10,4 điểm | thấp hơn vì bỏ nhãn tay của agent (đúng quy ước "không GT người"); không ảnh hưởng: first_seq 163/163 nhờ giải mã theo trang |
| 19 | KVK1884 số trang in góc | 126/163 khớp 169 − k (§1.4) | tự do 143/162; Viterbi 161/163 (lệch: canvas 8, 9) | +35 | mới đúng hơn (kNN riêng cho phông góc, CV 93,6 %) |
| 20 | KVK1884 phân loại 631 canvas QN | QN 295 / FR 313 / khác 23 (§1.5) | QN 295 (vol1 146 lẻ 27–317, vol2 149 lẻ 7–303) / FR 309 / khác 27 | QN 0; FR −4 | QN khớp; ranh FR/khác dịch 4 canvas trắng-gần-trắng, không dùng cho pipeline |
| 21 | KVK1884 QN dòng thơ | 3.261 thô; ≈3.252 dòng thật ước tính (§1.5) | **3.251** dòng vật lý (vol1 1.497 + vol2 1.754), chuỗi số in **3.253**, 1 dòng Hán lọt đã loại | −1 so ước tính | mới chính xác hơn (đếm, không ước); Nôm 3.256 − QN 3.251 = **5 câu chưa định vị** |
| 22 | KVK1884 QN bất thường đánh số | 1 bước nhảy 1581→1585 (§1.5) | 4 bất thường (verses.tsv): 1581→1583 (vol2 c23), 2220→2222 (c135), 2232 lặp (c137), 3053→3055 (c269); parity đảo ở 1583 và 2225, đảo lại ở 2235 | +3 | mới phát hiện thêm 3; kiểm ảnh c135: 11/11 dòng OCR đủ, số in 2225/2230 thật → nhảy là của bản in, không phải OCR mất dòng |
| 23 | KVK1884 QN trang trôi | 11 trôi + 28 mơ hồ /295 (§1.5) | 4 trôi (= 4 bất thường trên) + 22 mơ hồ + 7 xung đột + 14 không neo | −7 trôi | mới đúng hơn (2 phương pháp neo); 4 trôi còn lại là bản chất ấn bản |
| 24 | KVK1884 QN neo (đầu dòng) | 599/651 tìm, giá trị 417, vị trí 546/551 = 99,1 % (§1.5) | đầu dòng 552 tìm, giá trị 391 (70,8 %), vị trí 546 (98,9 %); gộp lề: 934 neo, vị trí 99,25 %, giá trị 73,9 %, phủ 595/650 = 91,5 % | ≈0 | khớp |
| 25 | KVK1884 QN parity 6/8 | 96,6 % (§1.5) | 3.123/3.251 = 96,1 % | −0,5 | khớp |
| 26 | Chrestomathie QN | 20 truyện, 610 dòng thân, 8.177 âm, OOV 4,83 % (§1.6) | 20; 608; 8.151; OOV 4,76 %; trang VN 29–53 = 25 | −2 dòng, −26 âm (0,3 %) | khớp (tesseract cùng phiên bản, khác cách gộp dòng tiêu đề) |
| 27 | Chrestomathie Nôm | 65 trang, 7 ô/trang, ≈416 cột, ≈8,7k chữ, pitch 133–143 (§1.6) | 65; 7 ô (48 trang đủ 7); 419 cột; 8.692 chữ (21,3 chữ/cột đầy, pitch↔run đồng ý 94,7 %); pitch 131–146 (median 140) | +3 cột | khớp |
| 28 | Chrestomathie bảng truyện↔trang | 20/20, I 106–109 … XX 167–170 (§1.6) | 20/20 giống hệt (`bang_truyen_trang.csv`, REF = agent đọc, cột `can_nguoi_ra=True`); tỉ lệ chữ/âm 0,99–1,24 (truyện 18: 1,42, 8 cột) | 0 | khớp; vẫn cần người biết Nôm rà |
| 29 | Chrestomathie ranh giới tự động | precision 18/18, recall 18/20 (§1.6) | lần chạy đầu 21/09: precision 18/24 = 0,75 → **sửa luật** (chỉ tính thiếu cột ở CUỐI trang, không tính ô trống nội bộ) → precision 18/18 = 1,0, recall 18/19 = 0,95 (bỏ đầu sách; sót 137/0) | 0 sau sửa | lỗi mã ở `chresto_map.py:boundaries_auto` đã sửa |
| 30 | Chrestomathie NomNaOCR định vị truyện | 7/7 gốc, 6/7 fine-tuned (§1.6) | 3 trang × 2 bộ trọng số: hạng 1 ở 6/6, LCS > đối chứng ngẫu nhiên 6/6 | 0 | khớp (mẫu nhỏ hơn, mặc định `--nomna-pages 3`) |
| 31 | Detector LVT ảnh gốc thr 0,2 | 66,7 % (n=6) / 64,4 % (n=9) (§1.7) | mẫu 9 trang cũ (`--page-ids`): **64,4 %** — tái lập đúng; mẫu 9 trang đều: 69,4 %; **27 trang (540 cột): 61,7 %** | 0 (cùng mẫu) | khớp; số 27 trang là số công bố |
| 32 | Detector LVT stretch / otsu thr 0,2 | 82,2 % (148/180) / 84,2 % (n=6) (§1.7) | mẫu cũ: 82,2 / 81,7; 27 trang: **77,6 / 76,3 %** (CI ±3,5) | 0 (cùng mẫu); −4,6 (mẫu lớn) | khớp trên cùng mẫu; mẫu 9 trang cũ lạc quan ~4,5 điểm |
| 33 | Detector KVK ảnh gốc / stretch / otsu thr 0,2 | 25,3 / 74,2 / 80,3 % (n=9, 178 cột) (§1.7) | mẫu cũ: 25,3 / 74,2 / 80,3 (178 cột, tái lập đúng); 27 trang (540 cột): **23,5 / 68,9 / 77,6 %** | 0 (cùng mẫu); −2,7 otsu (mẫu lớn) | khớp; **lỗi mã đã sửa**: `find_blocks` cắt lề trái 6 % → mất cột 10 ở 4/27 canvas KVK (đúng như §0.1 cảnh báo); nay 2 % → 27/27 trang 10+10 |
| 34 | Detector: cổng I5 ≥ 80 % cột M==N | đạt (82,2 / 80,3, n=9) (§7) | **không đạt** trên 27 trang: 77,6 / 77,6 % | — | §7 I5 hạ xuống ≥ 75 % kèm CI, hoặc giữ 80 % và coi thạch bản chưa đạt |
| 35 | Detector thr 0,3 (stretch) | LVT 24,2 %, KVK 19,2 % (§1.7) | LVT 25,7 %, KVK 20,6 % | ≈0 | khớp: thr 0,3 loại bỏ với thạch bản |
| 36 | Detector hộp h/pitch_y | LVT 1,12; KVK 1,08 (§1.7) | 1,132; 1,103 | ≈0 | khớp |
| 37 | Detector đối chứng STT | 70,4 % mọi cột theo N kim (n=54); 93,2 % cột OCR=QN (§1.1) | STT2/4/11 × 3 trang, raw thr 0,2: 90,1 % mọi cột (n=81); 92,7 % cột N kim == chiếu mực (n=55); recall hộp kim IoU 0,3 = 1,0; stretch/otsu không đổi (Δ ≤ 1 cột) | +20 / −0,5 | mốc 93,2 % tái lập; số 70,4 % cũ đo trên tập trang khác (kim đếm sai N), không so được |
| 38 | Đếm chữ chiếu ngang độc lập == N | lục 72 %, bát 63 % (LVT); 83/60 % (KVK) (§1.7 "dự phòng") | LVT 100 %, KVK 99,4 % (cùng thuật toán gom cụm neo pitch của layout) | +28…+39 | khác thuật toán: cũ tách ngưỡng thô, mới neo pitch hàng; số mới **không độc lập với giả định 6/8** (dùng pitch) → chỉ là kiểm chéo, không phải GT hộp |
| 39 | Ghim số 9 trong mã | ≥ 8 chỗ (§0.5) | 35 vị trí (24 ngoài lab/selftest), 9/9 vị trí đã biết tìm thấy (`docs/PIPELINE_FACTS.json`) | +27 | mới đầy đủ hơn; danh sách 8 chỗ cũ là tập con |
| 40 | Idempotence | không đo | `layout_lithograph --book LucVanTien1883 --limit 10` chạy 2 lần: summary.json md5 giống hệt, 4 CSV byte-identical; runner 2 lần (cache/OCR tươi): 6 tệp md5 giống | — | đạt |

### 11.2 Số nào thay đổi so với §0/§1/§7 (đã sửa tại chỗ, ghi "[measure_out 21/09]")

1. QN LVT: "2.092 thô, 7 trôi + 2 mơ hồ" → 2.088 dòng, 0 trôi, 5 xung đột giá trị + 1 không neo (#8, #9).
2. QN KVK: "3.261 thô, ≈3.252, 1 bước nhảy, 11 trôi + 28 mơ hồ" → 3.251 dòng vật lý, chuỗi 3.253, 4 bất thường đánh số, 4 trôi + 22 mơ hồ (#21–23). Lệch Nôm↔QN nay là **5 câu** (3.256 − 3.251), không phải 3–4.
3. Detector thạch bản: 82,2 / 80,3 % (n=9) → **77,6 / 77,6 %** (n=27, 540 cột/sách); cổng I5 ≥ 80 % không còn đạt (#32–34).
4. KVK số trang góc 126/163 → 161/163; số in đọc đúng 82,9 % → 72,5 % vì bỏ nhãn agent (#18, #19).
5. Chrestomathie: 610 dòng/8.177 âm/416 cột → 608/8.151/419 (#26–27).
6. Khe tầng KVK 299 → 260 px do đổi định nghĩa (#17).

### 11.3 Lệnh tái lập

```bash
.venv/bin/python scripts/measure/measure.py --all                       # ~8 phút, 0 token; đọc measure_out/REPORT.md
.venv/bin/python scripts/measure/detector_transfer.py --book all --stt-pages 0 --out /tmp/det_old9 \
  --page-ids "LucVanTien1883:010,030,050,070,090,100,020,060,095;KimVanKieu1884:0020,0050,0080,0100,0130,0160,0035,0120,0150"   # tái lập 82,2/80,3
.venv/bin/python scripts/measure/layout_lithograph.py --book LucVanTien1883 --limit 10 --out /tmp/idem_1   # chạy 2 lần, so md5 summary.json
```
