# KẾT QUẢ ĐO CUỐI — 3 sách Nhóm 1 mới (LVT1883, KVK1884, Chrestomathie1872) · 2026-09-21

Nguồn duy nhất: `measure_out/SUMMARY.json` + `measure_out/REPORT.md` do `scripts/measure/measure.py --all` sinh (0 token LLM, ~8 phút CPU, 118/118 invariants PASS, mã thoát 0). Đối chiếu với đặc tả 20/09: `docs/PIPELINE_SACH_MOI_2026-09-20.md` §11 (40 đại lượng). Quy ước cho phiên sau: **không đo lại bằng LLM**; chỉ đọc 2 tệp trên, nghi ngờ thì đổi tham số script hoặc thêm invariant (`CLAUDE.md`, `scripts/measure/README.md`).

## 1. Số liệu chốt

| Sách | Đại lượng | Giá trị chốt | n · phương pháp |
|---|---|---|---|
| LVT1883 Nôm | trang / tầng / cột | 105 trang, 105/105 hai tầng, 10 cột/tầng (trang 105: 5; trang 1 tầng dưới: 9) | chiếu mực Otsu; 2 phương pháp đếm cột đồng ý 210/210 tầng |
| | cặp lục bát → câu | 1.044 cặp = **2.088** câu; cột đúng 6/8 chữ 2.075/2.089 = 99,3 % | cột = 1 cặp (6 trên / 8 dưới) |
| | số in mỗi 5 câu | 397/417 đọc đúng (95,2 %); first_seq đúng 105/105 (Viterbi) | tesseract 8 lượt + kNN tự huấn (CV 99,8 %) |
| | hình học | pitch hàng 157, pitch cột 156, khe tầng 263 px, nghiêng p50 0,48° (max 2,3°) | n=105 |
| LVT1883 QN | dòng thơ | **2.088/2.088**, chuỗi 2.088, 0 trang trôi; 6 trang cần xem (5 xung đột giá trị, 1 không neo) | tesseract vie psm 4 TSV, lọc 897 dòng cước chú |
| | neo số câu | 714 neo (đầu dòng 345 + lề 369), vị trí mod 5 đúng 99,6 %, giá trị 80,3 %, phủ 97,6 % dòng có số | 2 phương pháp đồng ý 207/304, khi đồng ý đúng 206/207 |
| | chất lượng | parity 6/8 93,4 %; CER 7,2 % (16 dòng đọc mắt, 471 ký tự) | tham chiếu = người đọc ảnh 20/09 |
| KVK1884 Nôm | canvas / ánh xạ | 163 canvas chữ (4–166), `page = 167 − canvas`, first_seq = 20·(page−1)+1 đúng 163/163; số trang góc 161/163 = 169 − canvas | Viterbi ±1 trang |
| | cột / câu | 162/163 canvas 10+10 (canvas 4: 9/8); 1.628 cặp = **3.256** câu; cột đúng 6/8: 98,0 % | 2 phương pháp đếm cột đồng ý 326/326 |
| | số in | 472/651 đọc đúng (72,5 %, chữ số mảnh; không dùng nhãn tay) | kNN tự huấn CV 97,4 % |
| | hình học | pitch hàng 127, pitch cột 125, khe tầng 260 px (hộp đệm ±15 px), nghiêng p50 0,57° | n=163 |
| KVK1884 QN | phân loại 631 canvas | QN **295** (vol1 146 lẻ 27–317; vol2 149 lẻ 7–303), Pháp 309, khác 27 | vị trí số trang + tỉ lệ token Việt |
| | dòng thơ | **3.251 dòng vật lý** (vol1 1.497 + vol2 1.754); chuỗi số in **3.253**; 4 bất thường bản in: nhảy 1581→1583 (vol2 c23), 2220→2222 (c135), 2232 lặp (c137), 3053→3055 (c269) | kiểm ảnh c135: 11/11 dòng OCR đủ, số in thật |
| | neo | 934 neo, vị trí 99,25 %, giá trị 73,9 %, phủ 91,5 %; 40 trang cần xem (4 trôi, 22 mơ hồ, 7 xung đột, 14 không neo) | `pages_review.json` |
| | chất lượng | parity 6/8 96,1 %; CER 7,05 % (28 dòng, 808 ký tự); 1 dòng Hán lọt đã loại | |
| Chrestomathie | QN | 20 truyện (canvas 29–53 = 25 trang VN), 608 dòng thân, 8.151 âm, OOV 4,76 %; tiêu đề tự động khớp 17/20 | tesseract psm 4 |
| | Nôm | 65 trang (106–170), 7 ô/trang (48 trang đủ 7), **419 cột**, 21,3 chữ/cột đầy, ≈ **8.692** chữ; pitch↔run đồng ý 94,7 % | khung mở hình thái + tự tương quan |
| | ánh xạ | bảng truyện↔trang 20/20 (I 106–109 … XX 167–170), tỉ lệ chữ/âm 0,99–1,24 (truyện 18: 1,42); ranh giới tự động precision 18/18, recall 18/19; NomNaOCR xếp đúng truyện 6/6 | REF do agent đọc, **chưa người rà** |
| Detector CenterNet | thạch bản, thr 0,2, **27 trang/sách (540 cột)** | LVT raw 61,7 / **stretch 77,6** / otsu 76,3 %; KVK raw 23,5 / stretch 68,9 / **otsu 77,6 %** cột M==N; thr 0,3 rớt còn 20–29 % | mẫu 9 trang cũ tái lập đúng 82,2 / 80,3 (`--page-ids`) nhưng lạc quan ~4 điểm |
| | đối chứng STT2/4/11 (3 trang/sách, raw thr 0,2) | 90,1 % mọi cột (n=81); 92,7 % cột N kim == chiếu (n=55); recall hộp kim IoU 0,3 = 1,0; stretch/otsu không đổi kết quả STT | cùng gán cột như pipeline |
| | hình học hộp | h/pitch_y 1,13 (LVT) / 1,10 (KVK); w/rộng cột 1,13; số cột detector == chiếu 27/27 trang mỗi sách | |
| Mã | ghim '9' | 35 vị trí (24 ngoài lab/selftest), 9/9 vị trí đã biết; 9 bước CLI/54 tham số; 24 khoá JSON cache (8 không đọc); config chỉ 3 sách STT | `docs/PIPELINE_FACTS.json` 78,6 KB |

## 2. Invariants và kiểm chéo

- 118/118 PASS (code_facts 18, layout LVT 8, qn_ocr LVT 20, layout KVK 10, qn_ocr KVK 24, chresto_map 20, detector_transfer 18); 0 mềm, 0 SKIP; 2 mục STT trong `SOFT_INVARIANTS` (mẫu nhỏ) hiện PASS.
- Hai phương pháp độc lập: số cột (đỉnh chiếu vs chuỗi run) 100 % cả 2 sách; số câu in (tesseract vs kNN) đồng ý 77,6 % LVT / 52,3 % KVK, kNN CV 99,8 / 97,4 %; neo QN (đầu dòng vs lề) đồng ý 68 % / 63 %, khi đồng ý đúng 99,5 % / 99,0 %; số cột detector vs chiếu 27/27 trang; Nôm chữ/cột pitch vs run 94,7 %.
- Idempotence: `layout_lithograph --book LucVanTien1883 --limit 10` chạy 2 lần → `summary.json` md5 `290b0f15…` giống hệt, 4 CSV byte-identical; runner 2 lần (cache OCR / OCR tươi) 6 tệp md5 giống.
- Lệch đã sửa trong mã đo (21/09): (a) `detector_transfer.py` cắt lề trái 6 % → 2 % (mất cột 10 KVK ở 4/27 trang); thêm `--page-ids`; runner `--det-pages` 9 → 27. (b) `chresto_map.py` luật ranh giới "thiếu cột" chỉ tính cuối trang → precision 0,75 → 1,0. (c) `measure.py` nhãn "CER 44 dòng" → "CER dòng đọc mắt" + số dòng.
- Lệch do đổi định nghĩa/phương pháp, không sửa: khe tầng KVK 299 → 260 px (mực–mực vs hộp đệm); số in KVK 82,9 → 72,5 % (bỏ nhãn agent); FR/khác 313/23 → 309/27; trang đủ mọi số LVT 92 → 90 (final = kNN thay vì hợp tess+template); "tách chữ chiếu ngang" 72/63 % → 100/99,4 % (thuật toán neo pitch, không độc lập với giả định 6/8).

## 3. Lệnh tái lập

```bash
.venv/bin/python scripts/measure/measure.py --all                          # 7 bước, ~8 phút; mã thoát 0/1/2
.venv/bin/python scripts/measure/measure.py --all --report-only            # chỉ gom lại SUMMARY.json / REPORT.md
.venv/bin/python scripts/measure/measure.py --book KimVanKieu1884 --steps layout,qn_ocr
.venv/bin/python scripts/measure/detector_transfer.py --book all --stt-pages 0 --out /tmp/d9 \
  --page-ids "LucVanTien1883:010,030,050,070,090,100,020,060,095;KimVanKieu1884:0020,0050,0080,0100,0130,0160,0035,0120,0150"
.venv/bin/python scripts/measure/layout_lithograph.py --book LucVanTien1883 --limit 10 --out /tmp/i1   # ×2, so md5 summary.json
```
Sách mới: thêm vào `BOOKS` của `layout_lithograph.py` / `qn_print_ocr.py` và `ALL_BOOKS` của `measure.py` (xem `scripts/measure/README.md`), chạy `measure.py --book <Sách> --limit 8` trước khi chạy toàn bộ.

## 4. Việc còn mở (không thể đóng bằng máy)

1. **KVK 3.256 (Nôm) vs 3.251 (QN dòng vật lý)** — lệch 5 câu chưa định vị. Nôm không lệch lưới 20 câu/trang ở canvas nào; chuỗi QN có 4 bất thường bản in (parity đảo tại 1583 và 2225, đảo lại 2235 → giả thuyết bản QN bỏ câu có trong Nôm tại các chỗ nhảy). Người cần xem vol2 canvas 21–23, 133–137, 267–269 và đối chiếu cột Nôm quanh câu 1581–1585, 2220–2235, 3053–3055. Cho tới đó: ghép KVK **theo đoạn giữa neo** (`verses.tsv`: `verse_no`, `seq_no`, `page_flag`), không ghép theo chỉ số toàn sách.
2. **Trang QN cần người xem**: LVT 6 (`measure_out/LucVanTien1883/qn_ocr/pages_review.json`), KVK 40 (`…/KimVanKieu1884/qn_ocr/pages_review.json`) — chủ yếu mơ hồ/không neo, không phải trôi.
3. **Chrestomathie**: bảng truyện↔trang (`bang_truyen_trang.csv`, 20 dòng) và 6 ranh giới giữa trang (120/2, 122/5, 125/3, 139/4, 161/4, 167/5) cần người biết Nôm rà (~30 phút); truyện 18 (161–162, tỉ lệ chữ/âm 1,42) đáng ngờ nhất. Đơn vị ghép vẫn là truyện, chưa vào engine cột↔cột.
4. **Detector**: cổng "≥ 80 % cột đúng số hộp" **không đạt** trên 27 trang (77,6 / 77,6 %, CI ±3,5); §7 I5 hạ xuống ≥ 75 % hoặc chấp nhận thạch bản kém STT 13–15 điểm; chưa có GT hộp người vẽ (≥ 20 cột/sách) để đo precision thật.
5. **Số in KVK** đọc tự động chỉ 72,5 % (chữ số mảnh/vỡ); đủ cho giải mã theo trang (163/163) nhưng không đủ làm neo từng câu.
6. Số liệu Nôm↔QN trên đều **tự quy chiếu** (không GT người); mốc người kiểm ≥ 300 chữ/sách (§7) chưa thực hiện.
