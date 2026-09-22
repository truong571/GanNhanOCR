# Chốt kênh OCR (kim Hán-Nôm + Quốc ngữ) và quy hoạch gán cho 3 sách mới — 2026-09-23

Mọi số do script sinh, 0 token LLM. Tái lập:

```bash
.venv/bin/python scripts/measure/kim_channel_probe.py --steps params,crop,contrast,lang,tier_dp   # 107 lượt kim, đã cache → 0
.venv/bin/python scripts/measure/qn_print_ocr.py --book LucVanTien1883 --engine {tesseract|vietocr} --start 10 --limit 10 \
    --no-margin --out measure_out/qn_engine/LucVanTien1883_{tess|vietocr}      # KVK: --start 40 --limit 24
.venv/bin/python scripts/measure/qn_engine_compare.py
```
Đầu ra: `measure_out/kim_channel/SUMMARY.json` (+ `kim_cache/`, `kim_calls.json`, `img/`), `measure_out/qn_engine/SUMMARY.json`.
Ngân sách đã dùng: **107/120 lượt kim** (đếm bền trong `kim_calls.json`). Không sửa `pipeline/`, `core/`, `data/`; không commit.

## 0. Trả lời ngắn

| Câu hỏi | Trả lời |
|---|---|
| Kim đã gọi CHUẨN chưa? | **CHƯA.** Body đang gửi `lang_type = 1 = **Hán**` cho ba cuốn chữ **Nôm**. Đổi sang `lang_type = 2` (Nôm) nâng tỉ lệ chữ kim là cách đọc từ điển của âm QN **+18,4 / +20,2 / +10,1 điểm** và tỉ lệ khớp **dị bản độc lập +16,9 / +21,9 điểm** (§1). Chính máy chủ khi được hỏi cũng tự phân loại 3 trang này là `lang_type = 2`. |
| Cấu hình kim tốt nhất đo được | `ocr_id 1, **lang_type 2**, reading_direction 1, font_type 1`, ảnh **CẢ TRANG**, ảnh nguồn như hiện nay. `font_type 2`/`epitaph 1` (mô hình văn bia) và bước `image-preprocessing` của web: **không lợi** (§1). |
| Cắt cột / cắt tầng rồi mới OCR? | **KHÔNG.** Cột ≈ trang (kém 1,9/2,4 điểm, trong CI); tầng-cột **hỏng nặng**: kim chỉ đọc 34/60 (LVT) và 50/60 (KVK) chữ, đúng 4,3 % / 45,3 % (§2). Giữ nguyên adapter gọi theo trang. |
| Tiền xử lý cho kim | **Không cần.** Ảnh LVT/KVK vốn đã nền 255 nên `--contrast stretch` (LVT) và `--contrast otsu` (KVK) là **ánh xạ đồng nhất** — đo byte-identical. Chỉ Chresto có nền xám thật và ở đó `stretch` hơn `none` 2,1 điểm (n = 95, trong CI) (§3). |
| QN: tesseract hay VietOCR? | **tesseract `--psm 4` thắng mọi chỉ số** trên sách in thế kỷ 19: CER 0,072 vs 0,193 (đối chiếu mắt), 0,248 vs 0,337 và 0,118 vs 0,153 (đối chiếu phiên âm chuẩn), parity 6/8 91,1 % vs 78,3 %, và nhanh **225×** (0,09 vs 20–26 s/trang). **Không đổi** (§4). |
| Quy hoạch gán nên đổi gì | Ràng buộc đếm 6/8 + số câu in phải là **neo cứng** ở mức tầng, và QN — chứ không phải kim — mới là bên hay sai số đếm (87 % / 63 % ca M≠N) (§5–§6). |

## 1. Kim — tham số gọi (I.1)

Giá trị hợp lệ lấy từ mã trang web (`/js/hannom.js` + `<select>` trang chủ), không đoán:

| Tham số | Giá trị | Đang gửi (`core/ocr/ocr_api.py:324-330`) |
|---|---|---|
| `ocr_id` | −1 Tự động · 1 Văn bản thông thường · 2 Hành chính · 3 Ngoại cảnh · 4 Y học dân tộc · 5 Văn bia · 6 Kinh Phật (web quy 4/5/6 → 1, riêng 5 bật `epitaph=1`) | 1 |
| `lang_type` | 0 Tự động · **1 Hán** · **2 Nôm** | **1 (Hán)** ⚠ |
| `reading_direction` | 0 Tự động · 1 Dọc · 2 Ngang | 1 (đúng) |
| `font_type` | 0 Tự động · 1 In · 2 Viết tay (web: 2 → `ocr_id=5` → `epitaph=1`) | 1 (đúng, sách in) |
| khác | web còn gọi `structure-classification` (tự đoán 3 tham số trên) rồi `image-preprocessing` → `new_file_name` trước `image-ocr`; mã ta bỏ cả hai | — |

Đo trên **1 trang/sách** (139 / 139 / 153 ô), hai tiêu chí **không vòng tròn**: `in_R` = chữ kim ∈ tập đọc từ điển của âm QN cùng vị trí
(điều kiện cần của GOLD trực tiếp); `eq_ref` = chữ kim == chữ Nôm **dị bản độc lập** (Kiều 1871 LVĐ / LVT 1916 NF, bỏ vị trí PUA).
`eq_label` = đồng thuận với nhãn đang giao nộp — **vòng tròn** (nhãn GOLD sinh từ chính kim mặc định) nên chỉ đọc là "mức xáo trộn bộ nhãn".

| Cấu hình | LVT `in_R` / `eq_ref` | KVK `in_R` / `eq_ref` | Chresto `in_R` | `eq_label` (LVT/KVK/CHR) |
|---|---|---|---|---|
| `c0` hiện hành (lang 1 = Hán) | 66,2 / 57,7 | 80,6 / 78,8 | 78,4 | 92,9 / 99,1 / 94,5 |
| **`c1` lang_type 2 (Nôm)** | **84,9 / 76,1** | **92,1 / 92,9** | **83,7** | 86,9 / 87,5 / 84,3 |
| `c2` font_type 2 + epitaph 1 | 67,6 / 57,7 | 71,2 / 68,2 | 74,5 | 91,9 / 83,9 / 86,6 |
| `c3` + bước `image-preprocessing` | 66,2 / 57,7 | 80,6 / 78,8 | 78,4 | = `c0` **từng ký tự** |

Mở rộng **5 trang/sách** (`--steps lang`; `c0` lấy từ `kim_raw/` có sẵn = 0 lượt), chỉ tính **cột kim đếm đúng N** (ghép theo vị trí, không cần căn NW):

| Sách | `in_R` c0 → c1 [Wilson 95 %] | `eq_ref` c0 → c1 [Wilson 95 %] |
|---|---|---|
| LucVanTien1883 (42→37 cột) | 70,0 [66,2–73,6] → **88,4** [85,4–90,9] (n 587/518) | 55,8 [49,9–61,5] → **72,7** [66,8–78,0] (n 276/242) |
| KimVanKieu1884 (45→39 cột) | 72,5 [68,9–75,8] → **92,7** [90,2–94,6] (n 629/545) | 64,4 [59,6–69,0] → **86,3** [82,3–89,5] (n 399/350) |
| Chrestomathie1872 (29→18 cột) | 74,9 [71,4–78,1] → **85,0** [81,2–88,2] (n 638/394) | không có dị bản |

**Chứng chỉ không trôi mô hình**: `c0` gọi lại hôm nay trên đúng 3 ảnh trang ấy trả **cùng số hộp và cùng chuỗi ký tự** với
`prepared/<book>/kim_raw/` của lần chạy chốt (22/22 · 21/21 · 7/7 hộp; 143/141/154 ký tự, so khớp tuyệt đối) ⇒ chênh lệch ở
bảng trên là do **tham số**, không do máy chủ đổi mô hình.

Khoảng tin cậy **không chồng nhau** ở cả 3 sách. Mốc đối chiếu: nền dị bản 1871↔1872 = 82,9 % (`auto_precision`) → kim với `lang_type 2`
khớp 1871 ở mức **86,3 %**, tức *cao hơn mức hai ấn bản khác nhau khớp nhau*; đây là cận trên thực tế của phép đo này.

**Cái giá phải trả**: `lang_type 2` đọc ít chữ hơn chút nên tỉ lệ **cột đếm đúng N** giảm: LVT 84 → 74 %, KVK 90 → 78 %, Chresto 82,9 → **51,4 %**
(50/50/35 cột). Với Chresto mức giảm này lớn hơn cái được (+10 điểm `in_R`) ⇒ khuyến nghị khác nhau theo sách (§6).
`eq_label` giảm ~10 điểm nghĩa là ~1/10 nhãn hiện có sẽ đổi chữ — phải chạy lại toàn bộ và **không** còn byte-identical với bộ đã công bố.

## 2. Kim — mức ảnh: cả trang vs 1 cột vs 1 tầng-cột (I.2)

10 cột/sách "nhãn ổn định" (M==N, không `conflict`, ≥ 70 % ô GOLD), cùng ô tham chiếu; mức trang lấy từ cache (0 lượt), cột và tầng gọi mới.

| Mức ảnh | LVT: đếm đúng / `eq_label` / `eq_ref` | KVK | Chresto |
|---|---|---|---|
| CẢ TRANG (hiện hành) | **100 % / 93,8 / 60,2** [50,8–68,9] | **100 % / 94,2 / 67,7** [59,1–75,3] | 90 % / 97,8 |
| 1 CỘT (đã cắt) | 80 % / 91,2 / 58,3 [48,9–67,2] | 80 % / 90,1 / 65,3 [56,6–73,1] | **100 % / 96,6** |
| 1 TẦNG-CỘT (6 chữ) | 0 % / **8,3** / 4,3 — kim chỉ đọc **34/60** chữ | 20 % / **54,9** / 45,3 — đọc **50/60** | không có tầng |

**Kết luận**: cắt nhỏ ảnh làm kim *kém đi*, không tốt lên — mô hình cần ngữ cảnh trang; dải hẹp 6 chữ làm nó rụng gần nửa số chữ.
**Không đổi adapter.** (Đòn bẩy đã tưởng là lớn hoá ra âm; ghi lại để không ai thử lại.)

## 3. Kim — tiền xử lý (I.3)

5 cột/sách × {none, stretch, otsu} (cùng hàm `stretch_gray`/`otsu_gray` của adapter, áp ở mức **trang** rồi mới cắt):

- **LVT**: ảnh gốc đã có p2 = 0 và p90 = 255 ⇒ `stretch_gray` là **ánh xạ đồng nhất**; crop `none` và `stretch` **trùng md5**. `otsu` kém hơn
  (`eq_label` 88,1 vs 93,2; `eq_ref` 69,0 vs 72,4; đếm đúng 80 vs 100 %).
- **KVK**: nền đã 255 ⇒ `otsu_gray` cũng **đồng nhất** (`none` ≡ `otsu`, trùng md5). `stretch` đổi ảnh nhưng `eq_ref` 74,2 vs 72,7 — **trong CI**.
- **Chresto**: nguồn xám thật (p90 = 128, max 172) và kim đang chạy trên **ảnh gốc** (`kim_src: orig`). `eq_label` none 95,8 · stretch 97,9 · otsu 96,8;
  `in_R` **87,7 % cả ba** ⇒ khác biệt không đáng kể.

**Kết luận**: kim **không** mù nền xám (khẳng định lại) và tiền xử lý **không phải đòn bẩy cho kim** — khác hẳn detector CenterNet (cần `otsu`/`INTER_AREA`).
Hệ quả phụ đáng ghi: hai khoá `--contrast` đang đặt cho LVT/KVK **không làm gì cả** trên chính các bản quét này.

## 4. Kênh Quốc ngữ: tesseract `--psm 4` vs VietOCR (I.4)

Cùng tập trang, cùng mã trích dòng (`qn_print_ocr.py`), chỉ khác `--engine`. CER đối chiếu **phiên âm chuẩn dị bản** (căn cửa sổ neo + NW mức dòng;
CER này **gồm** khác biệt dị bản thật nên chỉ dùng để SO HAI ENGINE, không đọc tuyệt đối). CER "đối chiếu mắt" là mốc `visual_truth` có sẵn của script.

| Chỉ số | LVT 1883 (10 trang, 146 dòng) | KVK 1884 (12 trang, 97 dòng) |
|---|---|---|
| dòng thơ trích được: tess / vietocr | **146** / 143 | 97 / 97 |
| parity 6-8 đúng: tess / vietocr | **91,1 %** / 78,3 % | 91,8 % / 91,8 % |
| CER đối chiếu **mắt** (471 ký tự): tess / vietocr | **0,0722** / 0,1932 | — (3 trang mẫu ngoài tập) |
| CER đối chiếu **phiên âm chuẩn**: tess / vietocr | **0,2475** / 0,3369 (↔ LVT1916, dị bản xa) | **0,1183** / 0,1530 (↔ Kiều 1871) |
| âm ngoài từ điển: tess / vietocr | **7,29 %** (74/1015) / 7,68 % (73/950) | 5,10 % (35/686) / **4,39 %** (29/660) |
| giây/trang | **0,09** | 20,25 (LVT) · 25,77 (KVK) |

Chresto (không có phiên âm chuẩn): tesseract, 25 trang VN, 608 dòng thân, **âm ngoài từ điển 4,76 %** (n = 8.151) — `measure_out/Chrestomathie1872/chresto_map/summary.json`.

**Chốt**: **giữ tesseract `--psm 4`** cho sách in thế kỷ 19. Lưu ý phương pháp: `oov` một mình là tiêu chí **lệch** với engine có mô hình ngôn ngữ —
VietOCR có `oov` thấp hơn ở KVK nhưng CER cao hơn, vì nó "sửa" âm lạ thành từ có trong từ điển (sai mà hợp lệ). VietOCR vẫn đúng chỗ của nó ở STT
(trang QN chép tay/in hiện đại) — kết luận này **chỉ** cho sách in thế kỷ 19.

## 5. Quy hoạch gán đề xuất cho 3 sách mới (II.5)

### 5.1 Cấu trúc thật và bên nào hay sai

| Bằng chứng | LVT1883 | KVK1884 |
|---|---|---|
| cột kim đọc đủ 14 chữ | 1.024/1.044 (98,1 %) | 1.560/1.628 (95,8 %) |
| …trong đó tách tầng đúng **6+8** | **1.024/1.024 (100 %)** | **1.560/1.560 (100 %)** |
| dòng QN đủ 14 âm | 893/1.044 (85,5 %) | 1.511/1.628 (92,8 %) |
| …trong đó chia 6/8 đúng | 887/893 (99,3 %) | 1.508/1.511 (99,8 %) |
| cột M≠N: **lỗi do QN** / do kim / cả hai | 148 / 20 / 2 (**87 % là QN**) | 115 / 66 / 2 (**63 % là QN**) |

Đọc: **tách tầng hình học của kim là bằng chứng gần như hoàn hảo** (kim chưa bao giờ đọc đủ 14 chữ mà chia sai 6/8), còn **số đếm QN mới là khâu yếu**.
Nhưng `tier_split` của kim **được ghi vào cache rồi không bao giờ được đọc** (`grep tier_split pipeline/ core/` → chỉ `ingest` ghi + `dict_boost`),
và `pitch_target_count` lại **ưu tiên `n_qn`** (`align_production.py:625`) — tức pipeline đang tin bên yếu hơn.

### 5.2 Sơ đồ gán đề xuất (thứ tự ràng buộc: trang → tầng-cột → câu → chữ)

1. **Trang**: bố cục 10 cột × 2 tầng từ `measure_out/<book>/layout` (đã có). Cổng: đủ `n_columns`, `first_seq` khớp chuỗi số câu in.
2. **Câu (neo cứng #1)**: số câu in ở lề mỗi 5 câu → `verse_no`. Đã có (`qn_print_ocr.py` §4-5, hai phương pháp độc lập: inline + `--psm 7` số lề).
   Mỗi cột = đúng **một cặp** (câu lẻ, câu chẵn) ⇒ **không DP ở mức trang**, chỉ tra bảng.
3. **Tầng-cột (neo cứng #2 — MỚI)**: đơn vị gán là **TẦNG**, không phải cột. Bên Nôm: chữ kim thuộc tầng theo tâm y (đã tính, `tier_split`).
   Bên QN: số âm của câu lục = 6, câu bát = 8 **theo luật thể thơ**, không theo số âm OCR đọc được.
4. **Sửa số đếm QN trước khi DP (MỚI)**: với tầng có `n_syl ≠ {6,8}` mà kim cho đúng 6/8 → âm QN sai. Xử lý theo thứ tự:
   (a) tách/gộp âm theo từ điển âm tiết (dấu "-" của ấn bản, âm dính); (b) nếu có phiên âm chuẩn/dị bản (B1' KVK) → lấy dòng dị bản khi
   ≥ 0,9 tương đồng; (c) nếu vẫn ≠ 6/8 → đánh dấu `qn_count_unfixed`, DP tự do như hiện nay, **không** ép.
5. **DP chữ↔âm CHẠY TRONG TỪNG TẦNG** (6↔6, 8↔8) thay vì cả cột 14↔14. Giữ nguyên ma trận chi phí CALIB + `ANCHOR_CAP`
   (`anchor_align.py`), chỉ đổi phạm vi: băng hẹp lại `|i−j| ≤ |m−n| + 2` trong 6 hoặc 8 phần tử ⇒ khe ở câu lục **không thể** trôi sang câu bát.
6. **Hộp ảnh**: giữ `box_decoder: pitch` như hiện nay, nhưng `tier_n` lấy **6/8 theo luật** (và đối chiếu `tier_split` của kim) thay vì `len_odd` của QN.
7. **Tier nhãn**: giữ nguyên `consensus.decide_label` + cổng cơ chế B4'. Thêm cờ `tier_barrier_moved` cho ô đổi cặp khi bật rào tầng.
8. **Chresto (văn xuôi)**: không có 6/8 và không có số câu in ⇒ giữ DP mức **truyện** (đúng thiết kế hiện tại). Neo cứng khả dụng duy nhất là
   ranh giới truyện (`bang_truyen_trang.csv`) + tiêu đề; nên thêm neo mềm "số chữ/cột = pitch cột" (đã đo: pitch vs runs khớp ≤2 ở 94,7 %).

### 5.3 Khác gì thiết kế 9 cột của STT — và vì sao

| | STT (chép tay) | Sách mới (thạch bản in) |
|---|---|---|
| Nguồn cột Nôm | `boxes_to_columns` gom hộp theo `x_tol = **15 px cứng**` + `_merge_fragment_columns` (5 luật vá) + retry `frame_pad` để ép về 9 cột | adapter tự gán hộp vào (slot, tầng) theo **hình học đã đo** (`assign_boxes_to_columns`); **không** dùng `x_tol` cứng, không retry |
| Neo cấu trúc | số dòng in 1..9 trên trang QN đối diện | **số câu in ở lề** (mỗi 5 câu) + **luật 6/8** — mạnh hơn hẳn |
| Số âm/cột | biến thiên, không kiểm được | **cố định 14 = 6⧺8** ⇒ dùng làm ràng buộc cứng |
| Kênh QN | VietOCR (chép tay/in hiện đại) | **tesseract psm 4** (in thế kỷ 19) — §4 |
| Đơn vị DP | cả cột (buộc phải thế) | **nên là tầng** (§5.2 bước 5) |
| Ngưỡng/luật vá | nhiều, hiệu chuẩn trên chữ thảo | nên **bỏ bớt**: chữ tách rời, nền sạch ⇒ ràng buộc đếm thay cho ngưỡng hình ảnh |

## 6. Danh mục cải tiến, xếp theo lợi/chi phí (II.6)

| # | Thay đổi | Tệp | Kỳ vọng (từ số đo §1–§5) | Rủi ro với STT | Nghiệm thu |
|---|---|---|---|---|---|
| 1 | **`lang_type = 2` cho 3 sách Nôm** — thêm khoá theo sách (`books[].kim_lang_type`), mặc định 1 ⇒ STT không đổi | `core/ocr/ocr_api.py` (`recognize` nhận tham số), `book_layout.py`, `ingest_*_book.py` | `in_R` +18,4 (LVT) / +20,2 (KVK) / +10,1 (CHR) điểm, `eq_ref` +16,9 / +21,9 điểm ⇒ GOLD trực tiếp ước **62 % → ~79 %** (LVT) và **67 % → ~86 %** (KVK), tức **+2.400 / +4.200 ô GOLD** (giả định khoảng cách `in_R`→GOLD giữ nguyên ~7 điểm) | **0** nếu mặc định giữ 1 — phải kiểm `labels.csv` STT byte-identical (md5 `59e436d7…`) | chạy lại ingest 3 sách (333 lượt kim, ~25 phút) → `auto_precision --steps cross` khớp dị bản phải tăng ≥ 5 điểm; `box_ref` không giảm; `tools.selftest` |
| 2 | **Sửa số đếm QN theo 6/8 trước DP** (§5.2 bước 4) | `ingest_lithograph_book.py` (`lithograph_syllables` + bước mới), `verses_ref_fix.py` | chạm **151 cột LVT (14,5 %) + 117 cột KVK (7,2 %)**; các cột này đang có REVIEW cao hơn 4,9 điểm (25,7 vs 20,8 %) ⇒ ước **+250 / +150 ô** rời REVIEW | 0 (chỉ chạy khi `layout=lithograph`) | `n_cols_syll14` trong `manifest.json` phải tăng; `parity_rate` verses.tsv không giảm; khớp dị bản không giảm |
| 3 | **Rào tầng cho DP** (`realign_column` theo tầng, dùng `tier_split` + 6/8) | `align_production.py::_pair_new_state`, `anchor_align.py` | phơi nhiễm đã đo: ghép **xuyên tầng 18 cột/1.044 (LVT), 17/1.628 (KVK)**; DP theo tầng đổi **50 / 66 ô** (0,34 % / 0,29 %). Lợi **nhỏ nhưng là bảo đảm cấu trúc**; trên ô có dị bản n = 6/9 quá nhỏ để phân định (1 vs 0 và 0 vs 0) | 0 nếu chỉ bật khi có `tier_split` (STT không có khoá này) | `--steps tier_dp` trước/sau; `labels_final` chỉ đổi ≤ 120 ô, và ô đổi phải nằm trong danh sách `cells_cross` |
| 4 | **`tier_n` lấy 6/8 theo luật** thay `len_odd` của QN trong `pitch_decode` | `book_layout.py::expected_tier_counts` | tác động ở đúng 151/117 cột của #2; hộp bớt lệch khi QN đếm sai | 0 (`expected_tier_counts` chỉ đọc khi `box_decoder=pitch`) | `box_ref_eval.py` IoU ≥ 0,5 không giảm; `detector_low`/`ink_cut` không tăng |
| 5 | **Dọn khoá `--contrast` chết cho kim** — ghi rõ trong config rằng `stretch`(LVT)/`otsu`(KVK) là ánh xạ đồng nhất; cân nhắc `kim_src: prepared` + `stretch` cho Chresto | `config/pipeline_*.yaml`, `ingest_prose_book.py` | kim: **0 → +2,1 điểm** `eq_label` cho Chresto (n = 95, trong CI) ⇒ chủ yếu là **gỡ hiểu nhầm**, không phải nâng số | 0 | md5 `pages/*.png` LVT/KVK không đổi; Chresto chạy lại 65 lượt kim nếu đổi `kim_src` |
| — | **BÁC BỎ** (đo rồi, âm): OCR theo cột (§2), OCR theo tầng (§2), `font_type 2`/`epitaph` (§1), bước `image-preprocessing` (§1), đổi QN sang VietOCR (§4) | | | | |

Thứ tự thực hiện đề nghị: **1 → 2 → 4 → 3 → 5**. #1 bắt buộc phải chạy lại toàn bộ 3 sách nên gộp #2/#4 vào cùng một lần chạy.

## 7. Giới hạn của chính các phép đo trên

1. `eq_label` là tiêu chí **vòng tròn** (nhãn sinh từ kim mặc định) — không được đọc là "cấu hình mới làm sai đi".
2. `eq_ref` dùng **dị bản khác ấn bản**: trần thực tế ≈ 82,9 % (nền 1871↔1872); LVT chỉ có bản 1916 xa hơn nên số tuyệt đối thấp.
   Chresto **không có** dị bản ⇒ mọi kết luận cho Chresto chỉ dựa trên `in_R` (một chiều: đo "hợp từ điển", không đo "đúng chữ").
3. n của §1 mở rộng là 5 trang/sách (≈ 520–630 ô/cấu hình); §2–§3 là 10 và 5 cột/sách (60–220 ô) ⇒ CI ±6–12 điểm, chỉ kết luận được
   dấu, không kết luận được độ lớn chính xác. Tập cột "đếm đúng N" của `c0` và `c1` **không trùng nhau hoàn toàn** (42 vs 37 cột…).
4. Chưa đo `lang_type 0` (Tự động) — nhưng `structure-classification` trả `lang_type = 2` cho cả 3 trang nên Tự động ≈ `c1`.
5. Phép đo `lang_type` chỉ đo **văn bản**; ảnh hưởng tới **hộp/ảnh crop** (I5, bleed) chưa đo — phải chạy `box_ref_eval.py` sau khi đổi.
6. Ước lượng "+2.400/+4.200 ô GOLD" ở §6 #1 là **ngoại suy** từ khoảng cách đo được giữa `in_R` mức ô và tỉ lệ `s1_inter_s2_direct` hiện tại
   (LVT 70,0 vs 62,0; KVK 72,5 vs 67,2), **không** phải kết quả của một lần chạy thật.
7. Kim là dịch vụ trực tuyến: `c0` hôm nay trùng **tuyệt đối** với `kim_raw/` cũ (§1), nhưng đó là bằng chứng cho **hôm nay**;
   mọi lần chạy lại vẫn phải ghi ngày và giữ `kim_raw/`. Tài khoản đang đăng nhập được (không phải Guest) — nếu rơi về Guest,
   phải đo lại chứng chỉ này trước khi tin bất kỳ so sánh nào.
