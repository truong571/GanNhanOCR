# Vòng 8 — sửa 4 lỗi CĂN CHỈNH + crop giao nộp giữ ảnh gốc, chạy lại 5 bộ (23/09/2026)

Trên HEAD `014a0733fa` + thay đổi chưa commit của vòng 7. **Chưa commit gì.** `data/` chỉ ĐỌC.
Đề bài: `docs/RA_SOAT_CAN_CHINH_2026-09-23.md` §4 (5 đề xuất) — vòng này làm **#1, #2, #4** và phần
`pitch_target_count` của §3.1; #3 và #5 **KHÔNG làm** (§7).

> **Bốn dòng kết luận.**
> 1. **Đánh số câu 2 bộ IHR đã đúng parity**: bất biến A11 **61/988 → 0** (LucVanTien1916) và
>    **1.078/1.610 → 0** (TruyenKieu1872); A12 "số câu bị dùng hai lần" **1 → 0** ở cả hai bộ.
> 2. **Cột đếm âm QN hỏng nay là CỜ THẬT trong `labels.csv`** (`qn_count_unfixed`) và bị cổng cơ chế
>    hạ xuống REVIEW. Đo trên **nhãn người**: precision GOLD-ảnh **97,07 → 97,96 %** (LVT1916) và
>    **98,49 → 98,64 %** (TK1872); lỗi "khác hẳn" (không phải dị thể, không phải gần hình)
>    **111 → 31** và **19 → 1** ô. Giá phải trả: GOLD-ảnh −150 và −31 ô.
> 3. **Cổng bố cục thạch bản nay kiểm LUẬT 6/8** thay vì kiểm `num_syllables` do chính tesseract ghi:
>    `page_ok` 103/105 → **77/105** (LVT1883), 162/163 → **137/163** (KVK), 97/99 → **81/99** (LVT1916),
>    159/161 → **154/161** (TK1872). Đây là **cổng ĐẾM**, không loại ô nào — nó chỉ làm cột hỏng
>    **nhìn thấy được ở mức trang** thay vì đi lọt.
> 4. **Crop giao nộp của 5 sách nay cắt từ ẢNH QUÉT GỐC** (nền giấy 255 → 128–133 mức xám); bộ
>    `dataset/<Book>/` đã dựng lại toàn bộ, bản cũ giữ ở `dataset/<Book>_v7/`.
>
> **Hồi quy STT byte-identical**: `labels.csv` md5 `59e436d7641fa849bb6759868ac29259`, 556 dòng, 42 cột,
> `diff -rq` 0 tệp khác; `git status -- dataset_out prepared/SachThanhTruyen*` **trống**.

---

## 0. Tái lập

```bash
PY=.venv/bin/python
# (1) selftest — tất cả 0 FAIL
$PY -m pipeline.align_engine.book_layout_selftest          # 157/0   (vòng 7: 133)
$PY -m pipeline.phase1_engine_selftest                     # 288/0   (vòng 7: 270)
$PY -m pipeline.tools.ingest_ihr_selftest                  # 81/81   (vòng 7: 64)
$PY -m pipeline.remediation.mechanism_gates_selftest       # 113/0   (vòng 7: 94)
$PY -m pipeline.tools.ingest_lithograph_selftest           # 82/82
$PY -m pipeline.tools.ingest_prose_selftest                # 33
$PY -m pipeline.align_engine.char_detector.pitch_decode --selftest   # 22/0
$PY scripts/measure/verses_ref_fix.py --selftest           # 19/19
$PY scripts/measure/align_audit.py --selftest              # SELFTEST OK
$PY scripts/measure/ihr_endtoend_eval.py --selftest        # 24/24
$PY scripts/measure/code_facts.py --check                  # 18/18

# (2) hồi quy STT 3 trang — PHẢI byte-identical
printf 'book,page\nstt2,page_0024\nstt4,page_0050\nstt11,page_0100\n' > /tmp/pages3.csv
$PY -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --qd01-cells none \
    --force --pages /tmp/pages3.csv --out <scratch>/stt && md5 <scratch>/stt/labels.csv

# (3) chạy lại 5 bộ (≈2 h CPU, 0 gọi API — kim đọc từ cache kim_raw/)
NONINTERACTIVE=1 ./run_pipeline.sh --book all-new     # LVT1883, KVK1884, Chrestomathie1872
NONINTERACTIVE=1 ./run_pipeline.sh --book all-ihr     # LucVanTien1916, TruyenKieu1872 (tập ĐÁNH GIÁ)

# (4) nghiệm thu
$PY scripts/measure/align_audit.py --book all                 # 13 bất biến + trôi, ≈10 s
$PY scripts/measure/ihr_endtoend_eval.py --book all           # precision THẬT trên nhãn người
$PY scripts/measure/box_ref_eval.py --book all --pages 27     # 24/24 bất biến PASS
$PY scripts/measure/crop_source_samples.py --build prepared/LucVanTien1883/dataset_out \
    --build prepared_b1/KimVanKieu1884/dataset_out_b1 --build prepared/Chrestomathie1872/dataset_out
```

---

## 1. Bảng TRƯỚC → SAU, đủ 5 bộ

Nguồn: `measure_out/align_audit/SUMMARY.json` (ô/tier/M==N/trôi) + đếm tệp `.png` trong
`dataset/<Book>{_v7,}/{gold,syllable}` (ảnh export). "Trôi" = ô **đúng chữ, sai ô**
(`lech_vi_tri`), chuẩn = nhãn người với 2 bộ IHR, phiên âm dị bản với 2 bộ thạch bản.

| sách | ô | GOLD ảnh | GOLD_text_only | SYLLABLE | REVIEW | ảnh export | M==N | trôi |
|---|---|---|---|---|---|---|---|---|
| **LucVanTien1883** | 14.474 → 14.474 | 10.831 → **10.581** | 149 → 143 | 1.207 → 1.199 | 2.259 → **2.523** | 12.038 → 11.780 | 90,4 → 90,4 % | 7 → 7 |
| **KimVanKieu1884** | 22.750 → 22.750 | 19.303 → **19.018** | 727 → 720 | 618 → 614 | 1.971 → **2.267** | 19.921 → 19.632 | 96,9 → 96,9 % | 16 → 16 |
| **Chrestomathie1872** | 8.016 → 8.016 | 5.998 → 5.998 | 226 → 226 | 856 → 856 | 936 → 936 | 6.854 → 6.854 | 44,0 → 44,0 % | — |
| **LucVanTien1916** | 13.761 → 13.760 | 8.783 → **8.633** | 3.424 → 3.366 | 89 → 88 | 1.465 → **1.673** | 8.872 → 8.721 | 96,6 → 96,7 % | 91 → 91 |
| **TruyenKieu1872** | 22.499 → 22.499 | 13.465 → **13.434** | 6.694 → 6.670 | 48 → 48 | 2.278 → **2.333** | 13.513 → 13.482 | 95,4 → 95,4 % | 32 → 32 |

| sách | precision (chuẩn) | TRƯỚC | SAU |
|---|---|---|---|
| LucVanTien1916 | **nhãn NGƯỜI**, GOLD có ảnh | 97,07 % (n 8.775) | **97,96 %** (n 8.629) CI[97,64; 98,24] |
| LucVanTien1916 | nhãn NGƯỜI, GOLD kể text_only | 97,22 % (n 12.197, cov 0,872) | **98,00 %** (n 11.995, cov 0,857) |
| TruyenKieu1872 | **nhãn NGƯỜI**, GOLD có ảnh | 98,49 % (n 12.736) | **98,64 %** (n 12.706) CI[98,42; 98,83] |
| TruyenKieu1872 | nhãn NGƯỜI, GOLD kể text_only | 98,47 % (n 19.128, cov 0,893) | **98,61 %** (n 19.075, cov 0,891) |
| LucVanTien1883 | proxy dị bản LVT1916, GOLD sau cổng | 79,8 % (n 3.178) | 79,8 % (n 3.140) |
| KimVanKieu1884 | proxy dị bản Kiều 1871, GOLD sau cổng | 87,8 % (n 14.702) | 87,8 % (n 14.573) |
| Chrestomathie1872 | *không có chuẩn độc lập* | — | — |

**Đọc bảng — ba điều phải nói rõ.**

* **Chỉ số "trôi" KHÔNG đổi** (7 / 16 / 91 / 32). Đúng như thiết kế: bốn sửa của vòng này **không
  đụng phép ghép DP chữ↔âm**, chúng chỉ *phát hiện* và *hạ cấp* nhóm cột hỏng. Ô trôi vẫn còn trong
  `labels_gated.csv` ở tầng REVIEW — chỉ là không còn được giao nộp như GOLD.
* **Proxy dị bản không nhúc nhích** (79,8 % và 87,8 % ở cả hai bản). Đây là **giới hạn của phép đo**,
  không phải bằng chứng "sửa vô ích": `auto_precision.match_verses` chỉ nhận câu có ĐÚNG số âm bằng
  số chữ của dị bản, nên nó **mù** với đúng nhóm cột `n_qn ≠ 14` mà cổng mới nhắm tới
  (RA_SOAT §5 mục 2: chỉ 7/609 và 42/4.792 câu khớp đến từ nhóm này). Bằng chứng thật nằm ở
  hai dòng **nhãn người** phía trên.
* **Lỗi giảm ở đúng loại lỗi đáng lo.** Trên nhãn người, lỗi GOLD-ảnh rã ra:

  | | LVT1916 TRƯỚC → SAU | TK1872 TRƯỚC → SAU |
  |---|---|---|
  | dị thể (cùng âm, khác tự dạng) | 146 → 145 | 171 → 171 |
  | gần hình (`SinoNom_Similar`) | 0 → 0 | 0 → 0 |
  | GT vùng PUA | 0 → 0 | 2 → 1 |
  | **khác hẳn (lỗi đọc thật)** | **111 → 31** | **19 → 1** |

  Tức 80/111 và 18/19 ô "đọc sai hẳn" đã rời khỏi tầng GOLD-ảnh, trong khi số ô dị thể (vốn **không
  phải lỗi pipeline**) gần như không đổi. `bỏ PUA + dị thể`: 98,71 → **99,63 %** và 99,85 → **99,99 %**.

---

## 2. Bất biến căn chỉnh — TRƯỚC → SAU (8 bộ, `align_audit --book all`)

| Bất biến | STT2/4/11 | LVT1883 | KVK1884 | Chresto | LVT1916 | TK1872 |
|---|---|---|---|---|---|---|
| A1–A8 (cốt lõi: ô duy nhất, âm đúng vị trí, thứ tự đọc) | 1/2/0 → 1/2/0 *(a)* | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| A9 cột 14 âm, tách 6⧺8 | — | 27 → 27 | 26 → 26 | — | 22 → **21** | 7 → 7 |
| A10 không ghép xuyên tầng | — | 0 → 0 | 0 → 0 | — | 0 → 0 | 0 → 0 |
| **A11 parity câu lẻ/chẵn** | — | 0 → 0 | 0 → 0 | — | **61 → 0** | **1.078 → 0** |
| **A12 số câu trùng** | — | 0 → 0 | 0 → 0 | — | **1 → 0** | **1 → 0** |
| A12 số câu *thiếu* (trang bị cổng loại) | — | 0 | 0 | — | 83 → 86 | 20 → 20 |
| A13 span truyện liền | — | — | — | 0 → 0 | — | — |

*(a)* A3 ở STT = 3 ô `syllable` rỗng ở tầng REVIEW, có từ trước, không liên quan vòng này.

**Hai bất biến còn FAIL và lý do (KHÔNG sửa được ở vòng này):**

1. **A9 = 27/26/21/7 cột** — cột mà QN đếm ≠ 14 âm. Đây là lỗi **của bản OCR Quốc ngữ**, phải sửa ở
   adapter (`repair_tier_syllables`, đề xuất #3 của RA_SOAT) chứ không phải ở engine. Vòng 8 chỉ
   **gắn cờ + hạ cấp** chúng. LVT1916 giảm 22 → 21 là **hệ quả phụ của sửa (a)**: luật
   `drop_verse_number` của adapter chỉ áp cho câu có `verse_no % 5 == 0`, nên đánh số câu đúng
   parity làm luật ấy rơi vào đúng câu — cũng là lý do bộ này bớt đúng 1 ô (13.761 → 13.760).
2. **A12 "thiếu số câu"** (86 / 20) — các trang bị cổng `page_gate` của adapter loại (5 trang LVT1916,
   1 trang TK1872: không có ô cột, hoặc `n_cols ≠ ceil(số câu / 2)`). Số câu vẫn được **giữ chỗ**
   để không đảo parity ⇒ khoảng trống là hệ quả CỐ Ý, không phải lỗi sổ sách. Số câu **trùng** = 0.

---

## 3. Bốn sửa — mã, lý do, số đo

### 3.1 (a) `ingest_ihr_book.py` — bước tiến số câu theo CẶP

```python
def page_seq_step(n_verses: int) -> int:        # MỚI
    return 2 * math.ceil(max(int(n_verses), 0) / 2)
...
seq += page_seq_step(len(verses))               # CŨ: seq += len(verses)
```

`verse_pairs_for_page` gán cột *k* ↔ `(first_seq + 2k, +1)` nên nó **giả định `first_seq` LẺ**. Cộng
số câu THÔ làm một trang có số câu lẻ đảo parity cho **mọi** trang sau. Nay `first_seq` luôn lẻ
(selftest kiểm trên toàn bộ 104/161 trang thật của 2 sách). Trang có số câu lẻ được **ghi cờ**
`verses_odd=<n>` trong `page_gate` (cờ, KHÔNG chặn).

Đo: A11 **61 → 0** và **1.078 → 0**; số câu trùng **1 → 0** ở cả hai bộ; tập trang ghi ra **không
đổi** (99 và 161 trang, 5 và 1 trang bị loại, y như trước). Selftest `ingest_ihr` 64 → **81** phép
(+17, trong đó 9 phép riêng cho `page_seq_step`/parity/không-trùng-số-câu).

### 3.2 (b) Cờ `qn_count_unfixed` + cổng (e) của `mechanism_gates`

* `book_layout.qn_count_unfixed_columns()` (MỚI): lithograph có `tier_rule` → cột nào `len(qn) ≠ 14`;
  prose → cột nào `dp_ratio < PROSE_DP_RATIO_MIN = 0,75` (đọc `transcriptions/<page>.json`);
  **STT → LUÔN rỗng** (`tier_rule` None và không phải prose).
* `align_production.align_page` gắn cờ 0/1 vào từng ô và vào `col_states` (PASS 1b dùng lại).
* `build_dataset` thêm cột `qn_count_unfixed` vào `labels.csv` **chỉ khi có ô mang khoá** →
  STT giữ đúng 42 cột như cũ (byte-identical).
* `mechanism_gates` cổng **(e)**, ưu tiên `(c) crop_bad > (d) cross_similar > (e) qn_count_unfixed >
  (b) bridge/tone > (a) box`. Chế độ: `books[].qn_count_gate: review | text_only | off`, mặc định
  **`review` cho lithograph**, **`off` cho prose/STT**; `--qn-count-gate` ghi đè.

**Vì sao REVIEW chứ không phải GOLD_text_only** (đề bài yêu cầu "chọn theo số đo"): nhóm này sai ở
chính **nhãn văn bản**, không chỉ ở hộp — trên nhãn người, ô GOLD trong cột `n_qn ≠ 14` chỉ đúng
**52,2 %** (n 209) và **49,0 %** (n 51) so với 98,4 % / 98,7 % phần còn lại. `GOLD_text_only` **vẫn
giao nộp nhãn**, tức vẫn giao ~50 % nhãn sai; REVIEW thì không giao gì.

| sách | cờ = 1 (dòng) | GOLD trúng thô | **hạ xuống REVIEW** | chế độ |
|---|---|---|---|---|
| LucVanTien1883 | 364 (2,51 %) | 267 | **264** | review |
| KimVanKieu1884 | 350 (1,54 %) | 300 | **296** | review |
| LucVanTien1916 | 265 (1,93 %) | 226 | **212** | review |
| TruyenKieu1872 | 77 (0,34 %) | 57 | **54** | review |
| Chrestomathie1872 | 2.777 (34,6 %) | 2.010 | **0** | **off** — chỉ ghi cờ |

Chênh "GOLD trúng thô" ↔ "hạ": phần đã bị cổng mạnh hơn ((c) ảnh hỏng / (d) bất đồng dị bản) lấy trước.

**Chrestomathie để `off` có chủ ý.** Cờ prose theo `dp_ratio < 0,75` trúng **2.777/8.016 ô (34,6 %)**
mà sách này **không có bất kỳ chuẩn độc lập nào** để chứng minh nhóm đó xấu hơn phần còn lại — hạ
2.010 ô GOLD theo một giả định chưa đo là **không** chấp nhận được. Cờ vẫn được ghi ra `labels.csv`
nên bật lên chỉ cần `qn_count_gate: review` trong config, không phải dựng lại.

### 3.3 (c) `lithograph_gate` lấy LUẬT 6/8 làm kỳ vọng

```python
want = rule_n if rule_n else (expected_counts or {}).get(lid, lay.qn_per_column)
```

Cổng cũ so số âm của cột với `num_syllables` ghi trong `transcriptions/<page>.json` — mà số ấy **do
chính tesseract đếm**, nên cột 13/15 âm "khớp với chính mình" và đi lọt. Nay `tier_rule_for(lay)`
trả `(6, 8)` thì kỳ vọng = 14 tuyệt đối, và `bad_syl_cols` ghi thêm `"src": "tier_rule"`.

| `layout_gate.page_ok` | TRƯỚC | SAU |
|---|---|---|
| LucVanTien1883 | 103/105 | **77/105** |
| KimVanKieu1884 | 162/163 | **137/163** |
| LucVanTien1916 | 97/99 | **81/99** |
| TruyenKieu1872 | 159/161 | **154/161** |
| Chrestomathie1872 (prose, không có luật) | 65/65 | 65/65 — **không đổi** |

`page_ok` của sách thạch bản **chỉ đi vào `summary.json["layout_gate"]`** (thống kê); `build_dataset`
không dùng nó để loại ô nào (đã grep: 4 chỗ dùng, cả 4 là đếm/ghi báo cáo). Vì vậy số ô, tier và ảnh
export **không đổi một dòng nào** vì sửa này — nó thuần tuý biến "cột đếm hỏng" thành sự kiện
**nhìn thấy được ở mức trang**, và là cổng đôi cho cờ (b).

### 3.4 (d) `pitch_target_count` ưu tiên luật 6/8 thay `n_qn`

```python
if n_rule and n_qn != n_rule and n_ocr == n_rule:
    return n_rule, "pitch_rule"            # MỚI — chỉ layout=lithograph
```

Luật lục bát là **bất biến của bản in**, `n_qn` là số tesseract đếm; và 100 % cột `n_qn ≠ 14` đều
được kim đọc **đúng 14 chữ**. Nhánh `pitch_ocr` cũ chỉ bật khi `n_det == n_ocr` nên không cứu được
chúng. Điều kiện `n_ocr == n_rule` giữ cho sửa này **thận trọng**: kim cũng không đọc đủ 14 thì giữ
nguyên hành vi cũ. Bộ giải mã nhận `tier_n = (6, 8)` ở nhánh này (trước là `None`).

| `count_source` | TRƯỚC | SAU |
|---|---|---|
| LucVanTien1883 | pitch 14.155 · pitch_ocr 319 | pitch 14.035 · **pitch_rule 364** · pitch_ocr 75 |
| KimVanKieu1884 | pitch 22.494 · pitch_ocr 256 | pitch 22.361 · **pitch_rule 350** · pitch_ocr 39 |
| LucVanTien1916 | pitch 13.666 · pitch_ocr 95 | pitch 13.508 · **pitch_rule 237** · pitch_ocr 15 |
| TruyenKieu1872 | pitch 22.360 · pitch_ocr 139 | pitch 22.307 · **pitch_rule 67** · pitch_ocr 125 |
| Chrestomathie1872 (prose) | pitch 6.907 · pitch_ocr 1.109 | **y hệt** |

**Trung thực về (d):** mọi phép đo hiện có đều **không phân biệt được** nó tốt hay xấu.
`box_ref_eval` lấy N từ `transcriptions` nên không chạm nhánh này (24/24 bất biến PASS, ok50 98,1 %
LVT / 97,3 % KVK — không đổi); `crop_quality_flag` toàn sách LVT1883 đổi **`ok` 10.156 → 10.152,
`bleed` 2.219 → 2.223** (±4 ô, mức nhiễu); trôi không đổi. Giữ lại vì (i) nó làm số hộp bằng số chữ
**thật** trên giấy thay vì bằng số âm tesseract đếm nhầm, (ii) nó tạo giá trị `count_source` riêng
(`pitch_rule`) khiến 364/350/237/67 cột ấy **đếm được** ở hạ nguồn. Không có số nào xấu đi ⇒ không
hoàn nguyên; nếu muốn tắt: truyền `n_rule=None` (một dòng ở `_pair_new_state`).

---

## 4. `crop_source: original` — 5 sách, bộ giao nộp đã dựng lại

Cơ chế đã làm xong ở vòng 7 (`docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md` §2); vòng 8 **bật
cho 5 sách và dựng lại thật**. Sáu config khai `crop_source: original`
(LVT1883, KVK1884, KVK1884_b1, Chrestomathie1872, LVT1916, TK1872); `config/pipeline.yaml` (STT) **không đụng**.

`scripts/measure/crop_source_samples.py` trên chính bản dựng vừa chạy (6 ảnh, **0 FAIL**):

| ô mẫu | kích thước | tương quan mực | nền p90 | số mức xám |
|---|---|---|---|---|
| `lucvantien1883_page_0001_c08_097` | 158×169 | 0,999 | 255 → **128** | 57 → 83 |
| `lucvantien1883_page_0057_c06_083` | 157×123 | 0,999 | 255 → **128** | 57 → 84 |
| `kimvankieu1884_page_0001_c01_000` | 101×90 | 0,969 | 255 → **133** | 74 → 161 |
| `kimvankieu1884_page_0083_c07_092` | 91×107 | 0,974 | 255 → **133** | 73 → 165 |
| `chrestomathie1872_page_0001_c01_001` | 94×107 | 0,999 | 255 → **131** | 129 → 157 |
| `chrestomathie1872_page_0036_c05_089` | 96×110 | 0,998 | 255 → **132** | 129 → 160 |

Kiểm tệp thật: ảnh trong `dataset/LucVanTien1883/gold/` **trùng md5** với crop trong
`prepared/LucVanTien1883/dataset_out/gold/` (bản gốc) và **khác** bản trong `crops_bin/` (bản nhị
phân); `crops_bin/` **không** có trong `dataset/<Book>/` (đã liệt kê thư mục). `TruyenKieu1872`
(`contrast: none`, ảnh gốc vốn đã xám) **không đổi gì** — bật khoá cho nó là vô hại, ghi ở đây để
không ai tưởng có cải thiện.

**Bản cũ (crop nền trắng bệt) giữ nguyên ở `dataset/<Book>_v7/`** cho cả 5 sách — đối chiếu được,
chưa xoá.

---

## 5. Hồi quy STT + selftest

| Phép | Kết quả |
|---|---|
| STT 3 trang (`stt2/0024`, `stt4/0050`, `stt11/0100`) | `labels.csv` md5 **`59e436d7641fa849bb6759868ac29259`**, 556 dòng, **42 cột** (không có cột `qn_count_unfixed`); `diff -rq` **0 tệp khác** |
| `git status -- dataset_out prepared/SachThanhTruyen*` | **trống** |
| `book_layout_selftest` | **157 / 0** (+24: `[11] lithograph_gate tier_rule + qn_count_unfixed`) |
| `phase1_engine_selftest` | **288 / 0** (+18: `test_pitch_rule_and_qn_flag`) |
| `ingest_ihr_selftest` | **81 / 81** (+17: `page_seq_step`, parity, không trùng số câu) |
| `mechanism_gates_selftest` | **113 / 0** (+19: `[6] (e) qn_count_unfixed`) |
| `ingest_lithograph` / `ingest_prose` / `pitch_decode` / `verses_ref_fix` | 82/82 · 33 · 22/0 · 19/19 |
| `align_audit --selftest` / `ihr_endtoend_eval --selftest` | OK · 24/24 |
| `code_facts.py --check` | **18/18** |
| `box_ref_eval --book all --pages 27` | **24/24 bất biến PASS** (ok50 pitch-honest 98,1 % LVT · 97,3 % KVK) |
| `ihr_endtoend_eval --book all` | **14/14 bất biến PASS** cả hai bộ |

Ba chỗ selftest CŨ phải sửa vì hành vi đổi **có chủ ý** (ghi rõ để không ai tưởng là nới lỏng):

1. `book_layout_selftest`: "JSON ghi `num_syllables=13` khớp .txt → `page_ok` True" → nay
   **False** khi sách có luật 6/8; thêm phép đối chứng "lithograph `qn_syllables_per_column = 13`
   (không có luật) → vẫn PASS như cũ".
2. `ingest_ihr_selftest`: "`first_seq` tăng đều theo `len(verses)`" → "tiến theo **CẶP** câu".
3. Không phép nào bị xoá.

---

## 6. PTCL: bốn tệp vẫn gỡ, **tệp Kiều 1871 GIỮ LẠI**

Vòng 7 gỡ cả thư mục `data/TruyenKieuPhongTinhCoLuc/` khỏi phạm vi. Nhưng
`thamchieu_kieu_1871_LieuVanDuong_phienam.json` chỉ **nằm nhờ** trong thư mục đó: nội dung là bản
phiên âm **Truyện Kiều 1871 (Liễu Văn Đường)**, tức **tham chiếu dị bản của KimVanKieu1884**, không
phải dữ liệu của bản chép tay R.987. Người dùng đã khôi phục tệp; vòng 8 bật lại ba chỗ:

| Nơi | Vòng 7 | Vòng 8 |
|---|---|---|
| `config/pipeline_KimVanKieu1884_b1.yaml` khối `run.verses_ref_fix` | comment | **bỏ `#`** → B1' chạy lại (`ref_name nf1871`, `fuzzy_min 0,9`) |
| `scripts/measure/auto_precision.py` `CROSS_BOOKS["KimVanKieu1884"].refs` | chỉ `Kieu1872_DMT` | **`Kieu1871_LVD` LÊN TRƯỚC** (ref chính r0) + `Kieu1872_DMT` |
| `scripts/measure/qn_engine_compare.py` `REFS` | không có KVK | **có lại** (vẫn lọc theo tệp tồn tại) |

Bằng chứng khôi phục đúng: số trôi KVK trùng khít mốc đã công bố — **n 31.866 · đúng 80,58 % ·
lệch vị trí 16 · 1/4.792 câu trôi** (với 1 tham chiếu chỉ ra n 16.134 và lệch 13, không so được với
tài liệu cũ). "Nền dị bản 1871↔1872" vì thế tính được trở lại.

**Bốn tệp còn lại của PTCL vẫn ở trạng thái xoá trong worktree** (` D`) — KHÔNG `git rm`, KHÔNG khôi
phục. `measure.py` / `verses_ref_fix.py` / `run_pipeline.sh` giữ nguyên nhánh "thiếu tệp → cảnh báo,
bỏ bước" của vòng 7.

Đã sửa hộp cảnh báo trong: `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` §10,
`docs/CHAY_3_BO_CON_LAI_2026-09-23.md` (đầu tệp), `docs/CHAY_KVK1884_B1_2026-09-22.md` §lệnh B1',
`docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md` §4.

---

## 7. Việc KHÔNG làm ở vòng này và lý do

| Đề xuất RA_SOAT | Vì sao chưa làm |
|---|---|
| **#3** nới `repair_tier_syllables` (cột 13 âm → chèn `khongdoc` ở khe mực rộng nhất; cột 15 âm → gộp cặp token) | Đây là sửa **adapter**, mỗi lần đổi là ingest lại + build lại 4 sách. Trần lợi ích = đúng lượng ô mà cổng (e) vừa hạ (364+350+265+77 = 1.056 ô). Phải đo lại trên nhãn người mới biết cứu được bao nhiêu — cần một vòng riêng. |
| **#5** Chresto: pitch làm neo mềm cho DP mức truyện + hạ cột `dp_ratio < 0,75` | Cờ đã sẵn sàng (`qn_count_unfixed`, 2.777 ô) nhưng **chưa bật** vì sách này không có chuẩn độc lập nào; xem §3.2. Phần "neo mềm" là đổi thuật toán ghép của `ingest_prose_book`, không phải bật cờ. |
| detector v2 (`detector_ckpt` cho 5 sách) | **Chưa có checkpoint** — `docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md` §1.2. Không đổi ở vòng này. |

## 8. Giới hạn của chính các số trên

1. **Chỉ 2/8 bộ có nhãn người.** Hai con số precision đáng tin (97,96 % và 98,64 %) đến từ LVT1916 /
   TK1872. Suy cho LVT1883 / KVK1884 là **ngoại suy** (cùng adapter, cùng config, cùng cổng).
2. **Proxy dị bản mù với chính nhóm cột được sửa** (§1) — 79,8 %/87,8 % không đổi **không** bác bỏ gì.
3. **n nhỏ ở lõi lập luận**: tỉ lệ đúng 52,2 % / 49,0 % của nhóm cột `n_qn ≠ 14` dựa trên n = 209 và
   51 (Wilson 95 % ≈ 45–59 % và 35–63 %): **dấu** chắc chắn, **độ lớn** thì không.
4. **Chresto và STT không có chuẩn ngoài nào**; A1–A8 PASS ở đó chỉ chứng minh *tự nhất quán*.
5. **Trôi vẫn còn nguyên** (§1): vòng này không sửa phép ghép, chỉ sửa việc giao nộp. Muốn giảm trôi
   thật phải làm đề xuất #3.
6. **Chỉ số crop chỉ đo trên 6 ô mẫu** (`crop_source_samples`); bất biến "cùng vùng mực" là IoU mặt
   nạ, không phải kiểm thị giác toàn bộ 60.469 ảnh đã export.
