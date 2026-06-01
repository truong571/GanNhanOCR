# ver_new — Căn chỉnh Nôm↔Quốc-Ngữ bằng Banded Anchored DP + Consensus 3 tín hiệu

Bộ code đề xuất để **thay cách ghép nhãn hiện tại** và đo **kết quả đạt được**
trên dữ liệu thật (445 trang, 3 sách), **không cần model/GPU** (phần thị giác S3
để optional).

---

## 1. Vấn đề đang giải

Mục tiêu pipeline: dựng **gold dataset** — mỗi ảnh crop chữ Nôm gắn nhãn
`(âm tiết Quốc-Ngữ, chữ SinoNom)`. Nhãn được suy ra bằng cách đối chiếu **2 OCR**
trên bản song ngữ: trang Nôm (SinoNom OCR → ký tự + bbox) ‖ trang Quốc-Ngữ
(VietOCR → âm tiết). Bất biến: **1 chữ Nôm = 1 âm tiết Quốc-Ngữ**.

**Lỗi gốc của cách hiện tại** (`pipeline/step2_align.py`): ghép **theo vị trí
(index)** trong cột rồi **ép bằng số** (dư thì cắt bớt ký tự đầu, thiếu thì ghép
tiền tố). Hệ quả: chỉ cần lệch **1** ký tự, **toàn bộ đuôi cột bị dịch một bậc →
mọi crop sau đó gắn sai âm tiết**. Đo được: trên cột bị lệch số, tỉ lệ
dict-confirm chỉ **28–29%** (so với ~57–61% ở cột khớp số).

Ví dụ thật (`page_0014 col0`): index-pairing chỉ ghép đúng **5/22**; banded-DP
ghép đúng **17/22** — cùng dữ liệu, chỉ khác thuật toán.

---

## 2. Ý tưởng

1. **Từ điển QN↔Nôm là mỏ neo mạnh**: cặp `(ocr_char_i, syllable_j)` được
   *dict-confirm* nếu `ocr_char_i` là một âm Nôm hợp lệ của `syllable_j`
   (`ocr_char ∈ qn_to_nom[syllable]`). Độ phủ âm tiết 99%.
2. **Căn chỉnh chuỗi (Needleman–Wunsch) thay vì ghép vị trí**: chi phí khớp do
   từ điển điều khiển (confirm = 0.0, rẻ nhất) → mọi ký tự dict-confirm được
   **ghim đúng âm tiết**, chỉ đoạn quanh chỗ thật-sự thừa/thiếu mới "trôi".
3. **Dải biên (band)** `|i−j| ≤ |m−n| + 2`: chặn dịch-cả-đuôi **và** chặn aligner
   xé vụn khi thiếu neo (đây là thứ làm aligner toàn cục cũ sụp còn 4.133 cặp).
4. **Chi phí mềm (không −∞)**: một neo sai không kéo lệch cả đoạn.
5. **Nhãn = đồng thuận nhiều tín hiệu**, không bao giờ từ 1 tín hiệu đơn lẻ.

---

## 3. Kiến trúc & luồng dữ liệu

```
prepared/<book>/                         dict/
 ├─ detected/page_*_ocr_cache.json       ├─ QuocNgu_SinoNom_TongHop3.csv  (qn→[nom])
 │     S1: [{char, bbox}] mỗi cột        └─ SinoNom_Similar_Dic_v2.csv    (nom→[similar])
 └─ transcriptions/page_*.json
       S2: [syllables] mỗi cột
                 │
                 ▼
        ┌───────────────────────┐
        │  run_eval.py          │  nạp dữ liệu, ghép cột theo index
        │  (testbed A/B)        │
        └───────────┬───────────┘
            ┌────────┴─────────┐
            ▼                  ▼
   OLD = old_pairing     NEW = anchor_align.realign_column
   (index + cắt/ghép)    (banded anchored DP)        ← lõi đề xuất
            │                  │
            ▼                  ▼
        đếm dict-confirm   ops: match / del(thừa Nôm) / ins(thiếu Nôm)
                               │
                               ▼
                      consensus.decide_label   (S1 ⊕ S2 ⊕ S3?)
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
               GOLD         SILVER        REVIEW
          (dict-confirm   (Ext-B variant  (cột lệch / null /
           + anchored)     cần S3)         1 tín hiệu)
                               │
                               ▼
              results/  summary.json · pairs_new.csv · fixed_examples.txt
```

### 5 file mã

| File | Vai trò |
|---|---|
| **`anchor_align.py`** | Lõi: `realign_column()` — banded anchored DP cho **một cột**. `substitution_cost`, `is_confirmed`, `matched_pairs`. Phiên bản production đặt vào `core/align/anchor_align.py`, gọi từ `process_page_structural`. |
| **`consensus.py`** | `decide_label()` — gộp S1/S2/(S3) thành nhãn + tầng GOLD/SILVER/REVIEW + `rule_id` (truy vết). S3 optional. |
| **`visual_signal.py`** | **S3** — tín hiệu thị giác: bọc `DINOv2Ranker`, cắt crop Nôm theo bbox, xếp hạng ứng viên `{ocr_char} ∪ readings(syllable)` bằng cosine DINOv2 trên glyph **FontDiffusion** (89.898 ảnh) — fallback font. Có chốt loại crop trắng/đặc + ứng viên rác. |
| **`align_production.py`** | **Đường production**: sao chép Y HỆT phần phát hiện cột (`detect_nom_columns_v3`) + parse QN (`parse_v5` qua `_get_qn_lines`) của `step2_align.py`, chỉ **thay khối ghép-cặp**. `mode="old"` = logic hiện tại (ép-bằng-số + index); `mode="new"` = banded-DP. |
| **`bbox_fix.py`** | 🔧 **Sửa bug toạ độ crop** (chặn nhất): `frame_offset`+`correct_columns` cộng gốc khung (`detect_frame_hybrid`+`frame_pad`) → full-page (blank 31%→**0%**). `tighten_box` siết crop về ink bằng binary projection (bỏ nét cột/lề). Gọi trong `align_production._detect` + `build_dataset.save_crop`. |
| **`build_dataset.py`** | ⭐ **Build dataset 1 lệnh từ đầu đến cuối**: align banded-DP + đồng thuận 3 tín hiệu (đã áp `bbox_fix`) → **cắt crop PNG + xuất `labels.csv`**. `--use-s3` thêm tầng SILVER. Ra `dataset_out/{gold,silver}/*.png` + manifest. |
| **`run_full_eval.py`** | **Chạy FULL như production** trên cả 445 trang, **A/B OLD vs NEW** (so sánh, không cắt crop), 3 tín hiệu (`--use-s3`), đánh giá. |
| **`run_eval.py`** | Testbed nhanh (ghép cột theo index từ cache thô, không qua `parse_v5`) — chạy trong vài giây, để soi nhanh tác động thuật toán. |

---

## 4. Thuật toán `realign_column` (chi tiết)

Vào: `nom_chars` (chuỗi ký tự SinoNom theo thứ tự đọc), `syllables` (chuỗi âm
tiết QN). DP `dp[i][j]` = chi phí tối thiểu căn chỉnh `nom[:i]` với `syl[:j]`.

**Chi phí khớp** (`substitution_cost`):

| Trường hợp | Cost | Ý nghĩa |
|---|---|---|
| `ocr_char ∈ qn_to_nom[syl]` | **0.0** | dict-confirm (S1∩S2) — neo |
| `similar(ocr_char) ∩ qn_to_nom[syl] ≠ ∅` | 0.3 | cầu nối nhầm-OCR qua từ điển tương tự |
| `syl` có trong từ điển nhưng char không khớp | 1.0 | nghi sai/lệch |
| `syl` không có trong từ điển | 0.9 | không phán được → vẫn cho khớp |
| **del** (bỏ 1 ký tự Nôm) | 0.7 | box thừa / OCR tách đôi glyph |
| **ins** (bỏ 1 âm tiết) | 0.7 | Nôm OCR sót glyph thật |

Tham số chọn sao cho **đường chéo 1-1 luôn thắng ở cột khớp & sạch**, nhưng một
glyph thiếu/thừa thì **rẻ hơn** khi hấp thụ thành **một gap cục bộ** so với việc
ghép-sai phần còn lại (del+ins = 1.4 > 1.0 = một khớp-nghi-sai).

**Dải biên**: chỉ tính ô `|i−j| ≤ |m−n| + 2`. Giữ căn chỉnh sát đường chéo → không
dịch-cả-đuôi, không xé vụn.

Ra: danh sách `op` `match / del / ins`. `matched_pairs()` lấy các `match` =
nhãn xuất ra. Mỗi match có cờ `confirmed`.

> **O(m·n)** mỗi cột nhưng có band ⇒ thực tế O(m·band); cột ~22 ký tự → tức thì.

---

## 5. Gán nhãn đồng thuận & 3 tầng (`consensus.decide_label`)

3 tín hiệu độc lập mỗi crop: **S1** = `ocr_char`; **S2** = `qn_to_nom[syllable]`;
**S3** = khớp thị giác DINOv2/FontDiffusion `(char, cosine, margin)` — **optional**.

| Tầng | Điều kiện | Phụ thuộc S3? |
|---|---|---|
| **GOLD** | (cột khớp số **hoặc** *anchored*) **và** S1∩S2 dict-confirm (`ocr_char` là một âm Nôm hợp lệ của âm tiết, trực tiếp hoặc qua từ điển tương tự). `anchored` = match nằm trong **chuỗi confirm liền kề** ⇒ register cục bộ chắc kể cả khi cột lệch. | **Không** — chạy ngay, sàn an toàn |
| **SILVER** | S1∩S2 **không** khớp, **S3 phá thế** (cosine ≥ τ **và** margin ≥ δ): **(a)** một âm Nôm `r ∈ readings(syllable)` thắng thị giác ⇒ **vision sửa OCR**, nhãn = `r` (rule `s2_inter_s3_corrected`); **(b)** âm tiết ngoài từ điển (biến thể Ext-B) và `ocr_char` thắng ⇒ nhãn = `ocr_char` (rule `s1_inter_s3_out_of_dict`). | **Có** → thiếu S3 thì xuống REVIEW |
| **REVIEW** | còn lại: cột lệch chưa neo, `ocr_char=null`, S3 dưới ngưỡng. Không xuất làm nhãn. | — |

Chốt an toàn đã cài trong `visual_signal.py`: loại crop **trắng/đặc** (ink < 3%
hoặc > 97% → tránh cosine≈1.0 giả ở rìa trang) và ứng viên **không-CJK/rỗng**;
SILVER bắt buộc margin (winner − á quân) để chống chọn bừa khi mọi ứng viên đều
giống nhau. GOLD chỉ dựa từ điển ⇒ **không** chịu rủi ro S3.

---

## 6. Chỉ số đánh giá

**Dùng tỉ lệ dict-confirm** thay cho "lệch số ký tự thô" (số thô phụ thuộc cách
đếm — tách gạch nối cộng +2.830 âm tiết — nên không đo được độ đúng căn chỉnh).

- Dict-confirm là **proxy chính xác cao**: một cặp confirm ≈ đúng (OCR char là âm
  hợp lệ của đúng âm tiết). *Vắng* confirm ≠ sai (có thể là chữ Ext-B ngoài từ
  điển) ⇒ đây là **cận dưới** độ đúng.
- **DELTA OLD→NEW trên cùng input là hợp lệ**: nhiều confirm hơn = căn chỉnh
  đúng vị trí hơn. DP **đơn điệu + banded** nên không thể "ăn gian" bằng đảo thứ
  tự — chỉ chèn gap; confirm tăng ⇒ register tốt thật.

Để **bảo vệ trước hội đồng**, bổ sung (FLOW §8): đo `label-error rate` theo tầng
trên **mẫu audit người tách riêng** + Wilson 95% CI; báo Cohen's κ giữa 2 người.

---

## 7. Cách chạy & kết quả đạt được

### 7a. FULL — đúng đường production (khuyến nghị dùng để đánh giá)

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
.venv/bin/python evaluation/ver_new/run_full_eval.py          # ~vài phút (đọc ảnh thật)
# .venv/bin/python evaluation/ver_new/run_full_eval.py --limit 10   # smoke test
```

Kết quả **445/445 trang, 0 lỗi** (cùng `parse_v5` + `detect_nom_columns_v3` như
production, chỉ khác khối ghép-cặp; **chỉ từ điển, chưa dùng S3**):

| | Cặp xuất | Dict-confirm (≈đúng) | Tỉ lệ |
|---|---|---|---|
| **OLD** (ép-bằng-số + index) | 84.104 | 41.133 | **48,9%** |
| **NEW** (banded anchored DP) | 81.692 | **51.881** | **63,5%** |
| **Δ** | −2.412 (ít rác hơn) | **+10.748 nhãn đúng** | **+14,6 pp** |

Theo loại cột:

| Loại cột | OLD confirm | NEW confirm | Cải thiện |
|---|---|---|---|
| Khớp số | 63,2% | 64,6% | +1,4 pp |
| **Lệch số** | **14,2%** | **60,7%** | **+46,4 pp** |

Đối đầu từng cột: **NEW tốt hơn 1.405 — hoà 2.597 — NEW kém hơn 0** → **không hồi quy ở bất kỳ cột nào**.

**Dataset 3 tầng (kết quả cuối, S3-independent):**

| Tầng | OLD | **NEW** | Ý nghĩa |
|---|---|---|---|
| **GOLD** | 37.637 | **49.548** | nhãn dict-confirm + anchored, dùng train ngay — **+11.911 (+31,6%)** |
| SILVER | 0 | 0 | chờ S3 (chưa cấp glyph) |
| REVIEW | 46.467 | 32.144 | hàng đợi người — **−14.323 rác đẩy khỏi vùng "dùng được"** |

(+2.412 âm tiết không có box Nôm → hàng đợi phục hồi REVIEW.)

**Đọc một câu:** banded-DP cho **+10.748 nhãn ĐÚNG hơn** và **+11.911 nhãn GOLD**
trong khi **xuất ít hơn 2.412 cặp** (ít rác hơn) và **không làm xấu cột nào** —
đúng mục tiêu "nhãn đúng hơn, không phải nhiều hơn". Lợi ích dồn vào **cột lệch
số** (14,2% → 60,7%, +46,4 pp) — đúng chỗ trước đây gán sai cả đuôi.

Artefact: `results/summary_full.json`, `results/dataset_{gold,silver,review}.csv`
(từng nhãn: book, page, column, ocr_char, syllable, label, confirmed, rule, bbox),
`results/fixed_examples_full.txt`, `results/full_run.log`.

### 7b. Testbed nhanh (vài giây, để soi thuật toán)

```bash
.venv/bin/python evaluation/ver_new/run_eval.py
```

Ghép cột theo index từ cache thô (không qua `parse_v5`) → 81.232 vị trí thô,
NEW nâng dict-confirm 51,0% → 59,5%, GOLD 45.349, 0 hồi quy. Dùng để kiểm áp lực
nhanh; **con số chính thức lấy ở 7a** (đường production).

---

## 8. Tích hợp vào pipeline thật (bước tiếp)

| Bước | Việc | File |
|---|---|---|
| P1 | Copy `realign_column` → `core/align/anchor_align.py`; thay khối ghép-index | `pipeline/step2_align.py:164-214` |
| P0 | Bỏ ép-bằng-số: `count_ok=(actual==expected)`, ghi `count_delta` | `pipeline/step2_align.py:172` |
| P3 | `assign_label_consensus()` bọc tier1/2/3, vá singleton + sanity | `core/ranking/ranker.py:297,432` |
| P4 | **Cấp FD glyph + build cache DINOv2 + ĐO false-accept@0.80** trước khi bật SILVER | `kaggle_diffusion/`, `evaluation/audit_consensus.py` |
| P5 | Điền cột `tier` (đã có sẵn trong FIELDNAMES) + xuất 3 subset | `pipeline/step4_export.py:111` |

---

## 9. Giới hạn & rủi ro

- **S3 ĐÃ được cấp glyph đầy đủ**: `gannhanocr-fd/` chứa **89.898 ảnh
  FontDiffusion** (`U+*.png`) sinh sẵn — `visual_signal.py` dùng trực tiếp, log
  thực tế `font refs 0` (100% tham chiếu từ FD, không cần render font). DINOv2
  `dinov2_vitb14_reg` chạy MPS ~76 ms/embed; cache embedding `.npz` build lần
  đầu rồi tái dùng. *(Ghi chú cũ "chỉ 132 glyph" là NHẦM — đã đính chính.)*
  ⇒ Việc còn lại với SILVER **chỉ là hiệu chuẩn ngưỡng** `τ/δ`, không phải thiếu
  hạ tầng. **GOLD chỉ dựa từ điển + anchored, độc lập S3** ⇒ luôn là sàn an toàn.
- **Dict-confirm là cận dưới**: ~43% cặp ở cột khớp là chữ Ext-B ngoài từ điển
  (cần S3 để lên SILVER), nên tỉ lệ tuyệt đối 59,5% **không** phải "độ chính xác".
- **Testbed dùng QN của `parse_numbered_lines`** (đường báo cáo); production
  `parse_v5` sẽ có ít cột lệch hơn ⇒ lợi ích của NEW dồn đúng nơi cần.
- Ngưỡng `TAU/DELTA` trong `consensus.py` là **placeholder** — hiệu chuẩn trên
  tập held-out trước khi tin SILVER.
