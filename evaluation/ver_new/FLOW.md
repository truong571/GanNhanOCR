# ver_new — Pipeline gán nhãn Nôm↔Quốc-Ngữ (bản cải tiến) — Tài liệu thực hiện

Bộ code dựng **gold dataset**: mỗi ảnh crop chữ Nôm gắn nhãn `(chữ SinoNom, âm tiết
Quốc-Ngữ)`, bằng cách đối chiếu 2 OCR trên bản song ngữ (trang Nôm ‖ trang Quốc-Ngữ),
dựa bất biến **1 chữ Nôm = 1 âm tiết Quốc-Ngữ**.

Tài liệu này ghi **toàn bộ những gì đã làm**: (A) thay căn chỉnh, (B) sửa 3 lỗi cắt
crop, (C) port fix vào production, (D) phát hiện S3/DINOv2 hỏng.

> **Lệnh chạy 1 phát ra dataset:**
> ```bash
> cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
> .venv/bin/python evaluation/ver_new/build_dataset.py
> ```

---

## 0. Các file

| File | Vai trò |
|---|---|
| `anchor_align.py` | **Lõi căn chỉnh**: `realign_column` — banded anchored DP (thay ghép-index). |
| `bbox_fix.py` | **Sửa toạ độ crop**: `frame_offset`+`correct_columns` (dịch về full-page) và `tighten_box` (siết ink). |
| `align_production.py` | Đường production: phát hiện cột (`parse_v5`+`detect_nom_columns_v3`) + áp `bbox_fix` + banded-DP + **re-segment cột**. |
| `consensus.py` | `decide_label` — đồng thuận S1/S2/S3 → tầng GOLD/SILVER/REVIEW. |
| `visual_signal.py` | **S3** (DINOv2+FD) — *đang TẮT*, xem mục D. |
| `build_dataset.py` | ⭐ Build 1 lệnh: align → đồng thuận → **cắt crop + labels.csv** (4 tầng, 19 cột, split chống rò). |
| `to_standard.py` | Xuất 3 **chuẩn quốc tế**: `metadata.csv` (HF imagefolder) · `datapackage.json` (Frictionless) · `croissant.json` (MLCommons). |
| `sample_pdf.py` | Xuất PDF 100 mẫu (crop ↔ glyph FD) để soi tay. |
| `run_full_eval.py` · `run_eval.py` | A/B production OLD vs NEW · testbed nhanh. |
| `REPORT_dinov2_unsuitable.md` + `figures/` · `TRAIN_nom_embedding.md` · `dinov2_proof.py` · `make_figs.py` | Báo cáo DINOv2 + hình; hướng dẫn train model thay thế (Kaggle P100). |

> **Đã merge vào pipeline chính (2026-06):** `run_pipeline.sh` Step 2 nay gọi
> `build_dataset.py` + `to_standard.py` (thay step2-align-index / step3-DINOv2 /
> step4-export cũ). **Đã xoá code chết:** `core/align/parser_v3.py, parser_v4.py,
> probe.py, nom_column_cluster.py`, `core/text/alignment.py`, nhánh `--legacy`
> hỏng trong `step2_align.py`, và các log rác 0-byte. `run_full.py` /
> `export_dataset_v4.py` đã rút gọn còn hàm sống (`nom_cols_hybrid`,
> `resegment_col`). Đã verify: step1-4 + ver_new import OK.

---

## A. Căn chỉnh: banded anchored DP (thay ghép-index)

**Lỗi cũ** (`pipeline/step2_align.py`): ghép Nôm↔QN **theo vị trí (index)** rồi ép
bằng số. Lệch 1 ký tự → **dịch cả đuôi cột → sai nhãn hàng loạt**. Đo: cột lệch số
chỉ dict-confirm 28–29% (vs 57–61% cột khớp).

**Cách mới** — `anchor_align.realign_column`: quy hoạch động ghép chuỗi
`ocr_char[]` ↔ `syllable[]`, chi phí khớp do từ điển điều khiển:

| Trường hợp | Cost |
|---|---|
| `ocr_char ∈ qn_to_nom[syl]` (dict-confirm) | **0.0** |
| qua từ điển tương tự | 0.3 |
| `syl` có trong từ điển, char không khớp | 1.0 |
| `syl` ngoài từ điển | 0.9 |
| del (bỏ ký tự Nôm) / ins (bỏ âm tiết) | 0.7 |

**Dải biên** `|i−j| ≤ |m−n|+2`: chặn dịch-cả-đuôi và chặn xé vụn. **Chi phí mềm**
(không −∞): một neo sai không kéo lệch cả đoạn. Lệch 1 ký tự ⇒ **một gap cục bộ**,
phần còn lại vẫn đúng register.

**Kết quả A/B (445 trang, `run_full_eval.py`):** dict-confirm OLD 48,9% → NEW
**63,5%**; cột lệch số **14,2% → 60,7% (+46pp)**; đối đầu từng cột NEW thắng 1.405 /
hoà 2.597 / **thua 0**.

---

## B. Cắt crop: sửa 3 lỗi (đây là phần làm dataset "dùng được")

Sau khi căn chỉnh đúng nhãn, **crop ảnh vẫn sai** do 3 vấn đề. Sửa lần lượt:

### B1. Bug toạ độ frame-crop (lỗi nặng nhất, có sẵn trong production)
- **Triệu chứng:** ~30% crop **trắng** (cột 9: 98%, cột 8: 64%).
- **Nguyên nhân:** SinoNom OCR chạy trên ảnh **đã crop khung** (`framed=True`), bbox ở
  toạ độ ảnh-crop; nhưng cắt crop trên **ảnh full** không cộng offset → lưới cột lệch
  trái ~252px (~1,7 cột). Đo: 99% trang offset >70px. **Production cũng dính** (crop
  gốc `detected/crops/` cũng trắng).
- **Sửa** (`bbox_fix.frame_offset`): gốc khung `(max(0,fx0−pad), max(0,fy0−pad))` qua
  `detect_frame_hybrid`+`frame_pad`; cộng vào mọi bbox → full-page. **Pure translation,
  chính xác.** Gọi trong `align_production._detect` (1 điểm).

### B2. Siết bbox bằng binary projection
- **Triệu chứng:** crop dính nét cột (ruling line), lề thừa.
- **Sửa** (`bbox_fix.tighten_box`): bỏ hàng/cột rìa toàn-ink (nét cột) + crop sát ink
  bbox bằng projection. Áp trong `build_dataset.save_crop` (mặc định bật, `--no-tighten`
  để tắt).

### B3. Re-segment cột (hết crop dính 2 chữ)
- **Triệu chứng:** một số crop chứa 2 chữ kề (bbox-y của OCR lỏng).
- **Sửa** (`align_production._reseg_column`): bỏ chiều cao bbox lỏng của OCR; dùng
  **y-center (đáng tin) + ranh giới = trung điểm giữa 2 chữ kề** (margin 10% để không
  cắt cụt chữ cao). Mỗi crop không thể chứa tâm chữ hàng xóm ⇒ **không dính 2 chữ**.
  *(Đã thử valley-segmentation nhưng mis-pack khi cửa sổ cột dính ink hàng xóm → chuyển
  sang trung-điểm, bền hơn.)*

### Tiến triển chất lượng crop (50.502 GOLD)
| Giai đoạn | Crop trắng | Mực trung vị | Dùng được (≥10%) |
|---|---|---|---|
| Bug gốc | 29,7% | 9,8% | 53,6% |
| + B1 sửa offset | 0,1% | 14,9% | 93% |
| + B2 siết bbox | 0,02% | 18,9% | 99% |
| **+ B3 re-segment** | **0,02%** | **18,4%** | **99%, không dính chữ** |

---

## C. Gán nhãn đồng thuận & 4 tầng (`consensus.decide_label` + `build_dataset.py`)

3 tín hiệu/crop: **S1** chữ OCR, **S2** ứng viên từ điển `qn_to_nom[syl]`, **S3** khớp
thị giác (đang tắt — mục D). `label_level` tách giám sát **char** vs **syllable**.

| Tầng | `label_level` | Điều kiện | Nhãn (`label`) |
|---|---|---|---|
| **GOLD** | char | (a) `ocr_char ∈ R` → trực tiếp (kể cả cột diverged, char tự xác nhận) [#2]; hoặc (b) bridge tương tự duy nhất ∈ R khi anchored | `ocr_char` / **chữ-bridge** (đã sửa: KHÔNG còn ghi chữ OCR đọc nhầm) [#1] |
| **SILVER** | char | S1∩S2 lệch, **S3 (embedder Nôm đã train) phá thế**: vision chọn 1 âm Nôm hợp lệ của âm tiết (cos≥τ, margin≥δ) → sửa OCR đọc nhầm | char S3 chọn (+ cột `s3_cosine`) |
| **SYLLABLE** | syllable | char chưa chắc nhưng **âm tiết đúng & nhất quán xuyên-trang** (≥3 trang, purity≥0.6, ≥5 lần) — mượn-nghĩa dict không chứa được [#6] | `''` (chỉ giữ `syllable`) |
| **REVIEW** | — | còn lại (diverged-gapped / đơn lẻ / chất lượng kém). Giữ bbox trong manifest | — |

**Quan trọng:** không trộn char-level với syllable-level. Classifier ký tự train
`label_level=char`; reader Nôm→QN train cả char + syllable. Bất biến 1-chữ-1-âm giữ
tầng SYLLABLE hợp lệ.

---

## D. S3 — đã THAY DINOv2 bằng embedder Nôm tự train → SILVER đang BẬT

DINOv2 zero-shot **không phân biệt được chữ Nôm** (đã loại). Bằng chứng (3 thí
nghiệm, `REPORT_dinov2_unsuitable.md` + `figures/`):

| Test | Cùng chữ | Khác chữ | |
|---|---|---|---|
| T1 glyph FD sạch | 0,95 | 0,90 | dải trùng |
| T2 crop thật | 0,889 | 0,877 | ≈ bằng nhau |
| T3 retrieval top-1 | — | — | **0,0%** (chance 0,2%) |

**Giải pháp đã làm:** train embedder Nôm riêng (ResNet-18 + ArcFace,
`nom_classifier/`, Kaggle P100) → `nom-embed/best.pt`. Nghiệm thu trên test split:
**T2 separation +0,29 · T3 retrieval 76,5%** (DINOv2: +0,01 · 0%). `visual_signal.py`
nay dùng `NomEncoder` (`nom_classifier/infer.py`) thay `DINOv2Ranker`.

⇒ **SILVER BẬT**: chạy `build_dataset.py --use-s3` (đã set mặc định trong
`run_pipeline.sh`). Thiếu checkpoint → tự **degrade** (SILVER bỏ qua, GOLD/SYLLABLE
vẫn chạy). Ngưỡng `τ=0.62, δ=0.06` trong `consensus.py` (hiệu chuẩn theo thang
embedder mới: cùng-chữ ~0,80 / khác ~0,50).

---

## E. Port vào production (pipeline gốc cũng hết bug)

- `core/ocr/ocr_api.py`: thêm `_frame_offset`/`_translate_columns`/`load_columns_fullpage`.
  OCR mới ghi cache **full-page** (cờ `coords_space:"fullpage"`); cache cũ **tự dịch khi
  đọc** (không cần re-OCR).
- `pipeline/step2_align.py`: đọc cache qua `load_columns_fullpage`.
- `align_production` chốt cờ `coords_space` để **không dịch 2 lần**.
- ⇒ Chỉ cần **chạy lại step2** là production hết bug trên dữ liệu cũ.

---

## F. Chạy end-to-end & kết quả

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
# 1) Build dataset (GOLD, ~5 phút):
.venv/bin/python evaluation/ver_new/build_dataset.py
#    tuỳ chọn: --no-tighten | --pad 0.12 | --limit N | --out <dir> | --use-s3 (chưa nên)
# 2) Xuất 100 mẫu ra PDF để soi tay:
.venv/bin/python evaluation/ver_new/sample_pdf.py
# 3) (đánh giá) A/B alignment & báo cáo DINOv2:
.venv/bin/python evaluation/ver_new/run_full_eval.py
.venv/bin/python evaluation/ver_new/dinov2_proof.py && .venv/bin/python evaluation/ver_new/make_figs.py
```

**Output** `dataset_out/`: `gold/ · syllable/ ( · review/ với --crop-review)` PNG +
`labels.csv` (19 cột) + `summary.json`. Cột `labels.csv`:
`image, book, page, column, ocr_char, syllable, label, unicode, label_level, tier, rule,
ink_pct, crop_w, crop_h, image_md5, seg_flag, split, split_group, bbox`.

**Audit cuối (445 trang):** GOLD **51.195** (sửa **3.856** nhãn similar sai + cứu ~2.011
diverged) · SYLLABLE **7.681** (phục hồi mượn-nghĩa) · REVIEW 23.392 → **usable 58.876**
(char 51.195 + syllable 7.681) · blank 0,02% · 1.591 lớp chữ · split train/val/test
~82/9/9 **rò rỉ nhóm = 0** (atomic theo book/page/column, singleton→train).

---

## G. Giới hạn & việc tiếp

- **SILVER tắt** đến khi có model embedding Nôm (`TRAIN_nom_embedding.md`).
- **Per-char segmentation** đã tốt hơn nhiều nhờ B3 nhưng cột Nôm-OCR **đếm sai** (chữ
  dính/thiếu) thì re-segment không cứu được — đó là cột "diverged", đã đẩy REVIEW.
- Ngưỡng `τ/δ` trong `consensus.py` là placeholder — hiệu chuẩn khi bật lại SILVER.
- Để production dùng: chạy lại `step2_align.py` cho 3 sách (cache tự migrate).
