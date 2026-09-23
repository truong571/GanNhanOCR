# RÀ SOÁT TOÀN BỘ MÃ NGUỒN — 2026-09-13

**Phạm vi**: mọi mã trên đĩa (`core/`, `pipeline/`, `train_crop/`, `ArcFace/`, `NomNaOCR/`,
`kaggle_diffusion/`, `lab/`, `scripts/`, `re-dataset/check/`, cấu hình, tài liệu), đọc theo
chiều rộng; **đi sâu vào quy trình gán chữ Nôm ↔ Quốc ngữ** (`pipeline/align_engine/` + các bước
hậu xử lý `pipeline/remediation/`). Mọi con số trong tệp này tái sinh bằng
`.venv/bin/python scripts/_oneoff/ra_soat_2026-09-13_do_lai.py` trên commit `3e062b1fa3` + cây làm việc hiện
tại (chỉ hai tệp bẩn không liên quan: `docs/nghien_cuu_dang_do/README.md`,
`re-dataset/check/hannom_recheck.py`). Selftest: 105 · 146 · 56 · 171 · 62 · 81 · 44 — **665 passed,
0 failed** trước khi rà.

---

## 0. KẾT LUẬN NGẮN

Cảm giác "phần gán Nôm ↔ Quốc ngữ chưa tốt" là **đúng, và đo được**. Phần *văn bản* của căn chỉnh
(NW có băng + chi phí từ điển) hoạt động khá — xác nhận từ điển giả khi lệch thanh ghi chỉ 0,37%.
Ba chỗ hỏng nằm ở **những gì bao quanh** nó:

| # | Vấn đề gốc | Bằng chứng đo được | Ô bị ảnh hưởng |
|---|---|---|---|
| 1 | **Hình học và văn bản bị tách rời.** Hộp ký tự của kinhhannom là hộp **tổng hợp** (chia đều hộp cột cho số chữ OCR đọc được — `core/ocr/ocr_api.py:398`), không phải vị trí đo từ ảnh. DP văn bản quyết *chữ nào ↔ âm nào*, một DP hình học **khác** quyết *hộp nào ↔ chữ nào*; hai DP không biết nhau. | 27,3% cột có số chữ OCR ≠ số âm tiết; ở lớp `OCR>QN` **99,7% hộp là hộp tổng hợp**, ở mọi lớp ~26% hộp rơi về tổng hợp. Soi mắt 24 ô GOLD ở cột lệch ≥2: ~6 crop không phải chữ được gán; ở cột khớp số: ~1–2/24. | 16.529 ô usable (25,6% bộ giao nộp) nằm ở cột lệch số; 4.915 trong đó không có hộp nào từ ảnh |
| 2 | **Từ điển được dùng như chân lý, không phải như tiên nghiệm.** \|R(âm)\| trung vị 20, p90 55; OCR nhầm tự dạng thì hay nhầm sang đúng một chữ *khác cũng nằm trong R* → S1∩S2 "xác nhận" nhãn sai. Đây chính là cơ chế đã sinh lớp 㝵/"người" — và nó **không phải ca duy nhất**. | 1.718 ô GOLD (3,2%) mang nhãn *thiểu số* của (sách, âm) và nhãn ấy *nhìn giống* nhãn đa số: (cùng, 共→其) 158 · (được, 特→時) 39 · (xin, 嗔↔真) 74 · (một, 没→殳) 32 · (nói, 呐→内) 31… Cộng thêm luật L1 **sửa lại âm Quốc ngữ đang đúng** cho vừa từ điển ở 729/755 ô (nhiều→nhiêu ×67, vồ→vô ×70). | ≥2.400 ô GOLD nghi ngờ; 0 ô nào từng có người xem |
| 3 | **Tầng SYLLABLE đặt cổng sai chỗ.** Nó đo "chữ OCR có luôn đọc một âm không" (độ thuần ≥0,6 theo *chữ OCR*), trong khi thứ cần đo là "thanh ghi crop ↔ âm có đúng không". Chữ OCR là nhiễu; âm thì đến từ bản dịch in. | 4.938 ô đủ tần suất/trang nhưng rớt vì độ thuần (移/ri 124, 在/chăng 90, 君/này 84…); 2.198 ô REVIEW **kẹp giữa hai ô GOLD-trực-tiếp trong cột khớp số** vẫn bị bỏ; 5.007 ô `SILVER_uncalibrated` trong cột khớp số bị loại chỉ vì S3 từng chạm vào. | ~7.000–12.000 ô có thể thành chú giải âm tiết đáng tin |

Kèm **8 điểm lỗi mã** (§4; hai trong đó — BUG-1, BUG-2 — làm sai dữ liệu đã giao), một chuỗi bằng chứng đã **đứt lại** (§5.1), và một nhận xét cấu trúc:
logic gán nhãn hiện trải trên **6 mô-đun** (`consensus` → `build_dataset` L1/L3/L5 → `remediate` →
`confusion_fix` → `s3_unwind` → `glyph_fix`), tier đổi tên giữa đường, `USABLE_TIERS` định nghĩa **5
lần** với 2 nội dung khác nhau. S3 vẫn chạy trong build sản xuất nhưng **không ảnh hưởng một ô nào**
của bộ giao nộp (chứng minh §3.6).

---

## 1. BỨC TRANH TOÀN CẢNH (chiều rộng)

### 1.1 Đường chạy thật (`run_pipeline.sh`, 8 bước)

```
PDF ──step1_extract──> pages/ · pages_denoised/ · detected/*_ocr_cache.json (kinhhannom, 9 cột/trang)
                        transcriptions/*_qn_ocr_cache.json (VietOCR 2-pass)
     ──build_dataset──> [mỗi trang] detect_nom_columns_v3 → _get_qn_lines(parse_v5) → normalize_column
                        → realign_column (NW băng, chi phí từ điển) → _pick_reseg (CenterNet N=#âm | midpoint)
                        → decide_label (S1∩S2 ± cầu tự dạng ± S3) → L1/L3 (neo ngữ liệu) → L5 syllable_gate
                        → split theo trang → crop (carve seam + tighten) → labels.csv (22 cột)
     ──remediate──────> quarantine crop trùng, ép md5 về 1 split           → labels_remediated.csv
     ──confusion_fix──> hạ (người, 㝵) → REVIEW                              → labels_final.csv
     ──s3_unwind──────> SILVER → SILVER_uncalibrated; readmit demoted; chốt confusion
     ──glyph_fix──────> QĐ-01: mọi ô âm "người" còn REVIEW/SILVER → 𠊚 GOLD (2.014 ô)
     ──export─────────> re-dataset/ (GOLD 54.156 + SYLLABLE 10.369 = 64.525) + DATASHEET/README/xlsx
```

### 1.2 Tình trạng từng khối

| Khối | Vai trò | Tình trạng | Ghi chú rà soát |
|---|---|---|---|
| `core/ocr/ocr_api.py` | kinhhannom client, cache, chốt md5/pixel | **sống**, chắc | hộp ký tự tổng hợp (§3.1); retry/backoff ổn; auto-login ổn |
| `core/ocr/qn_ocr.py`, `line_detector.py` | VietOCR 2-pass + dò dòng | sống | `projection_deskew` ghim cứng — đúng |
| `core/pdf/pdf_parser.py` | tách trang, `parse_numbered_lines` | sống một phần | `parse_numbered_lines` trả **dòng**, không phải âm tiết → BUG-1 |
| `core/align/parser_v5.py`, `nom_detect_v3.py`, `run_full.py` | bóc 9 dòng QN, 9 cột Nôm | sống, tốt | 447/448 trang v5; 445/445 đủ 9 cột |
| `core/align/parser_v2.py`, `export_dataset_v4.py` | đường cũ | 1 hàm còn dùng (`resegment_col` chỉ trong mode `old`) | có thể gỡ |
| `core/image/*` | nhị phân, dò khung, dò cột, tách chữ theo hình chiếu | sống một phần | `char_segmenter` chỉ còn dùng ở nhánh `valley_*` (không phải sản xuất) |
| `core/ranking/*` (DINOv2, FontDiffusion, ranker) | Bước 3 cũ | **chết** (giữ để đối chiếu) | 1.063 dòng; không import từ đường sống |
| `core/text/*` | từ điển, chuẩn hoá âm, lexicon JSON | sống, tốt | `is_plausible_qn_syllable`, `strip_tone/strip_all` sạch |
| `pipeline/step1_extract.py` | Bước 1 | sống | xử lý trùng tên trang ổn |
| `pipeline/step2_align.py`, `step3_label.py`, `step4_export.py` | NGHỈ | **step2 vẫn là dependency sống**: `align_production` import `_get_qn_lines` từ nó | nên chuyển `_get_qn_lines` sang `core/align/` |
| `pipeline/align_engine/` | **Bước 2 hiện hành** | sống — trọng tâm §3 | `README.md` tự mô tả là "bản copy đóng băng của pipeline/align_engine" (trỏ vào chính nó) — chú thích lỗi thời |
| `pipeline/remediation/` | 5 bước hậu xử lý | sống, test 146 | logic tier rải rác; §5.2 |
| `pipeline/ground_truth/` | dựng mẻ chấm tay, ước lượng precision | sống, test 171 | **chưa có phán quyết nào ngoài QĐ-01** — `dataset_out/human_audit/` không tồn tại |
| `pipeline/publish/` | Frictionless/Croissant/Parquet/datasheet | sống, test 56 | có **hệ chia tách riêng** (`splits.py`) khác với chia tách trong `labels.csv` — hai định nghĩa `split` |
| `pipeline/consensus_fusion/` | Giai đoạn 2: hợp phiếu + Qwen verifier | **không nằm trong run_pipeline**; `score_s3` được `ground_truth` dùng | AI-verdict-không-thành-nhãn: đúng như ràng buộc đề tài |
| `pipeline/lab/` | bàn thí nghiệm T1–T7 | sống, test 81 | tốt: chuẩn nhiễu loạn có đáp án, tất định |
| `pipeline/tools/` | 18 công cụ | sống, test 105 | `variant_table.py` đã thấy đúng vấn đề dị thể nhưng **chưa ai quyết** |
| `train_crop/` | CenterNet R34+FPN, ràng buộc N + seam | sống (ckpt HF) | `enforce_count` chọn hộp theo **điểm tin cậy**, không theo hình học (§3.2) |
| `ArcFace/`, `nom-embed/` | encoder S3 | sống nhưng **vô hiệu** trong bộ giao nộp | §3.6 |
| `gannhanocr-fd/`, `font_diffusion/`, `kaggle_diffusion/` | 89.898 glyph phông | sống, chỉ S3 dùng | submodule `font_diffusion` là repo ngoài (1 GB) |
| `NomNaOCR/`, `lab/ocr_compare/` | recognizer thứ 2/3, Paddle/GLM | thí nghiệm | kết luận riêng của bạn: Paddle cứu 9,5% M2, không thay được kinhhannom |
| `re-dataset/check/` | bộ kiểm độc lập (người ngoài) | sống | diff chưa commit: thêm Unihan, không dùng CSV làm căn cứ (đúng, chống vòng tròn) |
| `scripts/` | check_consistency/evidence/clean_build | sống | **đang báo LỆCH** (§5.1) |

Hai điểm vệ sinh kho:
- `dict/dict -> /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/dict` là **symlink tự trỏ vào chính
  nó** (tạo 2026-09-09 01:57, không trong git). Mọi `rglob`/`find` đi qua `dict/` có thể lặp vô hạn.
- `/Users/truongmdn/TruongMDN/ThS/GanNhan` là **bản sao rời** (inode khác) của
  `DoAn/GanNhanOCR`, cùng HEAD `3e062b1fa3`, `.venv` trong bản sao vẫn trỏ về `DoAn/…/.venv`. Hai
  cây làm việc cùng nội dung sẽ trôi khỏi nhau ngay lần sửa đầu tiên.

---

## 2. QUY TRÌNH GÁN NÔM ↔ QUỐC NGỮ — HOẠT ĐỘNG THẬT SỰ THẾ NÀO

Mô tả này lấy từ mã, không từ tài liệu, vì hai bên có chỗ khác nhau.

1. **kinhhannom** trả về mỗi cột là *một* hộp tứ giác + *một* chuỗi chữ. `boxes_to_columns`
   (`core/ocr/ocr_api.py:389-406`) **chia đều chiều cao hộp cột cho số chữ** để tạo "bbox từng
   chữ". Nghĩa là: bbox chữ = `y_top + k·(H/n)`, `n` = số chữ OCR *đọc được*. OCR rụng một chữ thì
   **mọi hộp phía dưới lệch dần tới một ô**, không hộp nào đo từ ảnh.
2. **VietOCR** đọc trang Quốc ngữ; `parse_v5` bóc 9 dòng đánh số; `normalize_syllables` tách "-",
   nở tên thánh/địa danh theo `config/lexicon/*.json`, sửa nhầm OCR thường gặp; `normalize_column`
   vá dấu rụng bằng đọc âm của các chữ trong cột.
3. **`realign_column`** (Needleman–Wunsch, băng `|m−n|+2`, chi phí 0/0,3/0,9/1,0, del=ins=0,7) ghép
   chuỗi chữ OCR với chuỗi âm. Khe `ins` (âm không có chữ) và `del` (chữ không có âm) **bị vứt**: 1.734
   âm và 1.410 chữ không vào `labels.csv`.
4. **Hộp để cắt** không lấy từ DP trên. `_pick_reseg` gọi CenterNet lấy đúng **N = số âm tiết** hộp
   cho cột (`enforce_count`: thừa thì giữ N hộp điểm cao nhất, thiếu thì bổ đôi hộp cao nhất bằng
   seam), rồi `_monotone_assign` gán N hộp ấy cho **m tâm chữ tổng hợp** của OCR, thay hộp nào cách
   tâm tổng hợp > 0,35 bước bằng **hộp midpoint** (cũng dựng từ tâm tổng hợp). Nếu m > N thì
   trả về toàn midpoint.
5. **`decide_label`**: `ocr_char ∈ R(âm)` → GOLD trực tiếp (không cần cột khớp số, không cần neo);
   cầu tự dạng xuôi/ngược duy nhất + cột khớp/neo → GOLD; S3 → SILVER; còn lại REVIEW.
6. **L1** (`apply_am_sua_dau`): đổi *âm* sang âm khác cùng khung xương nếu chữ OCR có ≥5 ô GOLD-trực-tiếp
   với âm ấy trên ≥3 trang. **L3**: cột lệch + đúng 1 cầu + cặp đã chứng thực → GOLD.
   **L5** (`syllable_gate`): (chữ OCR, âm) ≥5 ô, ≥3 trang, chữ ấy đọc âm ấy ≥60% → SYLLABLE.
7. Sau build: `remediate` (trùng crop, rò split), `confusion_fix` (㝵/người ↓), `s3_unwind` (mọi SILVER
   → `SILVER_uncalibrated`, loại khỏi bộ giao nộp), `glyph_fix` (QĐ-01 người: "người" → 𠊚), export.

Điểm mấu chốt: **thanh ghi văn bản (bước 3) và thanh ghi hình học (bước 4) là hai bài toán được giải
riêng, ghép lại bằng chỉ số `nom_idx` của chuỗi OCR.** Khi số chữ OCR ≠ số glyph thật, chỉ số ấy
không còn định danh được glyph nào trên ảnh.

---

## 3. PHÁT HIỆN (chiều sâu)

### 3.1 Hộp ký tự đầu vào là hộp tổng hợp — và mã tin nó là thật

`align_production.py:188` viết *"Rebuild per-char boxes from the OCR y-CENTERS (which are reliable)"*.
Sai: tâm ấy là `y_top + (k+0,5)·H/n`, chỉ đúng khi OCR đọc đủ chữ **và** chữ cách đều — chữ viết
tay hành-thảo không cách đều (xem cột `stt11 page_0136 c9` trong hình rà soát: chữ cao thấp lệch
nhau tới 40%).

Đo trên 4.029 cột:

| lớp cột | số cột | ô usable | hộp midpoint (tổng hợp) |
|---|---|---|---|
| OCR = QN | 2.931 (72,7%) | 47.996 | 25,6% |
| OCR < QN | 746 | 11.614 | 26,9% |
| OCR > QN | 352 | 4.915 | **99,7%** |

Ở lớp `OCR > QN` (kinhhannom tách một chữ thành hai, hoặc VietOCR rụng/gộp âm — ví dụ `stt2
page_0302 c6`: "ôliva" đếm là 1 âm nên QN 16 < OCR 18 dù OCR đúng), `_monotone_assign` trả `None`
vì m > N và **toàn cột rơi về hộp tổng hợp** — 4.915 ô giao nộp không có bất kỳ bằng chứng ảnh nào
cho hộp của chúng. Ngay ở cột khớp số, 1/4 hộp cũng là midpoint (guard 0,35 bước loại hộp
CenterNet khi tâm tổng hợp lệch — mà tâm tổng hợp lệch là chuyện thường ở chữ viết tay).

Soi mắt (hình `gold_mism_neg2.png` / `gold_eq.png` trong scratchpad phiên này, 24 ô ngẫu nhiên mỗi
nhóm): ở cột lệch ≥2, khoảng 6/24 crop **không phải** chữ được gán hoặc cắt cụt nặng; ở cột khớp số
~1–2/24. Kết quả này khớp với phát hiện của chính bạn ngày 12-09 (`hoan_tat_4_huong.json`, hướng C:
*"tâm hộp detector lệch tâm mực > 0,25 bước ở 34–38% ô"*). Mẫu 24 là để định hướng, không phải số
công bố — cần mẻ chấm phân tầng theo lớp cột (§6).

`crop_quality_flag` **không bắt được** lớp lỗi này: nó báo `truncated` 0,5–0,9% và `bleed` ~5% ở mọi
lớp cột như nhau; nó đo mực ở mép, không đo "có đúng glyph không".

### 3.2 Ràng buộc N của CenterNet lấy N từ nguồn hay sai, và hoà giải bằng điểm tin cậy

`_pick_reseg` truyền `n = len(syllables)`. Nhưng số âm tiết sai ở mọi cột `OCR > QN` do từ mượn
chưa nở ("ôliva", "constantino"…) hoặc VietOCR gộp/rụng; `enforce_count` (`train_crop/infer_centernet.py:160`)
khi M > N **giữ N hộp điểm cao nhất** — hộp thật của một chữ mờ bị bỏ, hộp giả điểm cao ở lại; khi M
< N bổ đôi hộp cao nhất dù đó có thể là một chữ cao thật. Đo nhanh 216 cột (24 trang ngẫu nhiên):
detector không ràng buộc đếm đúng bằng QN chỉ 68% cột khớp số; ở cột lệch, nó đồng ý với OCR 21 lần,
với QN 10 lần, với không ai 20 lần. Ba nguồn đếm (OCR, QN, detector) hiện **không được hoà giải**,
chỉ có QN được tin tuyệt đối.

### 3.3 Phần văn bản của DP: tốt, nhưng khe không được dùng

- Xác nhận từ điển **giả** khi lệch thanh ghi 1 ô: **0,37%** (cầu tự dạng: 1,48%; âm ngẫu nhiên:
  0,73%). ⇒ `s1_inter_s2_direct` là bằng chứng thanh ghi rất mạnh — đây là phần **đúng** của thiết kế.
- Khi OCR rụng 1 chữ giữa cột, DP đặt khe `ins` **đúng chỗ 68,2%** (270/396). 32% còn lại khe trôi
  sang đoạn không có neo; các ô giữa khe thật và khe DP bị ghép lệch 1 → rơi REVIEW (không sai nhãn,
  nhưng mất ô). Khe này **không được truyền sang hình học** (§3.1), nên hộp của các ô ấy cũng không
  biết là mình đang lệch.
- `anchored` (`align_production.py:532-536`) dùng `is_confirmed` gồm cả cầu tự dạng *bất kỳ* (không
  cần duy nhất), yếu hơn tiêu chí GOLD; nhỏ nhưng là một chỗ lỏng.

### 3.4 Từ điển làm chân lý: lớp 㝵/"người" không phải ca đơn lẻ

`consensus.py:119`: `ocr_char ∈ R(âm)` → GOLD, bất kể cột lệch hay không, bất kể \|R\|. Với \|R\|
trung vị 20 (p90 = 55) và lỗi OCR có bản chất *tự dạng*, xác suất OCR nhầm sang một chữ **khác cũng
đọc âm ấy** không nhỏ — vì các dị thể Nôm của cùng một âm thường chia sẻ thành phần.

Đo trên 54.156 ô GOLD: 4.434 ô mang nhãn *thiểu số* của (sách, âm); **1.718 ô (3,2%)** nhãn thiểu
số ấy **nằm trong danh sách nhìn-giống** của nhãn đa số. Tách hai nhóm:

| nhóm | ví dụ (âm, đa số → thiểu số): ô | bản chất | việc cần làm |
|---|---|---|---|
| dị thể của cùng một chữ | (đức, 徳→德) 159 · (biết, 别→別) 82 · (lắm, 廪→廩) 40 · (càng, 强→強) 26 · (một, 没→沒) 18 · (đoạn, 段→断) 18 | Z-variant / giản-phồn; OCR không phân biệt được, và cũng **không nên** tính là hai lớp | bảng chuẩn hoá dị thể có người duyệt (`tools/variant_table.py` đã dựng ứng viên, chưa ai quyết) |
| nhầm hệ thống kiểu 㝵/người | (cùng, 共→其) 158 · (được, 特→時) 39 · (xin, 嗔↔真) 74 · (một, 没→殳) 32 · (nói, 呐→内) 31 · (giờ, 除→徐) 24 · (lần, 吝→各) 17 | OCR đọc sai sang chữ có mặt trong R; S1∩S2 "xác nhận" | mẻ chấm người theo cặp, rồi `confusion_fixes.yaml` mở rộng |

Cả hai nhóm hiện ở GOLD, **chưa ô nào có người xem**. Cơ chế phát hiện ở trên (thiểu số + nhìn
giống đa số trong cùng sách) rẻ, tất định, và chính là cách 㝵/người lẽ ra được bắt trước khi cần
Fisher test.

### 3.5 Luật L1 sửa lại văn bản Quốc ngữ đang đúng

`apply_am_sua_dau` (`build_dataset.py:106-160`) đổi **âm** để chữ OCR khớp từ điển. Đối chiếu 755 ô L1
với thế hệ trước (`2f0b9dc116`): **729/755 (97%) âm gốc là từ có trong từ điển** — tức là luật này
*không* có chốt "âm là từ có thật thì không đụng", chốt mà chính đề tài đặt ra cho
`normalize_column`/`fix_tone` (CHỐT 1 trong docstring `syllable_normalize.py`).

Hệ quả chia hai loại, mã không phân biệt được:

- Sửa đúng lỗi VietOCR: (chứa→chúa, 主) 14 · (lê→lễ, 礼) 47 · (quý→quỷ, 鬼) 6.
- **Ghi đè cách đọc thật của sách**: (nhiều→nhiêu, 僥) 67 — sách in "nhiều", 僥 là chữ Nôm cho
  "nhiều", chỉ từ điển thiếu cặp; (vồ→vô, 無) 70 — "Vít-vồ" là tên ("viết vồ" theo lexicon),
  Nôm mượn 無; (chẳng→chăng, 生) 41. Ở các ô này, nhãn chữ có thể đúng nhưng **cột `syllable` giao
  nộp nay sai so với bản in** — với tầng SYLLABLE đó là toàn bộ nhãn.

Việc đúng là **thêm cặp vào từ điển ngữ liệu** (僥→nhiều, 無→vồ) và giữ nguyên âm, hoặc ghi âm chuẩn
hoá ở một cột riêng. Không được sửa quan sát cho vừa tiên nghiệm.

### 3.6 Cổng SYLLABLE đo sai đại lượng; S3 chạy mà không tác dụng

`syllable_gate` yêu cầu *chữ OCR* đọc âm đó ≥60% số lần nó xuất hiện. Nhưng chữ OCR ở tầng này
theo định nghĩa là chữ **không tin được** (không nằm trong R). Kinhhannom có các "chữ hút" — 在, 君,
尺, 石, 乇, 要, 用 — nhả ra cho nhiều glyph thảo khác nhau; độ thuần của chúng luôn thấp dù âm ở mỗi ô
vẫn đúng (âm đến từ bản in và thanh ghi). Đo: **4.938 ô** đủ ≥5 ô/≥3 trang nhưng rớt độ thuần
(移/ri 124 vì 移 còn đọc "rê" 53 — cả hai đều là cách đọc hợp lệ của tên riêng).

Trong khi đó bằng chứng thanh ghi thật sự — cột khớp số + hai bên là GOLD-trực-tiếp — không được
dùng: **2.198 ô REVIEW** thoả điều kiện đó (xác suất ghép lệch < 0,4% theo §3.3) và **5.007 ô
`SILVER_uncalibrated` trong cột khớp số** vẫn bị bỏ.

Về S3: `s3_unwind` đổi mọi SILVER thành `SILVER_uncalibrated` (ngoài bộ giao nộp); cổng L5 gộp
SILVER vào bể (`be_day_du`), L1 áp cho mọi ô non-GOLD. Theo mã, **tập ô giao nộp khi chạy có S3 và
không có S3 là một** — S3 chỉ đổi tên rule (`below_visual_threshold` ↔ `unconfirmed_no_s3`,
`nghia_consensus_tu_silver` ↔ `nghia_consensus`). Build sản xuất vẫn nạp encoder + 89.898 glyph + nguyên mẫu 1.571 lớp và
chấm **33.309 ô** (`s3_cosine` khác rỗng) cho việc đó.

### 3.7 Cột "khớp số" vẫn được ghi như cột thường

`labels.csv` không có cột nào ghi `n_ocr`, `n_qn`, `n_det` hay nguồn hộp (detector / split / midpoint)
của từng ô. `page_cot_lech` chỉ ở mức trang (3 trang). Người dùng bộ dữ liệu không thể lọc 16.529 ô
cột-lệch, và mẻ chấm tay không thể phân tầng theo yếu tố rủi ro lớn nhất.

---

## 4. LỖI MÃ CỤ THỂ

| # | Vị trí | Lỗi | Tác động | Sửa |
|---|---|---|---|---|
| BUG-1 | `pipeline/step2_align.py:78-88` | fallback `parse_numbered_lines` trả `{n: [dòng vật lý]}`; `_get_qn_lines` trả thẳng → mỗi cột có **2 "âm tiết"** là nguyên hai dòng | `stt4/page_0146`: 18 ô rule `diverged_column` với `syllable` là cả câu; ~190 glyph của trang mất hẳn. Kết luận T2 "cứu 1 trang" là sai | nối qua `parse_v5._split_syllables(" ".join(lines), qn_dict)` trước khi trả |
| BUG-2 | `pipeline/align_engine/build_dataset.py:136-159` | L1 không có chốt "âm gốc là từ có thật" | §3.5 — 729 ô đổi âm đang đúng | thêm `if syl in qn_to_nom: continue` **hoặc** đổi thành "thêm cặp vào từ điển ngữ liệu, giữ âm" |
| BUG-3 | `pipeline/align_engine/align_production.py:187-216` | docstring khẳng định tâm OCR "reliable"; `_reseg_column` là đường rơi cho 1/4 hộp | §3.1 | ghi rõ nguồn hộp vào `labels.csv`; hạ midpoint xuống bậc cuối |
| BUG-4 | `train_crop/infer_centernet.py:138-183` | `enforce_count` M>N chọn theo điểm, không theo phủ/cách đều; M<N bổ đôi hộp cao nhất vô điều kiện | §3.2 | chọn tập con N hộp bằng DP theo bước lặp (tối thiểu \|gap − pitch\|), bổ đôi chỉ khi hộp cao > 1,6 pitch |
| BUG-5 | `dict/dict` | symlink tự trỏ | rủi ro lặp vô hạn khi `rglob` | `rm dict/dict` |
| BUG-6 | 5 tệp | `USABLE_TIERS` định nghĩa 5 lần (`export_final_dataset`, `s3_unwind` (khác nội dung), `census`, `suspicion`, `publish/splits`) | sửa một chỗ quên bốn chỗ — đã xảy ra với `SILVER_uncalibrated` | một hằng ở `pipeline/tiers.py` |
| BUG-7 | `pipeline/align_engine/README.md` | "bản copy đóng băng tách ra từ `pipeline/align_engine/`" — tự trỏ; docstring `build_dataset.py:20-21`, `to_standard.py:12` còn đường `evaluation/ver_new/` và đường tuyệt đối máy cá nhân | tài liệu sai đường | cập nhật |
| BUG-8 | `pipeline/phase1_engine_selftest.py` | **0 test** cho `realign_column`, `decide_label`, `apply_am_sua_dau`, `apply_cot_lech_cau_xuoi` — bốn hàm quyết toàn bộ nhãn | T3.4 trong VIEC_CAN_LAM còn mở | ≥15 assertion: khớp/rụng/thừa/khe hai đầu/băng; mỗi nhánh rule của `decide_label` |

---

## 5. QUẢN TRỊ & CẤU TRÚC

### 5.1 Chuỗi bằng chứng đã đứt lại sau `bff2a3bd28`

`bash scripts/check_consistency.sh` hôm nay: **3/4 phép kiểm hỏng** — `labels.csv`,
`labels_remediated.csv`, `labels_final.csv`, `re-dataset/labels.csv` đều LỆCH sha256 so với
`docs/EVIDENCE_INDEX.md`; `docs/BANG_SO_LIEU_CHINH_THUC.md` vẫn ghi **52.441 GOLD / 59.371 dòng /
commit `2f0b9dc116`** trong khi đĩa là 54.156 / 64.525 / `bff2a3bd28`. Đây đúng là "Lỗ hổng 1" mà
rà soát 2026-08-24 đã vá bằng khối `<!-- AUTO -->` — nhưng lệnh cập nhật vẫn phải **nhớ mà gõ**.
Chừng nào `run_pipeline.sh` chưa tự gọi `update_bang_so_lieu` + `evidence()` ở cuối `step_export`,
lớp lệch này sẽ tái diễn ở mỗi lần đổi flow.

### 5.2 Logic tier rải trên 6 mô-đun, trạng thái đổi tên giữa đường

`GOLD/SILVER/SYLLABLE/REVIEW` sinh ở build → `QUARANTINE` ở remediate → `|demoted_lowcos_s3` (tắt)
→ `REVIEW` bởi confusion_fix → `SILVER_uncalibrated` + `readmitted_from_s3_demotion` + đổi tên rule
bởi s3_unwind → `quyet_dinh_nguoi:*` bởi glyph_fix, mỗi bước có bất biến riêng phải giữ (AM_DA_QUYET
trong `consensus.py` tồn tại chỉ để bước 3 không giẫm lên bước 7). Cả chuỗi là các **bản vá hậu
nghiệm cho một quyết định đã sai ở build**. Một `decisions.yaml` duy nhất (dị thể, confusion, phán
quyết người, cặp từ điển ngữ liệu) áp **trong** `decide_label` sẽ thay được 4 trong 5 bước, và
`tier` sẽ chỉ được gán một lần.

### 5.3 Tỷ lệ công sức

Đếm dòng: engine căn chỉnh + gán nhãn (`align_engine/` 6 tệp lõi) **1.878 dòng**; hậu xử lý
`remediation/` 2.020 + chấm tay `ground_truth/` 6.779 + `publish/` 1.269 + `tools/` 2.927 + `lab/` 3.185
(trong đó selftest 3.540) — **≈ 16.000 dòng bao quanh 1.900 dòng quyết định nhãn**; tài liệu `docs/` 19 MB. Hạ tầng tái lập rất tốt (mọi selftest xanh, sha256, tất định),
nhưng **chưa có phép đo nào chạm vào câu hỏi trung tâm**: crop này có đúng glyph này không, và nhãn
này có đúng không. Cho tới khi có ~1.000 phán quyết người phân tầng, mọi số ở §2 của bảng số liệu là
số đếm, không phải chất lượng.

---

## 6. KHUYẾN NGHỊ — theo thứ tự ưu tiên

### P0 — sửa ngay, rẻ, không đổi thiết kế (≤ 1 ngày)

1. BUG-1 (fallback bóc dòng), BUG-2 (chốt từ-có-thật cho L1, hoặc chuyển L1 thành "thêm cặp từ điển
   ngữ liệu"), BUG-5, BUG-6.
2. **Ghi thêm cột** vào `labels.csv`: `n_ocr`, `n_qn`, `n_det_raw`, `box_source ∈ {detector, split,
   midpoint}`, `flank_gold` (0/1/2 ô GOLD-trực-tiếp kề). Không đổi nhãn, chỉ để lọc và phân tầng.
3. Tắt `--use-s3` trong `run_pipeline.sh` (hoặc giữ nhưng chỉ ghi cột điểm) — cùng bộ giao nộp
   (§3.6), bỏ được việc nạp encoder + glyph bank + chấm 33.309 ô và cả cơ chế cache nguyên mẫu
   từng tự đầu độc (`4ce957d14c`).
4. Nối `update_bang_so_lieu` + `evidence()` vào cuối `step_export`; chạy lại để chuỗi bằng chứng khớp.
5. Bổ sung test cho 4 hàm ở BUG-8.

### P1 — thiết kế lại phần ghép hình học ↔ văn bản (1–2 tuần, đây là "chưa tốt" thật)

Nguyên tắc: **một bài toán căn chỉnh ba chuỗi**, không phải hai bài toán ghép bằng chỉ số.

```
 G = hộp glyph từ ảnh (CenterNet KHÔNG ràng buộc, sau NMS)      ── nguồn hình học duy nhất
 C = chuỗi chữ OCR kinhhannom (kèm tâm tổng hợp chỉ làm TIÊN NGHIỆM vị trí)
 Q = chuỗi âm tiết Quốc ngữ (bản in, đáng tin nhất về SỐ và THỨ TỰ)
```

- Bước 1: DP hình học `G ↔ C` — chi phí = \|tâm hộp − tâm tổng hợp\|/pitch (mềm), del/ins cho hộp
  thừa/chữ rụng. Kết quả: mỗi chữ OCR gắn với **đúng một hộp ảnh hoặc không có hộp**.
- Bước 2: DP văn bản `C ↔ Q` như hiện nay, **nhưng thêm chi phí hình học**: một cặp (c, q) mà c không
  có hộp, hoặc vị trí hộp của c lệch xa vị trí kỳ vọng của q (q_idx/N·H), bị phạt.
- Bước 3: mọi ô giao nộp phải có hộp **từ ảnh**; ô không có hộp ảnh → REVIEW với lý do `no_glyph_box`,
  dù từ điển có xác nhận. Không còn hộp midpoint trong bộ giao nộp; thay vào đó ghi `box_source`.
- Đếm N: hoà giải ba nguồn — dùng \|G\| làm số glyph nếu \|G\| ∈ {\|C\|, \|Q\|}; nếu ba số khác nhau,
  cột vào `REVIEW_count_conflict` (ước tính < 2% cột theo mẫu 216 cột).
- Từ mượn: khi \|Q\| < \|C\| = \|G\| và cột có token không trong từ điển dài ≥5 ký tự, tách token
  ấy theo âm tiết hợp lệ (`is_plausible_qn_syllable`) sao cho \|Q\| = \|G\| — có đáp án kiểm được,
  không cần thêm vào lexicon tay từng tên.

Kiểm chứng: chuẩn nhiễu loạn `pipeline/lab/perturb.py` đã có; thêm loại hỏng "rụng hộp"/"tách hộp"
ở tầng G. Kỳ vọng đo được: tỷ lệ `IoU(hộp giao nộp, hộp G)` < 0,5 từ ~9% về ~0; ô cột-lệch không
còn khác ô cột-khớp về chất lượng crop.

### P1' — từ điển là tiên nghiệm, không phải chân lý

1. **Bảng dị thể** (Z-variant) có người duyệt, áp ở `decide_label`: gộp 徳/德, 别/別, 廪/廩, 强/強,
   没/沒, 段/断, 爲/為, 事/亊… thành một lớp, ghi cả hai mã. Giảm lớp giả, tăng nhất quán trong sách.
2. **Điểm tin cậy S1∩S2 theo \|R\|** thay vì nhị phân: `−log(|R(âm)|/|R_max|)` hay đơn giản là cột
   `dict_support = |R|`; ô có \|R\| ≥ 40 và nhãn thiểu số trong sách → ưu tiên chấm.
3. **Phát hiện confusion tự động** (§3.4) chạy mỗi build, xuất bảng ứng viên; các cặp đã có người
   quyết vào một `decisions.yaml` duy nhất (thay `confusion_fixes.yaml` + `quyet_dinh_glyph.yaml`).
4. **Từ điển ngữ liệu** (corpus lexicon): cặp (chữ, âm) có ≥5 ô GOLD-trực-tiếp trên ≥3 trang trở
   thành entry hợp lệ *cho ngữ liệu này* — đó là thứ L1/L3 đang làm ngầm; làm tường minh và giữ âm.

### P2 — tầng SYLLABLE theo bằng chứng thanh ghi

Thay cổng độ-thuần-chữ-OCR bằng: cột khớp số (hoặc \|G\| = \|Q\| sau P1) **và** ít nhất một ô kề là
GOLD-trực-tiếp (hoặc khoảng cách tới neo gần nhất ≤ 2). Ước tính cứu 7.000–12.000 chú giải âm tiết
(§3.6) mà xác suất ghép lệch < 1%. Chữ OCR ghi làm "giả thuyết" ở cột `ocr_char`, không phải nhãn.

### P3 — đo cái cần đo

Một mẻ chấm người ~1.000 ô, **phân tầng theo (lớp cột × box_source × dict_support × thiểu-số/đa-số)**,
hai câu hỏi tách bạch: (a) crop có đúng một glyph và là glyph của âm này không; (b) mã Unicode có
đúng không. Hạ tầng `pipeline/ground_truth/` đã sẵn; điều còn thiếu là chính mẻ chấm. Không có nó,
§3.1 và §3.4 vẫn chỉ là ước lượng của tôi từ 48 ô và các thống kê gián tiếp.

---

## 7. NHỮNG GÌ ĐANG TỐT — giữ nguyên

- `parse_v5` + `nom_detect_v3`: 447/448 và 445/445, nhánh projection không còn dùng.
- Chốt cache OCR theo md5/pixel, auto-login, retry — chắc.
- `realign_column`: phần văn bản đúng hướng; xác nhận giả 0,37% là con số đáng đưa vào luận văn.
- `carve_neighbor_ink` + `tighten_box`: đúng là "bộ phận chịu lực" như T4 kết luận.
- Chia tách theo trang, cột `label_in_train`, xuất 3 chuẩn, selftest 665 xanh, không RNG.
- Tinh thần tài liệu: ghi cả cái đã hỏng. Bản rà soát này chỉ nối tiếp tinh thần ấy.

---

*Hình dùng khi rà (không commit): `col_stt11_page_0106_c1.png`, `col_stt11_page_0136_c9.png`,
`mismatch_cols.png`, `gold_mism_neg2.png`, `gold_eq.png` trong scratchpad phiên; dựng lại bằng các
đoạn mã tương ứng trong `scripts/_oneoff/ra_soat_2026-09-13_do_lai.py` (phần A–G) và `_pick_reseg` cho hình.*
