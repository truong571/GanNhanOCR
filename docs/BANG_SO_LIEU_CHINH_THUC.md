# BẢNG SỐ LIỆU CHÍNH THỨC

**Đo ngày**: 2026-08-11 · **Bộ nhãn**: `dataset_out/labels_final.csv` (sinh bởi `run_pipeline.sh` 6 bước)

> **QUY TẮC BẤT DI BẤT DỊCH**: mọi con số trong luận văn (mọi chương, mọi bảng, mọi slide) **chỉ được trích từ file này**. Không chương nào được lấy số từ chỗ khác. Mỗi số dưới đây có: giá trị · file nguồn · lệnh tái sinh · ngày đo · **nguồn kiểm định** (người / AI / chưa đo).

---

## 0. NGUỒN GỐC — chuỗi nhãn bất biến

```
labels.csv  --[remediation apply]-->  labels_remediated.csv  --[confusion_fix]-->  labels_final.csv
```

| File | sha256 (16 ký tự đầu) | Lệnh tái sinh |
|---|---|---|
| `dataset_out/labels.csv` | `8a0739affed489cf` | `python -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --use-s3 --reseg detector` |
| `dataset_out/labels_remediated.csv` | `8bf513cfbdc8ce51` | `python -m pipeline.remediation --labels dataset_out/labels.csv --out dataset_out apply --tau 0.62` |
| `dataset_out/labels_final.csv` | `8eb1c5c773ce1f54` | `python -m pipeline.remediation.confusion_fix --in dataset_out/labels_remediated.csv --out dataset_out/labels_final.csv --fixes config/confusion_fixes.yaml` |
| `dataset/labels.csv` (bộ giao nộp) | `af385a2eb69472c6` | `python pipeline/export_final_dataset.py --labels dataset_out/labels_final.csv --src-root dataset_out --out dataset` |

✅ **Reproduction-check 2026-07-21**: chạy lại 2 bước cuối từ `labels.csv` ra thư mục tạm → `labels_remediated.csv` và `labels_final.csv` **byte-identical** (sha256 khớp) với bản đã đóng băng. Chuỗi remediation→confusion là **tất định**.

---

## 1b. SAU KHI GỠ S3 (`labels_final.csv`, đo **2026-08-19**) — BẢN HIỆN HÀNH

Bước 6 `s3_unwind` (xem §4b: mọi tín hiệu thị giác có CI chứa 0,5) đã chạy trên bộ nhãn.
sha256 mới: `3396915e0e62b8c7` · báo cáo: `dataset_out/s3_unwind_report.json`

| Tier | Số ô | Đổi | Ghi chú |
|---|---|---|---|
| **GOLD** | **50.063** | **+1.185** | 1.185 ô `demoted_lowcos_s3` trả về, gắn cờ `readmitted_from_s3_demotion` |
| SYLLABLE | 6.761 | 0 | cấp âm tiết, giữ nguyên |
| **SILVER_uncalibrated** | **10.890** | (đổi tên) | **0 verdict người**; nằm NGOÀI `USABLE_TIERS` → **rơi khỏi bộ giao nộp**, KHÔNG bị xoá |
| REVIEW | 14.555 | −1.185 | 10.934 ô đổi lý do `below_visual_threshold` → `no_s1_inter_s2` |

- **Bộ giao nộp = GOLD + SYLLABLE = 56.824 ô** (trước: 66.589; **−9.765, −14,7%**)
- **Precision GOLD: 97,98%** (777/793 verdict người) **CI95 [96,7 – 98,8]** — trước gỡ S3 là
  98,00% (784/800). **Không đổi trong sai số**, đúng như dự kiến: bước này KHÔNG sửa nhãn nào.
- **Số lớp ký tự trong bộ giao nộp: 1.582** — TĂNG so với 1.559 của bộ cũ (GOLD+SILVER),
  vì 1.185 ô trả về mang thêm lớp. Bỏ SILVER **không** làm mất phủ lớp.
- Căn cứ trả 1.185 ô về GOLD: luật `s1_inter_s2_similar` đo được **97,6%** (40/41,
  CI95 [87,1–99,9]), không phân biệt được với `s1_inter_s2_direct` **98,0%** (737/752,
  CI95 [96,7–98,9]).

⚠️ **Hai điều PHẢI khai báo trong datasheet**:
1. 97,98% vẫn là **POST-HOC** (tính trên chính mẫu đã dùng phát hiện lỗi 㝵/người); chưa có
   mẫu SRS xác nhận độc lập.
2. Trong 1.185 ô trả về GOLD chỉ có **2 ô** từng được người chấm → dùng cờ
   `readmitted_from_s3_demotion` để lọc được.

Lệnh tái sinh: `.venv/bin/python -m pipeline.remediation.s3_unwind --in dataset_out/labels_final.csv --out dataset_out/labels_final.csv --apply` (luỹ đẳng)

---

## 1. DATASET CUỐI (`labels_final.csv`, đo 2026-08-11) — TRƯỚC khi gỡ S3, giữ làm lịch sử

Tổng: **82.274 dòng**.

| Tier | Số crop | label_level | Nguồn kiểm định precision |
|---|---|---|---|
| **GOLD** | **48.893** | char | 👤 **NGƯỜI** — xem §2 |
| SILVER | 10.887 | char | 🤖 AI-vision (chưa hiệu chuẩn) — xem §3 |
| SYLLABLE | 6.809 | syllable | ⚪ **CHƯA ĐO** |
| REVIEW | 15.685 | — | loại khỏi tập có nhãn (~93% không có crop) |
| QUARANTINE | 0 | — | không còn hàng nào: lớp trùng lặp đã đóng ở gốc (engine-fix), census 22/07 ra 0 |

- **Dataset có-nhãn (usable)** = GOLD + SILVER + SYLLABLE = **66.589 crop** — khớp đúng
  `dataset/labels.csv` (66.589 dòng, 66.589 ảnh copy, 0 ảnh thiếu)
- **Số lớp ký tự phân biệt** (GOLD+SILVER, có unicode) = **1.552**
- Phạm vi: **3 sách** (SachThanhTruyen 2/4/11), **445 trang**, 1 nét chữ (đơn nguồn)

Tái sinh phân bố tier: `python -c "import pandas as pd; print(pd.read_csv('dataset_out/labels_final.csv')['tier'].value_counts())"`

---

## 2. PRECISION GOLD — con số công bố được (audit NGƯỜI)

Nguồn: `dataset_out/ground_truth/verdicts_reanchored.csv` (846 verdict người, neo lại theo book/page/bbox-IoU; **KHÔNG có cột source → toàn bộ là người chấm**). Báo cáo: `dataset_out/confusion_fix_report.json`.

| Chỉ số | Giá trị | Chi tiết |
|---|---|---|
| **Precision GOLD trước demote 㝵/người** | **97,08%** | 799/823 (loại unsure) |
| **Precision GOLD sau demote 㝵/người** | **98,00%** | 784/800 |
| Precision toàn mẫu 846 (mọi tier, loại unsure) | 97,04% | 819/844 |

⚠️ **Cảnh báo trung thực (phải viết trong luận văn)**: con số **98,00% là POST-HOC** — tính lại trên chính mẫu đã dùng để phát hiện lỗi 㝵/người. Theo acceptance sampling, cần một **mẫu SRS xác nhận MỚI** (GĐ4) mới được tuyên bố "≥97% (acceptance)". Trước GĐ4, trình bày trình tự: *audit → phát hiện (Fisher p=5,4e-8) → demote 1.926 crop → tính lại*.

Case study demote: 1 fix (người→㝵), demote **1.926 crop** sang REVIEW (`confusion_fix_report.json`: `total_demoted = 1926`). Precision GOLD 97,08 % → 98,00 % ở §2 tính trên mẫu neo `verdicts_reanchored.csv` (846 verdict), nên giữ nguyên tư cách **post-hoc** cho tới khi có mẻ SRS mới (mẻ 860 ô).

---

## 3. SILVER / SYLLABLE — CHƯA có precision NGƯỜI

| Tier | Trạng thái | Ghi chú |
|---|---|---|
| SILVER (10.887) | 🤖 chỉ có AI-audit | `audit_SILVER/verdicts_ai.jsonl` = 750 dòng **100% `source='ai_vision'`**. Con số 72,98% (`report.json`) là **AI chấm, KHÔNG phải người** — không được trình bày như human audit. |
| SYLLABLE (6.809) | ⚪ 0 verdict | grid đã dựng, chưa ai chấm. |

🔒 **Guard đã cài (GĐ3)**: `pipeline/ground_truth/estimate.py` — module sinh precision/CI — mặc định **LOẠI** verdict `source='ai_vision'`, raise lỗi to tiếng nếu toàn bộ là AI. Muốn dùng verdict máy phải khai `--include-ai-verdicts` tường minh. Nghĩa là **không thể vô tình** biến nhãn máy thành ground truth.

→ Đo precision người cho SILVER/SYLLABLE là việc của **GĐ4** (đường găng).

---

## 4. S3 (tín hiệu thị giác) — hai metric KHÁC NHAU, đừng trộn

| Metric | Giá trị | Đo cái gì |
|---|---|---|
| Retrieval@1 (S3 như bộ xếp hạng ứng viên) | ~0,89 | S3 xếp đúng chữ lên đầu bao nhiêu % |
| **Error-detection AUC (bank_cos)** | **0,566** [0,459–0,672] | S3 phân biệt nhãn ĐÚNG/SAI thật — **gần ngẫu nhiên** |
| ~~precision 0,9517 / 0,959 / 0,976~~ | ĐÃ BỊ BÁC | proxy circular tự sinh (GOLD-test do chính S1∩S2 sinh) |

### 4b. ArcFace retrain (Kaggle, Sub-center K=3 + SAM) — ĐO 2026-08-19, KHÔNG cứu được

| checkpoint | val head-top1 | **error-detection AUC** (826 verdict NGƯỜI) | CI95 bootstrap |
|---|---|---|---|
| `ArcFace/checkpoints/best.pt` | 0,806 (epoch tốt nhất/30) | **0,577** | [0,442 – 0,706] |
| `ArcFace/checkpoints/last.pt` | 0,806 (epoch 30, hoàn tất) | 0,558 | — |
| ~~S3 cũ (`nom-embed/best.pt`)~~ | — | 0,566 | [0,459 – 0,672] |

Tách theo loại lỗi (`best.pt`): chiều NHÃN (`wrong_label`, n=18) AUC **0,581** [0,423–0,736];
chiều CROP (`wrong_image`, n=6) AUC 0,564 [0,263–0,821].

**Kết luận**: bản retrain **không khác 0,5 có ý nghĩa thống kê** — CI của cả ba lát cắt đều
chứa 0,5, và không hơn S3 cũ (0,566). Val top-1 tăng 0,23→0,806 nhưng AUC bắt lỗi đứng yên:
xác nhận encoder giỏi **xếp hạng** chứ không **phát hiện nhãn sai**. → **Không có cổng thị
giác nào dùng được**; chiến lược phải là 2 tín hiệu văn bản (S1 ∩ S2) + chiều CROP đo bằng
hình học. Lệnh tái sinh: `python ArcFace/eval_human_verdicts.py --ckpt ArcFace/checkpoints/best.pt`

**Kết luận**: S3 là **ranker/filter**, KHÔNG phải cổng phát hiện lỗi (must-pass). Số `measured_precision=0.9517` còn nằm trong `pipeline/align_engine/s3_calibration.json` (file dữ liệu JSON engine đọc, không chú thích được) — **đây là proxy circular, chưa thay bằng đo người**; không trích vào luận văn.

---

## 5. CENSUS (trùng lặp crop) — số hiện tại vs lịch sử

Chi tiết before/after ở **`docs/census_history.md`**. Tóm tắt:

| Chỉ số | Lịch sử (trước engine-fix) | Hiện tại (`labels.csv`, đo 2026-07-21) |
|---|---|---|
| dup_bbox | 701 | **0** |
| cross_col | 1.686 | **8** |
| dup_defect union | 2.321 | **8** |
| provably-wrong | 1.177 | **4** |
| split-leak (md5 cross-split) | 288 | **0** |
| quarantine | (n/a) | **8** (đều conflict, 0 duplicate) |

Số "hiện tại" là bất biến selftest (kiểm bằng `bash scripts/run_all_selftests.sh` = **392/0**). Số "lịch sử" **không còn tái lập được** (thế hệ labels.csv trước dedup không còn trên đĩa) — bảo tồn làm bằng chứng engine-fix.

---

## 6. SPLIT công bố (`dataset_out/release/`, rebuild từ labels_final 2026-07-21)

| Split | Số crop | Rò rỉ |
|---|---|---|
| train | 61.909 | page-span=0, md5-span=0 |
| val | 1.980 | (414 lớp singleton ép về train) |
| test | 2.687 | |

- LOBO theo sách: holdout yen2/yen4/yen11, test ~22k mỗi cấu hình.
- `crops.csv` sha256 thật, `validate` **10/10 checks passed**, PK không null.
- Lệnh: `python -m pipeline.publish split && ... metadata && ... datasheet && ... validate`

---

## 7. KIỂM ĐỊNH (selftest)

**414 passed, 0 failed** — `bash scripts/run_all_selftests.sh` (mốc 2026-08-19; lịch sử 223 → 392 → 414).
Con số trích vào luận văn là **414 assertions**.

---

## Ghi chú bảo trì

Nếu `labels.csv` được tái sinh lại (thế hệ dữ liệu mới), các số §1/§5 sẽ đổi → phải chạy lại census + cập nhật file này + `census_history.md` + assertion selftest, rồi mới rút mẫu audit mới. **Đo một lần, không đo hai lần** (xem `THU_TU_THUC_HIEN_TONG_THE.md`).
