<!-- KHÔI PHỤC 2026-07-20: bản gốc bị xoá nhầm và chưa từng được commit.
     Nội dung dưới đây trích nguyên văn từ transcript agent đã đọc trọn file
     (101/101 dòng, bản Read đầy đủ). Đã đối chiếu: khớp mọi trích dẫn rời rạc
     trong 4 workflow journal. Nếu phát hiện sai khác, đây là bản phục dựng. -->

# FLOW TỔNG THỂ — BẢN CHỐT CHÍNH THỨC (cập nhật fresh 3-sách, 2026-07-15)

> Đo lại toàn bộ trên thế hệ dữ liệu HIỆN TẠI (3 sách, engine đã fix), verify bằng workflow
> 10 agent (3/3 kiểm chéo đối kháng CONFIRMED). Blocker data-currency đã GIẢI QUYẾT.

## TL;DR — trạng thái sau khi build lại 3 sách + neo lại audit

- ✅ **Dataset hiện tại đầy đủ 3 sách:** `labels.csv` = 82.274 pairs (yen2 28.135 + yen4 27.368 + yen11 26.771; 445 trang; 1.593 lớp chữ).
- ✅ **825/846 verdict human neo lại được** (97.5%, median IoU 1.0, 93% byte-identical) → ground truth dùng lại sạch.
- ✅ **Chuỗi remediation xác định đã dựng:** `labels.csv → labels_remediated.csv (defect) → labels_final.csv (confusion-fix)`.
- ✅ **㝵/người đã demote** (1.923 crop → REVIEW) → **precision GOLD 97.08% → 98.00%**.
- ⏭ **Việc mở lớn nhất còn lại:** audit SILVER (10.856) + SYLLABLE (6.751) — 17.607 crop precision CHƯA ĐO.

## 1. CON SỐ ĐỊNH LƯỢNG CUỐI (fresh, đã verify)

| Chỉ số                                 | Giá trị                                                                                    |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Precision GOLD (trước fix)**   | **97.08%** (799/823), Wilson95 [95.70, 98.03], CP95 [95.69, 98.12]                     |
| **Precision GOLD (sau demote 㝵)** | **98.00%** (784/800)                                                                   |
| Precision SILVER / SYLLABLE              | **CHƯA ĐO** (0 verdict — audit 100% GOLD)                                           |
| S3 bắt-lỗi (bank_cos) AUC              | **0.566** [0.459, 0.672] — trùm 0.5, mọi tín hiệu ≤0.6                           |
| Nhãn sai hệ thống kỳ vọng           | ~**1.341** (16 cặp đã chứng minh; top5 = 74%; **cả 16 đều dict_ok=True**) |
| **㝵 (U+3775)**                    | 2.008 crop, 98% "người", audit sai**34.8%** (Fisher p=5.4e-8 vs baseline 2.0%)       |
| 𠊛/𠊚/人 (Nôm chuẩn 'người')         | 𠊛=0, 𠊚=43, 人=0 nhãn → pipeline dùng 㝵:𠊚 = 46:1                                       |

**Dân số tier (labels.csv fresh):** GOLD 51.211 / SILVER 11.384 / SYLLABLE 6.751 / REVIEW 12.928.
**Sau chuỗi remediation (labels_final.csv):** GOLD **48.969** / SILVER 10.856 / SYLLABLE 6.751 / REVIEW 15.690 / QUARANTINE 8. REVIEW ~93% không có crop (chỉ bản ghi cột) → dataset công bố có-nhãn ≈ **66.576** (GOLD+SILVER+SYLLABLE).

**Đổi so với bản stale:** GOLD 48.6k→51.2k; 㝵 1.737→2.008; bug duplicate-export **ĐÃ HẾT** (dedup chỉ bỏ 5–8 dòng, không còn 2.299 quarantine); precision GOLD hội tụ ~97% (ổn định qua re-build). yen4 là ổ lỗi 㝵 nặng nhất (62.5%).

## 2. FLOW TỔNG THỂ (7 giai đoạn — trạng thái)

```
[1] EXTRACT      PDF → kim OCR (S1 Nôm+box) + VietOCR (S2 QN 9 cột)        ✅ step1_extract   (3 sách)
[2] ALIGN+TIER   banded DP → S1∩S2=GOLD | +S3 phá-hòa=SILVER | REVIEW       ✅ align_engine.build_dataset
                 ⚠ S3 = phá-hòa SILVER, KHÔNG phải cổng chân lý (AUC 0.566)
[3] REMEDIATION  defect: quarantine dup(8) + demote similar-bridge s3 thấp   ✅ remediation apply → labels_remediated.csv
[4] CENSUS       gom (âm→nhãn) theo dân số → worklist confusion               ✅ mine_confusions → confusions.csv
[5] AUDIT NEO    846 verdict cũ → NEO LẠI (book,page,IoU) 825 dùng được      ✅ reanchor_verdicts → verdicts_reanchored.csv
                 model = TRIAGE (bất đồng→audit), KHÔNG gán nhãn
[6] CONFUSION-FIX demote 㝵-người 1923 → REVIEW (từ confusion_fixes.yaml)     ✅ remediation.confusion_fix → labels_final.csv
[7] ESTIMATE+PUBLISH  precision per-tier + publish                            ⏭ estimate ✅(GOLD) / publish CHƯA
```

**Xếp lớp bất biến (idempotent, hàm thuần, không sửa tay):**
`labels.csv → [3] → labels_remediated.csv → [6] → labels_final.csv`. Verdict neo theo khoá bền (book,page,bbox-IoU).

## không claim nào bị bác)

| Quyết định                                 | Bằng chứng fresh                                                       |
| --------------------------------------------- | ------------------------------------------------------------------------ |
| **㝵/người → DEMOTE, KHÔNG remap**  | Fisher p=5.4e-8; remap→𠊛 bị bác (𠊛=0 trên trang, 65% audit đúng) |
| **S3 KHÔNG làm cổng must-pass**      | AUC 0.566, mọi tín hiệu CI trùm 0.5; combo overfit vẫn ≤0.6        |
| **3 model = TRIAGE, không gán nhãn** | cả 16 cặp lỗi đều dict_ok=True → consensus/gate không tự bắt    |
| **Công bố precision THEO TIER**       | chỉ GOLD có mẫu (98.0%); SILVER/SYLLABLE chưa đo                    |

## 4. VIỆC CÒN LẠI — ưu tiên (P0/P1.1 đã XONG)

- ✅ **P0** hợp nhất 3 sách + neo lại verdict + tái sinh remediated — XONG.
- ✅ **P1.1** demote 㝵-người (precision GOLD 97.08→98.00) — XONG.
- 🔴 **P1.2 [lớn nhất] AUDIT SILVER + SYLLABLE** (17.607 crop, 0% phủ). Lấy mẫu SRS ~300–400/tier (blinded grid `ground_truth`), chấm, `estimate`. **Đánh giá:** có CI precision cho SILVER → quyết ship 'char' hay hạ 'weak'.
- 🟠 **P1.3 Audit + fix confusion tiếp** theo `confusions.csv`: cặp đã sai còn lại (liên→連, phúc→福, viết→曰...) + cặp CHƯA audit dân số lớn (đời→代 195, mới→買 183 [bị 㝵 tràn], còn→群 173, nới→尼 166). Cặp nào audit ≥3 crop & sai cao → thêm vào `confusion_fixes.yaml`.
- 🟡 **P1.4** Soát 3.850 dòng `s1_inter_s2_similar` (nhóm bắc-cầu rủi ro cao nhất trong GOLD).
- 🟡 **P1.5** Phục hồi 15 orphan (nới ngưỡng neo IoU≥0.3 + trùng nhãn) + re-audit 6 label_changed.
- 🟢 **P2** Nối `run_pipeline.sh` (§5) + hạ vai fuse_stage (bỏ PROMOTE) + publish từ labels_final.

## 5. CẤU TRÚC `run_pipeline.sh` CUỐI

| --step | Chạy                                                                               | Guard                             |
| ------ | ----------------------------------------------------------------------------------- | --------------------------------- |
| 0–2   | setup → extract(3 sách) → build_dataset + to_standard → labels.csv              | FD-cache                          |
| 3      | `remediation apply` → labels_remediated.csv                                      | luôn chạy, xác định          |
| 4      | `consensus_fusion.mine_confusions` → confusions.csv                              | không cần verdict               |
| 5      | `ground_truth sample→grid` in HTML **rồi DỪNG**                          | THỦ CÔNG (cần người chấm)   |
| 5b     | `ground_truth.reanchor_verdicts` (khi có verdict cũ) → verdicts_reanchored.csv | skip nếu không có              |
| 6      | `remediation.confusion_fix` (confusion_fixes.yaml) → labels_final.csv            | skip nếu thiếu yaml             |
| 7      | `ground_truth estimate` + `publish` từ labels_final                            | estimate skip nếu thiếu verdict |

## 6. THÀNH PHẦN DATASET CÔNG BỐ + TUYÊN BỐ TRUNG THỰC

| Tier (labels_final)           | Crop             | label_level | Tuyên bố                                                                                  |
| ----------------------------- | ---------------- | ----------- | ------------------------------------------------------------------------------------------- |
| **GOLD** (đã trừ 㝵) | **48.969** | char        | **precision 98.0%** (CI ~96.9–98.8), đo được; tách stratum quality_flag (90.9%) |
| SILVER                        | 10.856           | char        | tier YẾU tách biệt,**precision CHƯA ĐO** (P1.2)                                  |
| SYLLABLE                      | 6.751            | syllable    | distant/weak, không nhãn char                                                             |
| REVIEW + QUARANTINE           | —               | —          | LOẠI khỏi bộ có nhãn                                                                   |

**Câu cho luận văn:** *"Tier GOLD (~48,9k crop) có precision char kiểm định bằng human-audit = 98,0% (CI95 ~96,9–98,8%, n=800) sau khi demote confusion hệ thống 㝵/người. SILVER/SYLLABLE là supervision yếu/mức-âm CHƯA kiểm định precision char. Không suy rộng 98% ra toàn bộ crop."*

## 7. ĐIỀU KHÔNG LÀM (có bằng chứng)

- ❌ S3/head làm cổng chân lý (AUC 0.566). ❌ Remap 㝵→𠊛 (hỏng ~713 nhãn đúng). ❌ Majority-vote 3 model (vòng tròn).
- ❌ Công bố SILVER như GOLD (chưa audit). ❌ Gộp 97/98% ra toàn dataset. ❌ Tin exp_wrong cặp a_n=1 (cần ≥10 audit/cặp).

## Công cụ đã build (repo)

- `pipeline/consensus_fusion/{score_s3,mine_confusions,fuse_stage}.py`
- `pipeline/ground_truth/reanchor_verdicts.py` — neo verdict theo bbox-IoU
- `pipeline/remediation/confusion_fix.py` + `config/confusion_fixes.yaml` — Stage 6
- Outputs: `dataset_out/{labels_remediated,labels_final}.csv`, `dataset_out/ground_truth/{verdicts_reanchored.csv,report.json}`, `dataset_out/fusion/*`
