# CHẨN ĐOÁN "ĐANG VƯỚNG Ở ĐÂU" — LVT1883, KVK1884, Chrestomathie1872 (23/09/2026)

Nguồn: `scripts/measure/loss_ledger.py` (mới, CLI `--book`) → `measure_out/loss_ledger/` (`summary.json`, `<Book>.csv`,
`<Book>_cells.csv`, `imgq.json`, `<Book>_kimapi_*`). Chạy lại: `.venv/bin/python scripts/measure/loss_ledger.py --book all`
(13 s, 0 API, **24/24 invariants PASS**). Bản chốt đọc: `dataset_out_LucVanTien1883`, `dataset_out_KimVanKieu1884_b1`,
`dataset_out_Chrestomathie1872` (`labels_gated.csv`) + `dataset_<Book>/columns.csv` + `dict/QuocNgu_SinoNom.csv` (7.464 âm / 41.502 chữ).
Mọi số kèm n. **Không có GT người** — mọi "đúng/sai" ở đây là proxy, ghi rõ từng chỗ.

---

## 1. Sổ kế toán mất mát (`measure_out/loss_ledger/<Book>.csv`)

Mỗi ô non-GOLD được gán **đúng một** nguyên nhân gốc theo thứ tự ưu tiên
`quarantine → khongkhop → crop_bad → box_low_conf → cross_similar → QN không hợp lệ → QN sai dấu → R(âm) rỗng → cầu gần hình → kim ∉ R → chữ lạ`
(hàm `classify()`; invariant `nguyen_nhan_phu_het_non_gold` kiểm tổng = số ô non-GOLD).

| Mục | LVT1883 | KVK1884 (B1') | Chresto1872 |
|---|---|---|---|
| **0. Ô lý thuyết** (câu × chữ) | **14.616** (2.088 câu = 1.044 cặp × 14) | **22.792** (3.256 câu = 1.628 × 14) | **8.692** (`chresto_map` `nom.total_chars_est`) |
| 1. Ô thực sinh (`labels_gated.csv`) | 14.476 (99,0 %) | 22.704 (99,6 %) | 8.303 (95,5 %) |
| 1b. Không sinh ra | 140 (1,0 %) | 88 (0,4 %) | **389 (4,5 %)** |
| **2. GOLD ảnh** | **8.650** = 59,8 % ô sinh / **59,2 %** ô lý thuyết | **13.908** = 61,3 % / **61,0 %** | **5.359** = 64,5 % / **61,7 %** |
| 3. Non-GOLD | 5.826 | 8.796 | 2.944 |
| **(b) KIM đọc sai/thiếu** | **4.597 = 78,9 %** non-GOLD | **7.145 = 81,2 %** | **2.267 = 77,0 %** |
|  · kim ∈ từ điển nhưng ∉ R(âm) | 3.652 | 5.519 | 1.949 |
|  · cầu gần hình `s1_inter_s2_similar` | 569 | 601 | 235 |
|  · chữ kim KHÔNG có trong từ điển Nôm | 182 | 333 | 83 |
|  · cổng (d) khác dị bản + gần hình | 194 | 692 | — (không dị bản) |
| **(a) QN sai/thiếu** | **1.071 = 18,4 %** | **767 = 8,7 %** | **289 = 9,8 %** |
|  · R(âm) rỗng (âm sai chính tả lọt lưới) | 618 | 462 | 187 |
|  · âm không hợp lệ `not_plausible` | 341 | 247 | 28 |
|  · QN sai dấu (kim khớp sau khi sửa dấu) | 112 | 58 | 74 |
| **(c) TỪ ĐIỂN thiếu** | **≈ 0** (đo ở §4) | **≈ 0** | **≈ 0** |
| **(d) HỘP / CROP** | **144 = 2,5 %** | **693 = 7,9 %** | **245 = 8,3 %** |
|  · (a') `ink_cut`/`detector_low` → text_only | 130 | 510 | 126 |
|  · (c) crop blank/truncated → REVIEW | 14 | 183 | 119 |
| **(e) CĂN CHỈNH** | **14 = 0,2 %** | **186 = 2,1 %** | **143 = 4,9 %** |
|  · âm `khongkhop` (verse-map hỏng) | 0 | 84 | 143 |
|  · quarantine F1 (1 ảnh 2 cột) | 14 | 102 | 0 |
| Khác (low_posterior, confusion_fix) | 0 | 5 | 0 |

Cắt chéo M==N (kim = QN) có trong `<Book>_cells.csv` (cột `m_eq_n`): ví dụ LVT `B_kim_ngoai_R` 3.059 ở cột M==N vs 593 ở cột M≠N
→ **lỗi kim chủ yếu KHÔNG phải do đếm lệch chữ**, mà do đọc sai chữ trong cột đã đếm đúng.

**Kết luận mục 1: 77–81 % mất mát nằm ở khâu KIM.** QN đứng thứ hai (9–18 %), hộp/crop 2,5–8,3 %, căn chỉnh 0,2–4,9 %, từ điển ≈ 0.

---

## 2. Ảnh/chữ 3 sách mới so với STT (`measure_out/loss_ledger/imgq.json`, 20 trang/sách)

Đo trên ảnh `prepared/<book>/pages` + hộp chữ trong cache kim. "Chữ dính" = chiếu mực trong cột: tại khe giữa 2 hộp chữ
liền nhau **cùng tầng**, lấy `min` số điểm mực trên 9 dòng quanh khe; dính nếu > 0,20 × trung vị mực của cột
(n = 2.390 / 2.412 / 2.437 khe ở sách mới; 3.355 / 3.668 / 3.698 ở STT2/4/11).

| Chỉ số | LVT1883 | KVK1884 | Chresto1872 | STT2 | STT4 | STT11 |
|---|---|---|---|---|---|---|
| **Chữ dính nhau** | **15,3 %** | **5,3 %** | 69,9 % | 73,1 % | 58,4 % | 67,8 % |
| px/chữ (cao × rộng, trung vị) | **154 × 140** | 125 × 109 | 73 × 104 | 99 × 106 | 63 × 70 | 93 × 105 |
| Mật độ mực trong cột | 11,1 % | **8,0 %** | 23,2 % | 22,1 % | 15,2 % | 17,5 % |
| Bước hàng (px) · CV | 153,5 · **0,018** | 124,5 · 0,049 | 73,0 · 0,031 | 98,5 · 0,047 | 63,0 · 0,053 | 93,0 · 0,048 |
| Bước cột (px) | 158,0 | 124,2 | 139,4 | 151,5 | 97,5 | 151,6 |
| Tương phản ảnh **gốc** `data/` (Michelson · tách lớp Otsu) | 0,943 · 0,993 | 0,236 · 0,932 | 0,786 · 0,948 | *(không có ảnh gốc trong repo)* | | |
| `bleed` trên ô có crop (`labels`) | **17,8 %** | 10,4 % | **2,0 %** | 6,4 % (gộp 3 cuốn STT, `dataset_out/labels.csv`) | | |

Caveat: ảnh `prepared/` của **cả 4** kho đã nhị phân hoá (2–129 mức xám) nên "tương phản" chỉ đo được trên `data/` (STT không có bản gốc trong repo);
"chữ dính" phụ thuộc độ ôm của hộp kim (cùng một nguồn hộp cho mọi kho nên so được).

**Kết luận mục 2 — nhận định "sách mới dễ nhận diện hơn, chữ ít dính hơn" ĐÚNG với 2 sách thạch bản thơ, SAI với Chrestomathie:**
LVT 15,3 % và KVK 5,3 % chữ dính so với 58–73 % ở STT (**thấp hơn 4–14 lần**), chữ to hơn (154/125 px vs 63–99 px), bước hàng đều hơn
(CV 0,018 LVT vs 0,047–0,053 STT). Chrestomathie **không** sạch hơn (69,9 % dính, 73 px/chữ, mực trong cột 23,2 % — đậm nhất cả 6 kho).
⟹ Với LVT/KVK, phần mất mát **không nằm ở ảnh**; nó nằm ở khâu đọc chữ (§3). Bằng chứng thứ hai cùng hướng:
tỉ lệ **kim ∈ R(âm)** (điều kiện cần duy nhất để thành GOLD) là **67,3 / 70,2 / 71,3 %** ở 3 sách mới so với **58,1 %** trên STT
(n = 13.531 / 21.912 / 7.948 vs 82.612 ô có âm trong từ điển) — kim đọc thạch bản **tốt hơn** chép tay 9–13 điểm.
Riêng `bleed` LVT 17,8 % > STT 6,4 % là lỗi **kích thước hộp** (hộp cao 1,14 × bước), không phải lỗi ảnh.

---

## 3. Chất lượng OCR kim (0 API cho §3.1–3.2; §3.3 dùng 54 lượt gọi)

### 3.1 Kim theo tier (`kim_trong_R_theo_tier`, toàn sách)

| kim ∈ R(âm) | GOLD | GOLD_text_only | SYLLABLE | REVIEW | Toàn sách |
|---|---|---|---|---|---|
| LVT (n 13.531) | 100 % (8.650) | 100 % (130) | 5,1 % (2.277) | 8,2 % (2.460) | **67,3 %** |
| KVK (n 21.912) | 100 % (13.908) | 100 % (510) | 1,6 % (3.783) | 23,2 % (3.609) | **70,2 %** |
| Chresto (n 7.948) | 100 % (5.359) | 100 % (126) | 6,8 % (1.249) | 7,7 % (1.214) | **71,3 %** |

GOLD = 100 % theo định nghĩa luật (`s1_inter_s2_direct`); con số dùng được là **toàn sách**: 29–33 % vị trí kim đọc ra chữ **không thuộc** tập chữ Nôm của âm ấy.

### 3.2 Kim so với phiên âm dị bản độc lập (`measure_out/auto_precision*/cross/*/cells.csv`, bỏ ref PUA)

| Tier | LVT ↔ LVT1916 (n 3.533) | KVK ↔ 1871+1872 (n 30.235) |
|---|---|---|
| GOLD: kim == ref | 68,9 % (n 2.796) | 74,3 % (n 23.136) |
| GOLD: khác nhưng **cùng âm, đều trong R** (dị bản thật) | 25,1 % | 21,9 % |
| GOLD: gần hình với ref (nghi kim sai) | 3,8 % | 2,5 % |
| GOLD: chữ khác hẳn | 2,2 % | 1,3 % |
| **REVIEW: kim == ref** | **0,0 %** (0/417) | **0,0 %** (0/2.901) |
| REVIEW: chữ khác hẳn / gần hình | 88,0 % / 12,0 % | 86,5 % / 13,5 % |
| **SYLLABLE: kim == ref** | 0,6 % (2/317) | 1,0 % (40/4.075) |
| SYLLABLE: chữ khác hẳn | 92,7 % | 94,1 % |

Phân loại lỗi kim: **(i) đọc nhầm chữ** = đại đa số (87–94 % ô REVIEW/SYLLABLE là **chữ khác hẳn**, không phải dị thể cùng âm, không phải gần hình);
**(ii) thiếu/thừa chữ trong cột** = 85 + 84 cột (LVT) · 101 + 82 (KVK) · 61 + 66 (Chresto) trên 1.041 / 1.628 / 417 cột
(`hist_n_ocr_tru_n_qn`), tức 16,2 / 11,2 / 30,5 % số cột; **(iii) dị thể** chỉ đáng kể ở tier GOLD (21,9–25,1 %) và phần lớn là **dị bản thật giữa hai bản in**,
không phải lỗi (nền dị bản 1871↔1872 = 82,9 %, `BAO_CAO_TONG_HOP §4`).
Đối chứng trên **GT người của IHR-NomDB** (`auto_precision/ihr`, 200 patch/sách, cắt cột đúng): kim thô đúng **49,2 %** (LVT1916, n 1.393) / **42,3 %** (Kiều1872, n 1.392).

### 3.3 Hai giả thuyết cải tiến — **ĐÃ ĐO, CẢ HAI KHÔNG ĂN** (54 lượt gọi kim, cache theo md5 → chạy lại 0 lượt)

| Phép so (chỉ số = % vị trí kim ∈ R(âm), không cần GT) | LVT | KVK | Chresto |
|---|---|---|---|
| (i) kim trên **cả trang** (cache sẵn) — 10 cột/sách | 62,1 % | 66,4 % | 58,7 % |
| (i) kim trên **crop 1 cột** (30 lượt gọi) | **60,0 %** | 67,1 % | **50,7 %** |
| (i) số cột kim trả đúng số chữ (trang / crop) | 10/10 · 10/10 | 10/10 · 10/10 | 10/10 · **7/10** |
| (ii) kim trên ảnh **gốc** `data/` vs **prepared** vs **otsu(gốc)** (24 lượt, 4 trang/sách) | 64,8 / 65,5 / 65,9 | 71,9 / 72,6 / 72,7 | 73,5 / 73,5 / 73,4 |
| (ii) Jaccard tập chữ so với prepared: gốc · otsu | 0,778 · 0,858 | 0,785 · 0,872 | **1,000** · 0,866 |

n = 481 vị trí ở (i), 4 trang × 3 sách ở (ii). Đọc: **crop 1 cột không cải thiện** (−2,1 / +0,7 / −8,0 điểm; Chresto còn mất số chữ ở 3/10 cột)
→ bỏ giả thuyết "kim đọc trang rộng nên nhầm cột". **Tiền xử lý gần như không đổi kết quả văn bản** (±1,1 điểm) nhưng kim **không ổn định về chữ**:
đổi ảnh một chút thì 13–22 % tập chữ khác đi (Jaccard 0,78–0,87) mà tỉ lệ đúng vẫn ≈ 65–73 % → sai số kim là **ngẫu nhiên quanh một trần cứng**,
không phải lệch hệ thống chữa được bằng tiền xử lý. Chresto `kim_src: orig` nên "gốc" trùng ảnh đã dùng: Jaccard **1,000** = kim **tất định** trên cùng ảnh
(kiểm chứng rằng 0,78–0,87 ở 2 sách kia là do ảnh đổi, không do API).

---

## 4. Chất lượng OCR quốc ngữ (tesseract) và câu hỏi "từ điển có thiếu không"

| | LVT1883 | KVK1884 (B1') | Chresto1872 |
|---|---|---|---|
| Câu QN đúng số âm 6/8 (`qn_ocr/summary.json`) | 93,4 % (n 2.088) | 96,1 % (n 3.251) | — (văn xuôi) |
| Âm ngoài từ điển QN dự án (`oov`) | 6,53 % (945/14.476) | 3,49 % (792/22.704) | 4,28 % (355/8.303) · tesseract `oov_rate` 4,76 % (n 8.151 âm) |
| Âm không hợp lệ `not_plausible` | 2,36 % (341) | 1,46 % (331) | 2,06 % (171) |
| Cột kim thiếu / thừa âm so với QN | 85 / 84 trên 1.041 (M==N **83,8 %**) | 101 / 82 trên 1.628 (**88,8 %**) | 61 / 66 trên 417 (**69,5 %**) |
| Nguồn QN của câu | OCR 2.088 (100 %) | 1871 exact 465 · 1871 fuzzy 1.544 · OCR 1.247 | OCR (mức truyện) |
| Conf tesseract trung bình | 0,828 | 0,851 | 88,9 (TSV) |
| **Lỗi vị trí dấu thanh** (hoà→hòa, hoạ→họa, luỵ→lụy) khiến R(âm) rỗng oan | 7 ô (7 ô kim khớp ngay) | **33 ô (29 khớp)** | 1 ô (1 khớp) |

**QN lớn cỡ nào trong sổ kế toán?** 1.071 / 767 / 289 ô non-GOLD (18,4 / 8,7 / 9,8 % non-GOLD) + 140 / 88 / 389 ô không sinh ra
= trần 1.211 / 855 / 678 ô (8,3 / 3,8 / 7,8 điểm GOLD). **QN là nút thắt số 2, không phải số 1.**
Với Chrestomathie, QN là nút thắt **kép**: QN chỉ có 8.151 âm cho 8.692 chữ Nôm (thiếu 6,2 %) và 143 ô mang âm `khongkhop`.
B1' đã chứng minh đường chữa cho KVK (+882 GOLD, `BAO_CAO_TONG_HOP §5`); LVT không dùng được vì 1916 chỉ khớp 29 % câu; Chresto không có phiên âm chuẩn.

**Từ điển KHÔNG thiếu** — hai phép đo độc lập:
1. Trên mọi ô có câu khớp dị bản: chữ của bản tham chiếu **luôn nằm** trong R(âm QN) — 100,0 % ở **mọi** tier
   (LVT n 3.533; KVK 1871 n 14.962 và 1872 n 15.273; chỉ 1 ca lệch ở 1872-REVIEW). Ghép cặp (chữ tham chiếu, âm) hoàn toàn
   độc lập với từ điển (`auto_precision.match_verses` gióng câu **chỉ bằng chuỗi âm**), nên đây không phải đo vòng tròn.
2. Trên GT người IHR: `kim_eq_gt_but_not_in_R` = **0/1.393** (LVT1916) và **6/1.392** (Kiều1872) = 0–0,43 %.
⟹ Nhánh (c) "từ điển thiếu" ≈ 0. 618 / 462 / 187 ô "R(âm) rỗng" là **âm QN sai chính tả lọt qua bộ lọc hợp lệ**
(`hỗi`, `nguơn`, `mãy`, `eho`, `nụa`, `tôt`, `gọp`…), trong đó 26,7 / 33,0 / 46,3 % có biến thể khác dấu nằm trong từ điển.

---

## 5. Xếp hạng nút thắt — trần lợi ích (`summary.json → tran_loi_ich`)

Các nhánh **loại trừ nhau** nên cộng được; đây là **cận trên** (chữa hoàn hảo, giữ nguyên các khâu khác).

| Chữa hoàn hảo khâu | LVT: +ô → GOLD (% ô lý thuyết) | KVK: +ô → GOLD | Chresto: +ô → GOLD | Chi phí / khả thi |
|---|---|---|---|---|
| **1. KIM (OCR Nôm)** | **+4.597** → 13.247 (59,2 → **90,6 %**) | **+7.145** → 21.053 (61,0 → **92,4 %**) | **+2.267** → 7.626 (61,7 → **87,7 %**) | **Rất cao.** kim là API ngoài, không huấn luyện lại được. Crop-1-cột và tiền xử lý **đã đo là vô ích** (§3.3). Đường còn lại: bộ đọc thứ 2 không vòng tròn + trọng tài từ điển/ngữ cảnh, hoặc chấp nhận nhãn mức âm. Cần huấn luyện. |
| **2. QN (tesseract + verse-map)** | +1.211 → 9.861 (**67,5 %**) | +855 → 14.763 (**64,8 %**) | +678 → 6.037 (**69,5 %**) | **Thấp–trung bình, 0 API.** (a) chuẩn hoá vị trí dấu thanh: +7/+33/+1 ô ngay lập tức; (b) B1' bằng phiên âm chuẩn — đã chạy cho KVK; (c) ràng buộc 6/8 khi giải mã tesseract; (d) Chresto cần bảng truyện↔cột chặt hơn. |
| **3. HỘP / CROP** | +144 → 8.794 (60,2 %) | **+693** → 14.601 (**64,1 %**) | +245 → 5.604 (64,5 %) | **Thấp.** INTER_AREA (cùng ckpt v1) đã đo: GOLD ảnh +83 / +257 / −15, text_only còn 1/3 — **đủ chạm trần của khâu này ở KVK**; bật `detector_resize: area` cho KVK là lợi ròng. Detector v2 ≈ 1 h Kaggle. |
| **4. CĂN CHỈNH** | +14 (0,1 điểm) | +186 (0,8) | **+143** (1,6) | **Thấp, 0 API.** Chresto: 143 ô `khongkhop` là lỗi verse-map văn xuôi; KVK: 102 ô quarantine F1. |

### 5b. ĐÃ THU ĐƯỢC BAO NHIÊU PHẦN CỦA TRẦN ẤY (đo lại 23/09 sau vòng 5)

Vòng 5 chạm **đồng thời** nút thắt (1) kim (gửi `lang_type = 2` = Nôm thay 1 = Hán) và một phần nút thắt (2) QN
(luật đếm 6/8 trước DP) + (4) căn chỉnh (rào tầng DP). Kết quả **thật**, không ngoại suy
(`docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` §3.4):

| | LVT1883 | KVK1884 | Chresto1872 |
|---|---|---|---|
| GOLD ảnh trước → sau | 8.650 → **10.831** (+2.181) | 13.908 → **19.303** (+5.395) | 5.359 → **5.998** (+639) |
| % ô lý thuyết (14.616 / 22.792 / 8.692) | 59,2 → **74,1 %** | 61,0 → **84,7 %** | 61,7 → **69,0 %** |
| Trần nếu chữa HOÀN HẢO khâu kim (§5) | 13.247 = 90,6 % | 21.053 = 92,4 % | 7.626 = 87,7 % |
| **Phần trần đã lấy được** | **47,4 %** (2.181/4.597) | **75,5 %** (5.395/7.145) | **28,2 %** (639/2.267) |
| Khớp dị bản GOLD (độc lập) | 1916: 72,6 → **78,0 %** | 1872: 73,0 → **78,2 %** | không có dị bản |
| Ô sinh ra | 14.476 → 14.474 | 22.704 → **22.750** | 8.303 → **8.016** (−287) |
| M==N | 83,8 → **90,4 %** | 88,8 → **96,9 %** | 69,5 → **44,0 %** ↓↓ |

Đọc: **§5 xếp hạng đúng** — khâu kim quả là nút thắt số 1, và chỉ một tham số gọi API đã lấy được từ 28 đến 76 %
cái trần ấy mà **không** huấn luyện gì. Hai điều chỉnh lại so với bản §5 viết buổi sáng:
1. "kim là API ngoài, không huấn luyện lại được" là đúng, nhưng **không có nghĩa là không cải thiện được**: phần lớn
   mất mát ở LVT/KVK là do gọi SAI kênh ngôn ngữ, không phải do giới hạn mô hình. Trần còn lại (LVT 52,6 %, KVK 24,5 %)
   mới là phần "cần bộ đọc thứ hai".
2. Với **Chrestomathie1872** cái giá lại nằm ở chỗ §5 không đo: `lang_type = 2` làm kim đọc **ít chữ hơn** nên
   M==N tụt 69,5 → 44,0 % và mất 287 ô sinh ra; GOLD vẫn tăng nhưng sách này không có dị bản để kiểm chéo.

**Ba nút thắt lớn nhất, theo đúng thứ tự: (1) kim đọc sai chữ — 78,9 / 81,2 / 77,0 % toàn bộ mất mát, trần +31,5 / +31,3 / +26,1 điểm GOLD;
(2) QN — 8,3 / 3,8 / 7,8 điểm; (3) hộp/crop — 1,0 / 3,0 / 2,8 điểm.** Hai khâu (2) và (3) cộng lại vẫn **nhỏ hơn 1/3** khâu (1).

---

## 6. Giới hạn của chính bản chẩn đoán này

1. **Không có GT người.** "kim sai" ở §3.2 nghĩa là *khác bản in dị bản*; nền dị bản 1871↔1872 chỉ 82,9 % nên con số GOLD 68,9–74,3 % là **cận dưới**.
   Ngược lại REVIEW/SYLLABLE 0,0–1,0 % trùng ref thì thấp hơn nền quá xa để giải thích bằng dị bản → kết luận "kim sai" ở nhánh này vững.
2. Nhánh nguyên nhân là **quy trách nhiệm theo thứ tự ưu tiên**: một ô vừa hộp xấu vừa kim sai được tính cho **hộp** (cổng chạy trước).
   Vậy trần của khâu kim (§5) là **cận dưới**, trần của khâu hộp là **cận trên**.
3. Ô lý thuyết Chresto (8.692) là **ước lượng theo bước chữ** (`chresto_map`, khớp pitch↔run 94,7 %), không phải đếm tay → mục "1b. không sinh 389" của Chresto có sai số.
4. "Chữ dính" đo bằng hộp kim: chỗ kim bỏ sót chữ thì khe ấy không được đếm → chỉ số **lạc quan** với sách kim đọc kém. Trang mẫu 20/sách (CI ≈ ±2 điểm).
5. §3.3 chỉ 481 vị trí / 12 trang: đủ bác bỏ "crop 1 cột giúp +10 điểm", **không** đủ bác bỏ mức lợi 1–2 điểm.
6. Mọi so sánh với STT dùng `dataset_out/labels.csv` (chưa qua cổng B4') còn sách mới dùng `labels_gated.csv` (đã qua cổng) — tier STT do đó **rộng hơn**
   (còn cả `confusion_fix`, `corpus_reading`, `lop_nham`: 1.611 dòng), trong khi sách mới GOLD **chỉ** từ `s1_inter_s2_direct`.

## 7. Việc nên làm tiếp (theo lợi ích/chi phí)

> **Cập nhật 23/09 sau vòng 5**: mục 3 ("khâu kim — cần thiết kế") đã có lời giải rẻ nhất và **đã thực thi**
> (`lang_type = 2`, §5b). Mục 1 (chuẩn hoá vị trí dấu thanh) vẫn chưa làm; mục 2 (`detector_resize: area`) vẫn chưa bật;
> mục 4 (SYLLABLE là dữ liệu mức âm dùng được) nay nhỏ đi vì SYLLABLE giảm mạnh (2.363 → 1.207 · 3.844 → 618 · 1.286 → 856)
> — phần lớn các ô ấy đã lên GOLD.

1. **0 API, ~1 giờ**: chuẩn hoá vị trí dấu thanh trước khi tra R(âm) (+7/+33/+1 ô, `loi_vi_tri_dau_thanh`); sửa verse-map văn xuôi Chresto (143 ô).
2. **0 API, đã có số**: bật `detector_resize: area` cho KVK1884 (khâu hộp ở KVK là nút thắt số 2 thật sự: 693 ô = 7,9 % non-GOLD).
3. **Cần thiết kế**: khâu kim. Trước khi bỏ công, lưu ý §3.3 đã loại 2 giả thuyết rẻ nhất; và §4 đã loại hẳn giả thuyết "mở rộng từ điển".
4. Nếu muốn tăng GOLD **ngay** mà không đụng kim: 2.363 / 3.844 / 1.286 ô SYLLABLE hiện có nhãn mức âm đúng (kim sai chữ) —
   đây là dữ liệu huấn luyện **mức âm** dùng được, không phải ô hỏng.
