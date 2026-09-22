# Phương án gán nhãn HOÀN TOÀN tự động cho sách thạch bản mới — đo độ đúng không cần người kiểm (2026-09-22)

Ràng buộc đề tài: không có người kiểm. Mọi con số về độ đúng dưới đây đến từ **đối chứng độc lập có sẵn trên máy**, do
`scripts/measure/auto_precision.py` sinh (tái lập, 0 token; kim cache theo md5 patch → chạy lại 0 lần gọi API):
`measure_out/auto_precision/{SUMMARY.json,REPORT.md, ihr/, cross/, gates/, kim_cache/, kim_calls.json}`.
Luật gán nhãn đối chiếu: `pipeline/align_engine/consensus.py` (GOLD = chữ kim ∈ tập đọc âm từ điển của âm QN; SYLLABLE/REVIEW theo `tier_v3.py`).

## 0. Ba con số thay cho người kiểm (đọc kèm §1–§3 để biết mỗi số đo CÁI GÌ)

| Phép đo | Đo cái gì | KVK1884 / TruyenKieu | LVT1883 / LucVanTien |
|---|---|---|---|
| (1) GT độc lập IHR-NomDB (mộc bản 1872 / 1916) | precision **văn bản** của luật GOLD (kim ∈ dict) trên cột đã cắt đúng, so với nom_text người soạn | **84,6 %** [81,7–87,1] (cov 49,5 %, n 689) | **89,3 %** [86,9–91,3] (cov 55,1 %, n 768) |
| (2) Đối chứng dị bản (thạch bản thật) | nhãn GOLD của pipeline == chữ Nôm cùng câu/cùng vị trí ở **dị bản khác** (1871 / 1916) | **80,4 %** [79,5–81,2] (n 8.403, bỏ ref PUA); nền 1871↔1872 chỉ **82,9 %** | **72,6 %** [71,0–74,3] (n 2.796) |
| (3) Cổng máy | GOLD bị hạ / proxy (2) trước→sau | tốt nhất `box+bridge+crop`: 76,5 → 77,7 %, coverage 81,6 % | `box+bridge`: 71,3 → 72,1 %, coverage 86,0 % |

Không phép đo nào ở đây đo được **ảnh crop có đúng chữ không** (lỗi vị trí hộp/F1 cross-col); §6.

## 1. (1) Precision trên GT độc lập — mộc bản IHR-NomDB (`ihr/`)

Thiết kế: `data/{LucVanTien1916,TruyenKieu1872}/manifest.tsv`, chỉ patch `len(nom_text)==n_qn_syll`, Nôm toàn CJK, có ảnh patch
(hợp lệ 1.943 / 3.048); mẫu **phân tầng theo trang, phân bổ tỷ lệ, seed 20260922, 200 patch/sách**. Kim (upload+recognize
`core.ocr.ocr_api`) ở độ phân giải gốc (~34 px/chữ) đọc **thiếu** (pilot ×1: LucVanTien1916: partial 4; TruyenKieu1872: partial 2, empty 2) → đo chính ở **phóng ×3 LANCZOS**
(≈100–130 px/chữ, cỡ chữ thạch bản). Chuỗi kim theo y; nếu số chữ ≠ N → căn chỉnh đơn điệu NW (match = kim ∈ R(âm)).
Luật tối thiểu: GOLD nếu kim ∈ R(âm) (nhãn = kim); GOLD_bridge nếu cầu duy nhất `SinoNom_Similar`∩R (luật `s1_inter_s2_similar`);
SYLLABLE-1 nếu |R| = 1 (proxy — pipeline thật không gán chữ ở tầng SYLLABLE); còn lại REVIEW. So `nom_text[i]` NFC; **không có bảng
tương đương PUA↔chuẩn trong repo** → vị trí GT là PUA/ExtG (U+F0000+, ≥U+30000) tách riêng.

| | TruyenKieu1872 (mộc bản) | LucVanTien1916 (mộc bản) |
|---|---|---|
| patch / vị trí / GT PUA-ExtG | 200 / 1392 / 79 | 200 / 1393 / 48 |
| kim ×3: rỗng / n_kim==N / n_kim≠N | 11 / 105 / 84 | 9 / 115 / 76 |
| **kim đúng thô** (chữ kim == GT) | **42,3 %** (44,9 % bỏ PUA); kim ∈ R 49,5 % | **49,2 %** (51,0 % bỏ PUA); kim ∈ R 55,1 % |
| **GOLD direct: coverage · precision [Wilson 95 %]** | cov 49,5 % · **84,6 %** [81,7–87,1] (n 689) | cov 55,1 % · **89,3 %** [86,9–91,3] (n 768) |
| GOLD direct, bỏ vị trí GT PUA | **86,6 %** [83,8–89,0] (n 673) | **90,5 %** [88,2–92,4] (n 758) |
| GOLD + cầu (`s1_inter_s2_similar`) | cov 54,7 % · 83,6 % [80,8–86,1]; riêng cầu: 74,0 % (n 73) | cov 60,0 % · 88,4 % [86,0–90,4]; riêng cầu: 77,9 % (n 68) |
| SYLLABLE-1 (|R|=1): n · precision | 1 · 100,0 % [20,7–100,0] | 4 · 100,0 % [51,0–100,0] |
| REVIEW | 45,2 % | 39,7 % |
| Lỗi GOLD theo loại | dong_am_di_the 90, gt_pua_extG 16 | dong_am_di_the 72, gt_pua_extG 10 |

Đọc: `dong_am_di_the` = kim sai nhưng vẫn ∈ R (GT là một cách đọc khác của cùng âm) — luật GOLD **không thể** bắt vì cả hai
chữ đều "đúng từ điển"; `gt_pua_extG` = không so được; `gt_not_in_R` = GT không có trong dict (dict thiếu, kim chọn chữ khác).
**Kết quả đo**: 100 % lỗi GOLD (72 + 90 ô) là `dong_am_di_the` hoặc GT PUA, **0** ca `gt_not_in_R` → luật GOLD không tạo lỗi "ngoài từ điển";
lỗi còn lại là kim chọn chữ đồng âm khác — cùng bản chất với 19,6 % bất đồng dị bản ở (2). Coverage GOLD chỉ ~50–55 % vì kim ×3 vẫn
đọc thiếu chữ (n_kim≠N 76–84/200 patch) trên mộc bản; luật cầu kém hơn direct ~11–12 điểm (74–78 %, n ≈ 70).
Caveat bắt buộc: (a) mộc bản 1872/1916 ≠ thạch bản 1883/1884 (khác miền in, kim là mô hình học chữ viết tay/in); (b) patch đã
cắt đúng cột → số này là precision **văn bản có điều kiện cắt đúng**, không gồm lỗi hộp; (c) phóng ×3 là điều kiện đo, không phải
điều kiện pipeline; (d) n = 200 patch/sách → CI ≈ ±2 điểm.

## 2. (2) Đối chứng chéo dị bản trên thạch bản — 0 API (`cross/`)

Cách làm: câu QN OCR của sách thạch bản (`prepared/<book>/transcriptions/page_*.json`, verse_odd/even) khớp câu tham chiếu
(json Nôm Foundation) khi **giống hệt** sau chuẩn hoá (`clean_line_text`/`split_to_syllables`/`normalize_tone_marks`/lower — như
pipeline) hoặc **≥ 75 % âm cùng vị trí, ứng viên duy nhất**; chỉ so ô ở **vị trí âm giống hệt**. Ô = `labels_final.csv` cùng
(page, column, syl_idx). Kết quả exact-only và ≥75 % **trùng nhau** (KVK 76,4 vs 76,5 %; LVT 71,3 vs 71,3 %) → khớp mềm không chệch.

| Sách ↔ tham chiếu | câu khớp (exact) | ô GOLD so | GOLD == ref | bỏ ref PUA | bất đồng: hình gần (Similar) | kim==ref theo tier |
|---|---|---|---|---|---|---|
| KVK1884 ↔ Kiều **1871** Liễu Văn Đường | 1.898 / 3.256 (462) | 8.830 | 76,5 % [75,6–77,4] | **80,4 %** [79,5–81,2] (n 8.403) | 22,7 % của 2.077 | GOLD 73,9 · SYLLABLE 0,5 · REVIEW 0,0 |
| KVK1884 ↔ Kiều **1872** Duy Minh Thị | 1.968 (491) | 9.216 | 69,1 % | 72,3 % | 24,4 % | GOLD 66,8 |
| **nền**: 1871 ↔ 1872 cùng câu, cùng vị trí, bỏ PUA | — | 9.961 vị trí | **82,9 %** | | | |
| LVT1883 ↔ LVT **1916** Nôm Foundation | 609 / 2.088 (121) | 2.850 | 71,3 % [69,6–72,9] | **72,6 %** [71,0–74,3] (n 2.796) | 23,7 % của 819 | GOLD 67,6 · SYLLABLE 0,6 · REVIEW 0,0 |

Đọc số: 100 % ô GOLD bất đồng có ref ∈ R(âm) (theo định nghĩa GOLD) → bất đồng là **đồng âm dị thể**, không phân biệt được
"kim đọc nhầm chữ đồng âm" với "dị bản dùng chữ khác" nếu không nhìn ảnh. Hai mốc tự động kẹp lại: (i) hai dị bản tham chiếu
1871↔1872 chỉ cùng chữ **82,9 %** → khớp 80,4 % của GOLD KVK gần trần dị bản (ước lượng thô lỗi văn bản GOLD ≈ 1 − 80,4/82,9 ≈ 3 %,
giả định độc lập); (ii) trong bất đồng chỉ 22,7–24,4 % là cặp **hình gần** (khả năng kim nhầm) → trần lỗi kim ≈ 0,23 × 19,6 % ≈ 4,5 %
(KVK), ≈ 0,24 × 27,4 % ≈ 6,5 % (LVT). LVT thấp hơn vì 1916 (Nôm Foundation) cách 1883 xa hơn về chính tả Nôm và QN OCR LVT xấu
(chỉ 29 % câu khớp). Tier SYLLABLE/REVIEW: chữ kim == ref 0–0,6 % → tier phân tách đúng chỗ "kim không đáng tin".
20 ô bất đồng mẫu mỗi sách (seed 20260922): `cross/<book>/disagreements_sample20.csv` — chủ yếu dị thể Unicode/dị bản
(台/𠄩 hai, 時/𣈜 ngày, 羅/𦋦 ra, 别/別, 沒/没, 為/爲, 難/难), vài ca luật cầu (滝/𥪞 trong, 偵/𥢆 riêng) đáng nghi.

## 3. (3) Cổng máy thay người (`gates/`) — GOLD `labels_final.csv`; proxy = khớp ref (2) trên ô GOLD nằm trong câu khớp

| Cổng | KVK1884: GOLD hạ / coverage / proxy | LVT1883: GOLD hạ / coverage / proxy |
|---|---|---|
| (none) | 0 / 100 % / 76,5 % | 0 / 100 % / 71,3 % |
| (i) n_det ≠ N → REVIEW | 6.060 / 59,8 % / 77,2 % | 3.258 / 66,3 % / 71,6 % |
| n_ocr ≠ n_qn (M≠N) | 1.658 / 89,0 % / 76,6 % | 1.434 / 85,2 % / 71,2 % |
| count_source = conflict | 5.319 / 64,7 % / 77,2 % | 2.438 / 74,8 % / 71,3 % |
| box_source ≠ detector | 2.094 / 86,1 % / 77,1 % | 815 / 91,6 % / 71,4 % |
| (ii) luật cầu `s1_inter_s2_similar` (kim ∉ dict) | 638 / 95,8 % / 77,0 % | 573 / 94,1 % / 71,9 % |
| `direct_am_sua_dau` (âm QN bị sửa dấu) | 124 / 99,2 % / 76,5 % | 112 / 98,8 % / 71,3 % |
| crop blank/truncated · bleed | 195 · 1.575 | 13 · 1.779 |
| (iii) độ tin cậy kim | **không có**: `kim_raw/*.json` chỉ transcription/points/`difficult` (0 hộp difficult=true) | như KVK |
| (iv) bất đồng dị bản → REVIEW | hạ 1.651 (18,7 % ô GOLD trong câu khớp; câu khớp phủ 58,6 % GOLD) — proxy sau = 100 % **theo định nghĩa**, không báo | hạ 765 (26,8 %; phủ 29,5 %) |
| **Tốt nhất, coverage ≥ 80 %** (quét 2^10 tổ hợp) | `box+bridge+crop`: hạ 2.771, cov 81,6 %, **77,7 %** [76,7–78,6] | `box+bridge`: hạ 1.354, cov 86,0 %, **72,1 %** [70,3–73,8] |

Kết luận (3): mọi cổng chỉ dịch proxy ≤ +1,2 điểm, **trong CI** → cổng (i)/conflict/box nhắm lỗi **vị trí hộp**, thứ mà proxy
văn bản (1)(2) **mù**; chỉ (ii) cầu và (iv) là cổng văn bản thật. Không nên chọn cổng bằng proxy này; chọn bằng lý do cơ chế (§4).

## 4. Phương án chuẩn cho sách mới không người kiểm

B0–B5 giữ như `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` §2 (config theo sách, `--plan-only` chọn verse-map, ingest kim, build
`--two-pass --box-rule syl_index`, remediation, export), thêm:
1. **B1'**: nếu có phiên âm chuẩn (Nôm Foundation/IHR) cho sách hoặc dị bản gần → dùng làm `verses.tsv` thay OCR QN (chữa 117–151 cột QN lệch số âm; nâng số câu khớp (2)).
2. **B4' cổng máy bắt buộc (cơ chế, không theo proxy)**: (a) `n_det ≠ N` hoặc `box_source ∈ {midpoint, split}` → **không export ảnh** ô đó vào GOLD (giữ nhãn văn bản ở tầng `GOLD_text_only`), vì lỗi hộp không đo tự động được; (b) `s1_inter_s2_similar` → SYLLABLE (kim ∉ dict, chữ cầu chưa có bằng chứng độc lập; (1) cho precision cầu 74,0 % (n 73, TruyenKieu1872)); (c) `direct_am_sua_dau` → SYLLABLE; (d) crop blank/truncated → REVIEW; (e) có dị bản → ô GOLD **bất đồng không hình-gần** giữ GOLD kèm cờ `di_ban_khac`, **bất đồng hình-gần** (SinoNom_Similar) → REVIEW (nghi kim nhầm chữ đồng âm).
3. **B6 báo cáo độ đúng tự động** (thay mục "người kiểm" §4 của HUONG_DAN): `auto_precision.py --all` → (1) precision văn bản trên GT độc lập cùng miền in gần nhất (mộc bản IHR khi chưa có GT thạch bản; ≤ 420 lần gọi kim, cache); (2) khớp dị bản + **nền dị bản↔dị bản** cùng bảng; (3) bảng cổng. Sách không có dị bản số hoá → chỉ (1) + cổng cơ chế.
4. Ngân sách: (1) 410 lần gọi kim/2 sách (~30 phút); (2)(3) 8 s CPU.

## 5. Trình bày trong luận văn — gọi đúng tên

- "**Precision văn bản của luật GOLD trên GT độc lập** (IHR-NomDB, mộc bản, n = 200 patch/sách, cột cắt đúng, kim ×3)": **84,6 %** [81,7–87,1] (cov 49,5 %, n 689) / **89,3 %** [86,9–91,3] (cov 55,1 %, n 768). KHÔNG gọi là precision của bộ dữ liệu thạch bản.
- "**Tỉ lệ khớp dị bản** của GOLD trên thạch bản": 80,4 % (KVK↔1871), 72,6 % (LVT↔1916), so với **nền dị bản 82,9 %** (1871↔1872) → GOLD KVK ở mức "không phân biệt được với một dị bản khác"; đây là **cận dưới** của precision, không phải precision.
- "**Trần lỗi kim đồng âm**" ≈ 4,5 % (KVK) / 6,5 % (LVT) từ tỉ lệ bất đồng hình-gần — ước lượng có giả định, ghi rõ.
- Không có số nào cho **độ đúng ảnh crop**; ghi rõ I5 (n_det==N) 59,2 % / 65,2 % và F1 cross-col census là chỉ số thay thế.
- Số cũ của STT (99 % GOLD sau audit máy) không được trích làm precision cho sách mới.

## 6. Việc vẫn KHÔNG đo tự động được

1. Ô GOLD có ảnh cắt sai chữ (hộp lệch ±1, midpoint/split) — không có GT hộp thạch bản; chỉ hạn chế bằng cổng cơ chế B4'(a).
2. Phân biệt "kim nhầm chữ đồng âm" với "dị bản dùng chữ khác" khi hai chữ đều ∈ R(âm) và không hình-gần.
3. Vị trí GT là PUA/ExtG (IHR: 79+48 vị trí; ref KVK 427 ô GOLD) — thiếu bảng tương đương.
4. Dị bản gần cho sách bất kỳ (KVK có 1871/1872, LVT chỉ 1916 xa hơn) — sách không có dị bản số hoá chỉ còn (1).
5. Chuyển miền mộc bản → thạch bản của chính phép đo (1); GT thạch bản chỉ có khi có người soạn.

## 7. Tái lập

```bash
.venv/bin/python scripts/measure/auto_precision.py --steps cross,gates          # 0 API, ~8 s
.venv/bin/python scripts/measure/auto_precision.py --steps ihr --budget 420     # kim; cache → 0 lần gọi khi chạy lại
.venv/bin/python scripts/measure/auto_precision.py --all --report-only          # gom SUMMARY.json / REPORT.md
```
Lần gọi kim tích luỹ: `measure_out/auto_precision/kim_calls.json` (410 / 420, gồm 2 pilot tay). Không sửa `pipeline/`, `core/`, `data/`; chưa commit.
