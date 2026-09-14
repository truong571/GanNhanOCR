# GÁN NHÃN NÔM ↔ QUỐC NGỮ — NGUYÊN NHÂN, KHAI THÁC TỪ ĐIỂN, KIỂM LÂN CẬN

**2026-09-13** · tiếp nối `docs/RA_SOAT_TOAN_BO_2026-09-13.md` (rà toàn bộ), tệp này chỉ nói về
**khâu gán nhãn**: vì sao nó chưa tốt, tận dụng từ điển đến đâu là hết, và làm sao biết một ô gán
đúng hay "đọc bị chạy" mà không cần chấm tay từng ô.

Mọi con số tái sinh bằng một lệnh, tất định, chỉ đọc repo:

```
.venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem.py all     # ~50 s, không cần GPU
.venv/bin/python lab/gan_nhan_2026-09-13/thuc_nghiem.py geo     # +40 s, cần detector (MPS/CPU)
```

Dữ liệu: 4.029 cột dựng lại từ `prepared/` bằng đúng hàm sản xuất (`nom_cols_hybrid` →
`_get_qn_lines` → `normalize_column` → `realign_column`), 4.022 cột join khớp từng ô với
`dataset_out/labels_final.csv` (7 cột lệch do dò cột khác nhánh, bỏ qua).

---

## 0. TRẢ LỜI NGẮN

| Câu hỏi | Trả lời | Bằng chứng chính |
|---|---|---|
| **Vì sao chưa tốt?** | Sáu nguyên nhân (§2). Hai nguyên nhân *cơ chế* quan trọng nhất: (N1) ma trận chi phí coi **một khe rẻ hơn một lần đọc sai** nên thanh ghi "chạy" chỉ vì một xác nhận từ điển tình cờ; (N2) hộp ảnh gán theo **tâm tổng hợp** của OCR, tách rời khỏi khe mà DP văn bản tìm ra. | 978 cặp lệch đường chéo trong cột *khớp số*, 604 là GOLD; đổi sang chi phí hiệu chuẩn theo likelihood: 975 → 503, cột có khe giả 345 → 54; hộp đúng trong cột rụng chữ 59% → 91% khi gán theo thứ tự sau khe |
| **Tận dụng từ điển tối đa thế nào?** | Từ điển tra thẳng đã *bão hoà* (Hán-Việt, Unihan kVietnamese thêm **0** ô). Nguồn còn lớn là **ngữ liệu tự nó**: cặp (chữ OCR, âm) lặp ở trang khác và **ngữ cảnh 2-gram** — độ phủ 8.000–17.000 ô với lift 58–371, tức mạnh hơn cầu tự dạng đang dùng (lift 10) từ 5 đến 30 lần. Nạp chúng **đệ quy** lại vào DP: khe đúng chỗ 60% → 85%, ghép sai khi OCR rụng chữ 3,9% → 1,4%. | bảng §3 |
| **Gán như thế có đúng không?** | Phần *ghép âm* (thanh ghi): GOLD-trực-tiếp ghép sai ≈ 1,3–1,7%, có thể đưa về 0,5% bằng xác suất hậu nghiệm. Phần *mã chữ*: cầu tự dạng bị bộ kiểm độc lập nghi sai **18,7%** (trực tiếp 5,2%); 128 lớp nhãn thiểu số nhìn-giống-nhãn-đa-số (1.453 ô GOLD) là ứng viên nhầm hệ thống kiểu 㝵/người. Precision mã chữ của GOLD ước **93–96%**, không phải "~99%" như docstring `consensus.py`. | §4, §5 |
| **"Đệ quy vài từ lân cận"?** | Đúng hướng và đo được: xác suất hậu nghiệm forward–backward (chính là tổng hợp bằng chứng của các ô lân cận) bắt 78% cặp ghép sai ở ngưỡng 0,5 (AUC 0,88); neo ngữ liệu từ trang khác nạp lại DP cắt lỗi thanh ghi 2,5–2,8 lần. Kiểm lệch cục bộ đơn giản (cửa sổ ±2) thì *không đủ*: chỉ bắt 7–14%. | §4 |

---

## 1. PHƯƠNG PHÁP KIỂM — năm phép đo không cần phán quyết người

Đề tài chưa có phán quyết người ngoài QĐ-01. Để đánh giá được vẫn cần **đáp án**, nên tôi dùng năm
nguồn đáp án gián tiếp, mỗi nguồn trả lời một câu khác nhau:

1. **Benchmark có đáp án** — lấy cột khớp số (m = n, đường chéo đúng ≈ 99,5%), *gây hỏng có chủ
   ý* (OCR rụng 1–2 chữ, tách đôi chữ, đọc nhầm chữ, QN rụng âm), chạy lại đúng mã sản xuất, chấm
   từng cặp ghép theo đáp án. Đo được **tỉ lệ ghép sai theo tier** và **khe đặt đúng chỗ**.
2. **Null lệch thanh ghi** — mỗi luật từ điển được bắn thử trên cặp (c_i, s_{j+1}) thay vì
   (c_i, s_j). Tỉ số bắn đúng / bắn lệch = **lift** = luật ấy tin được đến đâu để làm bằng chứng
   ghép. Không cần biết nhãn đúng.
3. **Xác suất hậu nghiệm** — forward–backward trên đúng lưới NW của sản xuất → P(ghép i↔j) cho
   từng ô, tức "các ô lân cận có ủng hộ ô này không" ở dạng có xác suất.
4. **Bộ kiểm độc lập** `re-dataset/check/` (từ điển ngoài, không dùng CSV của pipeline) — tham
   chiếu *không vòng tròn* cho câu hỏi mã chữ, dù bản thân nó là heuristic.
5. **Soi mắt có chọn lọc** — 24 + 24 + 24 ô cho ba giả thuyết (§2.2, §2.3, §4.3), chỉ để định hướng.

---

## 2. NGUYÊN NHÂN GỐC — sáu điểm, xếp theo mức tác động

### N1 · Ma trận chi phí tin rằng OCR rụng chữ thường hơn OCR đọc sai

`anchor_align.py:33-39`: khe (del/ins) = 0,7 < đọc-sai-từ-điển = 1,0. Đo trên ngữ liệu: một cặp
ghép đúng thì **không** khớp từ điển ở 41% trường hợp, còn OCR rụng/thừa chữ chỉ ~1–4% vị trí. Tỉ
số khả năng đúng phải là: khớp trực tiếp +5,1 nat, cầu tự dạng +2,6, không khớp −1,6, khe ≈ −3,5.
Ma trận đang dùng đảo ngược thứ tự này.

Hệ quả cơ học: với hai ô liền nhau cùng không khớp (chi phí 2,0), DP **mở một khe đôi** (0,7 + 0,7
+ 0 = 1,4) ngay khi có *một* xác nhận tình cờ ở thanh ghi lệch 1 — mà xác nhận tình cờ xảy ra
0,36% (trực tiếp) đến 1,5% (cầu tự dạng) mỗi ô. Đây chính là "đọc chữ bị chạy".

Ba cột thật (in đầy đủ bằng `thuc_nghiem.py posterior`, mục C; bảng dưới rút gọn):

```
stt2 page_0012 c5   OCR 裙(5) 署(6)  ↔  nước Rô        production: xoá 裙, ghép 署↔nước  → GOLD  ✗ (crop là "Rô")
stt2 page_0022 c5   OCR 摺(7) 打(8)  ↔  khỏi chết      production: ghép 摺↔chết (摺∈R!), xoá 打 → GOLD ✗
                                                        đường chéo: 打↔chết, cầu 折 ∈ R(chết) → GOLD ✓
stt2 page_0016 c8   OCR 衣薔移君車恪 ↔ I ghê rê xa nơi khác   m = n nhưng OCR vừa rụng vừa thừa: văn bản KHÔNG quyết được
```

Đo toàn ngữ liệu (cột m = n, 2.928 cột / 60.846 cặp):

| | production | hiệu chuẩn theo likelihood |
|---|---|---|
| cặp lệch đường chéo | **978 (1,61%)** — 604 GOLD *(tập cột ≥ 10 chữ dùng để so: 975)* | 503 (0,83%) |
| cột có khe giả | 345 | **54** |
| benchmark rụng 1 chữ: ghép sai | 4,67% | 3,82% |
| benchmark đọc nhầm 1 chữ: ghép sai | 2,09% | **1,15%** |
| khe đặt đúng chỗ (rụng 1 chữ) | 265/500 | 303/500 |
| ô giao nộp **đổi âm ghép** nếu đổi ma trận | — | GOLD 439 (0,81%) · SYLLABLE 42 · REVIEW 506 |

T3 (08-24) kết luận "giữ ma trận" vì `anchor_retention` không đổi — thước đo ấy chấm việc *giữ được
neo đã có*, không chấm việc *tạo khe giả*; và nó tiêm 1 indel/cột (≈5%/chữ, gấp 2–5 lần thật) nên
thiên về khe rẻ. Ma trận hiệu chuẩn ở đây (0 / 2,5 / 5,1 / 6,7 / khe 8,6) là −log tỉ số khả năng
đo trực tiếp, chưa tinh chỉnh — chỉ cần đúng *thứ tự*.

### N2 · Hộp ảnh ghép theo tâm tổng hợp, không theo khe của DP

`ocr_api.py:398` chia đều hộp cột cho số chữ OCR *đọc được* → khi OCR rụng một chữ, mọi tâm phía
dưới trôi dần tới một ô. `_monotone_assign` (`align_production.py:325`) gán hộp CenterNet cho các
tâm trôi ấy và thay hộp lệch > 0,35 bước bằng hộp midpoint (cũng dựng từ tâm trôi). Trong khi đó DP
văn bản *đã biết* khe ở đâu — nhưng không ai dùng.

Đo trên 42 trang, 171 cột khớp số mà detector đếm đúng, rụng 1 chữ OCR có chủ ý (3.343 ô):

| cách gán hộp | đúng hộp | hộp của chữ **khác** | rơi midpoint |
|---|---|---|---|
| production `_monotone_assign` | **59,1%** | 11,2% | 29,7% |
| theo **thứ tự** sau khe của DP văn bản | **91,0%** | — | — |

Trần 91% = độ chính xác đặt khe của DP (64% cột đúng tuyệt đối; sai thì chỉ vài ô quanh khe lệch).
Tâm tổng hợp *tự nó* không định vị được khe: phép gán "bỏ một hộp" theo tâm chỉ đúng 4% (`geo`).
Kết luận: hình học cho **số** và **vị trí** glyph; văn bản (từ điển + ngữ liệu) cho **khe**. Ghép
hai thứ theo thứ tự chỉ số, không theo khoảng cách tâm.

### N3 · Từ điển làm chân lý trong khi OCR thiên về khối URO

|R(âm)| trung vị 20, p90 55. OCR nhầm theo *tự dạng*, dị thể Nôm của cùng âm lại chia sẻ thành
phần, và kinhhannom hầu như không xuất Ext-B (𠊚, 𣈜, 𠰺 → 㝵, 時, 代). Nên "c ∈ R(âm)" xác nhận
cả nhãn *sai-nhưng-hợp-từ-điển*. Trên GOLD: 128 lớp (sách, âm, nhãn thiểu số) mà nhãn thiểu số
nhìn giống nhãn đa số, 1.453 ô — bộ kiểm độc lập nghi sai 22% các lớp này (10% với lớp không nhìn
giống). Ví dụ đã được bộ kiểm ngoài gắn cờ độc lập: (cùng 共→其) 158 ô · (xin 嗔↔真↔填) 84 ·
(được 特→時) 39 · (một 没→殳) 31 · (đến 旦→且) 15 · (ngày 𣈜→時) 64.

### N4 · Cầu tự dạng là luật yếu, được dùng như luật mạnh

Lift so với null lệch thanh ghi: trực tiếp **161**, cầu tự dạng duy nhất **15** (trên ô chưa xác
nhận: 10), cầu ngược L2: 9, hai bước (T7): **2**. Bộ kiểm độc lập nghi sai 18,7% ô cầu xuôi so với
5,2% ô trực tiếp. Cầu tự dạng chỉ đáng tin khi có ngữ cảnh đi kèm (§3).

### N5 · Cổng SYLLABLE đo độ thuần của chữ OCR — thứ vốn là nhiễu

`syllable_gate` đòi chữ OCR đọc một âm ≥ 60% số lần nó xuất hiện. Các "chữ hút" của kinhhannom
(在 君 尺 石 乇 要 用) đại diện cho nhiều glyph thảo khác nhau nên không bao giờ thuần, dù âm ở từng
ô vẫn đúng. Trong khi đó bằng chứng ngữ cảnh (2-gram lặp ở trang khác) có lift 85–371 và chưa được
dùng ở đâu.

### N6 · L1 sửa lại âm Quốc ngữ thật để vừa từ điển

729/755 ô L1 có âm gốc là từ có trong từ điển (nhiều→nhiêu ×67, vồ→vô ×70). Đã nêu ở rà soát
toàn bộ; nhắc lại vì nó là biểu hiện của N3: quan sát bị bẻ theo tiên nghiệm.

---

## 3. TẬN DỤNG TỐI ĐA NGUỒN TỪ ĐIỂN — đo từng nguồn, không đoán

Trên 57.875 cặp của cột khớp số (24.009 cặp có chữ OCR ∉ R(âm)). "Phủ thêm" = số ô *chưa xác
nhận* mà luật bắn; "null" = số lần bắn ở thanh ghi lệch 1; lift = tỉ số.

| Nguồn / luật | bắn đúng | bắn lệch | **lift** | phủ thêm | null | **lift (ô chưa xác nhận)** |
|---|---|---|---|---|---|---|
| **Từ điển tra thẳng** c ∈ R(âm) | 58,5% | 0,36% | 161 | — | — | — |
| Cầu tự dạng duy nhất (GOLD hiện hành) | 21,4% | 1,40% | 15 | 3.025 | 303 | **10** |
| Cầu tự dạng *tương hỗ* duy nhất | 23,8% | 0,78% | 30 | 2.496 | 167 | 15 |
| Cầu ngược L2 | 25,6% | 1,37% | 19 | 2.949 | 332 | 9 |
| Cầu hai bước (T7) | 18,5% | 10,95% | **2** | 4.524 | 2.517 | 2 |
| Âm cùng thanh khác (c ∈ R(s~thanh)) | 33,6% | 0,34% | 99 | 1.263 | 70 | 18 |
| Âm cùng khung xương (bỏ mọi dấu) | 42,0% | 0,61% | 69 | 1.618 | 108 | 15 |
| Hán-Việt (han_raw, 11.153 chữ) đúng âm | 18,1% | 0,02% | 1.045 | **1** | 5 | — |
| Hán-Việt ~thanh | 25,6% | 0,09% | 291 | 431 | 24 | 18 |
| Unihan kVietnamese (8.306 chữ) | 24,0% | 0,02% | 1.540 | **0** | 4 | — |
| **Ngữ liệu**: cặp lặp ≥ 2 trang khác | 86,2% | 2,22% | 39 | 16.769 | 731 | 23 |
| **Ngữ liệu**: cặp lặp ≥ 4 trang khác | 81,2% | 0,72% | 113 | 14.457 | 250 | **58** |
| **Ngữ cảnh**: 2-gram (chữ, âm) lặp ở trang khác | 56,8% | 0,18% | 310 | 8.129 | 96 | **85** |
| Ngữ cảnh: 2-gram ≥ 2 trang khác | 44,5% | 0,11% | 396 | 5.391 | 56 | 96 |
| Ngữ cảnh: 2-gram **cả hai phía** | 19,3% | 0,01% | 2.232 | 1.857 | 5 | **371** |
| Cầu tự dạng duy nhất **& 2-gram** | 13,4% | 0,01% | 1.111 | 957 | 7 | **137** |
| Cầu tự dạng duy nhất & cặp lặp ≥ 2 | 19,8% | 0,18% | 107 | 2.336 | 45 | 52 |
| Cầu tự dạng duy nhất & âm ~thanh/dấu | 9,9% | 0,03% | 303 | 169 | 3 | 56 |
| Âm ~thanh & cặp lặp ≥ 2 | 32,7% | 0,10% | 344 | 1.049 | 29 | 36 |

Đọc bảng:

1. **Từ điển ngoài đã hết**: Hán-Việt và Unihan gần như là tập con của `QuocNgu_SinoNom.csv`
   (thêm 1 và 0 ô). `dict_gap`/`kDefinition` trước đây cũng ra 0 — nhất quán. Không nên tốn thêm
   công vào việc "mở từ điển".
2. **Nguồn lớn nhất là ngữ liệu**: 14–17 nghìn ô chưa xác nhận có cùng cặp (chữ OCR, âm) ở trang
   khác; 8 nghìn ô có cùng **ngữ cảnh 2-gram** ở trang khác. Lift 58–371 nghĩa là bằng chứng
   *ghép* mạnh hơn cầu tự dạng 5–30 lần. Đây là "từ điển của chính ba cuốn sách", đang bị vứt.
3. **Cầu tự dạng chỉ nên dùng có điều kiện**: đơn độc lift 10; kèm 2-gram lift 137; kèm cặp lặp
   lift 52. Luật GOLD hiện hành nên đổi thành "cầu duy nhất **và** (2-gram ∨ cặp lặp ≥ 2)".
4. **Sửa dấu thanh** (tone/dia) lift 15–18 khi đứng một mình, 36 khi có cặp lặp; đủ để cho nhãn
   *âm tiết* (giữ nguyên âm gốc), không đủ để cho nhãn chữ một mình.

**Cách dùng đúng nguồn ngữ liệu = đệ quy hai lượt.** Lượt 1 căn chỉnh như hiện nay; rút ra
`pair_pages` và `bigram_pages` theo trang; lượt 2 căn chỉnh lại, coi cặp lặp ≥ 2 trang *khác* là
neo (chi phí 0). Loại-một-trang nên không có vòng tự khẳng định. Benchmark (600 cột):

| | khe đúng chỗ (rụng 1 chữ) | ghép sai khi rụng 1 chữ | ghép sai khi QN rụng 1 âm | ghép sai không hỏng |
|---|---|---|---|---|
| production, từ điển | 60% | 3,85% | 3,89% | 1,31% |
| production **+ neo ngữ liệu** | **82%** | **1,70%** | **1,48%** | 0,85% |
| hiệu chuẩn, từ điển | 64% | 3,14% | 3,04% | 0,54% |
| hiệu chuẩn **+ neo ngữ liệu** | **85%** | **1,42%** | **1,38%** | 0,63% |

Neo 2-gram trong DP không thêm gì ngoài neo cặp (đường ghép đã đúng); 2-gram phát huy ở bước
*phân hạng* (§5).

Cuối cùng, **từ điển phải là tiên nghiệm có trọng số**, không phải chân lý nhị phân: ghi
`dict_support = |R(âm)|` cho từng ô; xác nhận với |R| = 2 (chúa → 主) và |R| = 56 (thì) không thể
cùng tư cách. 25% ô GOLD có |R| ≥ 32, 10% có |R| ≥ 55.

---

## 4. "ĐỌC CHỮ BỊ CHẠY" — kiểm bằng lân cận, có xác suất

### 4.1 Xác suất hậu nghiệm thanh ghi (forward–backward)

Tổng theo mọi đường ghép trong băng với trọng số exp(−chi phí/T): P(i↔j). Một ô không khớp từ điển
vẫn có P cao nếu *các ô hai bên* neo chặt — đó chính là "đệ quy vài từ lân cận" ở dạng đúng đắn,
vì nó cân mọi cách ghép thay thế chứ không chỉ đếm hàng xóm.

Benchmark có đáp án (500 cột, production):

| hỏng | ghép sai | AUC bắt sai | p < 0,5 bắt / oan | p < 0,8 bắt / oan |
|---|---|---|---|---|
| OCR rụng 1 chữ | 4,0% | 0,877 | 78% / 18,5% | 88% / 36% |
| OCR rụng 2 chữ | 5,0% | 0,886 | 79% / 19% | 91% / 39% |
| OCR tách đôi 1 chữ | 4,2% | 0,873 | 79% / 18% | 88% / 36% |
| QN rụng 1 âm | 4,3% | 0,877 | 77% / 18% | 88% / 36% |
| OCR đọc nhầm 1 chữ | 2,1% | 0,728 | 51% / 21% | 73% / 39% |

Trên dữ liệu thật: GOLD-trực-tiếp p50 = 0,93 (chỉ 0,3% dưới 0,5); **GOLD cầu/L1/L3 p50 = 0,82,
20,7% dưới 0,5**; QĐ-01 14,9% dưới 0,5; REVIEW 62% dưới 0,5 nhưng 38% ≥ 0,5 (≈ 4.500 ô có thanh
ghi tin được mà chưa dùng). Hiệu chuẩn: với ma trận production, p dưới ước (p = 0,4 → đúng 95%);
với ma trận hiệu chuẩn thì trên ước ở dải giữa — cần chọn T trên benchmark trước khi lấy p làm
ngưỡng tuyệt đối; dùng để *xếp hạng* thì đã ổn.

### 4.2 Kiểm lệch cục bộ đơn giản không đủ

Cửa sổ ±2, thử offset ±1, ±2, gắn cờ nếu offset khác 0 khớp nhiều hơn ≥ 2 ô: chỉ bắt 7–14% cặp sai
(oan ≈ 0). Lý do: DP đặt khe sai *chính ở chỗ không có neo*, nên cửa sổ quanh đó cũng không có gì
để so. Phải dùng bằng chứng **ngoài cột** (ngữ liệu, §3) hoặc **ngoài văn bản** (hình học, §2 N2).

### 4.3 Khi m = n mà vẫn chạy

978 cặp lệch đường chéo trong cột khớp số; soi mắt 24 ô GOLD trong số đó: phần lớn *âm* ghép hợp
lý (OCR vừa rụng vừa thừa thật), nhưng 5–6 ô có **mã chữ** sai qua cầu tự dạng (又 cho "ngày" thay
𣈜, 箸 cho "nước" thay 渃, 易 cho "đã" thay 㐌, 代 cho "dạy" thay 𠰺). Tức ở vùng thanh ghi yếu, lỗi
trội là lỗi mã chữ của cầu — khớp với N3/N4.

---

## 5. GÁN NHƯ THẾ CÓ ĐÚNG KHÔNG — đánh giá theo tier, và đề xuất v3

### 5.1 Tier hiện hành dưới benchmark có đáp án (ghép âm)

| hỏng | GOLD (trực tiếp + cầu) | pool REVIEW |
|---|---|---|
| không hỏng | 1,33% | 2,10% |
| OCR rụng 1 chữ | 1,42% | 7,57% |
| OCR rụng 2 chữ | 1,70% | 9,72% |
| tách đôi 1 chữ | 1,32% | 8,47% |
| QN rụng 1 âm | 1,41% | 8,39% |

GOLD bền (từ điển không phụ thuộc vị trí) nhưng "sai mà từ điển vẫn xác nhận" ≈ 70/10.000 cặp mỗi
loại hỏng — đó là lớp N1. Pool REVIEW chứa 8–10% ghép sai khi cột bị hỏng: cổng SYLLABLE hiện lấy
từ pool này bằng độ thuần chữ OCR, không nhìn thanh ghi.

### 5.2 Ước lượng precision mã chữ của GOLD (54.156 ô)

Cộng ba lớp lỗi độc lập đã đo (không có chấm tay, đây là **ước lượng**):

- ghép sai thanh ghi ≈ 1,3–1,7% (benchmark) → ~800 ô;
- cầu tự dạng 4.808 ô × 17–19% nghi sai (bộ kiểm độc lập) ≈ 800–900 ô, một phần là báo động giả
  (寺/thì, 垩/thánh) nhưng phần (其/cùng, 真/xin, 時/được…) là nhầm hệ thống;
- nhãn thiểu số nhìn giống nhãn đa số trong cùng sách: 1.453 ô, trừ dị thể hợp lệ (徳/德, 别/別…)
  còn ≈ 700–900 ô nghi nhầm.

⇒ precision mã chữ **≈ 93–96%**, không phải "~99%" (`consensus.py:14`). Số này cần một mẻ chấm phân
tầng để chốt; xếp hạng ưu tiên chấm đã có ngay từ ba lớp trên.

### 5.3 Đề xuất phân hạng v3 — hai chiều tách bạch

Chiều **thanh ghi** (crop ↔ âm) và chiều **mã chữ** (glyph ↔ Unicode) là hai câu hỏi, cần hai bằng
chứng, ra hai nhãn:

```
CHAR_A  nhãn chữ, thanh ghi chắc : c ∈ R(âm)  ∧  P(ghép) ≥ 0,8
CHAR_B  nhãn chữ, cần soát        : c ∈ R(âm) ∧ P < 0,8   |   cầu duy nhất ∧ (2-gram ∨ cặp lặp≥2 ∨ ~thanh) ∧ P ≥ 0,8
SYL     chỉ nhãn âm               : P ≥ 0,5 ∧ (2-gram ∨ cặp lặp ≥ 4 ∨ ~thanh)      — chữ OCR ghi làm GIẢ THUYẾT
REVIEW  còn lại
+ decisions.yaml (phán quyết người theo LỚP) áp ngay trong decide_label — thay 4 bước hậu xử lý
```

Mô phỏng trên dữ liệu thật, **ma trận hiệu chuẩn + đệ quy neo ngữ liệu**:

| tier hiện hành → v3 | CHAR_A | CHAR_B | SYL | REVIEW |
|---|---|---|---|---|
| GOLD 54.097 | 47.481 | 3.294 | 2.018 | **1.304** |
| REVIEW 11.678 | 1 | 42 | **4.242** | 7.393 |
| SILVER_uncalibrated 6.316 | 0 | 0 | **2.415** | 3.901 |
| SYLLABLE 10.366 | 0 | 0 | 10.318 | 48 |
| (ô mới từ khe) 634 | 0 | 2 | 295 | 337 |
| **tổng v3** | **47.482** | **3.338** | **19.288** | 12.983 |

Bộ dùng được 64.525 → 70.108 ô, *và* 1.304 ô GOLD hiện hành bị trả về REVIEW (thanh ghi hoặc cầu
không có ngữ cảnh), 2.018 GOLD hạ xuống nhãn âm. (QĐ-01 chưa mô phỏng — trong v3 nó là một dòng
trong `decisions.yaml`, không phải bước riêng.) Benchmark có đáp án cho v3 (ma trận production):

| hỏng | v3 CHAR_A | v3 CHAR_B | v3 SYL | v3 REVIEW | *(tier cũ GOLD)* |
|---|---|---|---|---|---|
| không hỏng | **0,50%** | 5,8% | 0,91% | 3,7% | 1,33% |
| OCR rụng 1 chữ | **0,46%** | 6,2% | 0,79% | 11,2% | 1,42% |
| OCR rụng 2 chữ | 0,44% | 5,3% | 0,83% | 14,0% | 1,70% |
| tách đôi 1 chữ | 0,49% | 5,2% | 0,83% | 12,2% | 1,32% |
| QN rụng 1 âm | 0,50% | 5,4% | 0,95% | 12,2% | 1,41% |

CHAR_A giảm ghép sai 3 lần so với GOLD cũ; SYL ở mức < 1% dù lấy từ pool 8–14% sai; lỗi được dồn
vào CHAR_B (nhỏ, có cờ) và REVIEW.

### 5.4 Chiều mã chữ — phân xử theo LỚP, không theo ô

Việc người cần làm là quyết định theo lớp, mỗi quyết định lan ra hàng chục–hàng nghìn ô (QĐ-01 là
một lớp = 2.014 ô):

| hàng đợi | số lớp | số ô | công cụ có sẵn |
|---|---|---|---|
| GOLD thiểu số nhìn giống đa số (≥ 3 ô) | 128 | 1.453 | `thuc_nghiem.py classes` + `tools/variant_table.py` |
| GOLD bị bộ kiểm ngoài nghi sai, theo cặp (nhãn, âm) | 286 | 3.267 | `re-dataset/check/labels.xlsx` sheet `Cap_sai` |
| (chữ OCR, âm) ∉ R, ≥ 5 ô & ≥ 3 trang | 1.150 | 19.858 | top 300 lớp đã phủ 13.013 ô |

Mỗi lớp: xem 8–12 crop, chọn (a) đúng chữ → thêm vào *từ điển ngữ liệu*; (b) glyph thật là g →
ghi lớp nhầm (c, âm) → g; (c) dị thể → gộp; (d) hỗn hợp → để REVIEW. ~1.500 quyết định lớp thay cho
~25.000 quyết định ô.

---

## 6. LÀM THEO THỨ TỰ NÀO

| bước | việc | tác động đo được | công |
|---|---|---|---|
| 1 | Ma trận chi phí hiệu chuẩn (chỉ đổi 6 hằng số `anchor_align.py`) + test cho `realign_column` | khe giả 345 → 54 cột; 439 ô GOLD đổi âm về đúng đường chéo | giờ |
| 2 | Lượt đệ quy: `pair_pages` từ lượt 1 (loại-một-trang) làm neo lượt 2 | khe đúng 60 → 85%; ghép sai khi hỏng 3,9 → 1,4% | ngày |
| 3 | Hộp ảnh theo **thứ tự sau khe**: chỉ số glyph = chỉ số OCR + [i ≥ khe]; ô không có hộp ảnh → REVIEW `no_glyph_box`; ghi `box_source` | đúng hộp 59 → 91% ở cột rụng chữ; hết hộp midpoint trong bộ giao nộp | ngày |
| 4 | Posterior forward–backward + luật có lift → v3 (`CHAR_A/CHAR_B/SYL`), cột `p_register`, `dict_support`, `context_evidence` | CHAR_A ghép sai 0,5%; +5.600 ô dùng được; 1.304 GOLD yếu lộ diện | 2–3 ngày |
| 5 | `decisions.yaml` áp trong `decide_label`; gỡ `confusion_fix`/`s3_unwind`/`glyph_fix`/L1 khỏi đường chạy | tier gán một lần; bỏ ~1.000 dòng hậu xử lý | 1–2 ngày |
| 6 | Ba hàng đợi phân xử theo lớp (§5.4), ưu tiên 128 + 286 lớp GOLD | chốt precision mã chữ thay cho ước lượng 93–96% | người: vài buổi |
| 7 | Mẻ chấm ~600 ô phân tầng theo (tier v3 × lớp cột × box_source) để công bố CI | số công bố được | người |

Bước 1–3 không cần chấm tay, không cần GPU, và mỗi bước có benchmark tự động trong
`lab/gan_nhan_2026-09-13/thuc_nghiem.py` để kiểm hồi quy.

---

## 7. GIỚI HẠN CỦA CHÍNH BẢN NÀY

- Benchmark có đáp án lấy đường chéo của cột khớp số làm sự thật (đúng ≈ 99,5%, không phải 100%);
  các số "không hỏng 1,3%" gồm cả phần sai thật của sự thật ấy.
- Lift đo bằng null lệch 1 ô là bằng chứng về **ghép âm**, không về **mã chữ**. Mã chữ chỉ có bộ
  kiểm độc lập (heuristic) và soi mắt 72 ô — nên §5.2 là ước lượng, cần mẻ chấm để chốt.
- Ma trận hiệu chuẩn lấy thẳng từ tỉ số khả năng, chưa tinh chỉnh; T của posterior chưa hiệu chuẩn.
- Thí nghiệm hình học chỉ trên cột detector đếm đúng (171/216); cột detector đếm sai cần hoà giải
  ba nguồn đếm (OCR, QN, detector) — chưa làm.
