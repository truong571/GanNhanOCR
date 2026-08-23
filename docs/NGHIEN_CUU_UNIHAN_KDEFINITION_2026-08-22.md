# NGHIÊN CỨU: `unihan_kdefinition_full.csv` DÙNG ĐƯỢC VÀO VIỆC GÌ

**Ngày**: 2026-08-22 · **Câu hỏi**: có thể dùng nghĩa Hán (Unihan `kDefinition`) để kiểm tra lại
cặp (chữ Nôm, âm Quốc ngữ) trong bộ dữ liệu xem đúng không?

**Trả lời ngắn**: **được, nhưng chỉ một chiều và với sản lượng nhỏ.** Nghĩa Hán là bằng chứng
**xác nhận** tốt (điểm cao ⇒ gần như chắc đúng), nhưng **không phải bằng chứng bác bỏ** (điểm 0
không nói lên điều gì). Lý do có gốc ngôn ngữ học, đo được, và giải thích ở §2.

---

## 1. ĐỘ PHỦ — tài nguyên dùng được

`dict/unihan_kdefinition_full.csv`: 23.285 chữ, 3 cột `Codepoint, Character, kDefinition`
(nghĩa tiếng Anh).

| tier | lớp chữ | có kDefinition | độ phủ theo Ô |
|---|---|---|---|
| GOLD | 1.582 | 1.414 (89,4%) | **96,9%** |
| SILVER_uncalibrated | 788 | 715 (90,7%) | 91,1% |
| REVIEW (theo `ocr_char`) | 3.301 | 2.987 (90,5%) | 93,2% |

Độ phủ đủ cao để dùng. 10% lớp không có nghĩa Hán chính là **chữ Nôm riêng** (không phải chữ
Hán) — và bản thân sự vắng mặt đó là một tín hiệu (xem §4.3).

---

## 2. PHÁT HIỆN CỐT LÕI: CHỮ NÔM CHỌN THEO **ÂM**, KHÔNG THEO NGHĨA

Đây là lý do khiến ý tưởng ban đầu ("đối chiếu nghĩa để kiểm nhãn") chỉ đúng một phần.

Từ điển cho biết **20 chữ** đọc âm **"người"**. Nghĩa Hán của chúng:

```
仉  surname of the mother of Mencius     倘  if, supposing, in event of
儻  if, supposing, in case               匕  spoon, ladle; knife, dirk
命  life; destiny, fate, luck            徜  walking to and fro; lingering
昆  elder brother; descendants           皚  brilliant white
```

**Không chữ nào nghĩa là "người".** Chúng được mượn theo **ÂM** (thảng, tảng, bỉ, mệnh...), không
theo nghĩa. Âm "mà" cũng vậy: 傌 chửi rủa · 嘛 trợ từ · 女 phụ nữ · 痳 bệnh phong · 罵 mắng ·
蔴 cây gai · 調 chuyển.

⇒ **"Hồ sơ nghĩa" của một âm tiết phần lớn là nhiễu.** Bất kỳ phép kiểm nào giả định "chữ đúng
phải có nghĩa hợp với âm" sẽ sai trên đa số dữ liệu.

---

## 3. PHÉP THỬ ĐỊNH LƯỢNG

### 3.1 Thiết kế

Điểm nghĩa `score(C, S)` = cosine có trọng số IDF giữa tập từ trong `kDefinition(C)` và **túi
nghĩa của mọi chữ khác** mà từ điển cho là đọc âm `S` (loại chính `C` ra — leave-one-out).

- **Dương**: cặp (C, S) được từ điển xác nhận, lấy từ tier GOLD.
- **Âm khớp-âm-tiết**: giữ nguyên `S`, đổi `C` sang một chữ khác trong corpus **không** đọc âm đó.
  Cách này giữ hồ sơ nghĩa **giống hệt nhau**, nên loại sạch mọi yếu tố gây nhiễu của âm tiết.

### 3.2 Kết quả (n = 2.002 cặp)

| phép đo | AUC |
|---|---|
| **Điểm nghĩa** | **0,7418** |
| Đối chứng: độ dài định nghĩa | 0,4322 |
| Đối chứng: tổng IDF của định nghĩa | 0,4212 |

Hai đối chứng đều **dưới ngẫu nhiên** ⇒ tín hiệu **là thật**, không phải giả tạo do chữ phổ biến
có định nghĩa dài hơn.

Kiểm cơ chế: bỏ hết cụm dị thể (`same as` / `variant of` / `ancient form of`, chiếm 9,3% định
nghĩa) → AUC **0,7438**, không đổi. ⇒ tín hiệu **không** đến từ chú thích dị thể mà từ quan hệ
nghĩa thật.

### 3.3 Tính bất đối xứng — điểm quan trọng nhất

| | dương (đúng) | âm (sai) |
|---|---|---|
| tỉ lệ điểm **= 0** | **47,7%** | **93,2%** |

| ngưỡng | giữ lại dương | lọt âm | độ chính xác |
|---|---|---|---|
| > 0 | 1.047 | 137 | 88,4% |
| > 0,005 | 1.028 | 89 | 92,0% |
| > 0,02 | 724 | 8 | **98,9%** |
| > 0,05 | 303 | **0** | **100,0%** |
| > 0,10 | 64 | 0 | 100,0% |

**Đọc ra**: điểm cao là bằng chứng gần như kết luận được. Điểm 0 thì **vô nghĩa** — vì gần một
nửa cặp ĐÚNG cũng bằng 0.

### 3.4 Kiểm chứng ngoài — và đây là chỗ ý tưởng gốc gãy

| cặp | điểm | phân vị so với GOLD | thực tế |
|---|---|---|---|
| 㝵 "người" | **0,0** | 47,7% | lớp lỗi hệ thống đã chứng minh |
| 主 "chúa" | **0,0** | 47,7% | cặp Công giáo lõi, chắc chắn đúng |
| 妃 "bà" | 0,0 | 47,7% | nghi ngờ |
| 奴 "nó" | 0,0 | 47,7% | nghi ngờ |
| 麻 "mà" | 0,031 | 73,9% | cặp phổ biến nhất corpus (1.641 ô) |
| 時 "thì" | 0,059 | 88,2% | cặp phổ biến |

**Một lỗi đã chứng minh và một cặp chắc chắn đúng cho cùng một điểm.** Vì vậy: **tuyệt đối không
dùng điểm nghĩa để hạ cấp hay loại nhãn.**

---

## 4. SẢN LƯỢNG THỰC TẾ — dùng được vào đâu

### 4.1 Cứu ô REVIEW `no_s1_inter_s2`? — Gần như không

7.081 cặp / 10.933 ô. Ở ngưỡng an toàn: **> 0,02 chỉ được 78 cặp / 148 ô**; > 0,05 còn 14 cặp /
19 ô. Không phải con đường cứu REVIEW.

Nhưng các cặp điểm cao đáng xem: 炉 "tro" (3 ô, 炉 = lò lửa → tro) · 𢧐 "chén" · 泻 "rửa"
(泻 = tháo/xả nước → rửa). Đây là **mượn nghĩa thật mà từ điển âm chưa ghi**.

### 4.2 Tier SYLLABLE (chưa có nguồn nào xác nhận) — nhỏ nhưng có thật

315 cặp / 6.761 ô. Ngưỡng > 0,02: **8 cặp / 104 ô**. Vài cặp là mượn nghĩa rõ ràng:

| cặp | n | căn cứ |
|---|---|---|
| 肭 "thịt" | 10 | bộ 肉/月 (thịt) — mượn nghĩa trực tiếp |
| 煆 "lửa" | 5 | bộ 火 (lửa) — mượn nghĩa trực tiếp |
| 媛 "mẹ" | 10 | 媛 = người nữ đẹp → nghĩa nữ giới |

Với tier mà **Unihan xác nhận 0/316 theo âm**, việc xác nhận được ~8 cặp **theo nghĩa** là đóng
góp nhỏ nhưng có thật — và là 8 cặp không phải mang ra chấm tay.

### 4.3 Luật GOLD yếu nhất `s1_inter_s2_similar` — sản lượng lớn nhất

3.873 ô / 815 cặp. Ngưỡng > 0,02: **180 cặp / 1.041 ô**; > 0,05: 75 cặp / 451 ô.

Đây là chỗ đáng dùng nhất: luật cầu tự dạng là luật GOLD **yếu nhất** trong hệ thống, và điểm
nghĩa cho một **kênh xác nhận độc lập hoàn toàn** với cả tự dạng lẫn từ điển. Ví dụ điểm cao:
実/实 "thật" · 伝 "truyện" · 弌 "nhất" · 驭 "ngựa" (bộ 馬) · 噴 "phun" — đều đúng.

---

## 5. PHÁT HIỆN PHỤ: MÔ HÌNH LỖI CỦA OCR NÔM

Khi soi 15 cặp cầu tự dạng phổ biến nhất, một quy luật hiện ra: **OCR ánh xạ chữ Nôm lạ sang chữ
Hán CHIA SẺ THÀNH PHẦN với nó**.

| OCR đọc | nhãn đúng | âm | n | quan hệ |
|---|---|---|---|---|
| 韋 | 喡 | và | 63 | 喡 = 口 + **韋** — OCR **rụng bộ 口** |
| 等 | 寺 | thì | 45 | 等 = 竹 + **寺** — OCR **thêm bộ 竹** |
| 忝 | 𡗶 | trời | 34 | cùng phần trên **天** |
| 洋 | 群 | còn | 100 | cùng thành phần **羊** |
| 冕 | 晜 | con | 35 | cùng phần trên **日**-dạng |
| 不 | 丕 | vậy | 90 | 丕 = **不** + 一 |

Đây là một **mô hình lỗi mô tả được**: OCR SinoNom (huấn luyện trên chữ Hán) quy chữ Nôm chưa
biết về chữ Hán gần nhất **theo thành phần cấu tạo**, thường là rụng hoặc thêm một bộ thủ.

**Hệ quả cho đề tài**: đây mới là chỗ **IDS (chuỗi mô tả cấu tạo)** thật sự có ích — **không phải**
để huấn luyện mô hình thị giác như bản góp ý đề xuất, mà để **mô hình hoá lớp lỗi của OCR**: nếu
`IDS(ocr_char)` và `IDS(label)` chia sẻ thành phần, đó là bằng chứng ủng hộ phép bắc cầu; nếu
không chia sẻ gì (như 幾 → 没 "một", 46 ô), cặp đó đáng nghi.

Đây là một giả thuyết **kiểm được** bằng CHISE `ids.txt` (nguồn ngoài, miễn phí) và là hướng
nghiên cứu đáng làm hơn hẳn SVTR/IDS-cho-thị-giác.

---

## 6. KẾT LUẬN: DÙNG ĐƯỢC / KHÔNG DÙNG ĐƯỢC

### Dùng được

1. **Kênh xác nhận một chiều** — ngưỡng 0,05 cho độ chính xác 100% trên chuẩn 2.002 cặp. Áp cho
   `s1_inter_s2_similar` (1.041 ô ở ngưỡng 0,02) và SYLLABLE (104 ô). Ghi thành cột
   `sem_score` + `sem_confirmed` trong `labels.csv`. **Chỉ phong, không bao giờ hạ.**
2. **Trợ giúp chấm tay — giá trị cao nhất.** Gắn nghĩa Hán của `ocr_char` và của `label` vào mỗi ô
   trong HTML chấm. Người chấm nhìn thấy ngay "㝵 = (dạng cổ của 得) to get / to obstruct" khi
   đang chấm ô gán âm "người" — nhanh hơn và đúng hơn nhiều so với chấm mù.
3. **Xếp hạng ưu tiên chấm** — cùng với cột `fragility` (§T5 của chương trình thí nghiệm).
4. **Đóng góp ngôn ngữ học**: tách được nhóm **mượn nghĩa** khỏi nhóm **mượn âm** trong chính ngữ
   liệu — một bảng có giá trị học thuật, chưa từ điển nào của ngữ liệu Công giáo cổ này có.

### Không dùng được

1. **Không phát hiện được lỗi.** 47,7% cặp đúng cho điểm 0; 㝵/"người" (sai) và 主/"chúa" (đúng)
   cùng điểm 0.
2. **Không xác nhận được đa số cặp** — vì chữ Nôm chọn theo âm (§2).
3. **Không sinh được cặp từ điển mới** — tệp không chứa âm Quốc ngữ. (Trường sinh cặp là
   `kVietnamese`, nằm trong `Unihan_Readings.txt` mà repo chưa lưu, và đã đo được là chỉ thêm ròng
   **+109 cặp**.)
4. **Không xác nhận được luật cầu tự dạng** — luật đó là **đồ hình**, không phải ngữ nghĩa; điểm
   nghĩa bằng 0 ở đó là bình thường, không phải dấu hiệu sai.

---

## 7. VIỆC ĐỀ NGHỊ

| # | việc | công | giá trị |
|---|---|---|---|
| 7.1 | Đưa `sem_score` thành module `pipeline/tools/sem_score.py`, ghi 2 cột vào `labels.csv` | nửa ngày | kênh xác nhận độc lập thứ ba, không dính thị giác |
| 7.2 | Gắn nghĩa Hán vào HTML chấm tay (`audit_grid`) | 2 giờ | **giá trị cao nhất** — làm mẻ chấm sau này nhanh và đúng hơn |
| 7.3 | Xuất bảng "mượn nghĩa vs mượn âm" cho toàn corpus | nửa ngày | đóng góp ngôn ngữ học cho luận văn |
| 7.4 | Tải CHISE `ids.txt`, kiểm giả thuyết §5 (chia sẻ thành phần) | 1 ngày | mô hình hoá lớp lỗi OCR — hướng mạnh hơn SVTR/IDS |

**Không** đề nghị: dùng điểm nghĩa làm cổng hạ cấp, hay dùng `kDefinition` để mở rộng từ điển.

---

## PHỤ LỤC — lệnh tái sinh

Mã thí nghiệm: `pipeline/tools/sem_score.py` (sau khi làm 7.1). Bản nháp dùng để sinh các số trong
tài liệu này nằm ở scratchpad phiên làm việc; các bước tái lập:

1. Nạp `kDefinition`, tách token (bỏ stopword, độ dài > 2), tính IDF trên 23.285 định nghĩa.
2. `profile(S)` = túi từ của mọi chữ trong `qn_to_nom[S]`, **loại chính chữ đang chấm**.
3. `score(C,S)` = cosine có trọng số IDF giữa `tokens(C)` và `profile(S)`, tần suất kẹp trần 3.
4. Chuẩn đánh giá: dương = cặp GOLD dict-confirmed; âm = **cùng âm tiết, đổi chữ**.
