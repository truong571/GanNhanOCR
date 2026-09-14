# GÁN CHỮ HÁN-NÔM (OCR) ↔ QUỐC NGỮ: ĐANG Ở MỨC NÀO, VÀ HƯỚNG TẤN CÔNG

**2026-09-14** · nối tiếp `RA_SOAT_GAN_NHAN_2026-09-13.md` (nguyên nhân + từ điển). Tệp này trả lời hai
câu: *quy trình đang đạt tới đâu* (so với trần có thể đo), và *tấn công tiếp theo hướng nào* khi
ba nguồn — so khớp từ điển, từ điển tự dạng tương đồng, kho ảnh sinh — đã có sẵn.

Số liệu mới trong tệp này tái sinh bằng:

```
.venv/bin/python lab/gan_nhan_2026-09-13/thi_giac_am_tiet.py all    # cắt crop 15 s · huấn luyện ~15 phút MPS · 4 thí nghiệm
```

Số liệu của các mẻ trước (KB0, D, E, hướng B/C/D/2/3/4) lấy từ `docs/nghien_cuu_dang_do/*.json` —
**chưa qua phản biện**, tôi chỉ dùng những con số có đối chứng rõ và nhất quán giữa ≥2 mẻ.

---

## 0. TRẢ LỜI NGẮN

**Mức đạt.** Trên 82.780 ô: 56% có nhãn chữ nhờ từ điển tra thẳng, 6,7% nhờ cầu tự dạng + sửa dấu,
2,4% nhờ một phán quyết người, 12,5% chỉ có âm, 22% không có gì. Phần *ghép âm* đã gần trần
(≈98,5% đúng, có thể lên 99,5%); phần *mã chữ* của GOLD ước 93–96%. **Kênh ảnh đóng góp 0 ô** vào
bộ giao nộp — không phải vì chưa thử, mà vì đã thử 7 cách và đo được: kho ảnh sinh/phông thua tiên
nghiệm tần suất; crop thật làm tham chiếu thì làm được (71% khi có tham chiếu) nhưng 38% nhãn đúng
không có crop nào. Trần lý thuyết của mọi kênh "chọn trong R(âm)" là 97,7% ô; sàn tự động hiện
tại cho khối 27,7 nghìn ô OCR-đọc-sai là **0 ô ở chuẩn ≥95%**.

**Vì sao "so → tương đồng → so ảnh" không đủ.** Ba tầng ấy đều hỏi *cùng một câu* ("ô này là chữ
nào trong 4.700?") với bằng chứng ngày càng yếu (tra thẳng lift 161 → cầu tự dạng 10 → ảnh phông
≈ tần suất). Tầng ảnh bị giao câu hỏi khó nhất với tham chiếu tệ nhất (glyph in, nét rời) trên dữ
liệu 1-bit chữ thảo nét dính, và nhãn đích lại là **quy ước** (𠊚) chứ không phải hình trên giấy (㝵
ở 79–94% ô). Nó không thể thắng bằng cách "sinh ảnh giống hơn".

**Hướng tấn công.** Đổi *câu hỏi* giao cho ảnh, không đổi kho ảnh:
1. Ảnh trả lời **âm tiết** (~700 lớp, nhãn từ căn chỉnh — không nhiễm OCR), để **kiểm thanh ghi và
   định vị khe** — việc văn bản không làm được. Đo mới (§3): định vị hộp thiếu 72% (đoán mò 5%),
   bắt ô "đọc bị chạy" AUC 0,946; và ở mức lớp, 63 lớp/1.536 ô GOLD lộ ra là *một hình hai mã*.
2. Ảnh **gom cụm** các ô cùng (sách, âm) — cụm chính là "8 crop thật" mà mẻ D chứng minh là đủ lật
   một lớp; kho glyph/phông chỉ dùng để **đặt tên cụm** (vài trăm quyết định, người kiểm được).
3. Tách nhãn thành **hình quan sát** (cụm) và **mã chuẩn** (bảng quy ước có người ký) — hết nghịch lý
   㝵/𠊚, hết "accuracy đo tính nhất quán chứ không đo tính đúng".
4. Chọn trong R(âm) bằng CNN học trên chính bộ (+2,5 pp so tần suất, mẻ D) chỉ là **số hạng cộng
   mềm**, không phải lá phiếu quyết.

---

## 1. MỨC ĐẠT — đo trên bộ hiện hành, so với trần

### 1.1 Từng kênh đóng góp gì

| Kênh | Ô | % | Bằng chứng ghép (lift null lệch-1) | Bằng chứng mã chữ | Precision ước |
|---|---|---|---|---|---|
| S1∩S2 tra thẳng `c ∈ R(âm)` | 46.579 | 56,3 | 161 | OCR + từ điển đồng thuận; bộ kiểm ngoài nghi 5,2% | ghép 98,7% · mã ≈95% |
| + sửa dấu L1 | 755 | 0,9 | 18 (tone) | như trên, nhưng 97% đổi âm thật | mã tốt, **âm sai** ở ~140 ô |
| Cầu tự dạng xuôi/ngược/cột lệch | 4.808 | 5,8 | 10–15 | nghi sai 18,7%; CNN hạ nguồn đọc đúng 45–63% | mã ≈80% |
| Phán quyết người QĐ-01 | 2.014 | 2,4 | (REVIEW gốc) | người xem toàn bộ crop | theo quy ước |
| SYLLABLE (ngữ liệu, chỉ âm) | 10.369 | 12,5 | 58–371 nếu dùng ngữ cảnh (chưa dùng) | không có | âm ≈99% |
| S3 ảnh (SILVER → bị loại) | 6.328 | 7,6 | — | AUC bắt lỗi 0,54–0,58 (đo trên verdict máy) | không đo được |
| REVIEW | 11.927 | 14,4 | 38% có P(ghép) ≥ 0,5 | — | — |

### 1.2 Trần và sàn của khối "OCR đọc sai" (27,7 nghìn ô = REVIEW + SILVER + SYLLABLE)

| | giá trị | nguồn |
|---|---|---|
| trần lý thuyết mọi kênh chọn-trong-R(âm) | 97,7% ô (640 ô R rỗng; 329 bất khả) | hướng 2 |
| \|R(âm)\| trung vị / trung bình / max | 18 / 23 / 163 · đoán mò 6,6% | hướng 2 |
| tiên nghiệm "chữ phổ biến nhất của âm" | 87,9–91,9% **trên ô OCR-đọc-đúng**; **48,8%** trên tập chân trị OCR-đọc-sai | hướng 2, D, E |
| CNN học trên chính bộ, chọn trong R | 90,5% (+2,55 pp so tần suất); 48% ở nơi tần suất sai | hướng D |
| crop→crop (crop thật làm tham chiếu) | có điều kiện 71%; thô 42%; **38% nhãn đúng có 0 crop** | KB D |
| crop→glyph FD / phông Kai / phông in | 1–6% / 7,6% / 2% top-1 (4.700 lớp); trong R: 27,6% | KB 0, E |
| kênh ảnh cộng mềm vào tần suất | +0,5 đến +1,1 pp, KTC chứa 0 trên dân số thật | E, B |
| **sàn tự động hôm nay ở chuẩn ≥95%** | **0 ô** | hướng 2 |

### 1.3 Ba sự thật cứng về dữ liệu (đo, không đoán)

1. **Ngữ liệu là 1 bit** (PDF bpc=1): không có độ đậm nhạt bút; mọi nhị phân hoá cho cùng kết quả.
2. **Chữ thảo ở đây DÍNH nét, không BỚT nét**: điểm rẽ nhánh skeleton crop 26,8 vs FD 15,5; chiều dài
   skeleton crop/FD = 1,32. Kho sinh vẽ nét rời → sai miền theo cách không tăng cường nào mô phỏng được
   (hướng B bão hoà ở 11–12% top-1 mở).
3. **OCR gần như không xuất Ext-B**: 3,15% ô có `ocr_char` ngoài BMP, trong khi 39,8% ứng viên từ
   điển của chính các âm này nằm ngoài BMP; GOLD-trực-tiếp ngoài BMP: 369/46.579. Bất kỳ luật "OCR ∩
   từ điển" nào cũng kế thừa thiên lệch này — đó là gốc của 㝵/người, 時/ngày, 代/dạy.

---

## 2. VÌ SAO CHUỖI "SO → TƯƠNG ĐỒNG → SO ẢNH" ĐÃ CHẠM TRẦN

Cả ba tầng cùng hỏi một câu và đều quyết theo **từng ô, một tham chiếu**:

```
tầng 1  c ∈ R(âm)?                       lift 161    56% ô — hết ô dễ
tầng 2  ∃! b ∈ sim(c) ∩ R(âm)?           lift 10–15  6%  — 1/5 nghi sai
tầng 3  argmax_g cos(crop, glyph_g)?     ≈ tần suất  0%  — tham chiếu sai miền, câu hỏi 4.700 lớp
```

Ba lý do cấu trúc, mỗi lý do đã có số:

- **Tham chiếu**: cùng bộ mã hoá, crop→crop 53,4% vs crop→FD 5,6% (D); tức 90% hiệu năng khả dĩ mất
  ở tham chiếu, không ở thuật toán. Sinh lại kho theo phong cách bút chỉ đổi 3,8 → 7,6% (KB 0).
- **Câu hỏi**: hỏi "chữ nào trong 4.700" khi âm đã biết là hỏi thừa 4.680 lớp; hỏi "chữ nào trong
  R(âm) ∩ đã thấy" thì tần suất đã trả lời 88–92% và ảnh chỉ thêm +2,5 pp.
- **Nhãn đích là quy ước**: 2.014 ô QĐ-01 mang mã 𠊚 nhưng 79–94% có hình 㝵 trên giấy (đo mực bộ 亻:
  crop 0,148 ≈ mốc 㝵 0,145, ≠ mốc 𠊚 0,284; CNN độc lập học từ NomNaOCR cũng nói 㝵 94,2%). Một kênh
  ảnh *đúng* sẽ trả 㝵 — và bị chấm là sai. Không kho ảnh nào chữa được điều đó; chỉ một **bảng quy
  ước** mới chữa được.

Kết luận thiết kế: tầng ảnh phải được giao **câu hỏi mà văn bản không trả lời được** (thanh ghi,
khe, dị thể) và được đặt ở **đơn vị cụm** (nhiều crop thật) thay vì ô-so-với-glyph.

---

## 3. THÍ NGHIỆM MỚI — ảnh giám sát bằng ÂM TIẾT

Mô hình: CNN 8 tầng conv (1,4 M tham số), vào 64×64, **lớp = âm tiết** (mọi âm có ≥5 ô train),
huấn luyện trên GOLD+SYLLABLE của trang *train* (chia theo trang), nhãn âm đến từ căn chỉnh — không
dùng nhãn chữ, không dùng `nom-embed`, không dùng kho glyph. Đánh giá chỉ trên trang *test*.

### E1 · Ảnh nói được ÂM gì — 706 lớp, trang test chưa từng thấy

| ô trên trang test | n | top-1 | top-5 | hạng trung vị |
|---|---|---|---|---|
| GOLD-trực-tiếp | 4.899 | **77,8%** | 91,3% | 1 |
| GOLD-cầu tự dạng | 512 | 67,6% | 83,0% | 1 |
| SYLLABLE (chỉ có âm) | 1.131 | **81,9%** | 93,5% | 1 |
| REVIEW | 915 | 45,6% | 66,8% | 2 |

Đọc: một CNN 12 phút, chưa tinh chỉnh, nhìn crop và đoán đúng âm tiết trong 706 âm ở 78% ô. Ba
hệ quả: (a) tầng SYLLABLE — tầng chỉ có âm, không có chữ — **nhất quán về hình với âm của nó ngang
hoặc hơn GOLD**, tức nhãn âm ở đó tin được; (b) REVIEW thấp hẳn (45,6%): đúng như mong đợi vì ở đó
âm ghép hay sai hoặc crop hỏng — đây là lần đầu có một máy đo *độc lập với văn bản* nói điều đó;
(c) cầu tự dạng 67,6% < trực tiếp 77,8%: ô cầu có crop/âm kém hơn, khớp với 18,7% nghi sai.

### E2 · Ảnh làm trọng tài THANH GHI — 268 cột khớp số trên trang test

**E2a — hộp thiếu.** Bỏ ngẫu nhiên 1 hộp trong cột (mô phỏng detector sót glyph), DP hộp↔âm với phát
xạ `log P(âm | crop)` và phạt khe cố định, hỏi: âm nào không có hộp?

| phạt khe | 4 | 8 | 12 | 16 | 24 | đoán mò |
|---|---|---|---|---|---|---|
| chỉ đúng khe | 52,8% | 69,7% | 70,8% | **71,9%** | 71,9% | 5,0% |

So với thí nghiệm hình học hôm 13-09: tâm tổng hợp *không* định vị được khe (4%); DP văn bản định
vị 60–64%; **ảnh một mình 72%**, và độc lập với văn bản — hai nguồn cộng lại là chỗ H1 nhắm tới.

**E2b — thanh ghi trượt.** Từ vị trí k ngẫu nhiên, mọi ô nhận âm của ô kế tiếp (mô phỏng "đọc chữ bị
chạy"). `P(âm gán | crop)` bắt ô ghép sai: **AUC 0,946**; ngưỡng p < 0,05 bắt **93%** ô sai, oan 14,3%
ô đúng (cận trên, vì "ô đúng" là cặp của bộ hiện hành, tự nó ~1–2% sai). Xác suất hậu nghiệm văn bản
(13-09) đạt AUC 0,88 / bắt 78% ở cùng mức oan. Hai máy đo nhìn hai thứ khác nhau (từ điển vs điểm
ảnh) nên gộp được.

### E3 · Cùng (sách, âm), hai mã chữ nhìn giống nhau: MỘT hình hay HAI hình?

Phép đo: k-NN (k=5, loại-một-ô) giữa nhóm nhãn đa số và nhóm nhãn thiểu số trên embedding, kèm **ý
kiến thứ hai bằng HOG không học** (không thể bị nhãn âm "ép"), và đối chứng hoán vị. k-NN không hơn
đoán ở cả hai đặc trưng ⇒ cùng một hình, OCR đọc hai cách; k-NN ≥ 0,75 ⇒ hai hình thật.

| lớp (sách, âm, nhãn thiểu số ≥ 8 ô) | số lớp | số ô GOLD | ví dụ |
|---|---|---|---|
| **MỘT HÌNH** — nhìn giống theo SinoNom_Similar | 39 | 889 | (cùng 共/其) 158 · (đức 徳/德) 158 · (được 特/時) 39 · (một 没/殳) 31 · (đến 旦/且) 15 · (hồn 魂/塊) 10 |
| **MỘT HÌNH** — *không* có trong SinoNom_Similar | 24 | 647 | (stt4 thì 時/寺) **136** · (stt4 chăng 庄/生) **99** · (bỏ 補/𥙷) 36 · (một 没/蔑) 34 · (vì 爲/為) 33 |
| HAI HÌNH (dị thể thật, giữ cả hai mã) | 38 | 508 | (cũng 共/拱) · (đoạn 段/断) · (mình 命/𠇮) · (thánh 垩/圣/聖) · (stt2 thì 寺/時) |
| mờ | 11 | 254 | (xin 嗔/真/填) · (càng 强/強) |

Ba điều rút ra:
1. **1.536 ô GOLD** đang mang một mã chữ mà trên giấy không có hình khác với mã đa số của cùng
   (sách, âm). Với dị thể Unicode (徳/德, 爲/為, 别/別) đó là việc gộp; với (其/cùng, 時/được, 生/chăng,
   且/đến, 塊/hồn, 蔑/một) đó là nhãn sai hệ thống — đúng những cặp bộ kiểm độc lập đã gắn cờ, nay có
   bằng chứng *ảnh*. Mỗi lớp là **một** quyết định người, không phải 1.536 quyết định.
2. Phán quyết **theo sách**: (thì 時/寺) là một hình ở stt4 (136 ô) nhưng hai hình ở stt2 (17 ô). Bảng dị
   thể toàn cục là sai đơn vị; đơn vị đúng là (sách, âm).
3. 24/63 lớp một-hình **không có trong SinoNom_Similar** — tức luật cầu tự dạng và luật "nhìn giống"
   của báo cáo 13-09 bỏ sót 647 ô; ảnh bắt được vì nó so *trên chính nét bút của người chép*, không
   so trên glyph phông.

**"người" — 2.281 crop, gom cụm không giám sát (k = 4):** các cụm đi theo **SÁCH**, không theo chữ OCR:
cụm 3 = 684 ô, 661 stt2, OCR đọc 㝵 634/684; cụm 0 = 708 ô, 647 stt11, OCR đọc 早/㝵/昇 lẫn lộn;
cụm 2 = 606 ô, 594 stt4, OCR đọc 㝵/身/𭔿. Nghĩa là mỗi tay chép viết "người" một kiểu (một hình mỗi
sách), còn 㝵/早/昇/身 chỉ là nhiễu OCR trên cùng hình ấy — khớp với phép đo mực bộ 亻 của mẻ KB0 và
với QĐ-01. Cụm là đơn vị đúng để một người quyết định mã cho 2.281 ô bằng ba cái nhìn.

### E4 · Lan nhãn theo cụm — kiểm trên ô cầu tự dạng (OCR đọc sai, nhãn độc lập với ảnh)

398 ô cầu trên trang test có ≥3 nguyên mẫu GOLD-trực-tiếp cùng (sách, âm) ở trang khác:
nhãn cầu có trong bể nguyên mẫu **58,5%** (trần tồn kho — trùng 57,5% của mẻ D); k-NN đúng nhãn
cầu 43,2% vs "chữ phổ biến nhất của (sách, âm)" 42,7% — **không hơn tần suất**, y như mẻ D và E;
cổng chặt (≥4/5 phiếu, cos ≥ 0,6) bắn 16,3%, đúng 56,9%. Trên 18.602 ô chưa có mã chữ có nguyên
mẫu: 2.917 (15,7%) qua cổng chặt.

Kết luận thẳng: **chọn mã chữ cho từng ô bằng ảnh vẫn không thắng tần suất** — lần thứ tư kết quả
này lặp lại với một bộ mã hoá hoàn toàn khác. Nhưng E3 cho thấy vì sao con số này đánh giá thấp
ảnh: "chân trị" cầu tự dạng sai ở đúng những lớp mà ảnh nói là một hình (其/cùng…), nên mỗi lần
k-NN trả về nhãn đa số *đúng* thì bị chấm là *sai*. Vai trò của ảnh là ở **mức lớp** (E3), không ở
mức ô (E4).

---

## 4. HƯỚNG TẤN CÔNG — thứ tự và kỳ vọng

### H1 · Ảnh làm trọng tài THANH GHI (mới, chưa ai làm)

Ghép hộp ảnh ↔ âm bằng DP với phát xạ `log P(âm | crop)` từ mô hình §3, thay cho ghép chữ-OCR ↔ âm
rồi gán hộp theo tâm tổng hợp. Ở cột đếm khớp (M = N): ghép theo thứ tự, ảnh chỉ *kiểm*; ở cột lệch:
DP định vị hộp thiếu/thừa từ ảnh (§3 E2a), văn bản định vị chữ OCR thiếu/thừa (khe DP), hai khe
phải trùng — không trùng thì REVIEW. Kỳ vọng: xoá lớp lỗi N2 (hộp đúng 59% → ≥90% ở cột lệch),
thêm cột `p_visual_register` cho từng ô.

### H2 · Gom cụm cùng (sách, âm) → quyết định theo lớp

Với mỗi (sách, âm): embedding §3 → cụm (agglomerative, ngưỡng lấy từ E3: k-NN cân bằng ≤ 0,6 = một hình, ≥ 0,75 = hai hình). Ô GOLD-trực-tiếp
trong cụm cho cụm cái tên; cụm không có ô nào xác nhận → tên = ứng viên trong R(âm) ∪ sim(chữ hút)
có glyph (phông bút Kai, không phải FD) gần medoid nhất, **người xác nhận**. Mỗi quyết định lan ra
cả cụm. Đây là cách QĐ-01 (1 quyết định = 2.014 ô) trở thành quy trình: E3 đã cho sẵn 63 lớp/1.536 ô
GOLD cần gộp-hoặc-sửa và 38 lớp/508 ô là dị thể thật; toàn khối ~1.500 lớp thay ~25.000 ô.
Kho glyph 89.898 ảnh giữ lại đúng một vai: đặt tên cụm không có mẫu thật.

### H3 · Nhãn hai tầng: HÌNH quan sát và MÃ chuẩn

`glyph_cluster_id` (hình trên giấy, do ảnh gom) và `label` (mã Unicode chuẩn, do bảng quy ước có
người ký: 㝵→𠊚 cho "người", 徳=德, 别=別, …). Hai cột, hai câu hỏi, hai cách đánh giá: cụm đo bằng
độ thuần ảnh; mã đo bằng phán quyết người theo lớp. Hết chuyện "hai bộ nhãn cùng 89% accuracy".

### H4 · Chọn trong R(âm) bằng CNN học trên chính bộ — chỉ là số hạng cộng mềm

Mẻ D đo: +2,55 pp so tần suất, 48% ở nơi tần suất sai. Dùng với trọng số nhỏ trong điểm tổng hợp
(từ điển + ngữ liệu + ảnh), không dùng để quyết. Cần đo lại trên chân trị **cụm** (H2) thay vì chân
trị cầu tự dạng (18,7% nghi sai).

### H5 · KHÔNG làm — đã có số bác

| việc | số bác |
|---|---|
| sinh lại kho glyph theo phong cách bút / tăng cường kéo miền | 3,8→7,6% (KB0); bão hoà 11–12% (B); < tần suất |
| ≥2 kênh xác nhận (att + ext + Unihan) để cứu nhãn chữ | lift 1,07×, bơm ~1.400 nhãn sai (hướng 4) |
| bảng nhầm học từ S3 / ma trận nhầm phông | thua "đối thủ tầm thường" (hướng 3) |
| tần suất ngoài, bigram ngoài (NomNaOCR) | lift thật 1,4× / 1,04× (hướng 2) |
| siết cửa sổ crop theo tâm hộp | cắt cụt 47–63% ô; đúng là tái định tâm theo mực (C) |
| kênh ảnh làm lá phiếu quyết / phủ quyết / thu hẹp | AUC 0,666, mọi ngưỡng lỗ (E) |

### Lộ trình

| bước | việc | đo bằng | công |
|---|---|---|---|
| 1 | Ma trận chi phí hiệu chuẩn + đệ quy neo ngữ liệu + hộp theo thứ tự sau khe (báo cáo 13-09) | benchmark có đáp án; khe giả 345→54 cột | 2–3 ngày |
| 2 | Mô hình âm tiết (§3) vào pipeline: cột `p_visual_register`; DP hộp↔âm ở cột lệch | E2 trên trang test; ô cột-lệch không còn khác ô cột-khớp | 3–4 ngày |
| 3 | Gom cụm (sách, âm) + bảng lớp cần đặt tên + giao diện xem 12 crop/cụm | E3; số cụm/số ô; người: vài buổi | 1 tuần |
| 4 | Bảng quy ước mã chuẩn (dị thể + 㝵→𠊚 …) áp trong `decide_label`; hai cột nhãn | precision theo lớp có người ký | 2–3 ngày |
| 5 | Mẻ chấm 600 ô phân tầng để công bố CI | — | người |

---

## 5. GIỚI HẠN

- Mô hình §3 huấn luyện 15 phút, kiến trúc nhỏ, chưa tinh chỉnh; các con số là *sàn* của hướng này.
- Chân trị của E4 là nhãn cầu tự dạng (18,7% nghi sai) — độ chính xác tuyệt đối ở đó là cận dưới.
- E3 dùng k-NN trên embedding học từ âm tiết — mô hình chỉ cần phân biệt âm nên có thể ép hai dị thể
  của cùng âm lại gần nhau. Vì thế mỗi lớp được đo thêm bằng HOG không học; phán quyết "một hình"
  chỉ ra khi CẢ HAI đặc trưng đều không tách được, "hai hình" khi một trong hai tách được (≥ 0,75).
  HOG thô nên "hai hình" ở HOG yếu hơn ở CNN; 11 lớp "mờ" để người xem.
- Mọi kết luận về QĐ-01 dựa trên đo hình (mực bộ 亻, CNN độc lập), không phải phán quyết người thứ hai.
