# Kết hợp ba luồng OCR: kinhhannom (K) + PaddleOCR (P) + GLM-OCR (G) — cùng bộ mẫu M1/M2/M3 (seed 20260911)

Luồng dùng: K = ocr_char (cache kinhhannom); P = PP-OCRv6 chế độ CỘT, gióng đơn điệu theo toạ độ CTC (`paddle/ket_qua/*_col-PP-OCRv6_medium_rec.csv`, cột `pred_mono`); G = GLM-OCR col-ctx (`glm/out/*_col-ctx.csv`, cột `pred`). Luồng phụ (chỉ cho luật D2/D3): v6 crop, v5 crop, PaddleOCR-VL crop, GLM a-force. Trọng tài: R(âm) trong `mau/M*.csv`. Đối chứng: hoán vị P, G (và luồng phụ) độc lập giữa các ô, 500 lần (giữ phân bố đầu ra, phá liên hệ với ảnh); chữ ngẫu nhiên theo tần suất corpus ∈R = 0,64 %/ô (M2).

## 1. Bảng luật trên M2 (n = 600, K ∉ R theo định nghĩa; SILVER 219 / SYLLABLE 381)

| Luật | Bắn | ∈R (đề xuất) | ∈R / 600 [Wilson 95 %] | Hoán vị ∈R (TB) | Lift vs hoán vị | Lift vs ngẫu nhiên tần suất | SILVER ∈R | SYLLABLE ∈R | Trùng nhãn pipeline (SILVER) |
|---|---|---|---|---|---|---|---|---|---|
| (a) A2 P==G ∧ ∈R | 19 | 19 | 3,2 % [2,0–4,9] | 0,004 | > 4 700 (0/500 lần đạt) | 5,0 | 11/219 = 5,0 % | 8/381 = 2,1 % | 11/11 |
| (a) A2v P≈G (chuẩn hoá giản/phồn) ∧ ∈R | 20 | 20 | 3,3 % [2,2–5,1] | 0,004 | > 5 000 | 5,2 | 12/219 | 8/381 | 11/12 |
| (a) A1 P==G bất kể R | 90 | 19 | 21,1 % của ô bắn | 0,9 bắn | — | — | 33 bắn / 11 ∈R | 57 bắn / 8 ∈R | — |
| (a) A4 K==P ∨ K==G (engine mới XÁC NHẬN chữ ∉R của K) | 171 = 28,5 % [25,0–32,2] | 0 | 0 | 2,4 bắn | 70× (bắn) | — | 46 | 125 | — |
| (a) A5 K==P==G | 42 = 7,0 % | 0 | 0 | 0,0 | — | — | 13 | 29 | — |
| (b) B1 P∈R ? P : (G∈R ? G : ∅) | 100 | 100 | **16,7 % [13,9–19,9]** | 3,6 | **27,8** | 26,2 | 56/219 = 25,6 % [20,2–31,7] | 44/381 = 11,5 % [8,7–15,1] | 47/56 |
| (b) B3 chỉ P∈R | 57 | 57 | 9,5 % [7,4–12,1] | 2,1 | 27,5 | 14,9 | 31/219 | 26/381 | 24/31 |
| (b) B4 chỉ G∈R | 62 | 62 | 10,3 % [8,1–13,0] | 1,5 | 40,2 | 16,2 | 36/219 | 26/381 | 34/36 |
| (b) B6 đúng 1 ứng viên ∈R trong {P,G} | 100 | 100 | 16,7 % | 3,6 | 27,9 | 26,2 | 56 | 44 | 47/56 |
| (d) D1 ({P}∪P_all∪{G})∩R, duy nhất/nhiều cầu nhất | 101 | 101 | 16,8 % [14,1–20,0] | 5,7 | 17,7 | 26,5 | 56 | 45 | 47/56 |
| (d) D2 7 luồng (thêm crop v6/v5/VL/GLM-force)∩R | 150 | 150 | 25,0 % [21,7–28,6] | 10,7 | 14,1 | 39,3 | 74/219 = 33,8 % | 76/381 = 19,9 % | 59/74 |
| (d) D3 ∈R được CẢ họ Paddle lẫn họ GLM đọc ra | 33 | 33 | 5,5 % [3,9–7,6] | 0,02 | ≈ 1 400 | 8,6 | 19/219 | 14/381 | 18/19 |

Ghi chú: P và G KHÔNG bao giờ mâu thuẫn trong R trên M2 (0/600 ô có P∈R ∧ G∈R ∧ P≠G), nên B1 = B2 = B6. Hai engine bổ sung nhau: chỉ 19/100 ô cứu được có cả hai. Đếm cầu D1: 45 ô 1 cầu, 39 ô 2 cầu, 17 ô 3 cầu. Chữ cứu được 100 % là URO (chữ Hán thông dụng); 19/100 ô có K là Ext-B (chữ Nôm riêng) bị engine mới quy về chữ Hán ∈R.

### Độ chính xác thay thế (đo trên M1, n = 600, K ∈ R = kinhhannom đúng)
P(đề xuất == K | luật bắn): B1 427/468 = 91,2 % [88,3–93,5]; B3 90,7 %; B4 93,4 %; A2 95,5 % [91,9–97,5]; D1 91,4 %; D2 91,5 %; D3 93,6 %. Trong 41 ô B1 đề xuất ∈R nhưng ≠K: 16 là biến thể giản/phồn (OpenCC: 别/別, 爲/為…), 20 là chữ gần hình trong SinoNom_Similar (đa số dị thể: 徳/德, 庒/庄, 歳/歲, 亊/事), 5 khác hẳn (đồng âm trong R: 除/徐 giờ, 扛/江 giăng, 但/旦 đến, 共/其 cùng, 連/蓮 trên…). Nếu coi biến thể giản/phồn là đúng: B1 94,7 %, A2 96,4 %, D3 95,7 %. Ước lượng độc lập từ hoán vị (tỷ lệ ∈R giả kỳ vọng): B1 3,6/100 = 3,6 %, D1 5,7 %, D2 7,1 %, A2/D3 ≈ 0 %.

## 2. Luật (c) PHỦ QUYẾT — engine thứ hai như bộ phân loại "K sai" (M1 = K đúng, M2 = K sai)

| Luật phủ quyết | TPR (M2) | FPR (M1) | Lift | M3 bắn | Áp lên corpus: GOLD 53 604 bị oan / khối 27 770 bắt được → precision |
|---|---|---|---|---|---|
| C1 P≠K | 74,0 % | 44,0 % | 1,7 | 97,0 % | 23 586 / 20 550 → 46,6 % |
| C2 G≠K | 72,2 % | 31,0 % | 2,3 | 88,3 % | 16 617 / 20 041 → 54,7 % |
| C3 P≠K ∧ G≠K | 57,7 % | 20,5 % | 2,8 | 85,3 % | 10 989 / 16 014 → 59,3 % |
| C5 P==G≠K | 8,0 % | 3,8 % | 2,1 | 2,0 % | 2 055 / 2 222 → 51,9 % |
| C6 (P≠K ∧ P∈R) ∨ (G≠K ∧ G∈R) | 16,7 % | 7,7 % | 2,2 | 0,7 % | 4 110 / 4 628 → 53,0 % |

Kết luận (c): KHÔNG dùng được. Engine mới bất đồng với K ở 31–44 % ô mà K đúng; mọi luật phủ quyết có precision ≈ 47–59 % tức cứ 2 ô bị phủ quyết thì 1 ô oan. Riêng M3 (㝵) bị phủ quyết 85–97 % nhưng vì LÝ DO SAI (engine không có 㝵/𠊚 trong charset, không phải vì nhìn thấy 𠊚).

## 3. M3 — lớp 㝵/"người" (n = 300): PHÁ SẢN ở mọi luật
- 0 % 𠊚, 0 % 㝵, 0 % 𠊛 ở CẢ 6 luồng (P, G, v6 crop, v5 crop, VL, GLM-force): 0/1 800 lần đọc.
- Luật (b)/(d): bắn 2/300 = 0,7 % (命, đúng bằng mức hoán vị 2,0) → không tín hiệu.
- Luật (a) sau chuẩn hoá giản/phồn: P≈G bắn 67/300 = 22,3 %, trong đó 60 là 尋 (P đọc 尋 99 lần, G đọc 寻 142 lần) — ĐỒNG THUẬN HAI ENGINE VỀ MỘT CHỮ SAI (người phán 𠊚). Đây là bằng chứng lỗi TƯƠNG QUAN: hai engine huấn luyện trên chữ Hán in cùng quy chữ Nôm lạ về chữ Hán gần hình. Hệ quả: đồng thuận P==G mà ∉R KHÔNG được coi là bằng chứng (trên M2 có 71/90 ô như vậy).

## 4. Hiện tượng phụ: 28,5 % ô M2 được engine mới XÁC NHẬN chữ ∉R của kinhhannom
A4: 171/600 ô (SILVER 46, SYLLABLE 125) có P==K hoặc G==K; 42 ô cả ba trùng. Toàn bộ 171 chữ K đều có âm trong từ điển, nhưng chỉ 12 ô âm đó trùng âm QN sau bỏ dấu/thanh (詩/thì, 庄/chắng, 異/là — nghi VietOCR rụng dấu); 159 ô âm khác hẳn (而/làm, 解/năm, 百/làm). Hai cách hiểu, chưa phân xử được bằng R: (i) âm QN gióng lệch ô/cột hoặc từ điển thiếu âm → K đúng, "trượt" không do OCR; (ii) lỗi tương quan như M3 (K cũng quy chữ Nôm lạ về chữ Hán gần hình). Vì M3 cho thấy (ii) có thật, KHÔNG dùng A4 để phục hồi K.

## 5. Suy rộng lên khối 27 770 ô (hậu phân tầng theo tier×sách về pool SILVER+SYLLABLE R≠∅ = 16 991 ô; bootstrap phân tầng 2 000 lần)

| Luật | Tỷ lệ hậu phân tầng | SILVER+SYLLABLE (16 991) | REVIEW 11 545 — KỊCH BẢN "như SYLLABLE" (chưa đo) | Tổng khối 27 770 | ×độ chính xác M1 → ô "đúng" ước tính (SILVER+SYLLABLE) |
|---|---|---|---|---|---|
| B1/B6/D1 (1 trong 2 engine cột ∈R) | 17,1 % | **2 901 [2 417–3 427]** | 1 333 [1 006–1 749] | **4 234 [3 423–5 176] ≈ 15 %** | 2 647 [2 135–3 204] |
| D2 (7 luồng) | 25,3 % | 4 307 [3 736–4 882] | 2 303 [1 870–2 806] | 6 610 [5 611–7 682] ≈ 24 % | 3 941 [3 314–4 569] (7 % giả theo hoán vị) |
| A2 (P==G ∧ ∈R) | 3,2 % | 539 [321–784] | 242 [123–472] | 781 [444–1 256] ≈ 3 % | 515 [295–765] |
| D3 (cả hai họ engine ∈R) | 5,7 % | 960 [652–1 294] | 424 [254–701] | 1 384 [906–1 995] ≈ 5 % | 899 [599–1 215] |

Giả định phải ghi rõ: (1) M2 KHÔNG chứa ô REVIEW (11 442 ô = 41 % khối, không crop, S3 dưới ngưỡng); dòng REVIEW là kịch bản áp tỷ lệ SYLLABLE, có thể lạc quan vì REVIEW là phần khó hơn. (2) Pool SILVER dùng 6 706 ô R≠∅ trong khi khối chỉ tính 5 959 SILVER không cầu → dòng SILVER+SYLLABLE cao hơn khối thật ≈ 4 %. (3) "∈R" là điều kiện CẦN; độ chính xác thay thế đo trên M1 (ô dễ, K đúng) có thể cao hơn thực tế trên ô khó. (4) Trong phần SILVER, 84 % ô cứu được (47/56) chỉ XÁC NHẬN nhãn S2∩S3 sẵn có; phần ứng viên MỚI thật sự nằm ở SYLLABLE: 11,5 % [8,7–15,1] ≈ 1 190 [900–1 550] ô (+ REVIEW nếu kịch bản đúng).

## 6. Kết luận thẳng
- Có tín hiệu thật nhưng NHỎ: luật tốt nhất (b)/(d) cấp 1 engine cột ∈R thêm bằng chứng độc lập cho 16,7 % ô trượt (lift 28× so hoán vị, 26× so ngẫu nhiên tần suất), độ chính xác thay thế 91–95 %. Mở rộng 7 luồng lên 25 % nhưng 7 % trong đó là giả theo hoán vị. Đồng thuận 2 engine ∈R chắc nhất (≈ 96 %) nhưng chỉ 3 %.
- ≈ 75–83 % khối trượt vẫn KHÔNG có engine nào chạm được; lớp 㝵/người (1 183 ô) và mọi chữ Nôm Ext-B: 0.
- Phủ quyết bằng engine thứ hai: vô dụng (precision ≈ 50 %). Đồng thuận không qua R: nguy hiểm (M3: 22 % đồng thuận vào chữ sai 尋).
- Nếu triển khai: chạy P (cột, ~15 phút CPU/4 032 cột) + G (cột, ~1,5 s/cột GPU Metal) trên toàn khối; chỉ nhận đề xuất khi ∈R và hai engine không mâu thuẫn (B6/D1); gắn nhãn cấp "có 1 hoặc 2 cầu OCR độc lập", không gộp vào GOLD; loại hẳn lớp 㝵/người và chữ K Ext-B khỏi luật.
