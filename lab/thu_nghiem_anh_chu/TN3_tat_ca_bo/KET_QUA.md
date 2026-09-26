# THỬ NGHIỆM 3 — Hướng nào để ảnh và chữ trong GOLD phải đúng, trên cả 8 bộ đang có trên đĩa (26/09/2026)

Sinh tự động bởi `t03_report.py` từ `matrix.csv`, `t02_summary.json`, `t01_summary.json` (invariant t01: **ALL PASS**, t02: **ALL PASS**). 0 lần gọi API, không tải model, không mở ảnh bằng LLM, repo chỉ đọc ngoài `lab/thu_nghiem_anh_chu/TN3_tat_ca_bo/` và `measure_out/_thu_nghiem_anh_chu/TN3/`. MPS chỉ dùng cho bước chấm ứng viên sửa hộp của L83/KVK (TN2, 3 giây).

Mục tiêu GOLD: (1) crop là đúng **một** chữ, (2) chữ đó đúng là nhãn. Ưu tiên độ chính xác hơn số lượng. Mọi hướng chỉ **hạ** ô (không bao giờ sửa nhãn); ô bị hạ không mất, chỉ không được chứng nhận "ảnh + chữ".

Mức chắc: **CHẮC CHẮN THEO MÁY** = đo trên nhãn người IHR · **ƯỚC LƯỢNG** = suy từ văn bản người của dị bản + mô hình hiệu chuẩn trên IHR · **SUY ĐOÁN** = chuyển tỉ lệ từ sách khác, **cần người kiểm**.

## 0. Trả lời ngắn

1. **Chỉ TruyenKieu1872 đạt mục tiêu** theo tiêu chí đăng ký trước (CHẮC CHẮN THEO MÁY). Hướng giữ nhiều ô nhất mà vẫn đạt là **H3** (hình học + dị bản người): 12.654 ô (64,7 %), hai vế **99,72** [99,63–99,81]. H4 chặt hơn: 9.766 ô, **99,87** [99,80–99,93].
2. **LucVanTien1916 không hướng nào đạt 99,5 %.** Tốt nhất là H2 = H4 (sách không có văn bản người của dị bản nên H3 ≡ H1): 2.737 ô (23,6 %), **99,49** [99,14–99,77] — hụt 99,5 % ở điểm, cận dưới 99,14 %. CHẮC CHẮN THEO MÁY.
3. **Thạch bản (KVK, L83): H4 chính xác nhất** (điểm KVK 99,82 %, L83 99,81 %), nhưng **cận bi quan chưa tới 99 %** (KVK 98,03 %, L83 94,76 %; bỏ cận "không cần d" — độ nhạy thêm sau — còn 98,62 % / 97,90 %). ƯỚC LƯỢNG. Không chứng nhận được nếu không có người kiểm.
4. **Chrestomathie1872 và 3 cuốn STT: không hướng nào đạt.** Chr H2 = H4: điểm 99,11 %, cận bi quan 97,14 %. STT H2 = H4: điểm 98,23–98,44 %, cận bi quan ≤ 96,46 % (tham chiếu Rv3: điểm ≤ 99,09 %, bi quan ≤ 97,23 %). SUY ĐOÁN — cần người kiểm.
5. **H1 (hình học) là tầng bắt buộc đầu tiên ở mọi bộ.** Trên IHR nó cắt lỗi ảnh L16 252 → 55 ô (phần còn lại chủ yếu là "khe chưa xác định"), TK 134 → 5, chỉ hạ khoảng 5 % ô. Nhưng lỗi **chữ** gần như nguyên vẹn, nên một mình nó dừng ở 98,05 % (L16) / 98,85 % (TK).
6. **Vòng sửa hộp (H5, H6) gần như không đổi gì**: H5 thêm tổng 49 ô trên 8 bộ so với H1 (stt2 +2, stt11 +1, Chr +4, L16 +42); H6 thêm 55 ô so với H4 (stt2 +2, stt11 +1, Chr +4, L16 +48) và làm L16 tụt 99,49 → 99,43 %. → để **TẮT** mặc định.
7. **Hướng bền qua mọi loại sách: H4** (hình học → bộ kiểm ảnh → dị bản người nếu có). Theo điểm, H4 cao nhất (hoặc ngang) ở 8/8 bộ, trải cả 4 loại sách (mộc bản, thạch bản, in văn xuôi, chép tay); theo cận dưới/bi quan cao nhất ở 7/8 bộ (ngoại lệ: Chr — xem §4). Nhưng vế **chữ** chỉ được đẩy qua 99,5 % khi có văn bản người của dị bản: trên TK, bỏ dị bản (H2) chỉ còn 99,12 %, thêm dị bản (H4) lên 99,87 %.
8. **Bắt buộc người kiểm (hoặc đọc lại) trước khi gọi là "GOLD chính xác"**: L16 (sát ngưỡng), KVK, L83, Chr, stt2, stt4, stt11. Đọc lại STT bằng kim lt2 (H7) **không đủ**: kể cả khi xoá hết lỗi riêng của STT, cận bi quan của STT vẫn ≤ 97,31 % (xem §6).

## 1. Cách làm (viết trước khi tính; không đổi sau khi thấy kết quả)

### 1.1 Tám bộ

| Bộ | Loại sách | GOLD | Sự thật có trên máy | Mức chắc của số độ chính xác |
|---|---|---:|---|---|
| stt2 (SachThanhTruyen stt2) | chép tay (lt1) | 17.678 | không (Borg chỉ gián tiếp) | SUY ĐOÁN |
| stt4 (SachThanhTruyen stt4) | chép tay (lt1) | 17.440 | không (Borg chỉ gián tiếp) | SUY ĐOÁN |
| stt11 (SachThanhTruyen stt11) | chép tay (lt1) | 17.589 | không (Borg chỉ gián tiếp) | SUY ĐOÁN |
| Chr (Chrestomathie1872) | in văn xuôi | 4.238 | không | SUY ĐOÁN |
| L83 (LucVanTien1883) | thạch bản | 10.910 | văn bản người của dị bản: LucVanTien1916 IHR | ƯỚC LƯỢNG |
| KVK (KimVanKieu1884) | thạch bản | 19.958 | văn bản người của dị bản: Kiều 1871 LVD, TK1872 IHR | ƯỚC LƯỢNG |
| L16 (LucVanTien1916) | mộc bản IHR | 11.587 | chữ người IHR + hộp cột người vẽ | CHẮC CHẮN THEO MÁY |
| TK (TruyenKieu1872) | mộc bản IHR | 19.554 | chữ người IHR + hộp cột người vẽ; dị bản Kiều 1871 LVD | CHẮC CHẮN THEO MÁY |

### 1.2 Bảy hướng + một tham chiếu

| Mã | Nội dung | Ngưỡng / nguồn |
|---|---|---|
| H0 | giữ nguyên GOLD | — |
| H1 | cổng hình học: hạ nếu A0 (crop rỗng/cắt nét, ảnh ô khác, 1 hộp 2 cột) ∪ B0 (cờ "một chữ": bleed, truncated, blank, tall, dup, ov_heavy, f_tight_nb) ∪ cột có số chữ OCR ≠ số âm QN ∪ cờ trượt vòng 2 | cờ trượt chọn NGOÀI sách: L16 dùng `bc_k06v1` (chọn trên TK = G6 của TN1); mọi bộ khác `bc_k05dxv05` (chọn trên L16 = G7 của TN1) |
| H2 | H1 ∧ bộ kiểm ảnh ở τ = 0,995 | TN1/v3: L16 p_wood_T ∧ viss_T; TK p_wood_L ∧ viss_L; L83/KVK p_wood_T ∧ p_wood_L ∧ viss_X; Chr p_wood_T ∧ p_wood_L (C2); STT bộ kiểm viết tay LOBO r5 (q = 0,00015, ≥ 3 nguyên mẫu người) |
| H3 | H1 ∧ nhãn được văn bản người của dị bản chứng (`attested`) | chỉ TK, KVK, L83 có văn bản người của dị bản; L16/Chr/STT: H3 ≡ H1 |
| H4 | H1 ∧ H2 ∧ H3 | = cấu hình (e) của TN1 |
| H5 | H1 ∪ ô được vòng sửa hộp TN2 nhận (luật trùng hộp NGHIÊM), không dính A0/B0/cột lệch; ô được sửa dùng crop mới | θ TN2: L16 mô hình T θ = 0,25, TK mô hình L θ = 0,45 (LOBO); bộ khác: T (θ 0,45) và L (θ 0,25) cùng chọn một ứng viên |
| H6 | H4 ∪ ô được sửa (như H5) có `attested` ở bộ có dị bản | ô được sửa chỉ qua bộ kiểm của chính vòng sửa (m ≥ θ), không qua p_wood |
| Rv3 | **tham chiếu, không xếp hạng**: H4 ∧ ¬luật A (ảnh ô khác, rescue, cầu tự dạng, văn bản yếu) ∧ ¬M-OCR t50 | = luật cấp ô của chính sách v3, BỎ các cổng cấp bộ |
| H7 | chỉ mô tả, không chạy: đọc lại STT bằng kim lang_type = 2 (cần API) | §6 |

### 1.3 Cách ra con số độ chính xác hai vế (đúng hai vế = crop ở đúng khe của chữ **và** nhãn đúng chữ đó)

- **L16, TK — CHẮC CHẮN THEO MÁY.** Nhãn đúng = V1+(nhãn, chữ người IHR); ảnh đúng = `slot_ok = 1` (khe chưa xác định tính SAI). Ngưỡng của mỗi sách chọn trên sách kia (LOBO). CI 95 % bootstrap cụm theo trang, B = 2.000. Ô được sửa hộp: khe mới do TN2 chấm bằng hộp cột người vẽ.
- **KVK, L83 — ƯỚC LƯỢNG.**
  - Vế chữ: mỗi ô giữ thuộc một lớp dị bản (attested / contradicted / unattestable). Nhân số ô từng lớp với tỉ lệ nhãn sai **của cùng lớp, cùng hướng, đo trên TK** (hiệu chuẩn ngược; bootstrap trang TK, B = 2.000). **Giả định:** P(nhãn sai | lớp, hướng) chuyển nguyên từ TK (mộc bản, dị bản Kiều 1871, một tham chiếu) sang thạch bản (KVK: hai tham chiếu 1871/1872; L83: tham chiếu L16).
  - Cận bi quan vế chữ = lớn nhất của: CI trên; cận "không cần d" (coi mọi chênh với dị bản là lỗi, q cận trên CI = 0,346, KVK hai tham chiếu q = 1−(1−q)² = 0,573); CI trên của tổng p M-OCR vòng 2.
  - Vế ảnh: tổng p trượt từng ô của mô hình vòng 2 (`gate_v2`, logistic học trên IHR, bootstrap có sai số chuyển bộ σ = 0,393, B = 1.000). Hướng có bộ kiểm ảnh: (tổng p của phần giữ khi bỏ bộ kiểm) × tỉ lệ nhận trượt của bộ kiểm, đo trên IHR ở ô trượt/khe chưa xđ **sống sót H1**: 8/60 = 13,3 %. Cận bi quan = không cho bộ kiểm điểm nào. Ô được sửa hộp: tỉ lệ khe mới sai đo trên L16 = 2/55.
- **Chr, stt2/4/11 — SUY ĐOÁN.**
  - Vế ảnh như trên. STT: bể trượt thật 1.630–2.500 ô (vòng 4–5) phân bổ theo p trượt vòng 2; bộ kiểm viết tay nhận 11,3 % [6,8–15,7] crop trượt thật (đo trên Borg, LOBO theo sách).
  - Vế chữ: không có sự thật cục bộ. Chuyển tỉ lệ nhãn sai đo trên L16/TK ở **cùng hướng** (bỏ điều kiện dị bản). Điểm = trung bình L16/TK; cận bi quan = lớn nhất của L16, TK, L83, KVK. Riêng STT cộng thêm lỗi chỉ STT có: cầu tự dạng + nhãn rescue (p vòng 2) và bể "Hán hoá" do OCR lt1 750 [481–954] ô, rải đều trên 4.750 ô có âm Borg viết bằng chữ Nôm riêng.
- **Hai vế ngoài IHR**: số lỗi = lỗi ảnh + lỗi chữ (cộng, tức cận hợp). Khoảng = cộng hai đầu tương ứng (rộng hơn khoảng đồng thời).

### 1.4 Tiêu chí đạt (định trước)

- Đo được (L16, TK): điểm ≥ 99,5 %; ghi "chắc" khi cận dưới CI ≥ 99,5 %.
- Ước lượng / suy đoán: **cận bi quan ≥ 99,0 %**.
- Phải giữ ≥ 50 ô. Hướng tốt nhất của một bộ = hướng đạt mà giữ nhiều ô nhất. Rv3 là tham chiếu, không xếp hạng.

## 2. Ma trận bộ × hướng

### 2.1 Bảng gọn: `% ô giữ · độ chính xác hai vế % [khoảng]`

Khoảng: L16/TK = CI 95 % cụm trang; bộ khác = [cận bi quan – cận lạc quan]. ✔ = đạt tiêu chí §1.4.

| Bộ (mức chắc) | H0 | H1 | H2 | H3 | H4 | H5 | H6 | Rv3 |
|---|---|---|---|---|---|---|---|---|
| **stt2** (SUY ĐOÁN) | 100,0 · 90,90 [88,40–93,21] | 73,7 · 95,05 [93,52–96,66] | 13,6 · 98,38 [96,39–99,24] | 73,7 · 95,05 [93,52–96,66] | 13,6 · 98,38 [96,39–99,24] | 73,7 · 95,05 [93,52–96,66] | 13,6 · 98,36 [96,38–99,20] | 12,8 · 98,99 [97,13–99,64] |
| **stt4** (SUY ĐOÁN) | 100,0 · 91,95 [89,74–94,24] | 67,0 · 94,67 [93,05–96,39] | 8,2 · 98,23 [96,15–99,18] | 67,0 · 94,67 [93,05–96,39] | 8,2 · 98,23 [96,15–99,18] | 67,0 · 94,66 [93,05–96,39] | 8,2 · 98,21 [96,15–99,15] | 7,8 · 98,77 [96,80–99,53] |
| **stt11** (SUY ĐOÁN) | 100,0 · 92,00 [89,78–93,98] | 59,2 · 95,86 [94,49–97,21] | 14,3 · 98,44 [96,46–99,26] | 59,2 · 95,86 [94,49–97,21] | 14,3 · 98,44 [96,46–99,26] | 59,2 · 95,86 [94,49–97,21] | 14,3 · 98,42 [96,46–99,23] | 13,7 · 99,09 [97,23–99,69] |
| **Chr** (SUY ĐOÁN) | 100,0 · 91,99 [87,96–96,95] | 42,3 · 98,30 [97,27–98,95] | 7,3 · 99,11 [97,14–99,77] | 42,3 · 98,30 [97,27–98,95] | 7,3 · 99,11 [97,14–99,77] | 42,4 · 98,29 [97,26–98,94] | 7,4 · 99,05 [97,10–99,71] | 7,3 · 99,11 [97,14–99,77] |
| **L83** (ƯỚC LƯỢNG) | 100,0 · 97,95 [72,33–98,63] | 93,3 · 98,26 [72,41–98,79] | 17,0 · 98,24 [86,47–98,92] | 27,9 · 99,67 [88,27–99,80] | 5,9 · 99,81 [94,76–99,92] | 93,3 · 98,26 [72,41–98,79] | 5,9 · 99,81 [94,76–99,92] | 5,9 · 99,81 [94,76–99,92] |
| **KVK** (ƯỚC LƯỢNG) | 100,0 · 98,94 [79,38–99,21] | 97,7 · 99,09 [79,35–99,30] | 23,5 · 99,30 [96,74–99,55] | 72,6 · 99,66 [86,98–99,80] | 18,4 · 99,82 [98,03–99,92] | 97,7 · 99,09 [79,35–99,30] | 18,4 · 99,82 [98,03–99,92] | 18,3 · 99,82 [98,06–99,92] |
| **L16** (CHẮC CHẮN THEO MÁY) | 100,0 · 96,39 [95,71–97,04] | 94,5 · 98,05 [97,64–98,42] | 23,6 · 99,49 [99,14–99,77] | 94,5 · 98,05 [97,64–98,42] | 23,6 · 99,49 [99,14–99,77] | 94,9 · 98,07 [97,66–98,44] | 24,0 · 99,43 [99,07–99,72] | 23,6 · 99,49 [99,14–99,77] |
| **TK** (CHẮC CHẮN THEO MÁY) | 100,0 · 98,16 [97,87–98,40] | 94,7 · 98,85 [98,70–98,99] | 69,8 · 99,12 [98,95–99,28] | 64,7 · 99,72 [99,63–99,81] ✔ | 49,9 · 99,87 [99,80–99,93] ✔ | 94,7 · 98,85 [98,70–98,99] | 49,9 · 99,87 [99,80–99,93] ✔ | 49,9 · 99,87 [99,80–99,93] ✔ |

### 2.2 Bảng đủ theo từng bộ

#### stt2 — SachThanhTruyen stt2 (chép tay (lt1); GOLD 17.678; 968 lớp nhãn) — SUY ĐOÁN

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 17.678 (100,0 %) | 0 | 0 (0) | **90,90** [88,40–93,21] | 95,51 [94,06–96,91] | 95,40 [94,35–96,31] | 1.607,8 (bi quan 2.049,8) | không |
| H1 hình học | 13.025 (73,7 %) | 4.653 | 74 (90) | **95,05** [93,52–96,66] | 95,49 [94,13–96,90] | 99,56 [99,40–99,76] | 644,2 (bi quan 843,4) | không |
| H2 = H1 + kiểm ảnh | 2.400 (13,6 %) | 15.278 | 803 (6.891) | **98,38** [96,39–99,24] | 98,65 [96,81–99,36] | 99,73 [99,58–99,88] | 38,9 (bi quan 86,7) | không |
| H3 = H1 + dị bản người | 13.025 (73,7 %) | 4.653 | 74 (90) | **95,05** [93,52–96,66] | 95,49 [94,13–96,90] | 99,56 [99,40–99,76] | 644,2 (bi quan 843,4) | không |
| H4 = H1+H2+H3 | 2.400 (13,6 %) | 15.278 | 803 (6.891) | **98,38** [96,39–99,24] | 98,65 [96,81–99,36] | 99,73 [99,58–99,88] | 38,9 (bi quan 86,7) | không |
| H5 = H1 + sửa hộp | 13.027 (2 ô crop sửa) (73,7 %) | 4.651 | 74 (90) | **95,05** [93,52–96,66] | 95,49 [94,13–96,90] | 99,56 [99,40–99,76] | 644,4 (bi quan 843,6) | không |
| H6 = H4 + sửa hộp | 2.402 (2 ô crop sửa) (13,6 %) | 15.276 | 803 (6.891) | **98,36** [96,38–99,20] | 98,63 [96,81–99,32] | 99,73 [99,58–99,88] | 39,4 (bi quan 86,8) | không |
| Rv3 (tham chiếu) | 2.259 (12,8 %) | 15.419 | 816 (7.033) | **98,99** [97,13–99,64] | 99,23 [97,50–99,75] | 99,76 [99,63–99,89] | 22,8 (bi quan 64,8) | không |

Độ nhạy (thêm sau, không xếp hạng): nếu bể trượt STT **không** được H1 giảm (rải đều theo số ô, như quy tắc `stt_ok_allowed` của v3) thì cận bi quan hai vế: H1 89,38 %, H2 92,76 %, Rv3 93,79 %.

#### stt4 — SachThanhTruyen stt4 (chép tay (lt1); GOLD 17.440; 925 lớp nhãn) — SUY ĐOÁN

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 17.440 (100,0 %) | 0 | 0 (0) | **91,95** [89,74–94,24] | 95,16 [93,64–96,66] | 96,80 [96,10–97,59] | 1.403,1 (bi quan 1.788,8) | không |
| H1 hình học | 11.679 (67,0 %) | 5.761 | 105 (126) | **94,67** [93,05–96,39] | 95,17 [93,74–96,67] | 99,50 [99,31–99,72] | 623,1 (bi quan 811,7) | không |
| H2 = H1 + kiểm ảnh | 1.428 (8,2 %) | 16.012 | 811 (8.197) | **98,23** [96,15–99,18] | 98,69 [96,87–99,39] | 99,54 [99,28–99,79] | 25,3 (bi quan 55,0) | không |
| H3 = H1 + dị bản người | 11.679 (67,0 %) | 5.761 | 105 (126) | **94,67** [93,05–96,39] | 95,17 [93,74–96,67] | 99,50 [99,31–99,72] | 623,1 (bi quan 811,7) | không |
| H4 = H1+H2+H3 | 1.428 (8,2 %) | 16.012 | 811 (8.197) | **98,23** [96,15–99,18] | 98,69 [96,87–99,39] | 99,54 [99,28–99,79] | 25,3 (bi quan 55,0) | không |
| H5 = H1 + sửa hộp | 11.679 (67,0 %) | 5.761 | 105 (126) | **94,66** [93,05–96,39] | 95,17 [93,74–96,67] | 99,50 [99,31–99,72] | 623,2 (bi quan 811,7) | không |
| H6 = H4 + sửa hộp | 1.428 (8,2 %) | 16.012 | 811 (8.197) | **98,21** [96,15–99,15] | 98,68 [96,87–99,36] | 99,54 [99,28–99,79] | 25,5 (bi quan 55,0) | không |
| Rv3 (tham chiếu) | 1.358 (7,8 %) | 16.082 | 816 (8.502) | **98,77** [96,80–99,53] | 99,19 [97,45–99,72] | 99,58 [99,35–99,81] | 16,7 (bi quan 43,5) | không |

Độ nhạy (thêm sau, không xếp hạng): nếu bể trượt STT **không** được H1 giảm (rải đều theo số ô, như quy tắc `stt_ok_allowed` của v3) thì cận bi quan hai vế: H1 89,00 %, H2 90,76 %, Rv3 91,88 %.

#### stt11 — SachThanhTruyen stt11 (chép tay (lt1); GOLD 17.589; 876 lớp nhãn) — SUY ĐOÁN

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 17.589 (100,0 %) | 0 | 0 (0) | **92,00** [89,78–93,98] | 96,23 [94,92–97,41] | 95,76 [94,86–96,57] | 1.407,5 (bi quan 1.796,9) | không |
| H1 hình học | 10.407 (59,2 %) | 7.182 | 102 (129) | **95,86** [94,49–97,21] | 96,26 [95,03–97,44] | 99,60 [99,46–99,77] | 430,6 (bi quan 573,6) | không |
| H2 = H1 + kiểm ảnh | 2.511 (14,3 %) | 15.078 | 716 (6.701) | **98,44** [96,46–99,26] | 98,62 [96,77–99,35] | 99,81 [99,69–99,91] | 39,2 (bi quan 89,0) | không |
| H3 = H1 + dị bản người | 10.407 (59,2 %) | 7.182 | 102 (129) | **95,86** [94,49–97,21] | 96,26 [95,03–97,44] | 99,60 [99,46–99,77] | 430,6 (bi quan 573,6) | không |
| H4 = H1+H2+H3 | 2.511 (14,3 %) | 15.078 | 716 (6.701) | **98,44** [96,46–99,26] | 98,62 [96,77–99,35] | 99,81 [99,69–99,91] | 39,2 (bi quan 89,0) | không |
| H5 = H1 + sửa hộp | 10.408 (1 ô crop sửa) (59,2 %) | 7.181 | 102 (129) | **95,86** [94,49–97,21] | 96,26 [95,03–97,44] | 99,60 [99,46–99,77] | 430,7 (bi quan 573,6) | không |
| H6 = H4 + sửa hộp | 2.512 (1 ô crop sửa) (14,3 %) | 15.077 | 716 (6.701) | **98,42** [96,46–99,23] | 98,61 [96,77–99,32] | 99,81 [99,69–99,91] | 39,6 (bi quan 89,0) | không |
| Rv3 (tham chiếu) | 2.402 (13,7 %) | 15.187 | 724 (6.856) | **99,09** [97,23–99,69] | 99,26 [97,54–99,77] | 99,83 [99,69–99,92] | 21,9 (bi quan 66,5) | không |

Độ nhạy (thêm sau, không xếp hạng): nếu bể trượt STT **không** được H1 giảm (rải đều theo số ô, như quy tắc `stt_ok_allowed` của v3) thì cận bi quan hai vế: H1 90,29 %, H2 93,68 %, Rv3 94,66 %.

#### Chr — Chrestomathie1872 (in văn xuôi; GOLD 4.238; 705 lớp nhãn) — SUY ĐOÁN

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 4.238 (100,0 %) | 0 | 0 (0) | **91,99** [87,96–96,95] | 98,57 [97,73–99,00] | 93,42 [90,23–97,95] | 339,6 (bi quan 510,1) | không |
| H1 hình học | 1.794 (42,3 %) | 2.444 | 219 (312) | **98,30** [97,27–98,95] | 98,58 [97,83–99,01] | 99,72 [99,44–99,95] | 30,4 (bi quan 49,0) | không |
| H2 = H1 + kiểm ảnh | 309 (7,3 %) | 3.929 | 614 (2.569) | **99,11** [97,14–99,77] | 99,32 [97,62–99,81] | 99,79 [99,52–99,96] | 2,8 (bi quan 8,8) | không |
| H3 = H1 + dị bản người | 1.794 (42,3 %) | 2.444 | 219 (312) | **98,30** [97,27–98,95] | 98,58 [97,83–99,01] | 99,72 [99,44–99,95] | 30,4 (bi quan 49,0) | không |
| H4 = H1+H2+H3 | 309 (7,3 %) | 3.929 | 614 (2.569) | **99,11** [97,14–99,77] | 99,32 [97,62–99,81] | 99,79 [99,52–99,96] | 2,8 (bi quan 8,8) | không |
| H5 = H1 + sửa hộp | 1.798 (4 ô crop sửa) (42,4 %) | 2.440 | 219 (312) | **98,29** [97,26–98,94] | 98,58 [97,83–99,01] | 99,72 [99,43–99,93] | 30,7 (bi quan 49,3) | không |
| H6 = H4 + sửa hộp | 313 (4 ô crop sửa) (7,4 %) | 3.925 | 614 (2.569) | **99,05** [97,10–99,71] | 99,31 [97,62–99,78] | 99,74 [99,48–99,93] | 3,0 (bi quan 9,1) | không |
| Rv3 (tham chiếu) | 309 (7,3 %) | 3.929 | 614 (2.569) | **99,11** [97,14–99,77] | 99,32 [97,62–99,81] | 99,79 [99,52–99,96] | 2,8 (bi quan 8,8) | không |

#### L83 — LucVanTien1883 (thạch bản; GOLD 10.910; 1.610 lớp nhãn) — ƯỚC LƯỢNG

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 10.910 (100,0 %) | 0 | 0 (0) | **97,95** [72,33–98,63] | 98,24 [72,83–98,73] | 99,71 [99,50–99,89] | 223,5 (bi quan 3.019,3) | không |
| H1 hình học | 10.184 (93,3 %) | 726 | 45 (47) | **98,26** [72,41–98,79] | 98,35 [72,58–98,82] | 99,91 [99,83–99,97] | 177,4 (bi quan 2.809,5) | không |
| H2 = H1 + kiểm ảnh | 1.858 (17,0 %) | 9.052 | 1.231 (4.992) | **98,24** [86,47–98,92] | 98,30 [86,64–98,94] | 99,94 [99,83–99,98] | 32,8 (bi quan 251,4) | không |
| H3 = H1 + dị bản người | 3.040 (27,9 %) | 7.870 | 809 (2.302) | **99,67** [88,27–99,80] | 99,76 [88,44–99,83] | 99,91 [99,83–99,97] | 10,1 (bi quan 356,7) | không |
| H4 = H1+H2+H3 | 648 (5,9 %) | 10.262 | 1.413 (6.939) | **99,81** [94,76–99,92] | 99,87 [94,93–99,94] | 99,95 [99,83–99,99] | 1,2 (bi quan 33,9) | không |
| H5 = H1 + sửa hộp | 10.184 (93,3 %) | 726 | 45 (47) | **98,26** [72,41–98,79] | 98,35 [72,58–98,82] | 99,91 [99,83–99,97] | 177,4 (bi quan 2.809,5) | không |
| H6 = H4 + sửa hộp | 648 (5,9 %) | 10.262 | 1.413 (6.939) | **99,81** [94,76–99,92] | 99,87 [94,93–99,94] | 99,95 [99,83–99,99] | 1,2 (bi quan 33,9) | không |
| Rv3 (tham chiếu) | 648 (5,9 %) | 10.262 | 1.413 (6.939) | **99,81** [94,76–99,92] | 99,87 [94,93–99,94] | 99,95 [99,83–99,99] | 1,2 (bi quan 33,9) | không |

Cận bi quan vế chữ do thành phần nào quyết định: mọi hướng: cận không-d. Độ nhạy (thêm sau): bỏ cận "không cần d" → cận bi quan hai vế H2 97,45 %, H3 97,19 %, H4 97,90 %, Rv3 97,90 %.

#### KVK — KimVanKieu1884 (thạch bản; GOLD 19.958; 2.337 lớp nhãn) — ƯỚC LƯỢNG

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 19.958 (100,0 %) | 0 | 0 (0) | **98,94** [79,38–99,21] | 99,16 [79,72–99,31] | 99,78 [99,65–99,90] | 212,2 (bi quan 4.115,7) | không |
| H1 hình học | 19.506 (97,7 %) | 452 | 16 (17) | **99,09** [79,35–99,30] | 99,18 [79,53–99,33] | 99,91 [99,82–99,97] | 177,1 (bi quan 4.027,8) | không |
| H2 = H1 + kiểm ảnh | 4.689 (23,5 %) | 15.269 | 1.757 (8.572) | **99,30** [96,74–99,55] | 99,35 [96,92–99,56] | 99,95 [99,82–99,99] | 33,0 (bi quan 152,8) | không |
| H3 = H1 + dị bản người | 14.484 (72,6 %) | 5.474 | 302 (823) | **99,66** [86,98–99,80] | 99,76 [87,16–99,83] | 99,91 [99,82–99,97] | 48,7 (bi quan 1.886,2) | không |
| H4 = H1+H2+H3 | 3.676 (18,4 %) | 16.282 | 1.837 (9.653) | **99,82** [98,03–99,92] | 99,87 [98,21–99,94] | 99,95 [99,82–99,99] | 6,7 (bi quan 72,3) | không |
| H5 = H1 + sửa hộp | 19.506 (97,7 %) | 452 | 16 (17) | **99,09** [79,35–99,30] | 99,18 [79,53–99,33] | 99,91 [99,82–99,97] | 177,1 (bi quan 4.027,8) | không |
| H6 = H4 + sửa hộp | 3.676 (18,4 %) | 16.282 | 1.837 (9.653) | **99,82** [98,03–99,92] | 99,87 [98,21–99,94] | 99,95 [99,82–99,99] | 6,7 (bi quan 72,3) | không |
| Rv3 (tham chiếu) | 3.657 (18,3 %) | 16.301 | 1.837 (9.653) | **99,82** [98,06–99,92] | 99,87 [98,24–99,94] | 99,95 [99,82–99,99] | 6,6 (bi quan 70,9) | không |

Cận bi quan vế chữ do thành phần nào quyết định: mọi hướng: cận không-d. Độ nhạy (thêm sau): bỏ cận "không cần d" → cận bi quan hai vế H2 98,60 %, H3 98,53 %, H4 98,62 %, Rv3 98,99 %.

#### L16 — LucVanTien1916 (mộc bản IHR; GOLD 11.587; 1.923 lớp nhãn) — CHẮC CHẮN THEO MÁY

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 11.587 (100,0 %) | 0 | 0 (0) | **96,39** [95,71–97,04] | 98,27 [97,94–98,60] | 97,82 [97,15–98,45] | chữ 200 · ảnh 252 · hai vế 418 / 11.585 ô có GT | không |
| H1 hình học | 10.951 (94,5 %) | 636 | 36 (37) | **98,05** [97,64–98,42] | 98,29 [97,95–98,60] | 99,50 [99,20–99,75] | chữ 187 · ảnh 55 · hai vế 213 / 10.949 ô có GT | không |
| H2 = H1 + kiểm ảnh | 2.737 (23,6 %) | 8.850 | 1.410 (4.680) | **99,49** [99,14–99,77] | 99,52 [99,17–99,81] | 99,74 [99,46–99,96] | chữ 13 · ảnh 7 · hai vế 14 / 2.736 ô có GT | không |
| H3 = H1 + dị bản người | 10.951 (94,5 %) | 636 | 36 (37) | **98,05** [97,64–98,42] | 98,29 [97,95–98,60] | 99,50 [99,20–99,75] | chữ 187 · ảnh 55 · hai vế 213 / 10.949 ô có GT | không |
| H4 = H1+H2+H3 | 2.737 (23,6 %) | 8.850 | 1.410 (4.680) | **99,49** [99,14–99,77] | 99,52 [99,17–99,81] | 99,74 [99,46–99,96] | chữ 13 · ảnh 7 · hai vế 14 / 2.736 ô có GT | không |
| H5 = H1 + sửa hộp | 10.993 (48 ô crop sửa) (94,9 %) | 594 | 35 (36) | **98,07** [97,66–98,44] | 98,29 [97,95–98,61] | 99,51 [99,20–99,76] | chữ 188 · ảnh 54 · hai vế 212 / 10.991 ô có GT | không |
| H6 = H4 + sửa hộp | 2.785 (48 ô crop sửa) (24,0 %) | 8.802 | 1.403 (4.628) | **99,43** [99,07–99,72] | 99,50 [99,15–99,78] | 99,68 [99,37–99,93] | chữ 14 · ảnh 9 · hai vế 16 / 2.784 ô có GT | không |
| Rv3 (tham chiếu) | 2.737 (23,6 %) | 8.850 | 1.410 (4.680) | **99,49** [99,14–99,77] | 99,52 [99,17–99,81] | 99,74 [99,46–99,96] | chữ 13 · ảnh 7 · hai vế 14 / 2.736 ô có GT | không |

#### TK — TruyenKieu1872 (mộc bản IHR; GOLD 19.554; 2.682 lớp nhãn) — CHẮC CHẮN THEO MÁY

| Hướng | Ô giữ (%) | Ô hạ | Lớp mất hết ô (ô trong đó) | Hai vế [khoảng] | Vế chữ [khoảng] | Vế ảnh [khoảng] | Lỗi ước/đo (ô) | Đạt |
|---|---:|---:|---:|---|---|---|---:|---|
| H0 giữ nguyên | 19.554 (100,0 %) | 0 | 0 (0) | **98,16** [97,87–98,40] | 98,86 [98,71–99,00] | 99,28 [99,04–99,49] | chữ 211 · ảnh 134 · hai vế 342 / 18.550 ô có GT | không |
| H1 hình học | 18.511 (94,7 %) | 1.043 | 57 (60) | **98,85** [98,70–98,99] | 98,87 [98,72–99,01] | 99,97 [99,94–99,99] | chữ 198 · ảnh 5 · hai vế 202 / 17.515 ô có GT | không |
| H2 = H1 + kiểm ảnh | 13.643 (69,8 %) | 5.911 | 995 (2.060) | **99,12** [98,95–99,28] | 99,12 [98,95–99,28] | 99,99 [99,98–100,00] | chữ 113 · ảnh 1 · hai vế 113 / 12.908 ô có GT | không |
| H3 = H1 + dị bản người | 12.654 (64,7 %) | 6.900 | 743 (2.133) | **99,72** [99,63–99,81] | 99,76 [99,66–99,84] | 99,97 [99,93–100,00] | chữ 31 · ảnh 4 · hai vế 35 / 12.654 ô có GT | ĐẠT (chắc: cận dưới CI ≥ 99,5 %) |
| H4 = H1+H2+H3 | 9.766 (49,9 %) | 9.788 | 1.349 (3.826) | **99,87** [99,80–99,93] | 99,87 [99,80–99,93] | 100,00 [100,00–100,00] | chữ 13 · ảnh 0 · hai vế 13 / 9.766 ô có GT | ĐẠT (chắc: cận dưới CI ≥ 99,5 %) |
| H5 = H1 + sửa hộp | 18.511 (94,7 %) | 1.043 | 57 (60) | **98,85** [98,70–98,99] | 98,87 [98,72–99,01] | 99,97 [99,94–99,99] | chữ 198 · ảnh 5 · hai vế 202 / 17.515 ô có GT | không |
| H6 = H4 + sửa hộp | 9.766 (49,9 %) | 9.788 | 1.349 (3.826) | **99,87** [99,80–99,93] | 99,87 [99,80–99,93] | 100,00 [100,00–100,00] | chữ 13 · ảnh 0 · hai vế 13 / 9.766 ô có GT | ĐẠT (chắc: cận dưới CI ≥ 99,5 %) |
| Rv3 (tham chiếu) | 9.766 (49,9 %) | 9.788 | 1.349 (3.826) | **99,87** [99,80–99,93] | 99,87 [99,80–99,93] | 100,00 [100,00–100,00] | chữ 13 · ảnh 0 · hai vế 13 / 9.766 ô có GT | ĐẠT (chắc: cận dưới CI ≥ 99,5 %) |

## 3. Xếp hạng theo bộ

| Bộ | Loại sách | Mức chắc | Hướng tốt nhất đạt tiêu chí (ô giữ) | Các hướng đạt | Hướng chính xác nhất theo cận dưới/bi quan | Rv3 (tham chiếu) |
|---|---|---|---|---|---|---|
| stt2 | chép tay (lt1) | SUY ĐOÁN | **không hướng nào** | — | H2: 2.400 ô, 98,38 [96,39–99,24] | không: 98,99 [97,13–99,64] |
| stt4 | chép tay (lt1) | SUY ĐOÁN | **không hướng nào** | — | H2: 1.428 ô, 98,23 [96,15–99,18] | không: 98,77 [96,80–99,53] |
| stt11 | chép tay (lt1) | SUY ĐOÁN | **không hướng nào** | — | H2: 2.511 ô, 98,44 [96,46–99,26] | không: 99,09 [97,23–99,69] |
| Chr | in văn xuôi | SUY ĐOÁN | **không hướng nào** | — | H1: 1.794 ô, 98,30 [97,27–98,95] | không: 99,11 [97,14–99,77] |
| L83 | thạch bản | ƯỚC LƯỢNG | **không hướng nào** | — | H4: 648 ô, 99,81 [94,76–99,92] | không: 99,81 [94,76–99,92] |
| KVK | thạch bản | ƯỚC LƯỢNG | **không hướng nào** | — | H4: 3.676 ô, 99,82 [98,03–99,92] | không: 99,82 [98,06–99,92] |
| L16 | mộc bản IHR | CHẮC CHẮN THEO MÁY | **không hướng nào** | — | H2: 2.737 ô, 99,49 [99,14–99,77] | không: 99,49 [99,14–99,77] |
| TK | mộc bản IHR | CHẮC CHẮN THEO MÁY | **H3** (12.654 ô, 99,72 [99,63–99,81]) | H3, H4, H6 | H4: 9.766 ô, 99,87 [99,80–99,93] | ĐẠT (chắc: cận dưới CI ≥ 99,5 %): 99,87 [99,80–99,93] |

Ghi chú xếp hạng:
- TK: H3 giữ nhiều hơn H4 2.888 ô, đổi lại 743 lớp nhãn mất hết ô (H4: 1.349). Quy tắc "cần attested" không tham số, nhưng quyết định dùng nó dựa trên số đo TK và TK cũng là sách kiểm định ngược của text_attested → số của H3/H4 trên TK là **trong mẫu** đối với luật này (như TN1 §5).
- L16: G7 nguyên văn (ngưỡng trượt chọn trên chính L16) cho 98,13 % — không dùng vì không LOBO. H2 = H4 = Rv3 ở L16 vì L16 không có văn bản người của dị bản và không có ô luật A/M-OCR nào lọt qua H2.

## 4. Hướng bền qua loại sách và khuyến nghị cho pipeline

| Loại sách | Bộ | Hướng điểm cao nhất | H4: điểm | H4 cao nhất theo điểm? | Hướng cận dưới/bi quan cao nhất (giá trị) | H4: cận dưới/bi quan | H4 cao nhất theo cận? |
|---|---|---|---|---|---|---|---|
| chép tay (lt1) | stt2 | H2 (98,38 %) | 98,38 % | có | H2 (96,39 %) | 96,39 % | có |
| chép tay (lt1) | stt4 | H2 (98,23 %) | 98,23 % | có | H2 (96,15 %) | 96,15 % | có |
| chép tay (lt1) | stt11 | H2 (98,44 %) | 98,44 % | có | H2 (96,46 %) | 96,46 % | có |
| in văn xuôi | Chr | H2 (99,11 %) | 99,11 % | có | H1 (97,27 %) | 97,14 % | không |
| thạch bản | L83 | H4 (99,81 %) | 99,81 % | có | H4 (94,76 %) | 94,76 % | có |
| thạch bản | KVK | H4 (99,82 %) | 99,82 % | có | H4 (98,03 %) | 98,03 % | có |
| mộc bản IHR | L16 | H2 (99,49 %) | 99,49 % | có | H2 (99,14 %) | 99,14 % | có |
| mộc bản IHR | TK | H4 (99,87 %) | 99,87 % | có | H4 (99,80 %) | 99,80 % | có |

**Hướng bền = H4** (H1 → bộ kiểm ảnh → dị bản người nếu có): điểm cao nhất (hoặc ngang — hướng trùng tập ô với H4 như H2/H6 ở bộ không có dị bản hay không có ô sửa) ở 8/8 bộ; theo cận dưới/bi quan cao nhất ở 7/8 bộ. Ngoại lệ Chr: cận bi quan vế chữ của H2/H4 lấy tỉ lệ từ L83 ở cùng hướng (chủ yếu là tỉ lệ lỗi của lớp "không tham chiếu" trên TK), cao hơn chút so với H1 — chênh nằm trong độ bất định của phép chuyển, không phải H1 tốt hơn. Không hướng nào thay được người ở vế chữ khi thiếu văn bản người.

**Khuyến nghị cho pipeline (tệp phụ, không phá `labels.csv`):**

1. **Luôn chạy H1** cho mọi sách mới (rẻ, 0 tham số chọn trên sách đó; IHR hạ khoảng 5 %). Ô bị H1 hạ → `text_only`.
2. **Nếu có văn bản người của dị bản**: thêm H3 (chỉ nhận `attested`). Với mộc bản kiểu TK, H3 đã đủ (đo 99,72 %). Với thạch bản dùng **H4** và vẫn phải lấy mẫu người (§7), vì chuyển tỉ lệ TK → thạch bản chưa kiểm định được.
3. **Nếu không có văn bản người**: H4 (= H2) là trần của máy — đo được 99,49 % trên L16; ở Chr/STT chỉ là SUY ĐOÁN. Gắn nhãn `chưa chứng nhận` cho tới khi có mẫu người.
4. **STT (chép tay)**: thêm luật A + M-OCR (Rv3) trước khi cho người xem. Rv3 nâng điểm STT lên 98,77–99,09 % (từ 98,23–98,44 % ở H4), vẫn chưa đủ.
5. **Vòng sửa hộp: TẮT mặc định** (H5 thêm 49 ô, H6 thêm 55 ô trên cả 8 bộ; L16 H6 giảm độ chính xác).

## 5. Kiểm định và chẩn đoán (đọc để biết ước lượng tin được tới đâu)

**Vòng sửa hộp TN2 trên cả 6 book_set** (`t01_summary.json`; chạy thêm L83/KVK bằng bản sao s1a/s1b/s2 của TN2, chỉ đổi thư mục ra):

| book_set | ô bị cờ kim_off | T nhận thô / L nhận thô / cùng ứng viên | huỷ trùng hộp | được sửa | khe mới đúng (IHR) |
|---|---:|---|---:|---:|---|
| L16 | 234 | LOBO một mô hình | 44 | 55 | 53/55 |
| TK | 167 | LOBO một mô hình | 6 | 0 | — |
| Chr | 738 | 50 / 147 / 35 | 25 | 10 | — |
| STT | 1.562 | 88 / 312 / 57 | 32 | 25 | — |
| L83 | 111 | 0 / 6 / 0 | 0 | 0 | — |
| KVK | 61 | 6 / 16 / 4 | 3 | 1 | — |

Bất biến L83/KVK: crop cũ cắt lại trùng md5 tệp giao nộp (True), chữ kim tại nom_idx = ocr_char ở 100 % ô bị cờ (True); quyết định L16/TK/Chr/STT trùng từng ô với `TN2/decisions.csv`.

**Kiểm định ngược mô hình ảnh trên IHR** (học trượt trên một sách, dự tổng p trong phần giữ của sách kia; "thật U" = lệch khe xác nhận bởi IHR-2 ∪ CG3, "thật slot≠1" = tính cả khe chưa xác định):

| Sách · hướng | dự | thật U | thật slot≠1 |
|---|---:|---:|---:|
| TK|H0 | 131,1 | 94 | 134 |
| TK|H1 | 46,0 | 3 | 5 |
| TK|H3 | 33,7 | 2 | 4 |
| TK|H2(Σp_H1×tỉ lệ nhận) | 6,1 | 1 | 1 |
| L16|H0 | 137,8 | 202 | 252 |
| L16|H1 | 13,7 | 12 | 55 |
| L16|H3 | 13,7 | 12 | 55 |
| L16|H2(Σp_H1×tỉ lệ nhận) | 1,8 | 0 | 7 |

→ Sau H1, mô hình dự **thừa** ở TK (46,0 so với thật 3–5) và **đúng cỡ** ở L16 với lệch khe thật (13,7 so với 12), nhưng **không thấy** "khe chưa xác định" (L16 còn 55 ô slot≠1). Vì số IHR tính khe chưa xác định là sai còn số ước lượng chỉ đếm trượt thật, **số IHR khắt khe hơn số ước lượng** của các bộ khác ở cùng hướng.

**Cận "không cần d" trên TK** (vế chữ, so với lỗi nhãn đo được; hợp lệ khi cận ≥ đo):

| Hướng | c (tỉ lệ bị chống) | cận | đo | hợp lệ |
|---|---:|---:|---:|---|
| H0 | 0,167 | 4.574,3 | 211 | True |
| H1 | 0,166 | 4.301,9 | 198 | True |
| H2 | 0,137 | 2.636,1 | 113 | True |
| H3 | 0,166 | 1.333,7 | 31 | True |
| H4 | 0,137 | 823,3 | 13 | True |
| H5 | 0,166 | 4.301,9 | 198 | True |
| H6 | 0,137 | 823,3 | 13 | True |
| Rv3 | 0,137 | 823,3 | 13 | True |

→ Cận luôn hợp lệ nhưng **lỏng 22–63 lần** trên TK (chữ thật của TK khác Kiều 1871 ở khoảng 16,6 % vị trí dù nhãn đúng, tham số d của text_attested). Ở KVK (hai tham chiếu, c ở H2 = 0,013) cận này chặt hơn nhiều nhưng vẫn là thành phần quyết định cận bi quan. Đây là lý do thạch bản không đạt 99 % bi quan: **không có cặp dị bản thứ hai có nhãn người để kiểm định phép chuyển TK → thạch bản**.

**Bộ kiểm ảnh với ô trượt sống sót H1**: nhận 8/60 (13,3 %; L16 7/55, TK 1/5) — cao hơn hẳn tỉ lệ nhận ô sai khe nói chung của bộ kiểm p_wood (TN1 `err_class_catch.csv`: L16 2,7 %, TK 1,6 %): trượt nhỏ qua được hình học thì cũng dễ lừa bộ kiểm.

## 6. H7 — đọc lại STT bằng kim lang_type = 2 (chỉ mô tả; cần API, không chạy)

- Số đã có (báo cáo v2 §3.1, §6.1): STT được OCR bằng lt1; lt1 gần như mù chữ Nôm riêng (đúng 2,45 % so với lt2 96,4 % trên chữ Nôm riêng, CHẮC CHẮN THEO MÁY); bể "Hán hoá" khoảng 750 [481–954] ô (CÓ THỂ); lợi ích ước sửa khoảng 340–1.100 nhãn; chi phí 448 lượt API (khoảng 18 phút), đề xuất pilot 41 lượt. Không giải được vế "một chữ".
- **Trần lợi ích** (SUY ĐOÁN, tính từ ma trận: giả sử lt2 xoá **hết** lỗi riêng STT — Hán hoá, cầu tự dạng, rescue):

| Bộ · hướng | hiện tại: điểm / bi quan | H7 trần: điểm / bi quan | H7 trần, bể trượt không được H1 giảm: bi quan |
|---|---|---|---|
| stt2 · H1 | 95,05 / 93,52 % | 98,14 / 97,23 % | 93,08 % |
| stt2 · H2 | 98,38 / 96,39 % | 99,06 / 97,20 % | 93,57 % |
| stt2 · Rv3 | 98,99 / 97,13 % | 99,09 / 97,25 % | 93,91 % |
| stt4 · H1 | 94,67 / 93,05 % | 98,08 / 97,14 % | 93,08 % |
| stt4 · H2 | 98,23 / 96,15 % | 98,86 / 96,90 % | 91,52 % |
| stt4 · Rv3 | 98,77 / 96,80 % | 98,91 / 96,97 % | 92,05 % |
| stt11 · H1 | 95,86 / 94,49 % | 98,19 / 97,28 % | 93,08 % |
| stt11 · H2 | 98,44 / 96,46 % | 99,14 / 97,31 % | 94,53 % |
| stt11 · Rv3 | 99,09 / 97,23 % | 99,15 / 97,31 % | 94,74 % |

→ H7 nâng điểm STT ở H1 lên khoảng 98 %, nhưng ở H2/Rv3 (tập đã lọc) chỉ thêm vài phần mười điểm, và **cận bi quan vẫn dưới 99 %** vì hai phần H7 không chạm tới: (i) tỉ lệ nhãn sai chuyển từ sách khác (ở H2 bi quan tới 2,38 %, lấy từ L83 cùng hướng, chủ yếu là lớp "không có tham chiếu" của TK); (ii) crop trượt một phần mà bộ kiểm viết tay vẫn nhận. H7 đáng làm để **giảm lỗi thật**, không để **chứng nhận**.

## 7. Cái gì chỉ người kiểm mới chốt được

1. **Vế "đúng MỘT chữ" ở mọi bộ.** `slot_ok` chỉ xét tâm crop nằm đúng khe; không có sự thật nào về mực chữ kề lọt vào crop. Cờ B0 (bleed, tight_nb, tall…) được áp mà không đo được lỗi còn sót. CHẮC CHẮN (thiết kế đo).
2. **L16**: 2.737 ô H2/H4 ở 99,49 % — 7 ô lỗi vế ảnh đều là "khe chưa xác định" (TN1: sai khe 0), cộng 13 ô sai chữ; người xem các ô này là chốt được sách này ở tập H4.
3. **KVK, L83**: phép chuyển tỉ lệ lỗi nhãn TK → thạch bản chưa kiểm định được (một cặp dị bản duy nhất có nhãn người). Mẫu người trên tập H4 là cách duy nhất đưa cận bi quan lên ≥ 99 %.
4. **Chr**: không có văn bản người nào trên máy → vế chữ hoàn toàn chuyển từ sách khác.
5. **STT**: vế chữ (lt1, luật cầu tự dạng/rescue, quy ước người phiên Borg khác nhãn) và vế ảnh (trượt một phần) đều cần người.

**Cỡ mẫu người để chứng nhận một tập đã lọc** (lấy ngẫu nhiên từ phần giữ của hướng; cận trên Clopper–Pearson một phía 95 %): 0 lỗi trong **299** ô → sai ≤ 1 %; ≤ 1 lỗi trong **473** ô → sai ≤ 1 %; 0 lỗi trong **598** ô → sai ≤ 0,5 %. Cần làm riêng cho mỗi bộ (hoặc mỗi tầng trong bộ) muốn chứng nhận; ô bị hạ vẫn giữ trong bộ ở tầng thấp. Giao thức người v3 (`$SP/r6/policy_v3/human_protocol.md`) đã có sẵn công cụ và ước 13–16 giờ cho lượt tối thiểu.

## 8. Giới hạn

- Chỉ hai sách có nhãn người (L16, TK); TK "trong miền" của kim nên tỉ lệ lỗi nền thấp, có thể không đại diện cho thạch bản (báo cáo v2 §3.1).
- Số ước lượng cộng lỗi ảnh và lỗi chữ (cận hợp) và cộng hai đầu khoảng → khoảng rộng hơn khoảng đồng thời thật.
- Mô hình ảnh vòng 2 chỉ ước "trượt thật", không ước "khe chưa xác định" — số IHR khắt khe hơn.
- Cận bi quan của thạch bản dùng cận "không cần d" (định trước); đã biết nó lỏng 22–63 lần trên TK. Cột "độ nhạy" bỏ cận này được thêm **sau** khi thấy kết quả và **không** dùng để xếp hạng.
- STT: bể trượt 1.630–2.500 và bể Hán hoá 750 [481–954] là SUY ĐOÁN/CÓ THỂ từ vòng 4–5; cách phân bổ vào từng ô (theo p trượt vòng 2, rải đều trên ô âm Borg-"nom") là giả định.
- H6: ô được sửa không qua p_wood ở τ = 0,995 (điểm cho crop mới chưa tính); chỉ qua bộ kiểm của vòng sửa.
- Luật "cột lệch số chữ" và "tall" (trong H1) được thêm vào v3 sau khi nhìn phần dư IHR (TN1 §5).

## 9. Tệp và tái lập

- `tn2_l83kvk/{s1a_kimdet,s1b_candidates,s2_score}.py`: bản sao TN2, **chỉ** đổi thư mục ra (`measure_out/_thu_nghiem_anh_chu/TN3/tn2_l83kvk/`), danh sách bộ (L83, KVK) và cấu hình (KVK = `pipeline_KimVanKieu1884_b1.yaml` chính thức B1').
- `t01_fix_decisions.py` → `measure_out/_thu_nghiem_anh_chu/TN3/fix_decisions.csv`, `t01_summary.json` (dùng nguyên `propose`/`resolve` của TN2).
- `t02_matrix.py` → `matrix.csv` (8 bộ × 8 hướng, mọi cột số), `t02_summary.json` (invariant, tham số, kiểm định), `measure_out/_thu_nghiem_anh_chu/TN3/cells_masks.pkl` (ô nào giữ ở hướng nào — cho người kiểm/pipeline), `boot_img_sums.npy`.
- `t03_report.py` → tệp này + `tong_hop.html`.

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR; D=lab/thu_nghiem_anh_chu/TN3_tat_ca_bo
.venv/bin/python $D/tn2_l83kvk/s1a_kimdet.py               # hộp kim + detector thô L83/KVK (~50 s)
.venv/bin/python $D/tn2_l83kvk/s1b_candidates.py --workers 4   # ứng viên + crop (~5 s)
.venv/bin/python $D/tn2_l83kvk/s2_score.py                 # điểm bộ kiểm T/L (MPS, ~3 s)
.venv/bin/python $D/t01_fix_decisions.py                   # quyết định sửa hộp 6 book_set (~6 s)
.venv/bin/python $D/t02_matrix.py                          # ma trận + bootstrap (~35 s CPU)
.venv/bin/python $D/t03_report.py                          # KET_QUA.md + tong_hop.html
```

Đầu vào chỉ đọc: `$SP/r6/policy_v3/out/{base_v2.pkl, v02_cells_0.995.pkl}`, `$SP/r6/policy_v3/summary.json`, `$SP/gold_img_audit/gate_v2/` (mô hình trượt vòng 2), `$SP/r5/text_attested/out/summary.json`, `lab/.../TN1_cong_kiem/{t01_summary.json, projection.csv, ladder_b.csv, main_table.csv}`, `lab/.../TN2_vong_sua_hop/{s3_eval.py, s3_summary.json}`, `measure_out/_thu_nghiem_anh_chu/TN2/{scores.csv, decisions.csv}`, `dataset/_ALL/labels.csv`.

Invariant chính (`t02_summary.json`): H2/H4/V3 trùng TN1 (d)/(e)/v3 ở cả 6 book_set; H1 trùng G7 (TK và 4 bộ ngoài IHR) và G6 (L16); H0/H2/H4/Rv3 trên L16/TK tái lập đúng độ chính xác TN1 (a)/(d)/(e)/(f); p trượt vòng 2 tái lập `gate_v2.json`; σ chuyển bộ trùng; bể STT = [1.630, 2.500]; 4.750 ô âm Borg-"nom"; cận không-d hợp lệ trên TK ở mọi hướng.
