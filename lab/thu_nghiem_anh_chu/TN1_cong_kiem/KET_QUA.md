# THỬ NGHIỆM 1 — Công cụ kiểm ảnh ↔ chữ Hán Nôm cho ô GOLD: độ chính xác ↔ tỉ lệ giữ (26/09/2026)

Sinh tự động bởi `t03_report.py` từ `sweep.csv`, `ladder_b.csv`, `err_class_catch.csv`, `projection.csv`, `t01_summary.json` (invariant: **ALL PASS**). 0 API, chỉ CPU, repo chỉ đọc.

## 0. Cách đo (đọc trước)

- **Sự thật chỉ là nhãn người IHR**: chữ người `gt_char` (so theo quy ước dị thể V1+) và `slot_ok` (crop nằm đúng khe theo hộp cột người vẽ). Không dùng nhãn pipeline làm sự thật. Hai sách có nhãn người: LucVanTien1916 (L16), TruyenKieu1872 (TK); chỉ xét ô GOLD có chữ người (L16 11.585, TK 18.550).
- **Đúng hai vế** = nhãn đúng chữ người (V1+) **và** ảnh đúng khe. Khe chưa xác định tính là sai vế ảnh (định nghĩa v2/v3).
- **LOBO hai chiều**: ngưỡng của L16 chọn trên TK, ngưỡng của TK chọn trên L16. Mô hình verifier_ft dùng cho mỗi sách báo là mô hình học trên sách kia; ngưỡng điểm chọn trên phần B (1/4 trang mà mô hình không thấy khi học) của sách chọn. Riêng cổng hình học (b) chọn trên cả sách chọn.
- **Quét**: 7 mục tiêu τ = 0,98 · 0,985 · 0,99 · 0,9925 · 0,995 · 0,9975 · 0,999. Ở mỗi τ, trên sách chọn lấy ngưỡng giữ nhiều ô nhất mà độ chính xác ≥ τ, rồi áp nguyên ngưỡng đó lên sách báo. **τ = 0,995 là mục tiêu đăng ký trước**, nên là bảng chính.
- **CI 95 %**: bootstrap cụm theo trang, B = 2000, seed 20260926.
- **Công cụ chỉ hạ ô, không bao giờ sửa nhãn.** "Hạ" = không được chứng nhận "ảnh + chữ"; theo chính sách v3 ô vẫn được giao là GOLD văn bản (`uncertified`/`text_only`) hoặc chờ người (`human_check`).

| Mã | Cấu hình | Thành phần |
|---|---|---|
| a | pipeline hiện tại (không kiểm) | giữ mọi ô GOLD |
| b | chỉ hình học/trượt | thang lồng nhau G1…G7 (bảng §3); ở mỗi τ chọn bậc trên sách chọn |
| c | chỉ bộ kiểm ảnh verifier_ft (p_wood) | p_wood ≥ t (verifier_ft, vòng 4) |
| c2 | hai bộ kiểm ảnh (p_wood ∧ viss), không hình học | p_wood ≥ t1 ∧ viss ≥ t2 (thêm bộ xếp hạng hình vòng 3), không hình học |
| d | hình học + ảnh | hình học v3 (A0 ∪ B0 ∪ cột lệch số chữ ∪ cờ trượt vòng 2) + p_wood ≥ t1 ∧ viss ≥ t2; ngưỡng tái lập đúng p02b |
| e | hình học + ảnh + dị bản (text_attested) | d + nhãn phải được văn bản người của dị bản chứng (`text_attested`). L16 không có dị bản người trên máy → e = d |
| f | chính sách v3 đầy đủ | e + luật A (rescue, cầu tự dạng, văn bản yếu, ảnh ô khác) + M-OCR. Ở τ = 0,995 trùng **từng ô** với `gold_exact = ok` của chính sách v3 |

## 1. Bảng chính: τ = 0,995 (đăng ký trước) — CHẮC CHẮN THEO MÁY (đo trên nhãn người)

### Báo trên L16 (ngưỡng chọn trên TK)

| Cấu hình | Chính xác hai vế [CI] | Ô giữ | Lỗi ẢNH còn sót (sai khe + khe chưa xđ) | Lỗi CHỮ còn sót | Ô đúng bị hạ oan | Lỗi bị hạ / tổng lỗi | Chỉ vế chữ |
|---|---|---:|---:|---:|---:|---:|---:|
| (a) pipeline hiện tại (không kiểm) | **96,39** [95,71–97,04] | 11.585 (100,0 %) | 252 (187 + 65) | 166 | 0 / 11.167 | 0 / 418 | 98,27 |
| (b) chỉ hình học/trượt | **98,13** [97,73–98,49] | 10.919 (94,2 %) | 46 (1 + 45) | 158 | 452 / 11.167 | 214 / 418 | 98,29 |
| (c) chỉ bộ kiểm ảnh verifier_ft (p_wood) | **99,10** [98,57–99,54] | 2.986 (25,8 %) | 17 (5 + 12) | 10 | 8.208 / 11.167 | 391 / 418 | 99,33 |
| (c2) hai bộ kiểm ảnh (p_wood ∧ viss), không hình học | **99,30** [98,83–99,68] | 2.848 (24,6 %) | 13 (4 + 9) | 7 | 8.339 / 11.167 | 398 / 418 | 99,51 |
| (d) hình học + ảnh | **99,49** [99,14–99,77] | 2.736 (23,6 %) | 7 (0 + 7) | 7 | 8.445 / 11.167 | 404 / 418 | 99,52 |
| (e) hình học + ảnh + dị bản (text_attested) | **99,49** [99,14–99,77] | 2.736 (23,6 %) | 7 (0 + 7) | 7 | 8.445 / 11.167 | 404 / 418 | 99,52 |
| (f) chính sách v3 đầy đủ | **99,49** [99,14–99,77] | 2.736 (23,6 %) | 7 (0 + 7) | 7 | 8.445 / 11.167 | 404 / 418 | 99,52 |

### Báo trên TK (ngưỡng chọn trên L16)

| Cấu hình | Chính xác hai vế [CI] | Ô giữ | Lỗi ẢNH còn sót (sai khe + khe chưa xđ) | Lỗi CHỮ còn sót | Ô đúng bị hạ oan | Lỗi bị hạ / tổng lỗi | Chỉ vế chữ |
|---|---|---:|---:|---:|---:|---:|---:|
| (a) pipeline hiện tại (không kiểm) | **98,16** [97,87–98,40] | 18.550 (100,0 %) | 134 (63 + 71) | 208 | 0 / 18.208 | 0 / 342 | 98,86 |
| (b) chỉ hình học/trượt | **98,85** [98,70–98,99] | 17.515 (94,4 %) | 5 (1 + 4) | 197 | 895 / 18.208 | 140 / 342 | 98,87 |
| (c) chỉ bộ kiểm ảnh verifier_ft (p_wood) | **99,10** [98,86–99,32] | 7.778 (41,9 %) | 18 (1 + 17) | 52 | 10.500 / 18.208 | 272 / 342 | 99,32 |
| (c2) hai bộ kiểm ảnh (p_wood ∧ viss), không hình học | **98,93** [98,73–99,13] | 13.575 (73,2 %) | 29 (1 + 28) | 116 | 4.778 / 18.208 | 197 / 342 | 99,13 |
| (d) hình học + ảnh | **99,12** [98,95–99,28] | 12.908 (69,6 %) | 1 (1 + 0) | 112 | 5.413 / 18.208 | 229 / 342 | 99,12 |
| (e) hình học + ảnh + dị bản (text_attested) | **99,87** [99,80–99,93] | 9.766 (52,6 %) | 0 (0 + 0) | 13 | 8.455 / 18.208 | 329 / 342 | 99,87 |
| (f) chính sách v3 đầy đủ | **99,87** [99,80–99,93] | 9.766 (52,6 %) | 0 (0 + 0) | 13 | 8.455 / 18.208 | 329 / 342 | 99,87 |

**Đọc bảng.**
- Không kiểm (a): L16 96,39 %, TK 98,16 %. Lỗi ảnh L16 252, lỗi chữ 166; TK 134 và 208.
- Hình học (b) gần như **xoá sạch lỗi sai khe** (L16 187 → 1; TK 63 → 1) mà chỉ hạ oan 4,0 % / 4,9 % ô đúng. Nhưng lỗi chữ gần như nguyên (158 / 197), nên trần chỉ 98,13 / 98,85 %. Ở τ = 0,995 không bậc nào đạt τ trên sách chọn → lấy bậc chính xác nhất (G7).
- Bộ kiểm ảnh (c, c2) bắt cả lỗi chữ nhìn thấy được, nhưng phải hạ nhiều ô đúng.
- Chính sách đầy đủ (f): L16 **99,49 %** [99,14–99,77] giữ 23,6 %; TK **99,87 %** [99,80–99,93] giữ 52,6 %. Trên TK, bỏ đối chiếu dị bản (d) chỉ còn 99,12 % → **dị bản người là thứ đẩy TK qua 99,5 %**.
- Cái giá: hạ oan L16 8.445/11.167 (75,6 %) và TK 8.455/18.208 (46,4 %) ô đúng.

## 2. Quét 7 mức τ (chọn trên sách này, báo trên sách kia) — CHẮC CHẮN THEO MÁY

Mỗi ô: `% giữ / chính xác hai vế %` trên sách báo. "—" = trên sách chọn không có ngưỡng nào đạt τ với ≥ 50 ô, nên giữ 0 ô. Chi tiết (ngưỡng, CI, lỗi ảnh/chữ, hạ oan, số đo trên sách chọn): `sweep.csv`.

| Cấu hình | Sách báo | 0,98 | 0,985 | 0,99 | 0,9925 | 0,995 | 0,9975 | 0,999 |
|---|---|---|---|---|---|---|---|---|
| (b) | L16 | 100,0 / 96,39 | 100,0 / 96,39 | 94,2 / 98,13 | 94,2 / 98,13 | 94,2 / 98,13 | 94,2 / 98,13 | 94,2 / 98,13 |
| (b) | TK | 94,7 / 98,77 | 94,6 / 98,80 | 94,4 / 98,85 | 94,4 / 98,85 | 94,4 / 98,85 | 94,4 / 98,85 | 94,4 / 98,85 |
| (c) | L16 | 100,0 / 96,42 | 100,0 / 96,42 | 90,1 / 98,16 | 61,4 / 98,90 | 25,8 / 99,10 | 0,3 / 100,00 | 0,3 / 100,00 |
| (c) | TK | 100,0 / 98,19 | 99,6 / 98,37 | 97,7 / 98,49 | 95,5 / 98,63 | 41,9 / 99,10 | 24,4 / 99,23 | 7,3 / 99,70 |
| (c2) | L16 | 99,9 / 96,45 | 99,9 / 96,45 | 89,8 / 98,22 | 60,9 / 99,04 | 24,6 / 99,30 | 10,6 / 99,27 | — |
| (c2) | TK | 98,1 / 98,31 | 97,7 / 98,50 | 97,7 / 98,50 | 93,1 / 98,74 | 73,2 / 98,93 | 68,2 / 99,00 | 17,1 / 99,34 |
| (d) | L16 | 94,5 / 98,07 | 94,5 / 98,07 | 86,1 / 98,56 | 58,6 / 99,15 | 23,6 / 99,49 | 10,2 / 99,58 | — |
| (d) | TK | 94,4 / 98,85 | 94,4 / 98,85 | 92,6 / 98,90 | 88,4 / 99,00 | 69,6 / 99,12 | 64,9 / 99,19 | 16,3 / 99,47 |
| (e) | L16 | 94,5 / 98,07 | 94,5 / 98,07 | 86,1 / 98,56 | 58,6 / 99,15 | 23,6 / 99,49 | 10,2 / 99,58 | — |
| (e) | TK | 68,2 / 99,72 | 68,2 / 99,72 | 66,8 / 99,77 | 64,4 / 99,80 | 52,6 / 99,87 | 49,9 / 99,86 | 13,0 / 99,83 |
| (f) | L16 | 94,5 / 98,07 | 94,5 / 98,07 | 86,1 / 98,56 | 58,6 / 99,15 | 23,6 / 99,49 | 10,2 / 99,58 | — |
| (f) | TK | 68,2 / 99,72 | 68,2 / 99,72 | 66,8 / 99,77 | 64,4 / 99,80 | 52,6 / 99,87 | 49,9 / 99,86 | 13,0 / 99,83 |

Cổng (b) chọn bậc: τ 0,98: L16←TK G0, TK←L16 G5; τ 0,985: L16←TK G0, TK←L16 G6; τ 0,99: L16←TK G7, TK←L16 G7; τ 0,9925: L16←TK G7, TK←L16 G7; τ 0,995: L16←TK G7, TK←L16 G7; τ 0,9975: L16←TK G7, TK←L16 G7; τ 0,999: L16←TK G7, TK←L16 G7.

## 3. Thang hình học (b), từng bậc trên từng sách (mô tả, không chọn) — CHẮC CHẮN THEO MÁY

| Bậc | Nội dung | Sách | Giữ | Chính xác hai vế | Lỗi ảnh còn | Lỗi chữ còn | Hạ oan |
|---|---|---|---:|---:|---:|---:|---:|
| G0 | không cổng | L16 | 100,0 % | 96,39 [95,71–97,04] | 252 | 166 | 0 |
| G1 | A0: crop rỗng/cắt nét, 1 hộp 2 cột, ảnh ô khác | L16 | 100,0 % | 96,39 [95,71–97,04] | 252 | 166 | 0 |
| G2 | G1 + B0: mực chữ kề (bleed/tight_nb), hộp chồng nặng, hộp cao "tall" | L16 | 97,4 % | 96,41 [95,73–97,05] | 241 | 164 | 293 |
| G3 | G2 + cột có số chữ OCR ≠ số âm QN | L16 | 96,1 % | 96,55 [95,88–97,18] | 223 | 161 | 422 |
| G4 | G3 + lệch dọc hộp kim > 1,0 bước | L16 | 95,9 % | 96,70 [96,07–97,29] | 206 | 161 | 422 |
| G5 | G3 + lệch dọc > 0,75 ∨ vis_z ≤ −1 | L16 | 94,8 % | 97,86 [97,45–98,23] | 77 | 158 | 423 |
| G6 | G3 + lệch dọc > 0,6 ∨ vis_z ≤ −1 (cờ vòng 2 chọn trên TK) | L16 | 94,5 % | 98,06 [97,65–98,42] | 55 | 158 | 431 |
| G7 | G3 + lệch dọc/ngang > 0,5 ∨ vis_z ≤ −0,5 (cờ vòng 2 chọn trên L16) | L16 | 94,2 % | 98,13 [97,73–98,49] | 46 | 158 | 452 |
| G0 | không cổng | TK | 100,0 % | 98,16 [97,87–98,40] | 134 | 208 | 0 |
| G1 | A0: crop rỗng/cắt nét, 1 hộp 2 cột, ảnh ô khác | TK | 100,0 % | 98,16 [97,87–98,40] | 134 | 208 | 0 |
| G2 | G1 + B0: mực chữ kề (bleed/tight_nb), hộp chồng nặng, hộp cao "tall" | TK | 99,3 % | 98,17 [97,89–98,42] | 130 | 206 | 128 |
| G3 | G2 + cột có số chữ OCR ≠ số âm QN | TK | 95,0 % | 98,46 [98,21–98,66] | 74 | 198 | 865 |
| G4 | G3 + lệch dọc hộp kim > 1,0 bước | TK | 94,9 % | 98,47 [98,24–98,67] | 71 | 198 | 865 |
| G5 | G3 + lệch dọc > 0,75 ∨ vis_z ≤ −1 | TK | 94,7 % | 98,77 [98,61–98,91] | 18 | 198 | 866 |
| G6 | G3 + lệch dọc > 0,6 ∨ vis_z ≤ −1 (cờ vòng 2 chọn trên TK) | TK | 94,6 % | 98,80 [98,65–98,95] | 12 | 198 | 866 |
| G7 | G3 + lệch dọc/ngang > 0,5 ∨ vis_z ≤ −0,5 (cờ vòng 2 chọn trên L16) | TK | 94,4 % | 98,85 [98,70–98,99] | 5 | 197 | 895 |

**Lưu ý vòng tròn một phần (SUY ĐOÁN về mức độ):** `slot_ok = 0` đòi cả mô hình khe theo mẫu lẫn mô hình khe theo hộp dòng kim cùng chỉ khe khác; cờ lệch hộp kim (G4–G7) cũng dựa trên hộp kim. Vì vậy tỉ lệ bắt lỗi sai khe của các bậc G4–G7 có thể được thổi phồng. Mô hình theo mẫu thì độc lập với kim.

## 4. Công cụ bắt được loại lỗi nào (τ = 0,995) — CHẮC CHẮN THEO MÁY

Tỉ lệ ô bị hạ theo loại. Muốn công cụ tốt thì cột lỗi phải cao hơn hẳn dòng "đúng hai vế".

| Sách | Loại | Số ô GOLD | Hạ (b) | Hạ (c) | Hạ (d) | Hạ (f) |
|---|---|---:|---:|---:|---:|---:|
| L16 | đúng hai vế (đối chứng) | 11.167 | 4,0 % | 73,5 % | 75,6 % | 75,6 % |
| L16 | ẢNH: sai khe | 187 | 99,5 % | 97,3 % | 100,0 % | 100,0 % |
| L16 | ẢNH: khe chưa xác định | 65 | 30,8 % | 81,5 % | 89,2 % | 89,2 % |
| L16 | CHỮ: đồng âm (cả hai ∈ R(âm)) | 55 | 1,8 % | 96,4 % | 98,2 % | 98,2 % |
| L16 | CHỮ: gần hình | 37 | 5,4 % | 91,9 % | 94,6 % | 94,6 % |
| L16 | CHỮ: nhãn = chữ kề | 11 | 18,2 % | 81,8 % | 81,8 % | 81,8 % |
| L16 | CHỮ: mã PUA khác | 61 | 4,9 % | 96,7 % | 96,7 % | 96,7 % |
| L16 | CHỮ: khác | 2 | 0,0 % | 50,0 % | 100,0 % | 100,0 % |
| TK | đúng hai vế (đối chứng) | 18.208 | 4,9 % | 57,7 % | 29,7 % | 46,4 % |
| TK | ẢNH: sai khe | 63 | 98,4 % | 98,4 % | 98,4 % | 100,0 % |
| TK | ẢNH: khe chưa xác định | 71 | 94,4 % | 76,1 % | 100,0 % | 100,0 % |
| TK | CHỮ: đồng âm (cả hai ∈ R(âm)) | 76 | 5,3 % | 69,7 % | 38,2 % | 90,8 % |
| TK | CHỮ: gần hình | 75 | 2,7 % | 66,7 % | 45,3 % | 93,3 % |
| TK | CHỮ: nhãn = chữ kề | 5 | 0,0 % | 60,0 % | 40,0 % | 100,0 % |
| TK | CHỮ: mã PUA khác | 51 | 7,8 % | 96,1 % | 58,8 % | 98,0 % |
| TK | CHỮ: khác | 1 | 100,0 % | 100,0 % | 100,0 % | 100,0 % |

- Lỗi **ảnh**: mọi cấu hình có hình học hạ gần hết (sai khe: 100,0 % L16, 98,4 % TK ở (d)).
- Lỗi **chữ đồng âm/gần hình**: trên TK, bộ kiểm ảnh (d) hạ 38,2 % ô đồng âm, trong khi hạ 29,7 % ô đúng → **gần như mù**. Chỉ đối chiếu dị bản người (f) nâng lên 90,8 %. Trên L16, (d) phân biệt tốt hơn nhưng phải hạ 75,6 % ô đúng.

## 5. Điểm vận hành đề xuất

1. **Đề xuất: cấu hình (f) ở τ = 0,995 (đăng ký trước)**, tức chính sách v3. Đo được: TK 99,87 % [99,80–99,93] giữ 52,6 %; L16 99,49 % [99,14–99,77] giữ 23,6 %. CHẮC CHẮN THEO MÁY.
2. **Mục tiêu 99,5 %:** TK đạt, kể cả cận dưới CI. L16 (sách không có dị bản người) chỉ **ngang mục tiêu ở mức điểm**, cận dưới 99,14 %. Nâng τ lên 0,9975 cho 99,58 % [99,10–99,92] nhưng chỉ giữ 10,2 %; cận dưới không tốt hơn. → Với sách chỉ có công cụ ảnh (không dị bản), **chưa chứng minh được 99,5 % ở cận dưới 95 %**.
3. **Cổng hình học G7 nên chạy trước mọi thứ, ở mọi bộ.** Nó rẻ, bắt gần hết lỗi sai khe và hạ oan chỉ khoảng 4–5 %. Nhưng một mình nó không lên quá khoảng 98–99 %.
4. *Chỉ để tham khảo, không phải đề xuất:* ở sách có dị bản, (e) với τ = 0,9925 cho TK 99,80 % giữ 64,4 %. Mức này **chọn sau khi nhìn TK**. Chỉ có một sách IHR có dị bản, nên chưa kiểm định ngược được (SUY ĐOÁN).
5. Lưu ý trong mẫu: luật "cột lệch số chữ" và "tall" của v3 được thêm **sau khi nhìn phần dư IHR**. Luật "dị bản chống → không ok" dựa trên số đo TK. Vì vậy số (e)/(f) trên TK và phần cải thiện v2 → v3 là đo trên chính dữ liệu chọn luật. Ngưỡng điểm (p_wood/viss) và thang (b) thì LOBO sạch.

## 6. Điều công cụ KHÔNG làm được

- **Lỗi chữ đồng âm / gần hình khi không có dị bản người**: xem §4. Phần sai còn sót trong (f) gồm: TK 7 đồng âm (cả hai ∈ R(âm)), 5 gần hình, 1 mã PUA khác; L16 7 khe chưa xác định, 1 đồng âm (cả hai ∈ R(âm)), 2 gần hình, 2 nhãn = chữ kề, 2 mã PUA khác. CHẮC CHẮN THEO MÁY.
- **Vế "đúng MỘT chữ" (mực chữ kề lọt vào crop)**: `slot_ok` chỉ xét tâm crop nằm đúng khe, không thấy mực kề. Các cờ B0 (bleed, tight_nb, tall…) được áp mà **không có sự thật người** để đo. CHẮC CHẮN (thiết kế đo).
- **STT (chữ viết tay)**: verifier_ft học trên mộc bản. Bộ kiểm chữ viết tay giữ ngoài theo sách (vòng 5) vẫn nhận khoảng 11,5 % [7,0–16,7] crop trượt thật (trích báo cáo v2 §1.7, CÓ THỂ). → `stt_ok_allowed = False`, 0 ô ok.
- **Chrestomathie1872**: không có văn bản người nào trên máy, nên không chứng được vế chữ → 0 ok (309 ứng viên). CHẮC CHẮN (dữ liệu).
- **LucVanTien1883**: cận sai nhãn qua dị bản L16 là 3,3–4,9 %, vượt 0,5 %. Cổng cấp bộ của v3 → 0 ok (648 ứng viên chờ người). CÓ THỂ.
- **Hạ oan lớn**: công cụ không phân biệt được phần lớn ô đúng với ô sai ở vùng điểm thấp. Ô bị hạ không mất, nhưng muốn "chứng nhận" chúng thì cần người.

## 7. Chiếu sang 4 bộ không có nhãn người — ƯỚC LƯỢNG (số ô giữ/hạ, KHÔNG phải độ chính xác đo được)

Cùng cấu hình ở τ = 0,995, dùng quy ước của v3 cho bộ ngoài IHR:
- (b): bậc chặt hơn của hai chiều. Ở τ này cả hai chiều đều chọn G7.
- (c): p_wood của **cả hai** mô hình phải qua ngưỡng.
- (d): hình học v3 + điểm. Thạch bản dùng p_wood hai mô hình ∧ viss_X; Chr dùng p_wood hai mô hình (không có viss).
- **STT: (c)/(d) dùng bộ kiểm chữ viết tay LOBO (vòng 5, q = 0,00015, ≥ 3 nguyên mẫu người), không phải verifier_ft.**
- (f) = đếm `gold_exact = ok` của v3.
- Hai dòng IHR để đối chiếu.

Mỗi ô: `giữ (%) · hạ`.

| Bộ | GOLD | (b) | (c) | (d) | (e) | (f) |
|---|---:|---:|---:|---:|---:|---:|
| STT | 52.707 | 35.111 (66,6 %) · 17.596 | 9.427 (17,9 %) · 43.280 | 6.339 (12,0 %) · 46.368 | 6.339 (12,0 %) · 46.368 | 0 (0,0 %) · 52.707 |
| Chr | 4.238 | 1.794 (42,3 %) · 2.444 | 864 (20,4 %) · 3.374 | 309 (7,3 %) · 3.929 | 309 (7,3 %) · 3.929 | 0 (0,0 %) · 4.238 |
| L83 | 10.910 | 10.184 (93,3 %) · 726 | 1.806 (16,6 %) · 9.104 | 1.858 (17,0 %) · 9.052 | 648 (5,9 %) · 10.262 | 0 (0,0 %) · 10.910 |
| KVK | 19.958 | 19.506 (97,7 %) · 452 | 4.290 (21,5 %) · 15.668 | 4.689 (23,5 %) · 15.269 | 3.676 (18,4 %) · 16.282 | 3.657 (18,3 %) · 16.301 |
| L16 (đo được, IHR) | 11.587 | 10.921 (94,2 %) · 666 | 2.987 (25,8 %) · 8.600 | 2.737 (23,6 %) · 8.850 | 2.737 (23,6 %) · 8.850 | 2.737 (23,6 %) · 8.850 |
| TK (đo được, IHR) | 19.554 | 18.511 (94,7 %) · 1.043 | 8.204 (42,0 %) · 11.350 | 13.643 (69,8 %) · 5.911 | 9.766 (49,9 %) · 9.788 | 9.766 (49,9 %) · 9.788 |

- (c) có thể giữ ít hơn (d): khi dùng một mình, ngưỡng p_wood chọn trên sách chọn phải cao hơn mới đạt τ.
- L83 (e) 648 → (f) 0 là do cổng cấp bộ TA4 của v3 (§6). STT/Chr (f) = 0 là luật đăng ký trước.
- STT (b) hạ 1/3 chủ yếu vì cờ "cột lệch số chữ" và mực chữ kề. Đây là cờ, chưa phải lỗi đã đo.

Sai số còn lại của 3.657 ô ok ở KVK theo v3: khoảng 4–27 ô (0,13–0,76 %); cận bi quan 1,40 %. SUY ĐOÁN (chuyển tỉ lệ từ TK). Các bộ khác có 0 ô ok nên không có số tương ứng.

## 8. Tệp và tái lập

- `t01_sweep.py` → `sweep.csv`, `main_table.csv`, `ladder_b.csv`, `err_class_catch.csv`, `projection.csv`, `t01_summary.json` (invariant); `measure_out/_thu_nghiem_anh_chu/TN1/cells_ihr_decisions.pkl`.
- `t02_vi_du.py` → `vi_du.html` (80 ô: 4 nhóm × 10 ô × 2 sách), `anh/<md5>.png` (crop chép), `glyph/U+XXXX.png` (NomNaTong), `vi_du_manifest.csv`.
- `t03_report.py` → tệp này.
- Invariant (`t01_summary.json`): tái lập đúng ngưỡng p02b ở 4 mục tiêu × 2 chiều; cờ trượt tính lại trùng `bc_k06v1`/`bc_k05dxv05`; thang lồng nhau; (f) ở τ = 0,995 trùng từng ô với v3 (L16 2.737, TK 9.766 ô); (a) và (f) trùng số IHR của v3; chiếu (f) trùng `per_set.ok` của v3 ở cả 6 bộ.

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
D=lab/thu_nghiem_anh_chu/TN1_cong_kiem
.venv/bin/python $D/t01_sweep.py && .venv/bin/python $D/t02_vi_du.py && .venv/bin/python $D/t03_report.py   # tổng < 10 s CPU, 0 API
```
Đầu vào (chỉ đọc): `$SP/r6/policy_v3/out/{base_v2.pkl, v02_cells_0.995.pkl}`, `$SP/r6/policy_v3/summary.json`, `$SP/r4/policy/out/p02b_sweep.json`, `$SP/kim_bottleneck/harness/{cells_eval.csv, harness_lib.py}`, `$SP/r6/policy_v3/scripts/vlib.py`, `fonts/NomNaTong-Regular.ttf`, crop trong `dataset/_ALL/crops/`.
