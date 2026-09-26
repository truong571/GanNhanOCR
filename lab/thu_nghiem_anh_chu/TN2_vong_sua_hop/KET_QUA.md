# TN2 — Vòng lặp sửa hộp ảnh GOLD: độ chính xác trước khi đưa vào pipeline (26/09/2026)

Thư mục: `lab/thu_nghiem_anh_chu/TN2_vong_sua_hop/` (script, bảng nhỏ, `vi_du.html` + `img/`); tệp lớn: `measure_out/_thu_nghiem_anh_chu/TN2/` (crop mới, `candidates.csv`, `scores.csv`, `decisions.csv`). 0 lần gọi API, không tải model, repo chỉ đọc ngoài hai thư mục trên, không mở ảnh bằng LLM. Mọi số dưới đây do `s5_ket_qua.py` đọc từ tệp đã lưu (`s3_summary.json`, `s3_lobo_table.csv`, `s3_theta_curve.csv`, `t0*_*.json`, `s1b_summary.json`).

Nhãn độ chắc: **CHẮC CHẮN THEO MÁY** = đo trên sự thật người IHR (hộp cột người vẽ + chữ GT người) · **ƯỚC LƯỢNG** = đếm của máy, chưa có sự thật · **SUY ĐOÁN** = suy luận chưa đo.

## 1. Vòng lặp đã định (viết trước khi chấm)

- **Ô vào vòng**: ô GOLD có cờ `geo_f_kim_off = 1` (tâm hộp ảnh cách tâm hộp kim của CHÍNH chữ nhãn > 0,5 bước cột theo y hoặc > 0,5 bề rộng theo x; `gold_suspicion.csv`).
- **Ứng viên** (`s1b_candidates.py`): (a) `prev`/`next` = hộp kề trên/dưới trong cột (dịch chỉ số ±1; hộp phân biệt kề theo y trong bản ghi build mọi tầng); (b) `kim` = hộp cũ tịnh tiến cho tâm trùng tâm hộp kim `chars[nom_idx]` (dựng lại bằng `align_production._detect`, kim đọc ra đúng nhãn ở 100 % ô); (c) `det` = hộp detector THÔ (ckpt/ngưỡng/biên x theo sách) gần tâm kim nhất, chỉ khi cách ≤ 0,5 bước.
- **Cắt**: đúng `build_dataset.save_crop` (pad 0,12, carve láng giềng = hộp kề theo y của chính hộp ứng viên, tighten, ảnh gốc khi sách khai `crop_source: original`).
- **Chấm để CHỌN** (`s2_score.py`): bộ kiểm học trên nhãn người `r4/verifier_ft` (T học TK1872 phần A + Borg; L học LVT1916 phần A + Borg). `m = cos(crop, nguyên mẫu nhãn) − max cos(crop, đối thủ)`, đối thủ = chữ kim kề ±1/±2 ∪ đồng âm R(âm) ∪ top-5 gần hình, bỏ dị thể của nhãn.
- **Nhận** ứng viên m cao nhất khi `m ≥ θ` VÀ `m > m(crop đang giao)`; rồi trên trạng thái cuối của cột: hộp mới không trùng (IoU < 0,5) hộp của bất kỳ ô giao nộp nào còn giữ hộp, và tâm y nằm giữa ô còn giữ có nom_idx liền trước/liền sau; vi phạm → huỷ (hai ô cùng dời vào trùng nhau → hạ cả hai). **Một lượt duy nhất. Không nhận → HẠ. Nhãn không bao giờ đổi.**
- **Luật trùng hộp — hai bản**: **NGHIÊM (chính)** = ô bị hạ VẪN giữ hộp cũ (nó còn trong bộ dữ liệu ở tầng thấp), đúng nghĩa "không hai ô chung một hộp"; **NỚI (phụ)** = chỉ ô không bị hạ mới giữ hộp. Ở bản nới, 31/95 ô sửa ở L16, 4/4 ô sửa ở TK, 19/31 ô sửa ở Chr, 19/45 ô sửa ở STT lấy đúng hộp của một ô bị hạ (hai ô chung một ảnh, khác tầng).
- **LOBO**: sách L16 chọn bằng T, TK chọn bằng L (không mô hình nào thấy sách nó chấm). θ chọn trên sách KIA theo tiêu chí đăng ký trước: θ nhỏ nhất trên lưới −0,20…0,80 (bước 0,05) mà độ đúng của ô được sửa ≥ độ đúng của GOLD không bị cờ ở sách tune, và giữ được ở mọi θ lớn hơn.
- **Chấm kết quả** (`s3_eval.py`, `slotlib.py`): chỉ mô hình khe người (chép nguyên logic `h01_build_cells.py`; bất biến: tái lập slot_ok/gt_img/gt_char của `cells_eval.csv` 13760/13760 và 22499/22499 ô). Bộ kiểm dùng để chọn KHÔNG dùng để chấm.
  - **Chỉ số chính "ảnh = nhãn"**: chữ GT người ở khe mà crop nằm (hai mô hình khe cùng chỉ) tương đương V1+ với nhãn. Đây đúng là yêu cầu "crop là chữ của nhãn". Ghi chú minh bạch: lượt chạy đầu dùng "khe == syl_idx" làm chỉ số chính; đổi sang "ảnh = nhãn" vì vài ô sửa đưa crop tới đúng chỗ chữ nhãn nhưng nhãn lệch âm (nom_idx ≠ khe); cả hai đều báo.
  - Phụ: "khe" = slot_ok (crop ở khe syl_idx); "hai vế" = nhãn ~V1+ chữ GT tại syl_idx và slot_ok (định nghĩa báo cáo v2).
  - Mẫu số: GOLD trên trang có GT người (L16 11.587 ô; TK 18.550 ô, bỏ 1.004 ô trên trang không có GT); khe chưa xác định tính SAI. CI = bootstrap cụm theo trang, B = 2000.

**Bất biến tái lập** (CHẮC CHẮN): crop cũ cắt lại trùng md5 tệp giao nộp: L16 234/234, TK 167/167, Chr 738/738, STT 1499/1562 (63 ô STT lệch đều là luật self_training_rescue — crop 64×64 không cắt bằng save_crop); chữ kim tại nom_idx == ocr_char ở 100 % ô bị cờ; invariants s3: LucVanTien1916_old_slot_gtimg_reproduce_cells_eval=True, TruyenKieu1872_old_slot_gtimg_reproduce_cells_eval=True, LucVanTien1916_flagged_all_have_candidates=True, TruyenKieu1872_flagged_all_have_candidates=True, label_never_changed=True, selector_not_used_for_scoring=True.

## 2. Kết quả LOBO trên hai sách IHR (luật trùng hộp NGHIÊM) — CHẮC CHẮN THEO MÁY

| Sách chấm (θ chọn trên) | θ | bị cờ | nhận sửa | sửa đúng | sửa sai | sửa, khe mới chưa xác định | làm hỏng ô đúng | đổi hộp vẫn đúng | hạ (crop cũ sai) | hạ (crop cũ vốn đúng) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L16 (θ từ TK, bộ kiểm T) | 0,25 | 234 | 55 | 47 | 0 | 1 | 0 | 7 | 147 | 32 |
| TK (θ từ L16, bộ kiểm L) | 0,45 | 162 | 0 | 0 | 0 | 0 | 0 | 0 | 103 | 59 |

Độ chính xác "ảnh = nhãn" của tập GOLD (P0 không làm gì · P1 chỉ hạ mọi ô bị cờ · P2 vòng sửa):

| Sách | P0 [CI] (n) | P1 chỉ hạ [CI] (n) | P2 vòng sửa [CI] (n) | P2 − P1 [CI] | độ đúng của ô được sửa [CI cụm trang] · Clopper–Pearson |
|---|---|---|---|---|---|
| L16 | 96,38 [95,69–97,03] (11.587) | 98,02 [97,59–98,40] (11.353) | 98,02 [97,59–98,40] (11.408) | 0,00 [-0,02–0,01] điểm % | 54/55 = 98,2 % [93,5–100,0] · CP [90,3–100,0] |
| TK | 98,16 [97,89–98,43] (18.550) | 98,71 [98,52–98,88] (18.388) | 98,71 [98,52–98,88] (18.388) | 0,00 [0,00–0,00] điểm % | 0/0 = — % [—–—] · CP [—–—] |

Chỉ số phụ (cùng ô, cùng quyết định):

| Sách | khe P0 → P1 → P2 | hai vế P0 → P1 → P2 | ô sửa đúng theo chỉ-mô-hình-mẫu (độc lập hình học kim) |
|---|---|---|---|
| L16 | 97,81 → 99,45 → 99,43 % (sửa khe đúng 46, sai 2, làm hỏng 0) | 96,38 → 98,02 → 98,01 % | 100,0 % |
| TK | 99,28 → 99,82 → 99,82 % (sửa khe đúng 0, sai 0, làm hỏng 0) | 98,16 → 98,70 → 98,70 % | — % |

**So với "chỉ hạ"**: ở điểm vận hành LOBO, vòng sửa nhận 55 ô ở L16, 0 ô ở TK; trong đó sửa sai rõ 0, 0, làm hỏng ô đúng 0, 0, khe mới chưa xác định (tính sai) 1, 0 (L16, TK). Độ chính xác GOLD P2 − P1 = 0,00 [-0,02–0,01] điểm % (L16), 0,00 [0,00–0,00] điểm % (TK) — **không khác "chỉ hạ"**, chỉ **giữ lại thêm** 55 ô ở L16, 0 ô ở TK. Toàn bộ mức tăng so với P0 đến từ **việc hạ**.

**Bản luật NỚI** (ô bị hạ không giữ hộp; `s3_summary_lenient.json`) — cùng tiêu chí chọn θ:

| Sách | θ | nhận sửa | sửa đúng | sửa sai | chưa xác định | làm hỏng | đổi vẫn đúng | P1 → P2 [CI] (n P2) | P2 − P1 [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| L16 | 0,25 | 95 | 85 | 0 | 1 | 0 | 9 | 98,02 → 98,03 [97,59–98,41] (11.448) | 0,01 [-0,01–0,02] |
| TK | 0,45 | 4 | 3 | 0 | 0 | 0 | 1 | 98,71 → 98,71 [98,52–98,88] (18.392) | 0,00 [0,00–0,00] |

**Tham chiếu "chỉ dùng hộp kim, không bộ kiểm"** (luật nghiêm; mọi ô bị cờ nhận ứng viên `kim`, vẫn qua kiểm trùng/đơn điệu — không có tham số nên không cần LOBO):

| Sách | nhận | sửa đúng | sửa sai | chưa xác định | làm hỏng ô đúng | độ đúng ô sửa | GOLD P2 [CI] (n) |
|---|---:|---:|---:|---:|---:|---:|---|
| L16 | 221 | 181 | 2 | 0 | 16 | 91,9 % | 97,90 [97,46–98,29] (11.574) |
| TK | 156 | 82 | 3 | 12 | 43 | 62,8 % | 98,40 [98,20–98,61] (18.544) |

→ Hộp kim đặt đúng chỗ gần như mọi ô trượt thật (trần "có ít nhất một ứng viên đúng khe" = L16 189/191, TK 89/105 ô bị cờ có khe sai/chưa xác định), nhưng cờ kim_off cũng bắn vào ô mà **hộp kim mới là hộp lệch** (L16 43, TK 62 ô bị cờ vốn đúng khe). Không có bộ kiểm, các ô này bị làm hỏng và P2 **thấp hơn** "chỉ hạ". Bộ kiểm là điều kiện cần.

**Đường cong θ** (luật nghiêm, `s3_theta_curve.csv`; mỗi sách chấm bằng mô hình ngoài-sách của nó; KHÔNG phải LOBO vì θ nhìn chính sách đó):

| θ | L16: nhận / sửa đúng / sai+chưa xđ / làm hỏng · độ đúng | TK: nhận / sửa đúng / sai+chưa xđ / làm hỏng · độ đúng |
|---:|---|---|
| -0,20 | 162 / 148 / 1 / 1 · 98,8 % | 82 / 66 / 3 / 4 · 91,5 % |
| 0,00 | 118 / 106 / 1 / 0 · 99,2 % | 54 / 44 / 2 / 3 · 90,7 % |
| 0,10 | 93 / 84 / 1 / 0 · 98,9 % | 33 / 27 / 1 / 2 · 90,9 % |
| 0,20 | 69 / 60 / 1 / 0 · 98,6 % | 23 / 20 / 0 / 1 · 95,7 % |
| 0,25 | 55 / 47 / 1 / 0 · 98,2 % | 13 / 12 / 0 / 0 · 100,0 % |
| 0,30 | 39 / 33 / 1 / 0 · 97,4 % | 6 / 6 / 0 / 0 · 100,0 % |
| 0,40 | 13 / 11 / 1 / 0 · 92,3 % | 2 / 2 / 0 / 0 · 100,0 % |
| 0,45 | 11 / 10 / 0 / 0 · 100,0 % | 0 / 0 / 0 / 0 · — % |
| 0,50 | 6 / 5 / 0 / 0 · 100,0 % | 0 / 0 / 0 / 0 · — % |

Tiêu chí chọn θ cho ra θ = 0,25 khi tune trên TK và θ = 0,45 khi tune trên L16: ngưỡng **không chuyển tốt giữa hai sách/hai mô hình** (TK nhận θ 0,45 của L16 nên sửa được 0 ô, dù ở θ 0,25 TK có 13 ô sửa, đúng 13 — số không-LOBO). Lỗi ở θ thấp chủ yếu là **làm hỏng ô vốn đúng** (hộp kim lệch, bộ kiểm thích crop mới hơn), không phải sửa sai ô trượt.

**Phần không với tới**: lỗi "ảnh = nhãn" còn lại trong GOLD KHÔNG bị cờ: L16 225/11.353 (trong đó khe sai/chưa xác định 63), TK 238/18.388 (trong đó khe sai/chưa xác định 34) — phần lớn là nhãn đọc sai chữ (kim), vòng sửa HỘP không chữa được; ô trượt thật mà cờ kim_off bỏ sót ít (khe = 0 không cờ: L16 8, TK 2).

## 3. Chrestomathie1872 và SachThanhTruyen — ƯỚC LƯỢNG, chưa kiểm được

Không có sự thật người. Luật: hai bộ kiểm T (θ chọn trên L16, nơi T ngoài-sách) và L (θ chọn trên TK) phải **cùng chọn một ứng viên và cùng qua ngưỡng**, rồi kiểm trùng/đơn điệu.

| Bộ · luật | GOLD | bị cờ | T nhận thô | L nhận thô | hai bộ cùng ứng viên | **đề xuất sửa** | **đề xuất hạ** | huỷ vì trùng hộp | loại ứng viên |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Chr · nghiêm | 4.238 | 738 | 50 | 147 | 35 | **10** | 728 | 25 | kim 8, next 2 |
| STT · nghiêm | 52.707 | 1.562 | 88 | 312 | 57 | **25** | 1.537 | 32 | kim 22, next 3 |
| Chr · nới | 4.238 | 738 | 50 | 147 | 35 | **31** | 707 | 4 | next 11, kim 15, det 1, prev 4 |
| STT · nới | 52.707 | 1.562 | 88 | 312 | 57 | **45** | 1.517 | 12 | kim 30, prev 3, next 12 |

Mức chắc: **ƯỚC LƯỢNG** (số đếm máy). Độ đúng của đề xuất trên Chr/STT **chưa đo được**. Báo cáo v2 (§1 mục 7) ghi rằng ở STT một bộ kiểm chữ viết tay giữ ngoài theo sách vẫn nhận ≈ 11,5 % [7,0–16,7] crop trượt một phần; bộ kiểm dùng ở đây chưa được đo riêng trên STT (SUY ĐOÁN: rủi ro cao hơn IHR). Mẫu cho người xem: `vi_du.html` (nhóm 4–5) và danh sách đủ (luật nghiêm) `de_xuat_chr_stt.csv`.

## 4. Kết luận

1. **An toàn ở điểm vận hành, mẫu nhỏ** (CHẮC CHẮN THEO MÁY): trên hai sách IHR (luật nghiêm) nhận sửa 55 ô; sửa sai rõ 0, làm hỏng ô đúng 0, khe mới chưa xác định 1. CP 95 % cho tỉ lệ lỗi của ô sửa: L16 ≤ 9,7 % (1/55); TK: 0 ô được sửa.
2. **Lợi ích chỉ là độ phủ, không phải độ chính xác**: P2 = P1 trong sai số ở cả hai sách; vòng sửa giữ lại 55 ô (L16) và 0 ô (TK) (bản nới: 95 ô và 4 ô) trong số 234 và 162 ô bị cờ. Việc **HẠ** ô bị cờ mới là đòn bẩy độ chính xác (L16 96,38 → 98,02 %, TK 98,16 → 98,71 %).
3. **Ngưỡng không ổn định giữa sách** (θ 0,25 vs 0,45); ở θ thấp vòng sửa làm hỏng ô vốn đúng; bỏ bộ kiểm thì hại (P2 < P1).
4. **Khuyến nghị**: (a) đưa **"hạ ô bị cờ kim_off"** vào pipeline trước (tệp phụ, không phá dữ liệu; cái giá: hạ cả ô vốn đúng — L16 39/234, TK 59/162 ô bị cờ); (b) vòng sửa chỉ nên vào như **bước TUỲ CHỌN, mặc định TẮT**, cho sách khắc gỗ/thạch bản, luật trùng hộp nghiêm, ngưỡng bảo thủ θ = 0,45 và đòi hai bộ kiểm cùng chọn (như luật Chr/STT); (c) **không** bật cho STT/Chr cho tới khi người xem mẫu `vi_du.html` (nhóm 4–5). Lợi ích kỳ vọng nhỏ (SUY ĐOÁN: vài chục ô mỗi sách).

## 5. Giới hạn

- Chỉ hai sách IHR có sự thật; L16 nhận 55 ô, TK nhận 0 ô ở điểm LOBO nên kết luận "an toàn" dựa chủ yếu vào L16.
- Mô hình khe người có một thành phần dùng hộp DÒNG của kim, còn ứng viên `kim` dùng hộp CHỮ của kim: có chung nguồn hình học. Độ nhạy bằng chỉ-mô-hình-mẫu (độc lập kim): ô được sửa đúng L16 100,0 %.
- "Ảnh = nhãn" đo bằng tâm hộp (như báo cáo v2), không đo mực chữ kề lọt vào crop mới (vế "một chữ").
- Cờ vào vòng chỉ là kim_off; ô trượt mà kim cũng lệch theo (kim đọc chữ kề) không vào vòng.
- Luật nghiêm chặn mọi ô muốn lấy hộp của ô bị hạ, nên một chuỗi trượt chỉ sửa được khi cả chuỗi cùng được nhận.

## 6. Tái lập (≈ 5 phút, 0 API; MPS cho s2)

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR; D=lab/thu_nghiem_anh_chu/TN2_vong_sua_hop
.venv/bin/python $D/s1a_kimdet.py          # hộp kim + detector thô của trang có ô bị cờ (~4 phút)
.venv/bin/python $D/s1b_candidates.py --workers 4   # ứng viên + crop bằng save_crop (~15 s)
.venv/bin/python $D/s2_score.py            # điểm bộ kiểm T/L (~15 s, MPS)
.venv/bin/python $D/t01_slot_invariant.py  # bất biến mô hình khe
.venv/bin/python $D/s3_eval.py --share strict  # vòng lặp + LOBO + Chr/STT (~20 s)
.venv/bin/python $D/s3_eval.py --share lenient # bản luật nới
.venv/bin/python $D/t04_share_demoted.py decisions_lenient.csv  # ô sửa lấy hộp của ô bị hạ
.venv/bin/python $D/t02_diag.py            # chẩn đoán ứng viên/trần
.venv/bin/python $D/s4_html.py             # vi_du.html + img/
.venv/bin/python $D/s5_ket_qua.py          # KET_QUA.md + de_xuat_chr_stt.csv
```
