# THỬ NGHIỆM 4 — Crop chuẩn tự động cho ô GOLD (26/09/2026)

Sinh tự động bởi `s09_report.py` từ các tệp số trong `out_v1/`, `out_v2/` và `../TN3_tat_ca_bo/matrix.csv`. 0 lần gọi API, không tải model, không mở ảnh bằng LLM (ảnh chỉ do chương trình đọc). Repo chỉ đọc ngoài `lab/thu_nghiem_anh_chu/TN4_crop_chuan/` và `measure_out/_thu_nghiem_anh_chu/TN4/`. Ví dụ ảnh: `vi_du.html` (mở bằng file://, ảnh chép trong `anh/`).

Mục tiêu GOLD: (1) crop là đúng MỘT chữ, căn chuẩn theo chữ; (2) chữ đó đúng là nhãn. TN4 chỉ đụng vế (1): thay crop cũ (bbox + pad 0,12 + carve + tighten) bằng **crop chuẩn** tính lại từ ảnh trang, rồi đo (a) crop có sạch/đúng chữ hơn không, (b) bộ kiểm ảnh có giữ được nhiều ô hơn ở cùng độ chính xác không. Không sửa nhãn nào.

Mức chắc: **CHẮC CHẮN THEO MÁY** = đo trên nhãn/khe người IHR (L16, TK) hoặc nhãn người Borg · **ƯỚC LƯỢNG** = bộ ước lượng TN3 cho thạch bản · **SUY ĐOÁN** = Chr, STT (chuyển tỉ lệ từ sách khác, cần người kiểm). v2 là **hậu kiểm** (xem §1).

## 0. Trả lời ngắn

1. **Hai phiên bản.** v1 = thiết kế đăng ký trước (md5 `65eb98b892`). v1 có ba lỗi chẩn đoán được bằng số (§2.2) nên viết **v2 hậu kiểm** (md5 `d818443449`): đưa thuật toán về đúng đặc tả (ranh giới = trung điểm tâm; chỉ thành phần vắt qua mới cắt seam; bóc nét kẻ), **không dò tham số**. Số của v2 phải đọc là hậu kiểm (có thể lạc quan).
2. **(a) Crop sạch hơn thật, không trượt khe — CHẮC CHẮN THEO MÁY (khe người IHR).** Mực nằm ngoài khe người: L16 13,45 % → 7,13 %, TK 5,16 % → 3,45 %; ô có > 25 % mực ngoài khe: L16 1.746 → 935, TK 1.043 → 579. Khe đổi 1 → ≠1: L16 20 ô (0,18 %), TK 3; sửa được khe 17/6. Cái giá: L16 61 ô (0,54 %) mất > 25 % mực LÕI khe người (nguyên nhân SUY ĐOÁN: chữ nhỏ dính chữ kề, luật đầu nét lạ bỏ nhầm), TK 2.
3. **Cờ "một chữ" trên 8 bộ (118.954 ô).** Ô sạch một chữ: cũ 113.788 → v1 110.918 → v2 114.882. v2 **cứu** 4.541 ô B0 cũ thành crop sạch (chủ yếu bleed chép tay: stt4 1.666, stt11 1.702) và **gắn cờ mới** 3.447 ô mà crop cũ không cờ (tổng cờ của crop chuẩn trên 8 bộ: hai chữ 986, cắt nét 665, tall 612, stray 1.685). Thạch bản mất ròng (L83 -152, KVK -206).
4. **(b) Bộ kiểm ảnh KHÔNG hưởng lợi khi chỉ thay ảnh đầu vào** (bộ kiểm học trên crop cũ). AUC hai vế trên IHR giảm nhẹ (L16 p_wood -0,0132). H4 đo được (τ = 0,995, LOBO): L16 cũ 2.737 ô 99,49 % → v2 2.871 ô 99,48 % [99,18–99,73]; TK cũ 9.766 ô 99,87 % → v2 9.452 ô 99,88 % [99,81–99,95] (v1: 2.750 / 10.211). Nhưng chỉ riêng việc CHỌN NGƯỠNG (bootstrap trang tune, B = 300) đã làm số ô H4 dao động L16 1.343–6.670, TK 9.200–12.137; hiệu v2 − cũ ghép cặp: L16 trung vị +110 [-1.299; +1.609], TK +104 [-1.958; +1.223] → **không phân biệt được với 0**. Thạch bản (ƯỚC LƯỢNG): KVK 3.676 → 4.465, L83 648 → 741. Chép tay/Chr (SUY ĐOÁN) **giảm mạnh**: STT 6.339 → 4.009, Chr 309 → 185 — bộ kiểm viết tay nhận ít dương hơn hẳn (Borg DungLy, cùng FAR: 83,05 % → 68,97 %).
5. **Học lại bộ kiểm viết tay trên crop chuẩn (hậu kiểm)**: Borg DungLy nhận dương 66,94 % (FAR đồng âm 0,026 %, trượt 0,000 %) so với cũ/cũ 83,05 %; STT được chứng nhận 17,89 % (cũ) · 10,47 % (thay ảnh) · 6,82 % (học lại). Học lại **không phục hồi** tỉ lệ nhận ở điểm làm việc nghiêm (AUC dương/đồng âm 0,9972 so với cũ 0,9968, dương/trượt 0,9952 so với 0,9963): với chữ viết tay, crop chuẩn không giúp bộ kiểm. (Một lần học, một hạt giống; ngưỡng q = 0,00015 do vài âm cực trị quyết định — số nhận dương rất nhạy.) Xem §4.3.
6. **Cấu hình "lai" (hậu kiểm, dùng được ngay)**: giao crop chuẩn v2 + cổng H1 dùng cờ một chữ của crop chuẩn + bộ kiểm hiện tại chấm crop CŨ với ngưỡng TN1. H4: STT 6.339 → 6.782, Chr 309 → 325, L16 2.737 → 2.760 (99,46 %, đo với khe crop mới), TK 9.766 → 9.747 (99,87 %), KVK 3.676 → 3.647, L83 648 → 641; độ chính xác không đổi (trong khoảng).
7. **Khuyến nghị** (§7): đưa crop chuẩn v2 vào pipeline dưới dạng **tệp phụ + cờ**, dùng cờ một chữ của nó trong cổng hình học (cấu hình "lai"); **không** cho bộ kiểm hiện tại chấm crop chuẩn (thay ảnh không lợi; học lại bộ viết tay cũng không phục hồi); người xem mẫu ô L16 bị cắt nét trước khi thay crop giao nộp; không đổi kết luận TN3 (chỉ TK đạt mục tiêu chính xác).

## 1. Thuật toán crop chuẩn

Mã: `cclib.py` (v1, đăng ký trước, đóng băng trước lượt đo) và `cclib_v2.py` (v2, viết SAU khi thấy kết quả hình học v1; đóng băng trước lượt đo v2; không dò tham số). Đơn vị: bước chữ `p` = trung vị khoảng cách tâm các hộp trong cột; bề ngang cột `w` = trung vị bề ngang 5 hộp quanh ô. Không tham số nào chọn trên IHR hay bằng bộ kiểm.

| Bước | v1 (đăng ký trước) | v2 (hậu kiểm) |
|---|---|---|
| Láng giềng | hộp cùng cột (bản ghi build mọi tầng), bỏ hộp trùng khe (\|Δcy\| < 0,25·h) | như v1 |
| Dải khe dọc | ranh giới = SEAM ít mực trong trung điểm ± 0,25·d (hoà → hàng trên cùng); thiếu ô kề: cy ∓ 0,8·max(p,h) | ranh giới = ĐƯỜNG THẲNG qua trung điểm hai tâm; thiếu ô kề: trung điểm ảo 0,55·max(p,h); trung điểm xa hơn 0,8·max(p,h) bị kẹp |
| Dải khe ngang | xc ± 0,75·w, cắt ở trung điểm với tâm cột kề | như v1 |
| Nhị phân cục bộ | Otsu trên cửa sổ (dải + 0,30·p / 0,30·w), kẹp [64, 200] | như v1 (nới 0,35) |
| Nét kẻ | thành phần dài ≥ 1,2·p và mảnh ≤ 0,2·w (hoặc ngược lại) | BÓC trước: mở hình thái nhân dọc 1,2·p và ngang 1,2·w |
| Gán thành phần | trọn trong seam → của ô; vắt qua → giữ phần giữa hai seam; tâm x ngoài dải → cột khác | trọn một phía TRUNG ĐIỂM → gán trọn; chỉ thành phần vắt qua trung điểm mới cắt seam (trung điểm ± 0,25·d, phạt khoảng cách ≤ 20/255) |
| Đầu nét lạ / đốm | phần giữ < 20 % thành phần và < 10 % mực ô → bỏ; đốm < max(3 px, 0,0015·p·w) | như v1 |
| Kiểm "một chữ" (cờ, không ép) | blank < 0,02·p·w; cut = ranh giới cắt qua mực > 30 % số cột hoặc mực chạm mép cửa sổ; two = cao > 1,35·p (hoặc rộng > 1,35·w) có khe trắng ≥ 0,10·p chia 20/20; ink = mực/(p·w) ngoài [0,4; 2,5] × trung vị cùng (bộ, nhãn) (lớp < 5 ô: [0,25; 4] × trung vị bộ); cộng stray/border của pipeline đo trên crop chặt MỚI (bleed/truncated) | như v1 |
| Xuất | khung VUÔNG cạnh L + 2·ceil(0,10·L) quanh hộp mực, điểm ảnh từ trang GỐC (5 sách crop_source original; STT: trang đã xử lý như pipeline); mực ngoại lai (nở 1 px) và ngoài trang tô màu giấy (trung vị điểm không-mực); + bản 128×128 | như v1 |

Định nghĩa phân tích (đăng ký trước trong `out_v1/t00_freeze.json`, dùng NGUYÊN cho v2):

- `one_char_ok`: status=ok ∧ ¬{blank,cut,two} (cclib) ∧ ¬ink (tỉ lệ mực/trung vị cùng nhãn ngoài khoảng) ∧ crop_quality pipeline trên crop chặt MỚI ∉ {bleed,truncated}
- `B0_crop_old`: cqf∈{blank,truncated,bleed} ∪ blank ∪ truncated ∪ f_tight_nb ∪ seg_flag=tall (phần cấp crop của B0 TN1)
- `B0_box`: geo_f_dup_bbox ∪ geo_f_ov_heavy (cấp hộp, không đổi)
- `B0_new`: B0_box ∪ ¬one_char_ok ∪ tall_new
- `A0_new`: int_foreign ∪ geo_f_dup_bbox ∪ cclib blank ∪ cclib cut
- `H1_new`: ¬(A0_new ∪ B0_new ∪ CNT ∪ BC)  (CNT, BC như TN3: L16 bc_k06v1, bộ khác bc_k05dxv05)
- `H2_new`: H1_new ∧ bộ kiểm ảnh (công thức TN3) với điểm tính trên CROP MỚI; ngưỡng C5/C2 CHỌN LẠI trên sách tune phần B đúng thủ tục TN1 sel_C5/sel_C2 (τ=0,995, LOBO L16↔TK), lọc tune r2=¬(A0_new∪BC)
- `H4_new`: H2_new ∧ TA_OK (dị bản người, không đổi)
- `scorers`: verifier_ft: CNN/W/nguyên mẫu/logistic 'wood' GIỮ NGUYÊN (học trên crop cũ; logistic tái lập trên đặc trưng cũ, invariant p_wood cũ) — chỉ đổi ảnh đầu vào = khung vuông crop chuẩn qua cùng tiền xử lý 64×64. viss: chạy lại xếp hạng lại r01–r03 (bản chép) với view A = crop chuẩn cho MỌI ô 4 sách + nguyên mẫu 'oth'/'self' từ crop chuẩn GOLD; view B (cắt bbox) giữ nguyên. STT: bộ viết tay LOBO (mô hình/hiệu chuẩn Kinh) giữ nguyên, q=0,00015, n_hum≥3.
- `slot_new`: khe của crop mới = 2 mô hình khe của harness (mẫu t + dòng kim) áp cho TÂM HỘP MỰC mới; slot_ok cũ = tâm bbox (harness)
- `damaged`: chính: slot_ok cũ=1 ∧ slot_ok mới≠1; phụ: độ phủ mực khe người giảm > 0,25 so với crop cũ
- `rescued`: B0_crop_old ∧ one_char_ok ∧ ¬tall_new (IHR: thêm ¬damaged)
- `ihr_measure`: như TN3 measure_ihr: đúng hai vế = V1+(nhãn, chữ người) ∧ khe=1 (khe của crop đang xét); CI bootstrap cụm trang B=2000
- `projection`: 6 bộ còn lại: ô giữ + độ chính xác theo bộ ước lượng TN3 với mặt nạ mới (ƯỚC LƯỢNG/SUY ĐOÁN)
- `circularity`: tham số cclib không chọn trên IHR hay bộ kiểm; ngưỡng bộ kiểm chọn LOBO trên sách kia

Chạy thử trước khi đóng băng v1 (3 trang/bộ, chỉ để bắt lỗi chương trình): Trước khi đóng băng: chạy thử 3 trang mỗi bộ L16/stt2/KVK chỉ để bắt lỗi chương trình (không có lỗi). Thấy: (i) cờ blank của crop_quality pipeline (ngưỡng 128 toàn cục) bật trên crop chặt mới ở L16 mực nhạt -> KHÔNG dùng blank của pipeline cho crop mới (cclib có blank riêng theo ngưỡng cục bộ); (ii) KVK: stray_ink>0,08 trên crop mới ở 10/354 ô (mảnh mực sát mép) -> đưa bleed/truncated của pipeline đo trên crop chặt MỚI vào kiểm một chữ. Không đổi tham số cclib.

Crop mới ghi ra `measure_out/_thu_nghiem_anh_chu/TN4/<v1|v2>/crops/<cell_uid>.png` (khung vuông, độ phân giải trang) và `crops128/` (128×128); ô không phải GOLD của 4 sách harness ở `aux/`, Borg ở `aux_borg/`. Không ghi gì vào `dataset/`.

## 2. Crop cũ ↔ crop chuẩn: cờ "một chữ" trên 8 bộ

Crop cũ đo lại bằng chính luật `save_crop` dựng lại trên trang (md5 byte trùng tệp giao ở mẫu 150 ô/bộ — 24 ô lệch đều là ô rescue 64×64 của STT; cờ pipeline tính lại trùng 100 % `crop_quality_flag`/`seg_flag`). Sạch cũ = không cờ B0 cấp crop (bleed/truncated/blank/tight_nb/tall); sạch mới = `one_char_ok` ∧ ¬tall. Hỏng: chỉ đo được ở L16/TK (§3).

| Bộ | Loại | GOLD | sạch cũ | sạch v1 | sạch v2 | B0 cũ | cứu v2 | cờ mới v2 | bleed cũ→v2 | tall cũ→v2 | hai chữ v2 | cắt nét v2 | stray>0,08 cũ→v2 | hỏng v2 (đo) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| stt2 | chép tay | 17.678 | 17.044 | 16.119 | 17.196 | 634 | 575 | 423 | 102→56 | 335→155 | 147 | 82 | 109→56 | – |
| stt4 | chép tay | 17.440 | 15.581 | 15.514 | 16.159 | 1.859 | 1.666 | 1.088 | 1.683→822 | 162→41 | 382 | 10 | 1.800→822 | – |
| stt11 | chép tay | 17.589 | 15.661 | 16.287 | 16.492 | 1.928 | 1.702 | 871 | 1.215→519 | 712→181 | 312 | 39 | 1.283→520 | – |
| Chr | in văn xuôi | 4.238 | 4.108 | 4.026 | 4.103 | 130 | 115 | 120 | 86→26 | 47→22 | 79 | 3 | 86→26 | – |
| L83 | thạch bản | 10.910 | 10.907 | 10.663 | 10.755 | 3 | 2 | 154 | 2→63 | 1→13 | 5 | 76 | 2→63 | – |
| KVK | thạch bản | 19.958 | 19.789 | 18.989 | 19.583 | 169 | 69 | 275 | 16→146 | 153→185 | 5 | 69 | 16→147 | – |
| L16 | mộc bản IHR | 11.587 | 11.281 | 10.641 | 11.260 | 306 | 284 | 305 | 264→32 | 3→14 | 45 | 219 | 264→36 | khe 20 · lõi 61 |
| TK | mộc bản IHR | 19.554 | 19.417 | 18.679 | 19.334 | 137 | 128 | 211 | 110→21 | 0→1 | 11 | 167 | 110→29 | khe 3 · lõi 2 |
| **Tổng** | | 118.954 | 113.788 | 110.918 | 114.882 | 5.166 | 4.541 | 3.447 | 3.478→1.685 | 1.413→612 | 986 | 665 | 3.670→1.699 | |

"cứu" = B0 cũ ∧ crop mới sạch; "cờ mới" = cũ sạch ∧ crop mới bị cờ — cờ mới KHÔNG phải lỗi chắc chắn (kiểm "hai chữ" bắt hộp gộp hai chữ mà pipeline không cờ; "cắt nét" bắt chữ chạm chữ kề). bleed mới = stray_ink của pipeline đo trên crop chặt mới (mảnh mực tách rời sát mép).

### 2.1 Borg (chữ viết tay, nhãn người) — crop chuẩn trên ô keep ∨ keep_high

| Tập | n | sạch một chữ v1 | sạch v2 | cắt nét v1→v2 | hai chữ v1→v2 | tall v1→v2 | stray v1→v2 |
|---|---:|---:|---:|---|---|---|---|
| keep_v5=1 | 54.884 | 52.401 (95,5 %) | 53.779 (98,0 %) | 1.449→79 | 417→262 | 316→317 | 350→266 |
| keep_v5=0 | 17.067 | 15.999 (93,7 %) | 16.540 (96,9 %) | 509→41 | 276→168 | 94→87 | 189→134 |

### 2.2 Vì sao có v2 (chẩn đoán v1 bằng số, không mở ảnh)

- **Nét kẻ dính chữ**: L16 v1 gắn cờ cắt nét 850 ô, TK 674; chẩn đoán 56 ô ở 6 trang L16: 100 % là "mực ô chạm mép TRÁI/PHẢI cửa sổ" — nét kẻ cột/khung dính chữ thành một thành phần lưới mà luật "thành phần mảnh" của v1 không nhận ra. v2 bóc nét kẻ bằng mở hình thái → cắt nét L16 219, TK 167.
- **Ranh giới bám chữ kề**: KVK v1 làm stray_ink > 0,08 tăng 16 → 647 (mảnh chữ kề lọt vào): v1 lấy chính seam làm ranh giới, khi hoà năng lượng chọn hàng TRÊN CÙNG → ranh giới trên bám sát chữ kề; chữ đầu cột lấy tới 0,8·max(p,h). v2 (ranh giới thẳng ở trung điểm, trung điểm ảo 0,55) → 147.
- **Ranh giới dưới cắt chữ ⿱**: cùng quy tắc hoà, ranh giới DƯỚI bám sát tâm ô (khe trắng bên trong chữ trên-dưới) → L16 v1 73 ô mất > 25 % mực lõi khe; v2 61.

## 3. Còn đúng khe không, sạch hơn không — khe người IHR (CHẮC CHẮN THEO MÁY)

Khe của một crop = hai mô hình khe của harness (mẫu t + dòng kim, chép nguyên `h01_build_cells.py`) áp cho tâm crop (cũ: tâm bbox — tái lập 100 % `slot_ok` harness; mới: tâm hộp mực). Khe người để đo mực = hộp cột NGƯỜI vẽ × vị trí mẫu của khe; mực = xám < ngưỡng Otsu cục bộ của ô (cùng ngưỡng cho crop cũ và mới). "Lõi" = bỏ 15 % trên/dưới khe và 10 % hai bên cột (độ nhạy thêm sau khi thấy mép khe mẫu lệch và có nét kẻ). Đo mực trên ô GOLD có khe cũ = 1. Cột "cũ" lấy từ lượt đo v2 (ngưỡng Otsu cục bộ tính theo cửa sổ của v2; lượt v1 cho số "cũ" lệch ≤ 0,1 điểm %).

| Chỉ số | L16 cũ | L16 v1 | L16 v2 | TK cũ | TK v1 | TK v2 |
|---|---:|---:|---:|---:|---:|---:|
| khe = 1 | 11.333 | 11.333 | 11.330 | 18.416 | 18.413 | 18.419 |
| khe = 0 (sai khe) | 187 | 171 | 167 | 63 | 65 | 62 |
| khe chưa xác định | 67 | 83 | 90 | 1.075 | 1.076 | 1.073 |
| **làm hỏng khe** (1 → ≠1) | | 17 | 20 | | 7 | 3 |
| sửa được khe (≠1 → 1) | | 17 | 17 | | 4 | 6 |
| mực ngoài khe người (TB, %) | 13,45 | 8,75 | 7,13 | 5,16 | 4,47 | 3,45 |
| ô > 10 % mực ngoài khe | 5.647 | 2.792 | 2.679 | 2.933 | 2.385 | 2.238 |
| ô > 25 % mực ngoài khe | 1.746 | 1.354 | 935 | 1.043 | 1.113 | 579 |
| ô B0 cũ: mực ngoài khe (TB, %) | 18,30 | 10,92 | 9,87 | 20,62 | 15,00 | 8,84 |
| độ phủ mực khe — đầy đủ (TB, %) | 98,14 | 94,36 | 94,46 | 99,13 | 98,61 | 98,65 |
| độ phủ mực khe — lõi (TB, %) | 99,51 | 98,23 | 98,30 | 99,62 | 99,49 | 99,48 |
| ô phủ < 80 % — đầy đủ | 157 | 652 | 623 | 187 | 131 | 163 |
| ô phủ < 80 % — lõi | 70 | 173 | 139 | 98 | 50 | 68 |
| **mất > 25 % mực khe** — đầy đủ | | 171 | 150 | | 5 | 3 |
| **mất > 25 % mực khe — lõi** (= cắt vào chữ) | | 73 | 61 | | 1 | 2 |

Đọc: v2 giảm số ô có > 25 % mực lạ còn 54 % (L16) và 56 % (TK) so với crop cũ, ô B0 cũ giảm mực lạ từ 18,30 % xuống 9,87 % (L16), mà khe gần như không đổi. Mất mực ở mép khe mẫu phần lớn là mực chữ kề/nét kẻ rơi vào khe mẫu (độ phủ LÕI gần như giữ). Phần còn lại ở L16 (chữ ~33 px, nét mảnh dính chữ kề, luật "đầu nét lạ" bỏ nhầm nét của chính chữ) là **ô bị làm hỏng thật mà không cờ nào bắt**. TK (chữ tách rời rõ) gần như không hỏng.

## 4. Bộ kiểm ảnh trước ↔ sau

Bộ kiểm verifier_ft (CNN + nguyên mẫu + logistic "wood") và kênh viss (xếp hạng lại r01–r03, CHẠY LẠI với view A = crop chuẩn cho mọi ô 4 sách, nguyên mẫu "oth"/"self" từ crop chuẩn GOLD) được **học/hiệu chuẩn trên crop CŨ** và giữ nguyên ("thay ảnh", drop-in). Invariant: logistic tái lập đúng p_wood cũ (sai tối đa 0,0), nhúng lại ảnh cũ trùng (cos ≥ 0,9999; encoder v1+v2 cos ≥ 1,000000); cấu trúc ứng viên r02 trùng lượt gốc (1.919.412 hàng).

### 4.1 AUC trên IHR (LOBO: L16 ← mô hình T, TK ← mô hình L), ô GOLD — CHẮC CHẮN THEO MÁY

"khe" = tách crop đúng/sai khe (khe của chính crop được chấm); "nhãn" = tách nhãn đúng/sai (ô khe = 1); "hai vế" = cả hai. Hiệu = v2 − cũ, CI 95 % bootstrap cụm trang (B = 500, ghép cặp).

| Bộ | Điểm | Việc | AUC cũ | AUC v1 | AUC v2 | hiệu v2 [CI] | v2 khung chặt |
|---|---|---|---:|---:|---:|---|---:|
| L16 | p_wood | khe | 0,9465 | 0,9351 | 0,9394 | -0,0071 [-0,0174; -0,0015] | 0,9397 |
| L16 | p_wood | nhan | 0,7577 | 0,7451 | 0,7514 | -0,0063 [-0,0205; +0,0086] | 0,7504 |
| L16 | p_wood | hai_ve | 0,8583 | 0,8409 | 0,8451 | -0,0132 [-0,0227; -0,0047] | 0,8448 |
| L16 | viss | khe | 0,8909 | 0,8812 | 0,8850 | -0,0059 [-0,0180; +0,0024] | – |
| L16 | viss | nhan | 0,8307 | 0,8222 | 0,8235 | -0,0071 [-0,0176; +0,0015] | – |
| L16 | viss | hai_ve | 0,8648 | 0,8539 | 0,8561 | -0,0087 [-0,0156; -0,0025] | – |
| TK | p_wood | khe | 0,9659 | 0,9684 | 0,9645 | -0,0014 [-0,0120; +0,0141] | 0,9652 |
| TK | p_wood | nhan | 0,6416 | 0,6315 | 0,6329 | -0,0086 [-0,0223; +0,0067] | 0,6318 |
| TK | p_wood | hai_ve | 0,7170 | 0,7121 | 0,7094 | -0,0076 [-0,0174; +0,0037] | 0,7087 |
| TK | viss | khe | 0,8882 | 0,8866 | 0,8877 | -0,0005 [-0,0087; +0,0099] | – |
| TK | viss | nhan | 0,6473 | 0,6440 | 0,6417 | -0,0056 [-0,0128; +0,0002] | – |
| TK | viss | hai_ve | 0,7035 | 0,7022 | 0,6986 | -0,0049 [-0,0107; +0,0006] | – |

"v2 khung chặt" = độ nhạy hậu kiểm: cùng crop v2 nhưng đưa vào verifier_ft theo khung hộp mực + 4 px (giống khung crop cũ). Không khác khung vuông → AUC giảm KHÔNG do khung/tỉ lệ; nhiều khả năng do nội dung crop khác kiểu ảnh bộ kiểm đã học (mực chữ kề bị xoá, nét mảnh bị cắt ở L16) — SUY ĐOÁN.

### 4.2 Bộ kiểm chữ viết tay LOBO-sách trên Borg DungLy giữ ngoài (nhãn người) — CHẮC CHẮN THEO MÁY

Mô hình/nguyên mẫu/hiệu chuẩn chỉ từ Borg Kinh; ngưỡng q = 0,00015 (chính sách v3); cặp dương (crop, chữ người) / âm đồng âm / âm trượt (chữ kề); chỉ ô DungLy có crop chuẩn v2.

| Tập | Crop · bộ kiểm | dương | nhận dương | FAR đồng âm | FAR trượt | AUC dương/trượt | AUC dương/đồng âm |
|---|---|---:|---:|---:|---:|---:|---:|
| keep_v5 | cũ · cũ | 5.315 | 83,05 % | 0,033 % | 0,000 % | 0,9963 | 0,9968 |
| keep_v5 | v1 · cũ (thay ảnh) | 5.400 | 69,54 % | 0,033 % | 0,000 % | 0,9945 | 0,9964 |
| keep_v5 | v2 · cũ (thay ảnh) | 5.315 | 68,97 % | 0,032 % | 0,000 % | 0,9947 | 0,9963 |
| keep_v5 | v2 khung chặt · cũ | 5.315 | 68,99 % | 0,032 % | 0,010 % | 0,9947 | 0,9963 |
| keep_v5 | **v2 · HỌC LẠI trên v2** | 5.315 | 66,94 % | 0,026 % | 0,000 % | 0,9952 | 0,9972 |
| keep_or_high | cũ · cũ | 7.581 | 77,44 % | 0,035 % | 0,007 % | 0,9951 | 0,9965 |
| keep_or_high | v1 · cũ (thay ảnh) | 7.693 | 63,64 % | 0,040 % | 0,014 % | 0,9918 | 0,9954 |
| keep_or_high | v2 · cũ (thay ảnh) | 7.581 | 62,66 % | 0,036 % | 0,014 % | 0,9917 | 0,9952 |
| keep_or_high | v2 khung chặt · cũ | 7.581 | 62,80 % | 0,037 % | 0,028 % | 0,9917 | 0,9952 |
| keep_or_high | **v2 · HỌC LẠI trên v2** | 7.581 | 60,07 % | 0,026 % | 0,000 % | 0,9932 | 0,9967 |

STT (52.707 ô GOLD) được chứng nhận ở q = 0,00015: cũ 17,89 % → v1 thay ảnh 10,52 % → v2 thay ảnh 10,47 % → v2 học lại 6,82 %.

### 4.3 Học lại bộ kiểm viết tay trên crop chuẩn v2 (hậu kiểm)

`h01_train_tn4.py` = bản chép `r5/hand_lobo/h01_train.py` (chỉ đổi thư mục), cùng hạt giống/bước (2.800)/tập học (Borg Kinh keep fold 0–3 + IHR sách tune phần A), ảnh = crop chuẩn v2 (ô Borg không có crop chuẩn giữ ảnh cũ, không vào tập học). `s10b_hand_retrain_eval.py` chép đặc trưng (s01/h02) và hiệu chuẩn (h03: logistic đầy đủ Kinh fold 4; ngưỡng = lượng tử cross-fit hai nửa trang trên âm).

Ngưỡng q = 0,00015: học lại T 0,99764, L 0,99770 (cũ T 0.99433, L 0.99504).

| STT | H4 cũ | H4 v2 thay ảnh | H4 v2 học lại (H1 v2 ∧ chứng nhận ∧ n_hum ≥ 3) |
|---|---:|---:|---:|
| stt11 | 2.511 | 1.579 | 1.231 |
| stt2 | 2.400 | 1.473 | 864 |
| stt4 | 1.428 | 957 | 505 |

Độ chính xác của H4 học lại ở STT chưa ước lượng lại (cần tỉ lệ nhận trượt của bộ kiểm học lại trên Borg — cột FAR trượt ở §4.2 là số tương ứng); SUY ĐOÁN.

## 5. H4 (và H2, H1) của TN3 với crop cũ ↔ crop chuẩn

H1 dùng cờ của CROP MỚI (A0_new, B0_new; CNT và cờ trượt BC giữ nguyên); điểm bộ kiểm tính trên crop mới; ngưỡng τ = 0,995 CHỌN LẠI trên sách kia (LOBO) đúng thủ tục TN1 (invariant: thủ tục tái lập đúng ngưỡng TN1 và số ô H2/H4 cũ ở cả 6 book_set). L16/TK đo với KHE CỦA CROP MỚI. Bộ khác: bộ ước lượng TN3 chạy lại (`s06c_tn3m.py` = bản chép `t02_matrix.py`, chỉ đổi mặt nạ và khe) — ƯỚC LƯỢNG/SUY ĐOÁN như TN3. Cột "lai" (hậu kiểm) = crop chuẩn v2 + H1 v2 + bộ kiểm chấm crop CŨ với ngưỡng TN1.

Ngưỡng chọn lại (C5 = (p_wood, viss); C2 = p_wood): old_T: C5 [0.98744, 0.09307], C2 0.98835; old_L: C5 [0.95369, 0.13467], C2 0.9745; new_T: C5 [0.98736, 0.09176], C2 0.98957; new_L: C5 [0.95452, 0.09929], C2 0.96565 · v1: new_T: C5 [0.98736, 0.09446], C2 0.98957; new_L: C5 [0.93877, 0.14259], C2 0.97044.

### H4

| Bộ | Mức chắc | cũ: giữ (%) · chính xác [khoảng] | v1 | v2 | Δ ô v2 − cũ | lai (v2 + bộ kiểm cũ) | Δ lai − cũ |
|---|---|---|---|---|---:|---|---:|
| stt2 | SUY ĐOÁN | 2.400 (13,6 %) · **98,38** [96,39–99,24] | 1.327 (7,5 %) · **98,22** [96,49–99,15] | 1.473 (8,3 %) · **98,38** [96,50–99,25] | -927 | 2.455 (13,9 %) · **98,39** [96,40–99,24] | +55 |
| stt4 | SUY ĐOÁN | 1.428 (8,2 %) · **98,23** [96,15–99,18] | 965 (5,5 %) · **98,25** [96,45–99,22] | 957 (5,5 %) · **98,25** [96,25–99,21] | -471 | 1.567 (9,0 %) · **98,26** [96,19–99,19] | +139 |
| stt11 | SUY ĐOÁN | 2.511 (14,3 %) · **98,44** [96,46–99,26] | 1.644 (9,3 %) · **98,29** [96,62–99,16] | 1.579 (9,0 %) · **98,34** [96,47–99,19] | -932 | 2.760 (15,7 %) · **98,47** [96,48–99,28] | +249 |
| Chr | SUY ĐOÁN | 309 (7,3 %) · **99,11** [97,14–99,77] | 190 (4,5 %) · **99,03** [97,32–99,73] | 185 (4,4 %) · **99,00** [97,06–99,72] | -124 | 325 (7,7 %) · **99,10** [97,13–99,77] | +16 |
| L83 | ƯỚC LƯỢNG | 648 (5,9 %) · **99,81** [94,76–99,92] | 750 (6,9 %) · **99,83** [94,67–99,92] | 741 (6,8 %) · **99,84** [95,04–99,94] | +93 | 641 (5,9 %) · **99,81** [94,95–99,92] | -7 |
| KVK | ƯỚC LƯỢNG | 3.676 (18,4 %) · **99,82** [98,03–99,92] | 4.204 (21,1 %) · **99,83** [98,04–99,92] | 4.465 (22,4 %) · **99,85** [98,02–99,94] | +789 | 3.647 (18,3 %) · **99,81** [98,02–99,92] | -29 |
| L16 | CHẮC CHẮN THEO MÁY | 2.737 (23,6 %) · **99,49** [99,14–99,77] | 2.750 (23,7 %) · **99,49** [99,16–99,77] | 2.871 (24,8 %) · **99,48** [99,18–99,73] | +134 | 2.760 (23,8 %) · **99,46** [99,09–99,74] | +23 |
| TK | CHẮC CHẮN THEO MÁY | 9.766 (49,9 %) · **99,87** [99,80–99,93] ✔ | 10.211 (52,2 %) · **99,86** [99,79–99,93] ✔ | 9.452 (48,3 %) · **99,88** [99,81–99,95] ✔ | -314 | 9.747 (49,9 %) · **99,87** [99,80–99,93] ✔ | -19 |

### H2

| Bộ | Mức chắc | cũ: giữ (%) · chính xác [khoảng] | v1 | v2 | Δ ô v2 − cũ | lai (v2 + bộ kiểm cũ) | Δ lai − cũ |
|---|---|---|---|---|---:|---|---:|
| stt2 | SUY ĐOÁN | 2.400 (13,6 %) · **98,38** [96,39–99,24] | 1.327 (7,5 %) · **98,22** [96,49–99,15] | 1.473 (8,3 %) · **98,38** [96,50–99,25] | -927 | 2.455 (13,9 %) · **98,39** [96,40–99,24] | +55 |
| stt4 | SUY ĐOÁN | 1.428 (8,2 %) · **98,23** [96,15–99,18] | 965 (5,5 %) · **98,25** [96,45–99,22] | 957 (5,5 %) · **98,25** [96,25–99,21] | -471 | 1.567 (9,0 %) · **98,26** [96,19–99,19] | +139 |
| stt11 | SUY ĐOÁN | 2.511 (14,3 %) · **98,44** [96,46–99,26] | 1.644 (9,3 %) · **98,29** [96,62–99,16] | 1.579 (9,0 %) · **98,34** [96,47–99,19] | -932 | 2.760 (15,7 %) · **98,47** [96,48–99,28] | +249 |
| Chr | SUY ĐOÁN | 309 (7,3 %) · **99,11** [97,14–99,77] | 190 (4,5 %) · **99,03** [97,32–99,73] | 185 (4,4 %) · **99,00** [97,06–99,72] | -124 | 325 (7,7 %) · **99,10** [97,13–99,77] | +16 |
| L83 | ƯỚC LƯỢNG | 1.858 (17,0 %) · **98,24** [86,47–98,92] | 2.132 (19,5 %) · **98,49** [86,26–99,07] | 2.112 (19,4 %) · **98,38** [87,14–99,02] | +254 | 1.834 (16,8 %) · **98,23** [86,94–98,92] | -24 |
| KVK | ƯỚC LƯỢNG | 4.689 (23,5 %) · **99,30** [96,74–99,55] | 5.334 (26,7 %) · **99,39** [96,74–99,60] | 5.609 (28,1 %) · **99,39** [96,72–99,60] | +920 | 4.650 (23,3 %) · **99,29** [96,72–99,54] | -39 |
| L16 | CHẮC CHẮN THEO MÁY | 2.737 (23,6 %) · **99,49** [99,14–99,77] | 2.750 (23,7 %) · **99,49** [99,16–99,77] | 2.871 (24,8 %) · **99,48** [99,18–99,73] | +134 | 2.760 (23,8 %) · **99,46** [99,09–99,74] | +23 |
| TK | CHẮC CHẮN THEO MÁY | 13.643 (69,8 %) · **99,12** [98,95–99,28] | 14.436 (73,8 %) · **99,09** [98,92–99,24] | 13.269 (67,9 %) · **99,11** [98,93–99,27] | -374 | 13.613 (69,6 %) · **99,12** [98,95–99,28] | -30 |

### H1

| Bộ | Mức chắc | cũ: giữ (%) · chính xác [khoảng] | v1 | v2 | Δ ô v2 − cũ |
|---|---|---|---|---|---:|
| stt2 | SUY ĐOÁN | 13.025 (73,7 %) · **95,05** [93,52–96,66] | 12.326 (69,7 %) · **95,02** [93,48–96,62] | 13.158 (74,4 %) · **95,08** [93,53–96,66] | +133 |
| stt4 | SUY ĐOÁN | 11.679 (67,0 %) · **94,67** [93,05–96,39] | 11.663 (66,9 %) · **94,67** [93,06–96,36] | 12.172 (69,8 %) · **94,70** [93,08–96,38] | +493 |
| stt11 | SUY ĐOÁN | 10.407 (59,2 %) · **95,86** [94,49–97,21] | 10.877 (61,8 %) · **95,82** [94,44–97,16] | 11.017 (62,6 %) · **95,84** [94,44–97,18] | +610 |
| Chr | SUY ĐOÁN | 1.794 (42,3 %) · **98,30** [97,27–98,95] | 1.788 (42,2 %) · **98,30** [97,27–98,93] | 1.813 (42,8 %) · **98,32** [97,28–98,95] | +19 |
| L83 | ƯỚC LƯỢNG | 10.184 (93,3 %) · **98,26** [72,41–98,79] | 10.011 (91,8 %) · **98,25** [72,30–98,80] | 10.059 (92,2 %) · **98,25** [72,38–98,78] | -125 |
| KVK | ƯỚC LƯỢNG | 19.506 (97,7 %) · **99,09** [79,35–99,30] | 18.740 (93,9 %) · **99,08** [79,17–99,30] | 19.311 (96,8 %) · **99,08** [79,36–99,29] | -195 |
| L16 | CHẮC CHẮN THEO MÁY | 10.951 (94,5 %) · **98,05** [97,64–98,42] | 10.359 (89,4 %) · **98,06** [97,63–98,45] | 10.948 (94,5 %) · **98,05** [97,64–98,42] | -3 |
| TK | CHẮC CHẮN THEO MÁY | 18.511 (94,7 %) · **98,85** [98,70–98,99] | 17.828 (91,2 %) · **98,82** [98,66–98,96] | 18.440 (94,3 %) · **98,84** [98,69–98,99] | -71 |

Khoảng: L16/TK = CI 95 % bootstrap cụm trang (B = 2.000); bộ khác = [cận bi quan – cận lạc quan] của bộ ước lượng TN3. ✔ = đạt tiêu chí TN3 (đo: điểm ≥ 99,5 %; ước lượng/suy đoán: cận bi quan ≥ 99 %; ≥ 50 ô). STT/Chr H2 = H4 (không có văn bản người của dị bản).

### 5.1 Độ nhạy: dao động do chọn ngưỡng (hậu kiểm)

Bootstrap B = 300 lần các TRANG phần B của sách tune → chọn lại C5 → áp lên sách thử (cùng mẫu trang cho cũ/v1/v2). Đếm ô H4 có GT (lệch ≤ 2 ô so với bảng trên vì bỏ ô không có chữ người).

| Sách thử | Crop | ô giữ (lượt gốc) | ô giữ trung vị [5 %–95 %] | chính xác trung vị (5 %) | hiệu với cũ: trung vị [5 %; 95 %] | P(mới > cũ) |
|---|---|---:|---|---|---|---:|
| L16 | cũ | 2.736 | 2.980 [1.343–6.670] | 99,54 % (99,09 %) | – | – |
| L16 | v1 | 2.749 | 2.785 [1.389–6.198] | 99,50 % (99,33 %) | -164 [-1.476; +778] | 0,36 |
| L16 | v2 | 2.870 | 2.996 [1.456–7.127] | 99,51 % (99,16 %) | +110 [-1.299; +1.609] | 0,58 |
| TK | cũ | 9.766 | 10.318 [9.200–12.137] | 99,86 % (99,79 %) | – | – |
| TK | v1 | 10.211 | 10.414 [9.323–11.714] | 99,82 % (99,76 %) | -39 [-939; +1.395] | 0,48 |
| TK | v2 | 9.452 | 10.558 [7.527–12.002] | 99,83 % (99,77 %) | +104 [-1.958; +1.223] | 0,54 |

→ Chênh số ô H4 giữa crop cũ và crop chuẩn ở L16/TK nằm gọn trong dao động do chọn ngưỡng; kết luận "giữ nhiều/ít hơn" **không được hỗ trợ** bởi dữ liệu. (Thạch bản dùng ngưỡng chọn trên IHR nên cũng chịu dao động này.)

## 6. Invariant (tự kiểm; mọi mục True trừ ghi chú)

- **v1 s02**: `design_frozen`=True, `n_cells`=118954, `uid_unique`=True, `all_gold_have_crop_record`=True, `old_cqf_recomputed_eq`=1.0, `old_tall_recomputed_eq`=1.0, `slot_old_bbox_eq_harness_L16`=1.0, `slot_old_bbox_eq_harness_TK`=1.0, `md5_mismatch_n`=24, `md5_mismatch_all_rescue64`=True
- **v2 s02**: `design_frozen`=True, `n_cells`=118954, `uid_unique`=True, `all_gold_have_crop_record`=True, `old_cqf_recomputed_eq`=1.0, `old_tall_recomputed_eq`=1.0, `slot_old_bbox_eq_harness_L16`=1.0, `slot_old_bbox_eq_harness_TK`=1.0, `md5_mismatch_n`=24, `md5_mismatch_all_rescue64`=True
- **v1 s03**: `gold_new_missing`=25, `ihr_new_missing`=36, `reembed_ihr_T_maxabs`=0.000732421875, `reembed_ihr_T_cos_min`=0.9998581409454346, `feat_gold_old_reproduce_T`=0.0, `p_wood_T_old_reproduce_maxabs`=0.0, `reembed_ihr_L_maxabs`=0.000637054443359375, `reembed_ihr_L_cos_min`=0.9998534917831421, `feat_gold_old_reproduce_L`=0.0, `p_wood_L_old_reproduce_maxabs`=0.0, `ok_reembed`=True, `ok_p_wood_old`=True
- **v2 s03**: `gold_new_missing`=18, `ihr_new_missing`=22, `reembed_ihr_T_maxabs`=0.000732421875, `reembed_ihr_T_cos_min`=0.9998581409454346, `feat_gold_old_reproduce_T`=0.0, `p_wood_T_old_reproduce_maxabs`=0.0, `reembed_ihr_L_maxabs`=0.000637054443359375, `reembed_ihr_L_cos_min`=0.9998534917831421, `feat_gold_old_reproduce_L`=0.0, `p_wood_L_old_reproduce_maxabs`=0.0, `ok_reembed`=True, `ok_p_wood_old`=True
- **v2 s04b**: `reembed_old_cos_min`=0.9999998807907104, `ok_reembed`=True, `gold_missing`=18, `harness_gold_from_crops`=62009, `aux_missing`=13
- **v1 s05**: `borg_rows_with_new_crop`=71951, `reembed_stt_T_maxabs`=0.000396728515625, `p_T_old_reproduce_maxabs`=0.0, `reembed_stt_L_maxabs`=0.00048828125, `p_L_old_reproduce_maxabs`=0.0, `cert_old_eq_saved`=0.9998482175043163
- **v2 s05**: `borg_rows_with_new_crop`=71839, `reembed_stt_T_maxabs`=0.000396728515625, `p_T_old_reproduce_maxabs`=0.0, `reembed_stt_L_maxabs`=0.00048828125, `p_L_old_reproduce_maxabs`=0.0, `cert_old_eq_saved`=0.9998482175043163
- **v1 s06a**: `viss_old_join_eq`=True, `thresholds_old_reproduce_TN1`=True, `H2_old_eq_TN1_d_STT`=True, `H4_old_eq_TN1_e_STT`=True, `H2_old_eq_TN1_d_Chr`=True, `H4_old_eq_TN1_e_Chr`=True, `H2_old_eq_TN1_d_L83`=True, `H4_old_eq_TN1_e_L83`=True, `H2_old_eq_TN1_d_KVK`=True, `H4_old_eq_TN1_e_KVK`=True, `H2_old_eq_TN1_d_L16`=True, `H4_old_eq_TN1_e_L16`=True, `H2_old_eq_TN1_d_TK`=True, `H4_old_eq_TN1_e_TK`=True
- **v2 s06a**: `viss_old_join_eq`=True, `thresholds_old_reproduce_TN1`=True, `H2_old_eq_TN1_d_STT`=True, `H4_old_eq_TN1_e_STT`=True, `H2_old_eq_TN1_d_Chr`=True, `H4_old_eq_TN1_e_Chr`=True, `H2_old_eq_TN1_d_L83`=True, `H4_old_eq_TN1_e_L83`=True, `H2_old_eq_TN1_d_KVK`=True, `H4_old_eq_TN1_e_KVK`=True, `H2_old_eq_TN1_d_L16`=True, `H4_old_eq_TN1_e_L16`=True, `H2_old_eq_TN1_d_TK`=True, `H4_old_eq_TN1_e_TK`=True
- Ghi chú: `cert_old_eq_saved` = 0,99985 (8/52.707 ô STT nằm đúng ngưỡng; ngưỡng trong h03_dir_Kinh.json làm tròn 5 chữ số). `md5_mismatch_n` = 24 đều là ô rescue 64×64 (không qua save_crop). Invariant TN3 bên trong `s06c_tn3m.py` (H2 = TN1 (d)…) báo False là ĐÚNG KỲ VỌNG vì mặt nạ đã thay.

## 7. Khuyến nghị cho pipeline

1. **Đưa crop chuẩn v2 vào pipeline dưới dạng TỆP PHỤ + CỜ, chưa thay crop giao nộp.** Lý do (CHẮC CHẮN THEO MÁY trên IHR): mực lạ so với khe người giảm mạnh (ô > 25 % mực lạ còn 54 %/56 % ở L16/TK), khe gần như không đổi (hỏng ≤ 0,18 %), 4.541 ô B0 được cứu (8 bộ). Rủi ro: ~0,5 % ô L16 bị cắt vào nét (chữ nhỏ dính chữ kề) mà không cờ nào bắt → **cần người xem mẫu trước khi thay crop giao nộp** (dùng `vi_du.html` nhóm iv; cỡ mẫu chứng nhận ≤ 1 % lỗi theo TN3 §7: 0 lỗi/299 ô).
2. **Dùng kiểm "một chữ" của crop chuẩn (one_char_ok, two, cut, ink) thay phần cấp crop của B0 trong cổng H1** — H1 v2 giữ thêm ô ở chép tay (stt2 +133, stt4 +493, stt11 +610), bớt ở thạch bản (KVK -195, L83 -125) mà độ chính xác (đo/ước lượng) không đổi.
3. **KHÔNG cho bộ kiểm ảnh hiện tại chấm crop chuẩn** ("thay ảnh"): không lợi ở mộc bản (AUC giảm nhẹ, H4 trong nhiễu chọn ngưỡng) và hại rõ ở chép tay (Borg keep_v5: nhận dương giảm 14,1 điểm % ở cùng FAR). Cấu hình dùng được ngay là **"lai"**: giao crop chuẩn + H1 v2 + bộ kiểm chấm crop CŨ với ngưỡng TN1 (số ở §5, cột lai). Muốn bộ kiểm chấm crop chuẩn thì phải **học lại** bộ kiểm trên crop chuẩn; nhưng với chữ viết tay, kể cả HỌC LẠI trên crop chuẩn v2 cũng chỉ nhận 66,94 % dương Borg (cũ 83,05 %, thay ảnh 68,97 %) ở cùng mức FAR → với STT giữ bộ kiểm trên crop cũ.
4. **Không đổi kết luận TN3**: crop chuẩn không đưa L16, thạch bản, Chr, STT qua ngưỡng chính xác; chỉ TK đạt. Vế "một chữ" vẫn chưa có sự thật người — khe người IHR chỉ là thước đo gián tiếp.
5. **Tham số đề xuất (v2, `cclib_v2.PARAMS`)**: ranh giới dọc = trung điểm tâm (virt 0,55·max(p,h) khi thiếu ô kề, kẹp 0,8·max(p,h)); seam chỉ cho thành phần vắt qua, dải ± 0,25·d, phạt 20/255; dải ngang xc ± 0,75·w; Otsu cục bộ kẹp [64, 200]; bóc nét kẻ nhân 1,2·p / 1,2·w; đầu nét lạ 20 %/10 %; đốm max(3, 0,0015·p·w); cờ cut 30 %/chạm mép, two 1,35·p + khe 0,10·p + 20/20, ink [0,4; 2,5]; lề 10 % cạnh dài, 128×128. Việc nên thử tiếp (chưa đo): cờ thêm cho "đầu nét lạ bị bỏ ở chữ nhỏ" (L16), học lại verifier_ft in (T/L) trên crop chuẩn.

## 8. Giới hạn

- **v2 là hậu kiểm**: cấu trúc v2 chọn sau khi thấy v1 trên toàn dữ liệu (kể cả IHR); không dò tham số nhưng vẫn có thể lạc quan. v1 (đăng ký trước) là số "sạch" về thủ tục: v1 kém v2 về số ô sạch ở cả 8 bộ và về mực lạ ở L16/TK (§2, §3), nhưng hỏng khe L16 v1 ít hơn (v1 17, v2 20).
- Bộ kiểm ảnh học trên crop cũ; chỉ bộ kiểm viết tay được học lại (hậu kiểm). verifier_ft in và kênh viss chưa học lại.
- Khe người là khe MẪU (vị trí trung vị theo sách trong hộp cột người vẽ), không phải hộp chữ người; mép khe lệch vài px → dùng thêm "lõi". Mực đo bằng ngưỡng Otsu cục bộ của chính thuật toán mới (cùng ngưỡng cho cũ/mới).
- "Một chữ" không có sự thật người ở bộ nào; cờ bleed/tall/hai chữ là luật máy. Ô "cờ mới" không chắc là lỗi.
- STT: trang đã xử lý là ảnh nhị phân (không có nền giấy gốc); crop chuẩn STT giữ nền trắng như pipeline.
- Số ô H4 phụ thuộc mạnh vào chọn ngưỡng trên ~2.800–4.700 ô tune (§5.1); ước lượng ở bộ không có nhãn người thừa hưởng mọi giả định của TN3.
- Borg: crop chuẩn chỉ dựng cho ô keep ∨ keep_high (71.951), không cho ô không keep.

## 9. Tệp và tái lập

| Tệp | Việc |
|---|---|
| `cclib.py / cclib_v2.py` | thuật toán crop chuẩn v1 (đăng ký trước) / v2 (hậu kiểm) + dựng lại crop cũ |
| `ver.py` | chọn phiên bản qua biến môi trường TN4_VER = v1 hoặc v2 |
| `hslot.py` | mô hình khe người của harness cho tâm bất kỳ |
| `s01_crop.py` | cắt crop chuẩn (gold/aux/borg) + đo crop cũ, md5 byte, khe người |
| `s02_geom.py, s02b_cov_core.py` | bảng hình học 8 bộ, IHR, Borg; độ phủ lõi |
| `s03_vft.py` | p_wood trên crop chuẩn (bộ kiểm giữ nguyên); `TN4_FRAME=tight` = khung chặt |
| `s04a_rerank_copies.py, s04b_embed_sv.py, rerank_v1/, rerank_v2/` | chạy lại kênh viss (bản chép r02/r02b/r03 chỉ đổi thư mục) |
| `s05_hand.py` | bộ kiểm viết tay LOBO trên STT + Borg DungLy |
| `s06a_masks.py` | H1/H2/H4 mới + chọn lại ngưỡng LOBO |
| `s06b_make_tn3m.py → s06c_tn3m.py` | bộ ước lượng TN3 với mặt nạ mới (`TN4_SIMG=SIMGo` = cấu hình lai) |
| `s06d_boot_sel.py` | độ nhạy chọn ngưỡng |
| `s07_auc.py` | AUC IHR trước/sau |
| `s10a_hand_retrain_prep.py, h01_train_tn4.py, s10b_hand_retrain_eval.py` | học lại bộ kiểm viết tay trên crop v2 |
| `s08_vi_du.py → vi_du.html, anh/, vi_du_manifest.csv` | ví dụ 4 nhóm × 24 ô, phân tầng theo bộ |
| `s09_report.py → KET_QUA.md` | báo cáo này |
| `run_ver.sh` | chuỗi bước 2–9 cho một phiên bản |

```bash
cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR; L=lab/thu_nghiem_anh_chu/TN4_crop_chuan
for V in v1 v2; do
  for s in stt2 stt4 stt11 Chr L83 KVK L16 TK; do TN4_VER=$V .venv/bin/python $L/s01_crop.py --part gold --sets $s --workers 4 --tag $s; done  # ≈ 5 phút
  TN4_VER=$V .venv/bin/python $L/s01_crop.py --part aux --workers 4; TN4_VER=$V .venv/bin/python $L/s01_crop.py --part borg --workers 4
  TN4_VER=$V bash $L/run_ver.sh 1        # s02 → s07 (≈ 20 phút, MPS cho CNN)
done
TN4_VER=v2 TN4_FRAME=tight .venv/bin/python $L/s03_vft.py; TN4_VER=v2 TN4_FRAME=tight .venv/bin/python $L/s05_hand.py; TN4_VER=v2 TN4_FRAME=tight .venv/bin/python $L/s07_auc.py
TN4_VER=v2 TN4_SIMG=SIMGo .venv/bin/python $L/s06c_tn3m.py      # cấu hình lai
.venv/bin/python $L/s06d_boot_sel.py                              # độ nhạy chọn ngưỡng (~1 phút)
.venv/bin/python $L/s10a_hand_retrain_prep.py && H=measure_out/_thu_nghiem_anh_chu/TN4/v2/hand_retrain
.venv/bin/python $L/h01_train_tn4.py Kinh TruyenKieu1872 2800 T; .venv/bin/python $L/h01_train_tn4.py Kinh LucVanTien1916 2800 L   # 2 × ~10 phút MPS
.venv/bin/python $L/s10b_hand_retrain_eval.py
.venv/bin/python $L/s08_vi_du.py && .venv/bin/python $L/s09_report.py
```

Đầu vào chỉ đọc: `dataset/_ALL/labels.csv`, `prepared/<Book>/{pages,manifest.json,dataset_out/labels.csv,kim_raw}`, `dataset_out/labels.csv` (STT), `data/<Book>/` (ảnh gốc, manifest.tsv IHR), scratchpad `$SP/{r6/policy_v3/out/base_v2.pkl, r4/verifier_ft, r5/hand_lobo, kim_bottleneck/{harness,rerank}, gold_img_audit/m_ocr, r4/borg_align, r5/borg_quality}`, `lab/.../TN1_cong_kiem`, `TN3_tat_ca_bo`.

