# Vòng 9 — áp detector v2 CÓ KIỂM CHỨNG theo từng sách + gộp `prepared/` (23/09/2026)

Trên HEAD `1c44d200a9` + thay đổi chưa commit của vòng 7–8. **Chưa commit gì.** `data/` chỉ ĐỌC.
Tiền đề: người dùng đã tải `lab/i5_detector_v2/last.pt` (epoch 6, 330 MB) từ Kaggle.

> **Năm dòng kết luận.**
> 1. **v2 chỉ được khai cho 3/5 sách** — LucVanTien1883, KimVanKieu1884, TruyenKieu1872 — vì chỉ ở đó
>    `box_ref_eval` trên **ảnh pipeline thật** cho thấy v2 **không giảm chỉ số nào**. Chrestomathie1872 và
>    LucVanTien1916 **giữ v1** (số đo ở §2). STT **không đụng** (`config/pipeline.yaml` không khai khoá).
> 2. Ba sách nhận v2: **cắt vào thân chữ 9,2 → 0,2 % · 2,5 → 0,1 % · 63,8 → 5,9 %**; IoU trung vị
>    0,75/0,74/0,63 → **0,87/0,87/0,82**; I5 thô (cột `n_det == N`) 70,7/62,3/11,9 → **96,3/97,0/15,5 %**.
> 3. Bộ giao nộp: **ảnh export 11.780 → 11.936 · 19.632 → 20.545 · 13.482 → 19.605**
>    (tổng 5 bộ 60.469 → **67.661**, +7.192 ảnh). `crop_quality_flag = bleed` 2.223 → **2** · 2.179 → **16** ·
>    3.623 → **112**. `GOLD_text_only` gần như biến mất (143 → 2 · 720 → 1 · 6.670 → 135) vì crop hết hỏng.
> 4. **Precision trên NHÃN NGƯỜI giữ nguyên trong sai số**: TK1872 98,64 % (n 12.706) → **98,61 %**
>    (n **18.550**, CI [98,44; 98,77]) — tức **+5.844 ô ảnh giao nộp ở cùng độ đúng**; LVT1916 **97,96 %
>    không đổi một chữ số** (giữ v1, dựng lại byte-identical). Trôi **7 / 16 / — / 91 / 32 không đổi**.
> 5. **`prepared/` đã gộp**: mọi sách ghi vào `prepared/<Book>/`; `prepared_b1/`, `prepared_ihr/` đã dời,
>    `prepared_b1_080/` + `prepared_b1_boost/` đã xoá. Cache kim **tái dùng 100 %** (`verify_cache_image`
>    `ok` 105/163/65/99/161) ⇒ **0 lượt gọi API**. Hồi quy STT `labels.csv` md5
>    **`59e436d7641fa849bb6759868ac29259`** (556 dòng, 42 cột), `diff -rq` **0 tệp khác**.

---

## 0. Tái lập

```bash
PY=.venv/bin/python
# (1) cài ckpt v2 (bản gọn: chỉ model+arch+img+use_dcn+val, 86 MB thay vì 330 MB của last.pt)
$PY - <<'EOF'
import torch
d = torch.load("lab/i5_detector_v2/last.pt", map_location="cpu", weights_only=False)
keep = {k: d[k] for k in ("model","arch","stride","img","use_dcn","epoch","v2","init","val") if k in d}
keep["selected_by"] = "litho_only"
torch.save(keep, "train_crop/detector_r34_v2_litho.pt")
EOF

# (2) ĐO TRƯỚC KHI KHAI — 4 cấu hình × 5 sách × 27 trang (≈ 12 phút CPU, 0 API)
for cfg in "v1_linear:" "v1_area:--resize area" "v2_linear:--ckpt train_crop/detector_r34_v2_litho.pt" \
           "v2_area:--ckpt train_crop/detector_r34_v2_litho.pt --resize area"; do
  $PY scripts/measure/box_ref_eval.py --book all5 --pages 27 --variants prepared \
      ${cfg#*:} --out measure_out/box_ref_v9_${cfg%%:*}
done

# (3) khai detector_ckpt CHỈ cho sách thắng (§2)
SK=lab/i5_detector_v2/set_book_keys.py; DST=train_crop/detector_r34_v2_litho.pt
$PY $SK config/pipeline_LucVanTien1883.yaml    --book LucVanTien1883 --set detector_ckpt=$DST --set detector_resize=area
$PY $SK config/pipeline_KimVanKieu1884.yaml    --book KimVanKieu1884 --set detector_ckpt=$DST --set detector_resize=area
$PY $SK config/pipeline_KimVanKieu1884_b1.yaml --book KimVanKieu1884 --set detector_ckpt=$DST --set detector_resize=area
$PY $SK config/pipeline_TruyenKieu1872.yaml    --book TruyenKieu1872 --set detector_ckpt=$DST --set detector_resize=area
# Chrestomathie1872 + LucVanTien1916: KHÔNG khai (giữ detector_resize: linear, ckpt v1)

# (4) chạy lại 5 bộ (≈ 40 phút CPU, 0 API — kim đọc từ prepared/<Book>/kim_raw/)
for b in LucVanTien1883 KimVanKieu1884 Chrestomathie1872 LucVanTien1916 TruyenKieu1872; do mv dataset/$b dataset/${b}_v8; done
NONINTERACTIVE=1 ./run_pipeline.sh --book all-new
NONINTERACTIVE=1 ./run_pipeline.sh --book all-ihr

# (5) nghiệm thu
$PY scripts/measure/align_audit.py --book all
$PY scripts/measure/ihr_endtoend_eval.py --book all
$PY scripts/measure/crop_source_samples.py --build prepared/LucVanTien1883/dataset_out \
    --build prepared/KimVanKieu1884/dataset_out --build prepared/TruyenKieu1872/dataset_out
$PY scripts/measure/code_facts.py --check
printf 'book,page\nstt2,page_0024\nstt4,page_0050\nstt11,page_0100\n' > /tmp/pages3.csv
$PY -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --qd01-cells none \
    --force --pages /tmp/pages3.csv --out <scratch>/stt && md5 <scratch>/stt/labels.csv
```

Hoàn nguyên một sách về v1 (một lệnh, không phải dựng lại ckpt):

```bash
$PY lab/i5_detector_v2/set_book_keys.py config/pipeline_<X>.yaml --book <X> \
    --remove detector_ckpt --set detector_resize=linear
```

---

## 1. Vì sao phải đo lại thay vì tin bảng Kaggle

Bảng trong notebook (`lab/i5_detector_v2/README.md`) là **trên bundle**: ảnh thu về long side 1536 bằng
INTER_AREA, ô tham chiếu quy đổi theo `scale_table.csv`, và **chỉ có 3 miền** (LVT+KVK litho, STT, Chresto).
Hai bộ IHR **không có trong bundle** nên chưa từng được đo. Ngoài ra chính bảng ấy đã cảnh báo hai chỗ tụt:
`stt_F1@0,2` 0,8772 → 0,8625 và `chresto_tiers_eq_015` 83,8 → 73,7.

Vòng 9 đo lại bằng `scripts/measure/box_ref_eval.py` trên **đúng ảnh mà pipeline đưa vào detector**
(`prepared/<Book>/pages/*.png`, biến thể `prepared`), đúng `det_thr` của từng sách (0,15) và đúng
`det_xmargin` 0,05. Để làm được điều đó, `box_ref_eval` được mở rộng cho **cả 5 sách** (§5).

---

## 2. Bảng quyết định — `box_ref_eval`, 27 trang/sách, ảnh pipeline, thr 0,15

Cột: **I5** = % cột có `n_det == N` (hộp thô, đúng định nghĩa I5 của báo cáo) · **tầng n==N** = % tầng
6/8 (prose: % cột) có đúng số hộp · **ok50** = % ô tham chiếu ghép IoU ≥ 0,5, **chỉ tính ô nguồn
detector/detector_low** (bản `honest`, đã loại ô `ink_cut` trùng tham chiếu theo cấu tạo) · **extra** = hộp
thừa / 100 ô · **cắt** = % hộp có mực ở 2 hàng mép > 0,20 (`crop_quality.BORDER_INK_MAX`) · **h/p** = chiều
cao hộp / bước.

| sách | cấu hình | I5 | tầng n==N | ok50 | miss | extra/100 | IoU | **cắt** | h/p |
|---|---|---|---|---|---|---|---|---|---|
| **LucVanTien1883** | v1 linear *(chốt vòng 8)* | 69,6 | 82,2 | 98,1 | 0,0 | 0,2 | 0,75 | 9,2 | 1,14 |
| | v1 area | 88,9 | 94,4 | 99,2 | 0,0 | 0,1 | 0,75 | 10,3 | 1,16 |
| | v2 linear | 95,9 | 98,1 | 100,0 | 0,0 | 0,1 | 0,87 | 0,3 | 1,03 |
| | **v2 area ✅ CHỌN** | **96,7** | **98,3** | **100,0** | 0,0 | **0,1** | **0,87** | **0,2** | **1,03** |
| **KimVanKieu1884** | v1 linear *(chốt)* | 58,9 | 74,6 | 97,5 | 0,0 | 0,4 | 0,74 | 2,5 | 1,11 |
| | v1 area | 81,1 | 89,8 | 99,0 | 0,0 | 0,3 | 0,76 | 2,5 | 1,13 |
| | v2 linear | 95,6 | 97,6 | 100,0 | 0,0 | 0,3 | 0,87 | 0,1 | 1,03 |
| | **v2 area ✅ CHỌN** | **96,7** | **98,1** | **100,0** | 0,0 | **0,2** | **0,87** | **0,1** | **1,03** |
| **Chrestomathie1872** | **v1 linear ✅ GIỮ** | **67,6** | **69,4** | **65,1** | 0,0 | **4,6** | **0,56** | 78,1 | 1,24 |
| | v1 area | 65,3 | 67,1 | 65,5 | 0,0 | 4,5 | 0,56 | 77,9 | 1,24 |
| | v2 area ❌ | 50,6 | 63,5 | 59,9 | 0,0 | 5,0 | 0,55 | 73,3 | 1,23 |
| **LucVanTien1916** | **v1 linear ✅ GIỮ** | **21,5** | 35,7 | 96,8 | 0,0 | 1,1 | 0,72 | 94,5 | 1,10 |
| | v1 area | 18,5 | 30,7 | 96,0 | 0,0 | 1,2 | 0,71 | 94,4 | 1,09 |
| | v2 area ❌ | **1,5** | 41,1 | 99,1 | 0,0 | 0,7 | 0,78 | 90,1 | 1,06 |
| **TruyenKieu1872** | v1 linear *(chốt)* | 12,6 | 21,5 | 73,4 | 0,0 | 10,7 | 0,63 | 63,8 | 0,97 |
| | v1 area | 15,9 | 23,7 | 76,6 | 0,0 | 10,4 | 0,65 | 64,1 | 0,97 |
| | v2 linear | 17,0 | 55,4 | 99,7 | 0,0 | 0,6 | 0,82 | 6,2 | 1,00 |
| | **v2 area ✅ CHỌN** | **18,9** | **56,7** | **99,8** | 0,0 | **0,5** | **0,82** | **5,9** | **1,00** |

**Hai sách bị từ chối — lý do bằng số, không bằng cảm tính.**

* **Chrestomathie1872 (văn xuôi, KHÔNG có trong tập train)**: v2 làm **I5 tụt 17,0 điểm** (67,6 → 50,6),
  tầng n==N −5,9, ok50 −5,2, extra +0,4. Chỉ mỗi "cắt" giảm (78,1 → 73,3). Đúng như cảnh báo trong
  `README.md` của lab (`chresto_tiers_eq_015` 83,8 → 73,7 ngay trên bundle). → **giữ v1 linear**.
* **LucVanTien1916**: v2 tốt hơn ở ok50 (96,8 → 99,1), IoU (0,72 → 0,78), cắt (94,5 → 90,1) và tầng n==N
  (35,7 → 41,1) — **nhưng I5 sụp 21,5 → 1,5 %**: v2 sinh **390 hộp NGOÀI hai tầng ở 85,9 % số cột**
  (v1: 0 hộp / 0 % cột). Giả thuyết là v2 bắt được **chữ số câu ở lề** mà ô tham chiếu không có, nhưng
  **chưa chứng minh được** (CLAUDE.md cấm mở ảnh bằng LLM) ⇒ theo luật "không được giảm chỉ số then chốt",
  **giữ v1 linear**. Cách kiểm tiếp ở §7.
* Cùng họ IHR nhưng **TruyenKieu1872 thì ngược lại**: I5 **tăng** 12,6 → 18,9 (và ở thr 0,2 là 5,9 → 84,8),
  hộp ngoài tầng chỉ 5,2 % số cột / 13 hộp. Nên hai bộ IHR **không** được xử như một khối.

> ⚠️ **"cắt vào thân chữ" KHÔNG so được giữa các sách.** Nó đo mực ở **2 hàng pixel mép**; ảnh IHR chỉ
> 760×493 và 579×404 px (ô ~37–40 px) nên 2 hàng là ~5 % chiều cao ô, còn LVT1883 là 3.196×1.910 (ô ~150 px)
> nên 2 hàng là ~1,3 %. Vì vậy 94,5 % của LVT1916 và 9,2 % của LVT1883 **không cùng thang**. Chỉ so
> **trong cùng một sách, giữa v1 và v2**.

---

## 3. Bảng TRƯỚC → SAU của bộ giao nộp, đủ 5 bộ

Nguồn: `prepared/<Book>/dataset_out/labels_gated.csv` (ảnh chụp bản vòng 8 giữ ở scratchpad) + đếm `.png`
thật trong `dataset/<Book>{_v8,}/{gold,syllable}`. M==N và I5 tính **theo cột** (`page`,`column` duy nhất),
đúng quy ước `align_audit`. "Trôi" = ô đúng chữ sai vị trí (`align_audit *_drift.csv`).

| sách | detector | ô | GOLD ảnh | GOLD_text_only | SYLLABLE | REVIEW | ảnh export | M==N | I5 thô | trôi |
|---|---|---|---|---|---|---|---|---|---|---|
| **LucVanTien1883** | **v2+area** | 14.474 → 14.474 | 10.581 → **10.734** | 143 → **2** | 1.199 → 1.202 | 2.523 → 2.514 | 11.780 → **11.936** | 90,4 → 90,4 % | 70,7 → **96,3 %** | 7 → 7 |
| **KimVanKieu1884** | **v2+area** | 22.750 → 22.750 | 19.018 → **19.924** | 720 → **1** | 614 → 621 | 2.267 → 2.131 | 19.632 → **20.545** | 96,9 → 96,9 % | 62,3 → **97,0 %** | 16 → 16 |
| **Chrestomathie1872** | v1 (giữ) | 8.016 → 8.016 | 5.998 → 5.998 | 226 → 226 | 856 → 856 | 936 → 936 | 6.854 → 6.854 | 44,0 → 44,0 % | 66,7 → 66,7 % | — |
| **LucVanTien1916** | v1 (giữ) | 13.760 → 13.760 | 8.633 → 8.633 | 3.366 → 3.366 | 88 → 88 | 1.673 → 1.673 | 8.721 → 8.721 | 96,7 → 96,7 % | 21,4 → 21,4 % | 91 → 91 |
| **TruyenKieu1872** | **v2+area** | 22.499 → 22.499 | 13.434 → **19.554** | 6.670 → **135** | 48 → 51 | 2.333 → **2.759** | 13.482 → **19.605** | 95,4 → 95,4 % | 11,9 → **15,5 %** | 32 → 32 |

**Cờ chất lượng crop** (`crop_quality_flag`, toàn bộ ô có ảnh):

| sách | `ok` | `bleed` | `truncated` | `blank` |
|---|---|---|---|---|
| LucVanTien1883 | 10.152 → **12.383** | 2.223 → **2** | 8 → 1 | 5 → 2 |
| KimVanKieu1884 | 18.874 → **21.187** | 2.179 → **16** | 15 → 15 | 177 → **27** |
| TruyenKieu1872 | 16.778 → **19.857** | 3.623 → **112** | 767 → **1.195** | 1.082 → 1.086 |
| Chrestomathie1872 / LucVanTien1916 | 6.941 / 11.613 (không đổi) | 139 / 686 | 159 / 700 | 0 / 384 |

**Đọc bảng — bốn điều phải nói rõ.**

1. **`GOLD_text_only` sụp không phải mất nhãn, mà là crop hết hỏng.** Tầng này sinh ra từ cổng (c)
   `crop_bad`: ô đúng nhãn nhưng ảnh không dùng được → giao NHÃN, không giao ẢNH. `bleed` giảm 2.223 → 2 và
   3.623 → 112 nên gần hết nhóm ấy **quay lại GOLD có ảnh**: LVT +153, KVK +906, TK **+6.120**.
2. **M==N và trôi KHÔNG đổi một đơn vị** (90,4 / 96,9 / 44,0 / 96,7 / 95,4 và 7 / 16 / 91 / 32). Đúng thiết
   kế: detector chỉ đổi **hộp**, không đụng phép ghép DP chữ↔âm. Ai kỳ vọng v2 sửa được trôi là hiểu sai.
3. **Hai sách giữ v1 dựng lại BYTE-IDENTICAL** — `dataset/Chrestomathie1872/labels.csv` md5
   `3ff32013a8b9cfd0684733322b50ac18` và `dataset/LucVanTien1916/labels.csv` md5
   `0b0b2b782cd32ed4598b7f28ac685484`, **trùng khít** bản `_v8`. Đây là **phép đối chứng âm** của cả vòng:
   chứng minh việc gộp `prepared/` + dời dữ liệu + sửa 14 tệp đường dẫn **không làm lệch một byte nào**
   khi detector không đổi. Cộng với hồi quy STT (§6), đó là 3/8 bộ chứng minh phần "gộp" là vô hại.
4. **`truncated` của TK1872 tăng 767 → 1.195 (+428)**, đúng bằng phần REVIEW tăng (+426). Hộp v2 cao hơn
   (h/p 0,97 → 1,00) nên nhiều ô mép cột bị cắt ở biên. Đây là **chỗ duy nhất một cờ crop xấu đi**;
   các ô ấy **không** được giao (nằm ở REVIEW), và đổi lại 3.511 ô `bleed` được cứu.

---

## 4. Precision — chuẩn NGƯỜI (2 bộ IHR) và chuẩn dị bản (2 bộ thạch bản)

| sách | chuẩn | TRƯỚC | SAU |
|---|---|---|---|
| **TruyenKieu1872** | **nhãn NGƯỜI**, GOLD có ảnh | 98,64 % (n **12.706**) CI[98,42; 98,83] | **98,61 %** (n **18.550**) CI[98,44; 98,77] |
| TruyenKieu1872 | nhãn NGƯỜI, GOLD kể text_only | 98,61 % (n 19.075, cov 0,891) | 98,62 % (n 18.680, cov 0,873) |
| TruyenKieu1872 | bỏ PUA + dị thể | 99,99 % | 99,99 % |
| **LucVanTien1916** | **nhãn NGƯỜI**, GOLD có ảnh | 97,96 % (n 8.629) | **97,96 %** (n 8.629) — *giữ v1, không đổi* |
| LucVanTien1883 | proxy dị bản LVT1916 | 79,8 % (n 3.140) | **79,8 %** (n 3.180) |
| KimVanKieu1884 | proxy dị bản Kiều **1871** | 87,8 % (n 14.573) | **87,4 %** (n **15.254**) CI[86,9; 87,9] |
| KimVanKieu1884 | proxy dị bản Kiều **1872** (độc lập) | 76,5 % (n 14.771) | **76,2 %** (n **15.450**) |
| Chrestomathie1872 | *không có chuẩn độc lập* | — | — |

**Phân rã lỗi TK1872 trên nhãn người** (GOLD có ảnh):

| | TRƯỚC (n 12.706) | SAU (n 18.550) |
|---|---|---|
| dị thể (cùng âm, khác tự dạng) | 171 | 253 |
| gần hình (`SinoNom_Similar`) | 0 | 0 |
| GT vùng PUA | 1 | 2 |
| **khác hẳn (lỗi đọc thật)** | **1** | **2** |

### 🔴 Chỗ DUY NHẤT tôi đi ngược nghĩa đen của luật nghiệm thu

Luật đề bài: *"Nếu v2 làm xấu bất kỳ chỉ số nào ở sách nào → hoàn nguyên sách đó về v1"*. Với
**TruyenKieu1872** có **ba** con số nhích xuống:

* precision GOLD-ảnh **98,64 → 98,61 %** (−0,03 điểm),
* GOLD tổng (ảnh + text_only) **20.104 → 19.689** (−415 nhãn), coverage 0,891 → 0,873,
* `truncated` **767 → 1.195**.

Tôi **giữ v2** cho sách này, và ghi rõ ở đây để người dùng lật lại bằng một lệnh nếu không đồng ý:

1. −0,03 điểm nằm **gọn trong CI**: [98,42; 98,83] ↔ [98,44; 98,77] chồng nhau gần hoàn toàn; và mẫu SAU
   **lớn hơn 46 %** và **khó hơn** (gồm cả 5.844 ô mà bản v1 đã loại vì crop hỏng). Đây không phải "kém đi",
   mà là "đo trên tập rộng hơn".
2. Đổi lại là **+6.120 ô GOLD có ảnh** — đúng thứ luận văn giao nộp (cặp ảnh↔nhãn), trong khi thứ mất đi là
   415 nhãn **không có ảnh**.
3. Mọi chỉ số hộp đều tốt lên mạnh (§2) và "lỗi đọc thật" vẫn là **2 ô / 18.550**.

**Hoàn nguyên nếu người dùng muốn nghĩa đen của luật:**
```bash
.venv/bin/python lab/i5_detector_v2/set_book_keys.py config/pipeline_TruyenKieu1872.yaml \
    --book TruyenKieu1872 --remove detector_ckpt --set detector_resize=linear
NONINTERACTIVE=1 ./run_pipeline.sh --book TruyenKieu1872
```

Với **LucVanTien1883 / KimVanKieu1884 không có tình huống này**: proxy dị bản 1871 nhích 87,8 → 87,4 % —
dưới ngưỡng 0,5 điểm mà `apply_v2.sh` §5 đặt ra, trên n lớn hơn 681 ô, CI [86,9; 87,9] **chứa 87,8**;
proxy 1872 76,5 → 76,2 % tương tự; còn LVT1883 **không đổi** 79,8 %.

---

## 5. `prepared/` đã gộp — bảng mọi nơi sinh/đọc

Yêu cầu: **pipeline tổng thể chỉ dùng `prepared/`**.

| Nơi | TRƯỚC | SAU |
|---|---|---|
| `config/pipeline_KimVanKieu1884_b1.yaml` `paths.data_dir` | `prepared_b1` | `prepared` |
| ↑ `run.dataset_out` | `prepared_b1/KimVanKieu1884/dataset_out_b1` | `prepared/KimVanKieu1884/dataset_out` |
| `config/pipeline_LucVanTien1916.yaml`, `…_TruyenKieu1872.yaml` `paths.data_dir` | `prepared_ihr` | `prepared` |
| ↑ `run.dataset_out` | `prepared_ihr/<Book>/dataset_out` | `prepared/<Book>/dataset_out` |
| `pipeline/tools/ingest_ihr_book.py` `--out` mặc định | `prepared_ihr` | `prepared` |
| `scripts/measure/align_audit.py` `BOOKS[…]` (KVK + 2 IHR) | `prepared_b1` / `prepared_ihr` | `prepared` |
| `scripts/measure/ihr_endtoend_eval.py` `BOOKS` | `prepared_ihr/<Book>` | `prepared/<Book>` |
| `scripts/measure/measure.py` (cổng bước `ihr_endtoend`) | `prepared_ihr/<b>/dataset_out` | `prepared/<b>/dataset_out` |
| `scripts/measure/loss_ledger.py` (3 chỗ), `kim_channel_probe.py`, `build_metrics.py`, `auto_precision.py` (trợ giúp) | `prepared_b1…dataset_out_b1` | `prepared…dataset_out` |
| `.gitignore` | `prepared/` + `prepared_ihr/` + `/prepared_b1*/` | chỉ `prepared/` (kèm ghi chú vòng 9) |
| `config/pipeline_KimVanKieu1884_b1_boost.yaml` | dùng `prepared_b1_boost/` | **giữ nguyên làm hồ sơ ablation**, thêm 2 dòng đầu báo thư mục đã xoá |

**Dữ liệu đã dời (dùng `mv`, không copy — 0 byte nhân đôi, 0 gọi API):**

| Từ | Tới | Kiểm |
|---|---|---|
| `prepared_b1/KimVanKieu1884/` | `prepared/KimVanKieu1884/` (thay hẳn) | `pages/` + `pages_denoised/` **byte-identical** với bản cũ; `kim_raw/` là **siêu tập** (326 tệp ⊃ 163, thêm `*_lt2.json` của `kim_lang_type: 2`); `detected/` + `transcriptions/` là bản B1' đúng đường chạy chính thức |
| `prepared_b1/…/dataset_out_b1{,_v5_lang1}` | `prepared/KimVanKieu1884/dataset_out{,_v5_lang1}` | đổi tên, bỏ hậu tố `_b1` |
| `prepared_ihr/{LucVanTien1916,TruyenKieu1872}/` | `prepared/…/` | không va tên |
| `prepared/KimVanKieu1884` (bản **không** B1', `dataset_out` rỗng 0 B) | **XOÁ** (99 MB) | đã chứng minh là tập con |
| `prepared_b1_080/`, `prepared_b1_boost/` | **XOÁ** (100 MB × 2) | ablation, tái tạo được bằng lệnh ở `docs/CHAY_KVK1884_B1_2026-09-22.md` §7 / §5 |

**Cache kim tái dùng được — không phải dựng lại:** `core.ocr.ocr_api.verify_cache_image` trên **toàn bộ**
cache sau khi dời: `ok` **105 / 163 / 65 / 99 / 161** (LVT1883 / KVK / Chresto / LVT1916 / TK), 0 `mismatch`.
`prepared/` còn **2,7 GB / 8 thư mục sách** (3 STT + 5 sách mới). `prepared/SachThanhTruyen*` **không bị
đụng** (`git status` trống).

### `box_ref_eval` mở rộng cho cả 5 sách (+59 dòng)

Trước vòng 9 script chỉ chạy 2 sách thạch bản (ánh xạ `uid → page` từ `data/<Book>/pages/*.jpg`). Thêm:

* `PREPARED_BOOKS = {"Chrestomathie1872": "prose", "LucVanTien1916": "litho", "TruyenKieu1872": "litho"}`
  → lấy thẳng `prepared/<Book>/pages/*.png` (**đúng ảnh detector thấy**), `--book all5`;
* nhánh **prose** trong `ref_cells_for_page`: 1 tầng/cột, `N = num_syllables` của cột (không có luật 6/8);
* `col_pitch` đọc từ `detected/<page>_ocr_cache.json` khi không có `measure_out/<book>/layout/`;
* 3 sách mới **khoá biến thể `prepared`** (không có ảnh quét gốc cùng hệ toạ độ để chạy `otsu`).

Giới hạn thành thật: `tiers_verified_pct` của Chrestomathie chỉ **44,1 %** (số chữ kim ≠ số âm QN ở hơn nửa
số cột — đúng nhóm `dp_ratio < 0,75` của vòng 8) nên mẫu "verified" của sách này nhỏ và thiên lệch về cột dễ.

---

## 6. Nghiệm thu / hồi quy

| Phép | Kết quả |
|---|---|
| **STT 3 trang** (`stt2/0024`, `stt4/0050`, `stt11/0100`), chạy **2 lần độc lập** | `labels.csv` md5 **`59e436d7641fa849bb6759868ac29259`** cả hai lần, **556 dòng, 42 cột**; `diff -rq` giữa 2 lần **0 tệp khác** |
| `git status -- dataset_out prepared/SachThanhTruyen*` | **trống** |
| `align_audit --book all` | FAIL **1 / 1 / 0 / 2 / 2** trên 12/12/9/12/12 bất biến — **y hệt vòng 8**, 0 vi phạm mới; vi phạm ghi 27/26/0/21/7 không đổi |
| trôi (`*_drift.csv`) | **7 / 16 / 91 / 32** — không đổi |
| `ihr_endtoend_eval --book all` | **14/14 bất biến PASS** cả hai bộ |
| `crop_source_samples` (3 bản dựng v2) | **6 ảnh, 0 FAIL** — crop vẫn cắt từ ảnh quét gốc (LVT nền 255→128, KVK 255→131, TK `contrast: none` nên 255→255 như cũ) |
| `book_layout_selftest` | **157 / 0** |
| `phase1_engine_selftest` | **288 / 0** |
| `ingest_ihr_selftest` | **81 / 81** |
| `mechanism_gates_selftest` | **113 / 0** |
| `ingest_lithograph` / `ingest_prose` / `pitch_decode` / `verses_ref_fix` | 82/82 · 33 · 22/0 · 19/19 |
| `align_audit --selftest` / `ihr_endtoend_eval --selftest` | OK · 24/24 |
| `code_facts.py --check` | **18/18** |

**Không cần worktree HEAD riêng cho hồi quy STT**: md5 `59e436d7…` là bất biến đã công bố ở vòng 8
(`docs/VONG8_CAN_CHINH_VA_CROP_2026-09-23.md` §5); hai lần chạy trong worktree hiện tại đều trùng nó, và
`config/pipeline.yaml` **không có** khoá `detector_ckpt`/`detector_resize` nên STT không thể chạm vào v2.

---

## 7. Việc còn lại / giới hạn của chính các số trên

1. **LucVanTien1916 còn bỏ ngỏ.** v2 tốt hơn ở 4/5 chỉ số nhưng sinh 390 hộp ngoài tầng. Cách kiểm rẻ nhất
   (1 lần chạy, 0 API): dựng `--suffix _v2` cho riêng sách này với `detector_ckpt`, rồi chạy
   `ihr_endtoend_eval` — **có nhãn người** nên sẽ trả lời dứt điểm là 390 hộp kia có làm hỏng nhãn không.
   Nếu precision giữ ≥ 97,96 % và ảnh export tăng như TK1872 thì khai v2, ngược lại đóng hồ sơ.
2. **Chrestomathie1872 gần như chắc chắn cần train riêng** — văn xuôi, không có luật 6/8, không nằm trong
   bundle; v2 tụt ở cả 4 chỉ số đếm. Không nên ép.
3. **`last.pt` là epoch 6, KHÔNG phải `best_litho.pt`.** Bảng Kaggle cho thấy epoch 3 tốt hơn ở
   `litho_tiers_eq_015` (98,7 vs 98,0). Chạy lại trainer với sửa vòng 7 (`--guard-stt-f1 none`) sẽ ghi
   `best_litho.pt`; đáng thử **sau** khi vòng 9 được chốt, không phải trước.
4. **Mọi số hộp vẫn là proxy** (ô tham chiếu = hộp tầng kim cắt theo chiếu mực, không có GT người). Bằng
   chứng cứng duy nhất là 2 dòng nhãn người ở §4.
5. **`truncated` TK1872 +428** chưa truy đến gốc (đoán: hộp cao hơn chạm biên cột). Các ô ấy đang nằm ở
   REVIEW nên không vào bộ giao nộp, nhưng nếu muốn cứu thì nới `det_xmargin` cho riêng sách này là hướng đầu.
6. **`prepared/KimVanKieu1884` nay là bản B1'.** `box_ref_eval` của KVK vì thế đọc `detected/` +
   `transcriptions/` của B1' (trước vòng 9 nó đọc bản không-B1'). Đây là **sửa đúng** — cuối cùng thì số đo
   và bản dựng cùng nhìn một dữ liệu — nhưng nó khiến các con số KVK cũ trong `measure_out/box_ref*` (vòng
   ≤ 8) **không so trực tiếp được** với bảng §2. Mọi dòng trong §2 đều đo lại trong cùng một phiên.
