# BÁO CÁO TỔNG HỢP 10 BỘ — 2026-09-28

> Tệp SINH TỰ ĐỘNG bởi `scripts/bao_cao_tong_hop.py` lúc 2026-09-28 23:52:46 (git `c5e5170624`),
> chỉ ĐỌC đầu ra đang có trên đĩa (0 API, 0 token). Ô ghi **chưa có** = tệp nguồn thiếu, KHÔNG suy ra số.
> Mức chắc của số độ chính xác: **ĐO** (so nhãn người từng chữ) · **ƯỚC LƯỢNG** (so dị bản do người số hoá — khác chữ hợp lệ bị tính sai) · **SUY ĐOÁN** (không có sự thật người). Đừng sửa tay — chạy lại script.

## 1. Bảng 10 bộ

Số ô = mọi tầng của bản dựng (`labels_final` STT / `labels_gated` sách). "xuất" = dòng `dataset/<bộ>/labels.csv`. GX = trạng thái GOLD chính xác (`dataset/_ALL/gold_exact.csv`, policy 2026-09-28.3, config 17af2cd476a04d07); ok+text_only+uncertified+review = GOLD ảnh của bộ gộp.

| bộ | vai trò | loại bản | ô | GOLD | text_only | SYLLABLE | REVIEW | QUARANT. | xuất | GX ok | GX text_only | GX uncert. | GX review | mức chắc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| STT2 | giao nộp | viết tay Công giáo | 28.485 | 17.678 | 0 | 6.345 | 4.454 | 8 | 24.023 | 2.737 | 1.050 | 11.513 | 2.378 | SUY ĐOÁN |
| STT4 | giao nộp | viết tay Công giáo | 27.451 | 17.440 | 0 | 6.389 | 3.575 | 47 | 23.829 | 1.828 | 1.722 | 11.631 | 2.259 | SUY ĐOÁN |
| STT11 | giao nộp | viết tay Công giáo | 27.303 | 17.589 | 0 | 6.169 | 3.499 | 46 | 23.758 | 3.677 | 1.540 | 10.520 | 1.852 | SUY ĐOÁN |
| LucVanTien1883 | giao nộp | thạch bản | 14.474 | 10.910 | 2 | 1.087 | 2.453 | 22 | 11.999 | 642 | 838 | 9.276 | 154 | ƯỚC LƯỢNG |
| KimVanKieu1884 | giao nộp | thạch bản | 22.750 | 19.958 | 1 | 616 | 2.102 | 73 | 20.575 | 3.628 | 645 | 15.491 | 194 | ƯỚC LƯỢNG |
| Chrestomathie1872 | giao nộp | sách in văn xuôi | 8.016 | 4.288 | 82 | 741 | 2.905 | 0 | 5.111 | 336 | 2.484 | 1.468 | 0 | SUY ĐOÁN |
| LucVanTien1916 | đánh giá (IHR) | mộc bản | 13.760 | 11.786 | 55 | 23 | 1.896 | 0 | 11.864 | 2.812 | 575 | 8.399 | 0 | ĐO |
| TruyenKieu1872 | đánh giá (IHR) | mộc bản | 22.499 | 19.589 | 144 | 51 | 2.715 | 0 | 19.784 | 10.322 | 1.092 | 8.175 | 0 | ĐO |
| SachKinhThayCaBinh | đánh giá (Borg) | viết tay Công giáo | 90.747 | 11.862 | 474 | 18.504 | 59.905 | 2 | 30.840 | 1.495 | 6.728 | 3.639 | 0 | ĐO |
| SachDungLyHoThan | đánh giá (Borg) | viết tay Công giáo | 19.348 | 369 | 17 | 3.055 | 15.907 | 0 | 3.441 | 115 | 208 | 46 | 0 | ĐO |
| **TỔNG 10 bộ** |  |  | **274.833** | **131.469** | **775** | **42.980** | **99.411** | **198** | **175.224** | **27.592** | **16.882** | **80.158** | **6.837** |  |

Bộ gộp `dataset/_ALL`: 175.224 dòng · 174.449 tệp crop · 65.929 dòng `evaluation_only` · 8 thư mục bộ (SachThanhTruyen, LucVanTien1883, KimVanKieu1884, Chrestomathie1872, LucVanTien1916, TruyenKieu1872, SachKinhThayCaBinh, SachDungLyHoThan) · bất biến gộp 25/25 PASS.

## 2. Độ chính xác — ĐO / ƯỚC LƯỢNG / SUY ĐOÁN

| bộ | mức chắc | đại lượng | giá trị | CI 95 % | n | nguồn |
|---|---|---|---:|---:|---:|---|
| LucVanTien1916 | **ĐO** (nhãn người IHR) | GOLD ảnh: nhãn = chữ người | 98,03 % | [97,76–98,27] | 11.784 | `measure_out/LucVanTien1916/ihr_endtoend/summary.json` |
| LucVanTien1916 | **ĐO** | GOLD ảnh bỏ PUA + dị thể | 99,69 % |  | 11.784 | cùng tệp |
| LucVanTien1916 | **ĐO** (chỉ báo) | gold_exact = ok: đúng hai vế (V1+ ∧ crop chuẩn) | 99,32 % | [98,95–99,63] | 2.811 | `measure_out/gold_exact/summary.json` |
| TruyenKieu1872 | **ĐO** (nhãn người IHR) | GOLD ảnh: nhãn = chữ người | 98,62 % | [98,44–98,78] | 18.584 | `measure_out/TruyenKieu1872/ihr_endtoend/summary.json` |
| TruyenKieu1872 | **ĐO** | GOLD ảnh bỏ PUA + dị thể | 99,99 % |  | 18.584 | cùng tệp |
| TruyenKieu1872 | **ĐO** (chỉ báo) | gold_exact = ok: đúng hai vế (V1+ ∧ crop chuẩn) | 99,89 % | [99,82–99,95] | 9.747 | `measure_out/gold_exact/summary.json` |
| SachKinhThayCaBinh | **ĐO** (nhãn người Borg) | GOLD: nhãn = chữ người, V1+ | 91,30 % | [90,64–91,94] | 10.490 | `measure_out/SachKinhThayCaBinh/borg_endtoend/summary.json` |
| SachKinhThayCaBinh | **ĐO** | GOLD strict (trùng hẳn) | 85,70 % | [84,89–86,45] | 10.490 | cùng tệp |
| SachKinhThayCaBinh | **ĐO** | gold_exact = ok (n ok 1.495): V1+ | 1279/1292 | [98,29–99,41] Wilson | 1.292 | cùng tệp |
| SachDungLyHoThan | **ĐO** (nhãn người Borg) | GOLD: nhãn = chữ người, V1+ | 91,81 % | [87,45–95,64] | 293 | `measure_out/SachDungLyHoThan/borg_endtoend/summary.json` |
| SachDungLyHoThan | **ĐO** | GOLD strict (trùng hẳn) | 85,32 % | [81,13–90,35] | 293 | cùng tệp |
| SachDungLyHoThan | **ĐO** | gold_exact = ok (n ok 115): V1+ | 83/83 | [95,58–100,00] Wilson | 83 | cùng tệp |
| LucVanTien1883 | **ƯỚC LƯỢNG** (dị bản người) | GOLD = chữ dị bản `LVT1916_NF` (cận dưới) | 79,8 % | [78,5–81,0] | 3.906 | `prepared/LucVanTien1883/dataset_out/auto_precision_verify/SUMMARY.json` |
| LucVanTien1883 | **ƯỚC LƯỢNG** | GOLD = chữ `LVT1916_NF` hoặc dị thể cùng âm | 99,5 % |  | 3.906 | cùng tệp |
| KimVanKieu1884 | **ƯỚC LƯỢNG** (dị bản người) | GOLD = chữ dị bản `Kieu1871_LVD` (cận dưới) | 87,4 % | [86,9–87,9] | 15.254 | `prepared/KimVanKieu1884/dataset_out/auto_precision_verify/SUMMARY.json` |
| KimVanKieu1884 | **ƯỚC LƯỢNG** | GOLD = chữ `Kieu1871_LVD` hoặc dị thể cùng âm | 100,0 % |  | 15.254 | cùng tệp |
| KimVanKieu1884 | **ƯỚC LƯỢNG** (dị bản người) | GOLD = chữ dị bản `Kieu1872_DMT` (cận dưới) | 76,2 % | [75,5–76,9] | 15.450 | `prepared/KimVanKieu1884/dataset_out/auto_precision_verify/SUMMARY.json` |
| KimVanKieu1884 | **ƯỚC LƯỢNG** | GOLD = chữ `Kieu1872_DMT` hoặc dị thể cùng âm | 99,9 % |  | 15.450 | cùng tệp |
| STT2 | **SUY ĐOÁN** | tập "lai" (bộ ước lượng TN3, không có sự thật người) | 98,39 % | [96,40–99,24] |  | `borg_endtoend_eval.STT_EST` ← docs/GOLD_CHINH_XAC_2026-09-27.md §6 |
| STT4 | **SUY ĐOÁN** | tập "lai" (bộ ước lượng TN3, không có sự thật người) | 98,26 % | [96,19–99,19] |  | `borg_endtoend_eval.STT_EST` ← docs/GOLD_CHINH_XAC_2026-09-27.md §6 |
| STT11 | **SUY ĐOÁN** | tập "lai" (bộ ước lượng TN3, không có sự thật người) | 98,47 % | [96,48–99,28] |  | `borg_endtoend_eval.STT_EST` ← docs/GOLD_CHINH_XAC_2026-09-27.md §6 |
| Chrestomathie1872 | **SUY ĐOÁN** | không có nhãn người lẫn dị bản số hoá | chưa có |  |  | — |

CI: IHR = Wilson; Borg = bootstrap cụm trang (B = 2000) trừ khi ghi Wilson; dị bản = Wilson (điểm %).

## 3. Độ đúng kim (OCR chữ Nôm HCMUS) theo loại sách

| bộ | loại bản | mức chắc | đại lượng | giá trị | CI 95 % | n |
|---|---|---|---|---|---:|---:|
| LucVanTien1916 | mộc bản | **ĐO** | kim ở ô (mọi ô có GT) | 95,23 % | [94,86–95,57] | 13.743 |
| LucVanTien1916 | mộc bản | **ĐO** | kim ở ô GOLD | 98,03 % |  | 11.839 |
| TruyenKieu1872 | mộc bản | **ĐO** | kim ở ô (mọi ô có GT) | 97,46 % | [97,24–97,66] | 21.369 |
| TruyenKieu1872 | mộc bản | **ĐO** | kim ở ô GOLD | 98,61 % |  | 18.724 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim ở ô strict | 50,51 % | [49,69–51,32] | 75.247 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim ở ô V1+ | 53,64 % | [52,75–54,49] | 75.247 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim cả trang (LCS): precision / recall strict | 49,21 % / 37,66 % | [48,30–50,07] | 91.388 |
| SachDungLyHoThan | viết tay | **ĐO** | kim ở ô strict | 38,03 % | [36,53–39,54] | 13.522 |
| SachDungLyHoThan | viết tay | **ĐO** | kim ở ô V1+ | 40,08 % | [38,56–41,66] | 13.522 |
| SachDungLyHoThan | viết tay | **ĐO** | kim cả trang (LCS): precision / recall strict | 32,70 % / 26,41 % | [31,26–34,23] | 19.521 |
| LucVanTien1883 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `LVT1916_NF` (mọi tầng) | 70,5 % |  | 4.499 |
| KimVanKieu1884 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `Kieu1871_LVD` (mọi tầng) | 84,5 % |  | 16.032 |
| KimVanKieu1884 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `Kieu1872_DMT` (mọi tầng) | 73,8 % |  | 16.216 |
| STT2/4/11, Chrestomathie1872 | viết tay / sách in | — | không có nhãn người -> không đo | chưa có |  |  |

**Cờ CNT của gold_exact trên chữ viết tay (ĐO, Borg).** GOLD có chữ người, tỉ lệ đúng V1+ tách theo cờ `cnt` (`measure_out/<Borg>/borg_endtoend/cells.csv` ⋈ `dataset/_ALL/gold_exact.csv`, Wilson):

| bộ | nhóm | n | đúng V1+ | CI 95 % |
|---|---:|---:|---:|---:|
| SachKinhThayCaBinh | CNT = 1 (số chữ OCR cột ≠ số âm QN) | 10.114 | 91,18 % | [90,61–91,72] |
| SachKinhThayCaBinh | CNT = 0 | 376 | 94,41 % | [91,61–96,32] |
| SachDungLyHoThan | CNT = 1 (số chữ OCR cột ≠ số âm QN) | 288 | 91,67 % | [87,90–94,34] |
| SachDungLyHoThan | CNT = 0 | 5 | 100,00 % | [56,55–100,00] |

Gần như mọi ô GOLD Borg mang CNT = 1 (cột viết tay hiếm khi đếm bằng), nên cờ này chủ yếu dồn ô sang `text_only` chứ không tách được ô sai khỏi ô đúng; so hai nhóm ở bảng trên (nhóm CNT = 0 rất nhỏ).

## 4. Bất biến (invariants)

`measure_out/SUMMARY.json` (2026-09-28T23:52:39, `measure.py --all`): **245 PASS · 0 FAIL · 2 FAIL mềm · 0 SKIP** trên 17 bước (sập 0).

| bước | bộ | mã | PASS | FAIL | mềm | SKIP | giây |
|---|---|---:|---:|---:|---:|---:|---:|
| code_facts | (repo) | 0 | 18 | 0 | 0 | 0 | 4 |
| layout | LucVanTien1883 | 0 | 8 | 0 | 0 | 0 | 46 |
| qn_ocr | LucVanTien1883 | 0 | 20 | 0 | 0 | 0 | 1 |
| layout | KimVanKieu1884 | 0 | 10 | 0 | 0 | 0 | 84 |
| qn_ocr | KimVanKieu1884 | 0 | 24 | 0 | 0 | 0 | 2 |
| chresto_map | Chrestomathie1872 | 0 | 18 | 0 | 0 | 0 | 12 |
| ihr_layout | LucVanTien1916 | 1 | 6 | 0 | 2 | 0 | 2 |
| ihr_endtoend | LucVanTien1916 | 0 | 7 | 0 | 0 | 0 | 1 |
| ihr_layout | TruyenKieu1872 | 0 | 8 | 0 | 0 | 0 | 1 |
| ihr_endtoend | TruyenKieu1872 | 0 | 7 | 0 | 0 | 0 | 1 |
| borg_endtoend | SachKinhThayCaBinh | 0 | 7 | 0 | 0 | 0 | 16 |
| borg_endtoend | SachDungLyHoThan | 0 | 7 | 0 | 0 | 0 | 6 |
| detector_transfer | LVT+KVK+STT | 0 | 18 | 0 | 0 | 0 | 115 |
| box_ref | LVT+KVK | 0 | 24 | 0 | 0 | 0 | 80 |
| gold_exact | (_ALL) | 0 | 29 | 0 | 0 | 0 | 20 |
| borg_human | Borg18+34 | 0 | 20 | 0 | 0 | 0 | 17 |
| stt_lt2 | STT | 0 | 14 | 0 | 0 | 0 | 12 |

| phép đo | PASS | FAIL |
|---|---:|---:|
| gold_exact_eval (bản giao dataset/_ALL/gold_exact.csv) | 29/29 | 0 |
| ihr_endtoend LucVanTien1916 | 7/7 | 0 |
| ihr_endtoend TruyenKieu1872 | 7/7 | 0 |
| borg_endtoend SachKinhThayCaBinh | 7/7 | 0 |
| borg_endtoend SachDungLyHoThan | 7/7 | 0 |

## 5. Thời gian chạy

Lượt `clean_rebuild_all.sh --run` (`logs/clean_rebuild_20260928_221152_thoi_gian.tsv`):

| bước | lệnh | mã | thời gian | bắt đầu | kết thúc | log |
|---|---|---:|---:|---|---|---|
| 1 | run_pipeline_all | 0 | 57 ph 50 s | 2026-09-28T22:16:30 | 2026-09-28T23:14:20 | `logs/clean_rebuild_20260928_221152_1_run_pipeline_all.log` |
| 2 | borg_human | 0 | 11 ph 7 s | 2026-09-28T23:14:20 | 2026-09-28T23:25:27 | `logs/clean_rebuild_20260928_221152_2_borg_human.log` |
| 3 | measure_all | 1 | 6 ph 53 s | 2026-09-28T23:25:27 | 2026-09-28T23:32:20 | `logs/clean_rebuild_20260928_221152_3_measure_all.log` |
| 4 | nghiem_thu | 1 | 0 ph 57 s | 2026-09-28T23:32:20 | 2026-09-28T23:33:17 | `logs/clean_rebuild_20260928_221152_4_nghiem_thu.log` |
| 5 | bao_cao_tong_hop | 0 | 0 ph 2 s | 2026-09-28T23:33:17 | 2026-09-28T23:33:19 | `logs/clean_rebuild_20260928_221152_5_bao_cao_tong_hop.log` |

| bộ / bước | bắt đầu | các bước (giây) | tổng | ghi chú |
|---|---|---|---|---|
| STT (3 quyển chung) | 2026-09-28T22:16:33 | setup 0s · extract 6s · build 430s · remediate 3s · rescue 7s · export 79s | 8 ph 45 s |  |
| LucVanTien1883 | 2026-09-28T22:25:18 | setup 2s · ingest 3s · build 192s · remediate 2s · gates 6s · export 5s · measure 2s | 3 ph 32 s |  |
| KimVanKieu1884 | 2026-09-28T22:28:50 | setup 3s · ingest 5s · build 229s · remediate 2s · gates 17s · export 8s · measure 5s | 4 ph 29 s |  |
| Chrestomathie1872 | 2026-09-28T22:33:19 | setup 0s · ingest 56s · build 134s · remediate 1s · gates 1s · export 2s · measure 0s | 3 ph 14 s |  |
| LucVanTien1916 | 2026-09-28T22:36:34 | setup 0s · ingest 1s · build 166s · remediate 2s · gates 1s · export 4s · measure 0s | 2 ph 54 s |  |
| TruyenKieu1872 | 2026-09-28T22:39:28 | setup 1s · ingest 1s · build 207s · remediate 2s · gates 1s · export 8s · measure 0s | 3 ph 40 s |  |
| SachKinhThayCaBinh | 2026-09-28T22:43:08 | setup 0s · ingest 7s · build 875s · remediate 4s · gates 1s · export 23s · measure 17s | 15 ph 27 s |  |
| SachDungLyHoThan | 2026-09-28T22:58:35 | setup 0s · ingest 2s · build 223s · remediate 2s · gates 1s · export 2s · measure 5s | 3 ph 55 s |  |
| gộp dataset/_ALL | 2026-09-28T23:03:51 | merge 2838s | 47 ph 18 s |  |
| GOLD chính xác (B8) | 20260928_230351 | mã thoát 0 · 622s | 10 ph 22 s |  |

## 6. Giới hạn

1. Độ chính xác chỉ **ĐO** được ở 2 bộ mộc bản IHR (LucVanTien1916, TruyenKieu1872) và 2 bộ chép tay Borg (SachKinhThayCaBinh, SachDungLyHoThan) — đều là TẬP ĐÁNH GIÁ, không giao nộp. 3 bộ STT và Chrestomathie1872 (giao nộp) chỉ có **SUY ĐOÁN**; LucVanTien1883/KimVanKieu1884 chỉ có **ƯỚC LƯỢNG** qua dị bản (cận dưới: dị bản khác chữ hợp lệ bị tính sai).
2. Borg: QN đầu vào là phiên âm NGƯỜI theo trang ⇒ số đo là **cận trên** của phương pháp trên chữ viết tay; chỉ câu "đếm bằng" được chấm. Kim trên chữ viết tay kém hẳn mộc bản (mục 3) — GOLD vẫn cao nhờ luật `kim ∈ R(âm)` lọc.
3. Số ô `gold_exact = ok` phụ thuộc ngưỡng bộ kiểm ảnh↔chữ (`config/gold_exact.yaml`); ở Borg rất ít ô ok có chữ người (xem mục 2, n nhỏ ⇒ CI rộng).
4. Bảng thời gian ghép từ nhiều lượt khi chưa có lượt `clean_rebuild_all.sh --run` trọn vẹn.

