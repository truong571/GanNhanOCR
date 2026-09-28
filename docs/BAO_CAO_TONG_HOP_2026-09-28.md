# BÁO CÁO TỔNG HỢP 10 BỘ — 2026-09-28

> Tệp SINH TỰ ĐỘNG bởi `scripts/bao_cao_tong_hop.py` lúc 2026-09-28 17:12:54 (git `3a3ac43d64`),
> chỉ ĐỌC đầu ra đang có trên đĩa (0 API, 0 token). Ô ghi **chưa có** = tệp nguồn thiếu, KHÔNG suy ra số.
> Mức chắc của số độ chính xác: **ĐO** (so nhãn người từng chữ) · **ƯỚC LƯỢNG** (so dị bản do người số hoá — khác chữ hợp lệ bị tính sai) · **SUY ĐOÁN** (không có sự thật người). Đừng sửa tay — chạy lại script.

## 1. Bảng 10 bộ

Số ô = mọi tầng của bản dựng (`labels_final` STT / `labels_gated` sách). "xuất" = dòng `dataset/<bộ>/labels.csv`. GX = trạng thái GOLD chính xác (`dataset/_ALL/gold_exact.csv`, policy 2026-09-28.2, config fda278c992131d9f); ok+text_only+uncertified+review = GOLD ảnh của bộ gộp.

| bộ | vai trò | loại bản | ô | GOLD | text_only | SYLLABLE | REVIEW | QUARANT. | xuất | GX ok | GX text_only | GX uncert. | GX review | mức chắc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| STT2 | giao nộp | viết tay Công giáo | 28.485 | 17.678 | 0 | 6.345 | 4.454 | 8 | 24.023 | 2.864 | 1.050 | 11.386 | 2.378 | SUY ĐOÁN |
| STT4 | giao nộp | viết tay Công giáo | 27.451 | 17.440 | 0 | 6.389 | 3.575 | 47 | 23.829 | 1.872 | 1.722 | 11.587 | 2.259 | SUY ĐOÁN |
| STT11 | giao nộp | viết tay Công giáo | 27.303 | 17.589 | 0 | 6.169 | 3.499 | 46 | 23.758 | 3.750 | 1.540 | 10.447 | 1.852 | SUY ĐOÁN |
| LucVanTien1883 | giao nộp | thạch bản | 14.474 | 10.910 | 2 | 1.087 | 2.453 | 22 | 11.999 | 642 | 838 | 9.276 | 154 | ƯỚC LƯỢNG |
| KimVanKieu1884 | giao nộp | thạch bản | 22.750 | 19.958 | 1 | 616 | 2.102 | 73 | 20.575 | 3.628 | 645 | 15.491 | 194 | ƯỚC LƯỢNG |
| Chrestomathie1872 | giao nộp | sách in văn xuôi | 8.016 | 4.238 | 153 | 745 | 2.880 | 0 | 5.136 | 325 | 2.425 | 1.488 | 0 | SUY ĐOÁN |
| LucVanTien1916 | đánh giá (IHR) | mộc bản | 13.760 | 11.587 | 195 | 25 | 1.953 | 0 | 11.807 | 2.758 | 639 | 8.190 | 0 | ĐO |
| TruyenKieu1872 | đánh giá (IHR) | mộc bản | 22.499 | 19.554 | 135 | 51 | 2.759 | 0 | 19.740 | 10.322 | 1.114 | 8.118 | 0 | ĐO |
| SachKinhThayCaBinh | đánh giá (Borg) | viết tay Công giáo | 90.747 | 11.587 | 325 | 17.786 | 58.885 | 2.164 | 29.698 | 1.120 | 6.988 | 3.479 | 0 | ĐO |
| SachDungLyHoThan | đánh giá (Borg) | viết tay Công giáo | 19.348 | 354 | 30 | 3.024 | 15.703 | 237 | 3.408 | 88 | 209 | 57 | 0 | ĐO |
| **TỔNG 10 bộ** |  |  | **274.833** | **130.895** | **841** | **42.237** | **98.263** | **2.597** | **173.973** | **27.369** | **17.170** | **79.519** | **6.837** |  |

Bộ gộp `dataset/_ALL`: 173.973 dòng · 173.111 tệp crop · 64.653 dòng `evaluation_only` · 8 thư mục bộ (SachThanhTruyen, LucVanTien1883, KimVanKieu1884, Chrestomathie1872, LucVanTien1916, TruyenKieu1872, SachKinhThayCaBinh, SachDungLyHoThan) · bất biến gộp 24/24 PASS.

_Ghi chú kiểm: LucVanTien1916, SachKinhThayCaBinh có cùng số ô GOLD (11.587) — trùng số tình cờ, hai bản dựng khác nhau (tổng ô và tầng khác xem bảng)._

## 2. Độ chính xác — ĐO / ƯỚC LƯỢNG / SUY ĐOÁN

| bộ | mức chắc | đại lượng | giá trị | CI 95 % | n | nguồn |
|---|---|---|---:|---:|---:|---|
| LucVanTien1916 | **ĐO** (nhãn người IHR) | GOLD ảnh: nhãn = chữ người | 98,05 % | [97,78–98,29] | 11.585 | `measure_out/LucVanTien1916/ihr_endtoend/summary.json` |
| LucVanTien1916 | **ĐO** | GOLD ảnh bỏ PUA + dị thể | 99,68 % |  | 11.585 | cùng tệp |
| LucVanTien1916 | **ĐO** (chỉ báo) | gold_exact = ok: đúng hai vế (V1+ ∧ crop chuẩn) | 99,46 % | [99,09–99,74] | 2.757 | `measure_out/gold_exact/summary.json` |
| TruyenKieu1872 | **ĐO** (nhãn người IHR) | GOLD ảnh: nhãn = chữ người | 98,61 % | [98,44–98,77] | 18.550 | `measure_out/TruyenKieu1872/ihr_endtoend/summary.json` |
| TruyenKieu1872 | **ĐO** | GOLD ảnh bỏ PUA + dị thể | 99,99 % |  | 18.550 | cùng tệp |
| TruyenKieu1872 | **ĐO** (chỉ báo) | gold_exact = ok: đúng hai vế (V1+ ∧ crop chuẩn) | 99,87 % | [99,80–99,93] | 9.746 | `measure_out/gold_exact/summary.json` |
| SachKinhThayCaBinh | **ĐO** (nhãn người Borg) | GOLD: nhãn = chữ người, V1+ | 91,20 % | [90,46–91,82] | 10.294 | `measure_out/SachKinhThayCaBinh/borg_endtoend/summary.json` |
| SachKinhThayCaBinh | **ĐO** | GOLD strict (trùng hẳn) | 85,69 % | [84,86–86,49] | 10.294 | cùng tệp |
| SachKinhThayCaBinh | **ĐO** | gold_exact = ok (n ok 1.120): V1+ | 962/970 | [98,38–99,58] Wilson | 970 | cùng tệp |
| SachDungLyHoThan | **ĐO** (nhãn người Borg) | GOLD: nhãn = chữ người, V1+ | 91,73 % | [88,49–95,09] | 278 | `measure_out/SachDungLyHoThan/borg_endtoend/summary.json` |
| SachDungLyHoThan | **ĐO** | GOLD strict (trùng hẳn) | 84,89 % | [81,02–89,72] | 278 | cùng tệp |
| SachDungLyHoThan | **ĐO** | gold_exact = ok (n ok 88): V1+ | 65/65 | [94,42–100,00] Wilson | 65 | cùng tệp |
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
| LucVanTien1916 | mộc bản | **ĐO** | kim ở ô GOLD | 98,02 % |  | 11.780 |
| TruyenKieu1872 | mộc bản | **ĐO** | kim ở ô (mọi ô có GT) | 97,46 % | [97,24–97,66] | 21.369 |
| TruyenKieu1872 | mộc bản | **ĐO** | kim ở ô GOLD | 98,62 % |  | 18.680 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim ở ô strict | 50,26 % | [49,41–51,07] | 73.415 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim ở ô V1+ | 53,37 % | [52,49–54,23] | 73.415 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim cả trang (LCS): precision / recall strict | 49,21 % / 37,66 % | [48,30–50,07] | 91.388 |
| SachDungLyHoThan | viết tay | **ĐO** | kim ở ô strict | 37,71 % | [36,25–39,22] | 13.274 |
| SachDungLyHoThan | viết tay | **ĐO** | kim ở ô V1+ | 39,75 % | [38,22–41,30] | 13.274 |
| SachDungLyHoThan | viết tay | **ĐO** | kim cả trang (LCS): precision / recall strict | 32,70 % / 26,41 % | [31,26–34,23] | 19.521 |
| LucVanTien1883 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `LVT1916_NF` (mọi tầng) | 70,5 % |  | 4.499 |
| KimVanKieu1884 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `Kieu1871_LVD` (mọi tầng) | 84,5 % |  | 16.032 |
| KimVanKieu1884 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `Kieu1872_DMT` (mọi tầng) | 73,8 % |  | 16.216 |
| STT2/4/11, Chrestomathie1872 | viết tay / sách in | — | không có nhãn người -> không đo | chưa có |  |  |

**Cờ CNT của gold_exact trên chữ viết tay (ĐO, Borg).** GOLD có chữ người, tỉ lệ đúng V1+ tách theo cờ `cnt` (`measure_out/<Borg>/borg_endtoend/cells.csv` ⋈ `dataset/_ALL/gold_exact.csv`, Wilson):

| bộ | nhóm | n | đúng V1+ | CI 95 % |
|---|---:|---:|---:|---:|
| SachKinhThayCaBinh | CNT = 1 (số chữ OCR cột ≠ số âm QN) | 9.942 | 91,08 % | [90,50–91,62] |
| SachKinhThayCaBinh | CNT = 0 | 352 | 94,60 % | [91,72–96,52] |
| SachDungLyHoThan | CNT = 1 (số chữ OCR cột ≠ số âm QN) | 272 | 91,54 % | [87,63–94,30] |
| SachDungLyHoThan | CNT = 0 | 6 | 100,00 % | [60,97–100,00] |

Gần như mọi ô GOLD Borg mang CNT = 1 (cột viết tay hiếm khi đếm bằng), nên cờ này chủ yếu dồn ô sang `text_only` chứ không tách được ô sai khỏi ô đúng; so hai nhóm ở bảng trên (nhóm CNT = 0 rất nhỏ).

## 4. Bất biến (invariants)

`measure_out/SUMMARY.json` (2026-09-28T17:12:50, `measure.py --all --report-only`): **230 PASS · 0 FAIL · 2 FAIL mềm · 0 SKIP** trên 16 bước (sập 0).

| bước | bộ | mã | PASS | FAIL | mềm | SKIP | giây |
|---|---|---:|---:|---:|---:|---:|---:|
| code_facts | (repo) | 0 | 18 | 0 | 0 | 0 | 4 |
| layout | LucVanTien1883 | 0 | 8 | 0 | 0 | 0 | 56 |
| qn_ocr | LucVanTien1883 | 0 | 20 | 0 | 0 | 0 | 1 |
| layout | KimVanKieu1884 | 0 | 10 | 0 | 0 | 0 | 83 |
| qn_ocr | KimVanKieu1884 | 0 | 24 | 0 | 0 | 0 | 2 |
| chresto_map | Chrestomathie1872 | 0 | 18 | 0 | 0 | 0 | 12 |
| ihr_layout | LucVanTien1916 | 1 | 6 | 0 | 2 | 0 | 2 |
| ihr_endtoend | LucVanTien1916 | 0 | 7 | 0 | 0 | 0 | 1 |
| ihr_layout | TruyenKieu1872 | 0 | 8 | 0 | 0 | 0 | 1 |
| ihr_endtoend | TruyenKieu1872 | 0 | 7 | 0 | 0 | 0 | 1 |
| borg_endtoend | SachKinhThayCaBinh | 0 | 7 | 0 | 0 | 0 | 16 |
| borg_endtoend | SachDungLyHoThan | 0 | 7 | 0 | 0 | 0 | 6 |
| detector_transfer | LVT+KVK+STT | 0 | 18 | 0 | 0 | 0 | 116 |
| box_ref | LVT+KVK | 0 | 24 | 0 | 0 | 0 | 79 |
| gold_exact | (_ALL) | 0 | 28 | 0 | 0 | 0 | 20 |
| borg_human | Borg18+34 | 0 | 20 | 0 | 0 | 0 | 16 |

| phép đo | PASS | FAIL |
|---|---:|---:|
| gold_exact_eval (bản giao dataset/_ALL/gold_exact.csv) | 28/28 | 0 |
| ihr_endtoend LucVanTien1916 | 7/7 | 0 |
| ihr_endtoend TruyenKieu1872 | 7/7 | 0 |
| borg_endtoend SachKinhThayCaBinh | 7/7 | 0 |
| borg_endtoend SachDungLyHoThan | 7/7 | 0 |

## 5. Thời gian chạy

Lượt `clean_rebuild_all.sh --run` (`logs/clean_rebuild_20260928_155231_thoi_gian.tsv`):

| bước | lệnh | mã | thời gian | bắt đầu | kết thúc | log |
|---|---|---:|---:|---|---|---|
| 1 | run_pipeline_all | 1 | 48 ph 1 s | 2026-09-28T15:54:55 | 2026-09-28T16:42:56 | `logs/clean_rebuild_20260928_155231_1_run_pipeline_all.log` |
| 2 | borg_human | 0 | 11 ph 48 s | 2026-09-28T16:42:56 | 2026-09-28T16:54:44 | `logs/clean_rebuild_20260928_155231_2_borg_human.log` |
| 3 | measure_all | 2 | 7 ph 23 s | 2026-09-28T16:54:44 | 2026-09-28T17:02:07 | `logs/clean_rebuild_20260928_155231_3_measure_all.log` |
| 4 | bao_cao_tong_hop | 0 | 0 ph 2 s | 2026-09-28T17:02:07 | 2026-09-28T17:02:09 | `logs/clean_rebuild_20260928_155231_4_bao_cao_tong_hop.log` |

| bộ / bước | bắt đầu | các bước (giây) | tổng | ghi chú |
|---|---|---|---|---|
| STT (3 quyển chung) | 2026-09-28T15:54:57 | setup 0s · extract 7s · build 446s · remediate 3s · rescue 8s · export 79s | 9 ph 3 s |  |
| LucVanTien1883 | 2026-09-28T16:04:01 | setup 2s · ingest 3s · build 197s · remediate 3s · gates 7s · export 5s · measure 2s | 3 ph 39 s |  |
| KimVanKieu1884 | 2026-09-28T16:07:40 | setup 3s · ingest 5s · build 243s · remediate 3s · gates 18s · export 9s · measure 6s | 4 ph 47 s |  |
| Chrestomathie1872 | 2026-09-28T16:12:27 | setup 0s · ingest 56s · build 95s · remediate 1s · gates 1s · export 2s · measure 0s | 2 ph 35 s |  |
| LucVanTien1916 | 2026-09-28T16:15:02 | setup 0s · ingest 2s · build 98s · remediate 2s · gates 0s · export 5s · measure 0s | 1 ph 47 s |  |
| TruyenKieu1872 | 2026-09-28T16:16:49 | setup 0s · ingest 2s · build 124s · remediate 2s · gates 1s · export 7s · measure 0s | 2 ph 16 s |  |
| SachKinhThayCaBinh | 2026-09-28T16:19:05 | setup 1s · ingest 6s · build 732s · remediate 5s · gates 1s · export 28s · measure 16s | 13 ph 9 s |  |
| SachDungLyHoThan | 2026-09-28T16:32:15 | setup 0s · ingest 2s · build 163s · remediate 2s · gates 1s · export 2s · measure 5s | 2 ph 55 s |  |
| gộp dataset/_ALL | 2026-09-28T16:36:35 | merge 2498s | 41 ph 38 s |  |
| GOLD chính xác (B8) | 20260928_163635 | mã thoát 0 · 315s | 5 ph 15 s |  |

## 6. Giới hạn

1. Độ chính xác chỉ **ĐO** được ở 2 bộ mộc bản IHR (LucVanTien1916, TruyenKieu1872) và 2 bộ chép tay Borg (SachKinhThayCaBinh, SachDungLyHoThan) — đều là TẬP ĐÁNH GIÁ, không giao nộp. 3 bộ STT và Chrestomathie1872 (giao nộp) chỉ có **SUY ĐOÁN**; LucVanTien1883/KimVanKieu1884 chỉ có **ƯỚC LƯỢNG** qua dị bản (cận dưới: dị bản khác chữ hợp lệ bị tính sai).
2. Borg: QN đầu vào là phiên âm NGƯỜI theo trang ⇒ số đo là **cận trên** của phương pháp trên chữ viết tay; chỉ câu "đếm bằng" được chấm. Kim trên chữ viết tay kém hẳn mộc bản (mục 3) — GOLD vẫn cao nhờ luật `kim ∈ R(âm)` lọc.
3. Số ô `gold_exact = ok` phụ thuộc ngưỡng bộ kiểm ảnh↔chữ (`config/gold_exact.yaml`); ở Borg rất ít ô ok có chữ người (xem mục 2, n nhỏ ⇒ CI rộng).
4. Bảng thời gian ghép từ nhiều lượt khi chưa có lượt `clean_rebuild_all.sh --run` trọn vẹn.

