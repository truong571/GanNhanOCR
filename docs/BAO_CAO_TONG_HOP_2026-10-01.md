# BÁO CÁO TỔNG HỢP 10 BỘ — 2026-10-01

> Tệp SINH TỰ ĐỘNG bởi `scripts/bao_cao_tong_hop.py` lúc 2026-10-01 20:48:39 (git `782d6dc621`),
> chỉ ĐỌC đầu ra đang có trên đĩa (0 API, 0 token). Ô ghi **chưa có** = tệp nguồn thiếu, KHÔNG suy ra số.
> Mức chắc của số độ chính xác: **ĐO** (so nhãn người từng chữ) · **ƯỚC LƯỢNG** (so dị bản do người số hoá — khác chữ hợp lệ bị tính sai) · **SUY ĐOÁN** (không có sự thật người). Đừng sửa tay — chạy lại script.

## 1. Bảng 10 bộ

Số ô = mọi tầng của bản dựng (`labels_final` STT / `labels_gated` sách). "xuất" = dòng `dataset/<bộ>/labels.csv`. GX = trạng thái GOLD chính xác (`dataset/_ALL/gold_exact.csv`, policy 2026-09-28.3, config 17af2cd476a04d07); ok+text_only+uncertified+review = GOLD ảnh của bộ gộp.

| bộ | vai trò | loại bản | ô | GOLD | text_only | SYLLABLE | REVIEW | QUARANT. | xuất | GX ok | GX text_only | GX uncert. | GX review | mức chắc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| STT2 | giao nộp | viết tay Công giáo | 28.589 | 17.126 | 0 | 3.673 | 7.790 | 0 | 20.799 | 3.147 | 1.479 | 12.164 | 336 | SUY ĐOÁN |
| STT4 | giao nộp | viết tay Công giáo | 27.526 | 16.586 | 0 | 3.627 | 7.305 | 8 | 20.213 | 2.057 | 2.448 | 11.863 | 218 | SUY ĐOÁN |
| STT11 | giao nộp | viết tay Công giáo | 27.427 | 18.179 | 0 | 2.917 | 6.315 | 16 | 21.096 | 4.156 | 2.187 | 11.402 | 434 | SUY ĐOÁN |
| LucVanTien1883 | giao nộp | thạch bản | 14.474 | 10.911 | 2 | 1.087 | 2.452 | 22 | 12.000 | 642 | 838 | 9.277 | 154 | ƯỚC LƯỢNG |
| KimVanKieu1884 | giao nộp | thạch bản | 22.750 | 20.584 | 1 | 617 | 1.475 | 73 | 21.202 | 3.629 | 980 | 15.775 | 200 | ƯỚC LƯỢNG |
| Chrestomathie1872 | giao nộp | sách in văn xuôi | 8.016 | 5.946 | 98 | 742 | 1.230 | 0 | 6.786 | 455 | 3.473 | 2.018 | 0 | SUY ĐOÁN |
| LucVanTien1916 | đánh giá (IHR) | mộc bản | 13.760 | 13.126 | 59 | 24 | 551 | 0 | 13.209 | 3.058 | 697 | 9.371 | 0 | ĐO |
| TruyenKieu1872 | đánh giá (IHR) | mộc bản | 22.499 | 21.951 | 154 | 52 | 342 | 0 | 22.157 | 11.443 | 1.271 | 9.237 | 0 | ĐO |
| SachKinhThayCaBinh | đánh giá (Borg) | viết tay Công giáo | 90.747 | 81.972 | 543 | 1.418 | 6.812 | 2 | 83.933 | 7.579 | 52.785 | 21.608 | 0 | ĐO |
| SachDungLyHoThan | đánh giá (Borg) | viết tay Công giáo | 19.348 | 13.293 | 22 | 302 | 5.731 | 0 | 13.617 | 2.432 | 9.284 | 1.577 | 0 | ĐO |
| **TỔNG 10 bộ** |  |  | **275.136** | **219.674** | **879** | **14.459** | **40.003** | **121** | **235.012** | **38.598** | **75.442** | **104.292** | **1.342** |  |

Bộ gộp `dataset/_ALL`: 235.012 dòng · 234.133 tệp crop · 132.916 dòng `evaluation_only` · 8 thư mục bộ (SachThanhTruyen, LucVanTien1883, KimVanKieu1884, Chrestomathie1872, LucVanTien1916, TruyenKieu1872, SachKinhThayCaBinh, SachDungLyHoThan) · bất biến gộp 25/25 PASS.

## 2. Độ chính xác — ĐO / ƯỚC LƯỢNG / SUY ĐOÁN

| bộ | mức chắc | đại lượng | giá trị | CI 95 % | n | nguồn |
|---|---|---|---:|---:|---:|---|
| LucVanTien1916 | **ĐO** (nhãn người IHR) | GOLD ảnh: nhãn = chữ người | 97,95 % | [97,69–98,18] | 13.117 | `measure_out/LucVanTien1916/ihr_endtoend/summary.json` |
| LucVanTien1916 | **ĐO** | GOLD ảnh bỏ PUA + dị thể | 99,68 % |  | 13.117 | cùng tệp |
| LucVanTien1916 | **ĐO** (chỉ báo) | gold_exact = ok: đúng hai vế (V1+ ∧ crop chuẩn) | 99,21 % | [98,80–99,54] | 3.057 | `measure_out/gold_exact/summary.json` |
| TruyenKieu1872 | **ĐO** (nhãn người IHR) | GOLD ảnh: nhãn = chữ người | 98,60 % | [98,43–98,75] | 20.850 | `measure_out/TruyenKieu1872/ihr_endtoend/summary.json` |
| TruyenKieu1872 | **ĐO** | GOLD ảnh bỏ PUA + dị thể | 100,00 % |  | 20.850 | cùng tệp |
| TruyenKieu1872 | **ĐO** (chỉ báo) | gold_exact = ok: đúng hai vế (V1+ ∧ crop chuẩn) | 99,88 % | [99,81–99,94] | 10.832 | `measure_out/gold_exact/summary.json` |
| SachKinhThayCaBinh | **ĐO** (nhãn người Borg) | GOLD: nhãn = chữ người, V1+ | 94,40 % | [94,09–94,70] | 72.693 | `measure_out/SachKinhThayCaBinh/borg_endtoend/summary.json` |
| SachKinhThayCaBinh | **ĐO** | GOLD strict (trùng hẳn) | 90,96 % | [90,63–91,30] | 72.693 | cùng tệp |
| SachKinhThayCaBinh | **ĐO** | gold_exact = ok (n ok 7.579): V1+ | 6652/6765 | [98,00–98,61] Wilson | 6.765 | cùng tệp |
| SachDungLyHoThan | **ĐO** (nhãn người Borg) | GOLD: nhãn = chữ người, V1+ | 96,70 % | [96,31–97,08] | 12.345 | `measure_out/SachDungLyHoThan/borg_endtoend/summary.json` |
| SachDungLyHoThan | **ĐO** | GOLD strict (trùng hẳn) | 93,84 % | [93,32–94,34] | 12.345 | cùng tệp |
| SachDungLyHoThan | **ĐO** | gold_exact = ok (n ok 2.432): V1+ | 2168/2219 | [96,99–98,25] Wilson | 2.219 | cùng tệp |
| LucVanTien1883 | **ƯỚC LƯỢNG** (dị bản người) | GOLD = chữ dị bản `LVT1916_NF` (cận dưới) | 79,8 % | [78,5–81,0] | 3.906 | `prepared/LucVanTien1883/dataset_out/auto_precision_verify/SUMMARY.json` |
| LucVanTien1883 | **ƯỚC LƯỢNG** | GOLD = chữ `LVT1916_NF` hoặc dị thể cùng âm | 99,5 % |  | 3.906 | cùng tệp |
| KimVanKieu1884 | **ƯỚC LƯỢNG** (dị bản người) | GOLD = chữ dị bản `Kieu1871_LVD` (cận dưới) | 87,4 % | [86,9–87,9] | 15.416 | `prepared/KimVanKieu1884/dataset_out/auto_precision_verify/SUMMARY.json` |
| KimVanKieu1884 | **ƯỚC LƯỢNG** | GOLD = chữ `Kieu1871_LVD` hoặc dị thể cùng âm | 100,0 % |  | 15.416 | cùng tệp |
| KimVanKieu1884 | **ƯỚC LƯỢNG** (dị bản người) | GOLD = chữ dị bản `Kieu1872_DMT` (cận dưới) | 76,2 % | [75,6–76,9] | 15.613 | `prepared/KimVanKieu1884/dataset_out/auto_precision_verify/SUMMARY.json` |
| KimVanKieu1884 | **ƯỚC LƯỢNG** | GOLD = chữ `Kieu1872_DMT` hoặc dị thể cùng âm | 99,9 % |  | 15.613 | cùng tệp |
| STT2 | **SUY ĐOÁN** | tập "lai" (bộ ước lượng TN3, không có sự thật người) | 98,39 % | [96,40–99,24] |  | `borg_endtoend_eval.STT_EST` ← docs/GOLD_CHINH_XAC_2026-09-27.md §6 |
| STT4 | **SUY ĐOÁN** | tập "lai" (bộ ước lượng TN3, không có sự thật người) | 98,26 % | [96,19–99,19] |  | `borg_endtoend_eval.STT_EST` ← docs/GOLD_CHINH_XAC_2026-09-27.md §6 |
| STT11 | **SUY ĐOÁN** | tập "lai" (bộ ước lượng TN3, không có sự thật người) | 98,47 % | [96,48–99,28] |  | `borg_endtoend_eval.STT_EST` ← docs/GOLD_CHINH_XAC_2026-09-27.md §6 |
| Chrestomathie1872 | **SUY ĐOÁN** | không có nhãn người lẫn dị bản số hoá | chưa có |  |  | — |

CI: IHR = Wilson; Borg = bootstrap cụm trang (B = 2000) trừ khi ghi Wilson; dị bản = Wilson (điểm %).

## 3. Độ đúng kim (OCR chữ Nôm HCMUS) theo loại sách

| bộ | loại bản | mức chắc | đại lượng | giá trị | CI 95 % | n |
|---|---|---|---|---|---:|---:|
| LucVanTien1916 | mộc bản | **ĐO** | kim ở ô (mọi ô có GT) | 95,23 % | [94,86–95,57] | 13.743 |
| LucVanTien1916 | mộc bản | **ĐO** | kim ở ô GOLD | 97,95 % |  | 13.176 |
| TruyenKieu1872 | mộc bản | **ĐO** | kim ở ô (mọi ô có GT) | 97,46 % | [97,24–97,66] | 21.369 |
| TruyenKieu1872 | mộc bản | **ĐO** | kim ở ô GOLD | 98,59 % |  | 20.999 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim ở ô strict | 50,49 % | [49,67–51,29] | 75.190 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim ở ô V1+ | 53,62 % | [52,72–54,46] | 75.190 |
| SachKinhThayCaBinh | viết tay | **ĐO** | kim cả trang (LCS): precision / recall strict | 49,21 % / 37,66 % | [48,30–50,07] | 91.388 |
| SachDungLyHoThan | viết tay | **ĐO** | kim ở ô strict | 38,03 % | [36,53–39,53] | 13.519 |
| SachDungLyHoThan | viết tay | **ĐO** | kim ở ô V1+ | 40,08 % | [38,56–41,66] | 13.519 |
| SachDungLyHoThan | viết tay | **ĐO** | kim cả trang (LCS): precision / recall strict | 32,70 % / 26,41 % | [31,26–34,23] | 19.521 |
| LucVanTien1883 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `LVT1916_NF` (mọi tầng) | 70,5 % |  | 4.499 |
| KimVanKieu1884 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `Kieu1871_LVD` (mọi tầng) | 84,5 % |  | 16.032 |
| KimVanKieu1884 | thạch bản | **ƯỚC LƯỢNG** | kim ở ô = chữ dị bản `Kieu1872_DMT` (mọi tầng) | 73,8 % |  | 16.216 |
| STT2/4/11, Chrestomathie1872 | viết tay / sách in | — | không có nhãn người -> không đo | chưa có |  |  |

**Cờ CNT của gold_exact trên chữ viết tay (ĐO, Borg).** GOLD có chữ người, tỉ lệ đúng V1+ tách theo cờ `cnt` (`measure_out/<Borg>/borg_endtoend/cells.csv` ⋈ `dataset/_ALL/gold_exact.csv`, Wilson):

| bộ | nhóm | n | đúng V1+ | CI 95 % |
|---|---:|---:|---:|---:|
| SachKinhThayCaBinh | CNT = 1 (số chữ OCR cột ≠ số âm QN) | 71.006 | 94,39 % | [94,22–94,56] |
| SachKinhThayCaBinh | CNT = 0 | 1.687 | 94,96 % | [93,81–95,91] |
| SachDungLyHoThan | CNT = 1 (số chữ OCR cột ≠ số âm QN) | 12.025 | 96,70 % | [96,36–97,00] |
| SachDungLyHoThan | CNT = 0 | 320 | 96,88 % | [94,34–98,29] |

Gần như mọi ô GOLD Borg mang CNT = 1 (cột viết tay hiếm khi đếm bằng), nên cờ này chủ yếu dồn ô sang `text_only` chứ không tách được ô sai khỏi ô đúng; so hai nhóm ở bảng trên (nhóm CNT = 0 rất nhỏ).

## 4. Bất biến (invariants)

`measure_out/SUMMARY.json` (2026-10-01T20:48:33, `measure.py --all --report-only`): **244 PASS · 1 FAIL · 2 FAIL mềm · 0 SKIP** trên 17 bước (sập 0).

| bước | bộ | mã | PASS | FAIL | mềm | SKIP | giây |
|---|---|---:|---:|---:|---:|---:|---:|
| code_facts | (repo) | 0 | 17 | 1 | 0 | 0 | 4 |
| layout | LucVanTien1883 | 0 | 8 | 0 | 0 | 0 | 45 |
| qn_ocr | LucVanTien1883 | 0 | 20 | 0 | 0 | 0 | 1 |
| layout | KimVanKieu1884 | 0 | 10 | 0 | 0 | 0 | 80 |
| qn_ocr | KimVanKieu1884 | 0 | 24 | 0 | 0 | 0 | 2 |
| chresto_map | Chrestomathie1872 | 0 | 18 | 0 | 0 | 0 | 12 |
| ihr_layout | LucVanTien1916 | 1 | 6 | 0 | 2 | 0 | 2 |
| ihr_endtoend | LucVanTien1916 | 0 | 7 | 0 | 0 | 0 | 1 |
| ihr_layout | TruyenKieu1872 | 0 | 8 | 0 | 0 | 0 | 1 |
| ihr_endtoend | TruyenKieu1872 | 0 | 7 | 0 | 0 | 0 | 1 |
| borg_endtoend | SachKinhThayCaBinh | 0 | 7 | 0 | 0 | 0 | 16 |
| borg_endtoend | SachDungLyHoThan | 0 | 7 | 0 | 0 | 0 | 6 |
| detector_transfer | LVT+KVK+STT | 0 | 18 | 0 | 0 | 0 | 114 |
| box_ref | LVT+KVK | 0 | 24 | 0 | 0 | 0 | 79 |
| gold_exact | (_ALL) | 0 | 29 | 0 | 0 | 0 | 27 |
| borg_human | Borg18+34 | 0 | 20 | 0 | 0 | 0 | 28 |
| stt_lt2 | STT | 0 | 14 | 0 | 0 | 0 | 14 |

| phép đo | PASS | FAIL |
|---|---:|---:|
| gold_exact_eval (bản giao dataset/_ALL/gold_exact.csv) | 29/29 | 0 |
| ihr_endtoend LucVanTien1916 | 7/7 | 0 |
| ihr_endtoend TruyenKieu1872 | 7/7 | 0 |
| borg_endtoend SachKinhThayCaBinh | 7/7 | 0 |
| borg_endtoend SachDungLyHoThan | 7/7 | 0 |

## 5. Thời gian chạy

Lượt `clean_rebuild_all.sh --run` (`logs/clean_rebuild_20261001_191055_thoi_gian.tsv`):

| bước | lệnh | mã | thời gian | bắt đầu | kết thúc | log |
|---|---|---:|---:|---|---|---|
| 1 | run_pipeline_all | 0 | 73 ph 31 s | 2026-10-01T19:13:19 | 2026-10-01T20:26:50 | `logs/clean_rebuild_20261001_191055_1_run_pipeline_all.log` |
| 2 | borg_human | 0 | 13 ph 25 s | 2026-10-01T20:26:50 | 2026-10-01T20:40:15 | `logs/clean_rebuild_20261001_191055_2_borg_human.log` |
| 3 | measure_all | 1 | 7 ph 12 s | 2026-10-01T20:40:15 | 2026-10-01T20:47:27 | `logs/clean_rebuild_20261001_191055_3_measure_all.log` |
| 4 | nghiem_thu | 1 | 1 ph 12 s | 2026-10-01T20:47:27 | 2026-10-01T20:48:39 | `logs/clean_rebuild_20261001_191055_4_nghiem_thu.log` |

| bộ / bước | bắt đầu | các bước (giây) | tổng | ghi chú |
|---|---|---|---|---|
| STT (3 quyển chung) | 2026-10-01T19:13:22 | setup 0s · extract 5s · build 799s · remediate 17s · rescue 6s · hai_luot 7s · export 69s | 15 ph 3 s |  |
| LucVanTien1883 | 2026-10-01T19:28:25 | setup 2s · ingest 3s · build 183s · remediate 2s · gates 6s · chon_chu 124s · export 5s · measure 2s | 5 ph 27 s |  |
| KimVanKieu1884 | 2026-10-01T19:33:52 | setup 3s · ingest 4s · build 230s · remediate 2s · gates 16s · chon_chu 155s · export 8s · measure 5s | 7 ph 3 s |  |
| Chrestomathie1872 | 2026-10-01T19:40:56 | setup 0s · ingest 55s · build 130s · remediate 1s · gates 1s · chon_chu 52s · export 2s · measure 0s | 4 ph 1 s |  |
| LucVanTien1916 | 2026-10-01T19:44:57 | setup 1s · ingest 1s · build 163s · remediate 2s · gates 0s · chon_chu 41s · export 5s · measure 0s | 3 ph 33 s |  |
| TruyenKieu1872 | 2026-10-01T19:48:30 | setup 0s · ingest 2s · build 204s · remediate 2s · gates 1s · chon_chu 58s · export 9s · measure 0s | 4 ph 36 s |  |
| SachKinhThayCaBinh | 2026-10-01T19:53:06 | setup 0s · ingest 7s · build 866s · remediate 4s · gates 2s · chon_chu 248s · export 43s · measure 14s | 19 ph 44 s |  |
| SachDungLyHoThan | 2026-10-01T20:12:50 | setup 1s · ingest 1s · build 221s · remediate 2s · gates 1s · chon_chu 72s · export 5s · measure 5s | 5 ph 8 s |  |
| gộp dataset/_ALL | 2026-10-01T20:19:40 | merge 3978s | 66 ph 18 s |  |
| GOLD chính xác (B8) | 20261001_201940 | mã thoát 0 · 421s | 7 ph 1 s |  |

## 6. Giới hạn

1. Độ chính xác chỉ **ĐO** được ở 2 bộ mộc bản IHR (LucVanTien1916, TruyenKieu1872) và 2 bộ chép tay Borg (SachKinhThayCaBinh, SachDungLyHoThan) — đều là TẬP ĐÁNH GIÁ, không giao nộp. 3 bộ STT và Chrestomathie1872 (giao nộp) chỉ có **SUY ĐOÁN**; LucVanTien1883/KimVanKieu1884 chỉ có **ƯỚC LƯỢNG** qua dị bản (cận dưới: dị bản khác chữ hợp lệ bị tính sai).
2. Borg: QN đầu vào là phiên âm NGƯỜI theo trang ⇒ số đo là **cận trên** của phương pháp trên chữ viết tay; chỉ câu "đếm bằng" được chấm. Kim trên chữ viết tay kém hẳn mộc bản (mục 3) — GOLD vẫn cao nhờ luật `kim ∈ R(âm)` lọc.
3. Số ô `gold_exact = ok` phụ thuộc ngưỡng bộ kiểm ảnh↔chữ (`config/gold_exact.yaml`); ở Borg rất ít ô ok có chữ người (xem mục 2, n nhỏ ⇒ CI rộng).
4. Bảng thời gian ghép từ nhiều lượt khi chưa có lượt `clean_rebuild_all.sh --run` trọn vẹn.

