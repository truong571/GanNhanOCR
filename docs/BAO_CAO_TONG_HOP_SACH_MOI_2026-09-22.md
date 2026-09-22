# BÁO CÁO TỔNG HỢP — gán nhãn tự động cho sách mới (LVT1883, KVK1884, Chrestomathie1872): vòng 1 + vòng 2 + I5 + run_pipeline

Ngày 2026-09-22 (chiều, vòng 3) · đã commit trên main: 59fde272c4 (engine pitch), 6575d4d764 (run_pipeline --book + measure + train_crop), 10e6fa1b3f (docs) — kế hoạch: `docs/KE_HOACH_COMMIT_VONG3_2026-09-22.md`.
Báo cáo này **gộp và thay** `BAO_CAO_TONG_THE_SACH_MOI_2026-09-22.md` (vòng 1–2) và `PHUONG_AN_TU_DONG_2026-09-22.md` (hai tệp giữ làm lịch sử).
Mọi con số sinh bởi script (0 token LLM), nguồn ghi ngay cạnh số; **không có người kiểm** (ràng buộc đề tài).

## 0. Câu trả lời ngắn

| Câu hỏi | Trả lời |
|---|---|
| Chạy được sách mới bằng pipeline STT không? | **Có, 1 lệnh**: `./run_pipeline.sh --book LucVanTien1883 \| KimVanKieu1884 \| Chrestomathie1872 \| all-new` (B0→B6, 0 API khi có cache kim; 169 / 236 / 137 s). Đường STT (`./run_pipeline.sh` không tham số) **không đổi byte** (§2). |
| Kết quả chốt cuối (sau pitch + cổng a')? | LVT **8.650** GOLD ảnh + 130 text_only → **11.013 ảnh**; KVK (B1') **13.908** + 510 → **17.752 ảnh**; Chresto **5.359** + 126 → **6.645 ảnh** (§3). |
| Nhãn đúng bao nhiêu (không người)? | Precision **văn bản** GOLD trên GT độc lập IHR-NomDB (mộc bản): **89,3 %** (LVT1916) / **84,6 %** (Kiều 1872); khớp dị bản trên chính thạch bản: KVK↔1871 **81,3 %** sau cổng (a)(b)(c), ↔1872 độc lập 78,5 % (B6), nền dị bản 1871↔1872 chỉ 82,9 % (§4). **Độ đúng hộp/ảnh không đo được** — chỉ có proxy ô tham chiếu tự động (§6). |
| I5 (detector đếm lệch ±1) đã chữa chưa? | Chữa phần **hoà giải** bằng `box_decoder: pitch` (hộp IoU ≥ 0,5 với ô tham chiếu 97,4 → 98,1 % LVT, 94,1 → 97,0 % KVK) và cổng (a') theo ô; **gốc mô hình chưa chữa** (I5 thô 65,2 / 59,2 / 70,7 % giữ nguyên theo định nghĩa); phương án B (huấn luyện v2) có script, chưa chạy dài (§6). |
| Quyết định pitch? | **PITCH = bản chốt** cho cả 3 sách (tier không giảm, khớp dị bản không giảm, ảnh lỗi ở mức crop giảm; §3.2). Bản legacy giữ ở `*_v3_legacy/`. |

## 1. Dữ liệu (`data/`, 13 thư mục, 3 nhóm — `data/README.md`, `data/*/SOURCE.md`, `data/KIEM_TRA_KHOP_1-1_2026-09-20.md`)

| Nhóm | Thư mục | Loại · QN cùng nguồn | Trạng thái |
|---|---|---|---|
| 1 — cặp Nôm–QN cùng nguồn (đầu vào gán nhãn) | SachThanhTruyen2/4/11 | viết tay thật; QN trang đối diện | kho chính STT (không đổi) |
| | **LucVanTien1883** | thạch bản, 10 cột × 2 tầng (cột = cặp 6⧺8), **2.088** câu; 139 trang QN in cùng sách | **đã chạy đủ 105/105** |
| | **KimVanKieu1884** | thạch bản, `page = 167 − canvas`, **3.256** câu (QN 3.251 dòng, lệch 5 chưa định vị); 295 trang QN xen dịch Pháp | **đã chạy đủ 163/163** (B1') |
| | **Chrestomathie1872** | thạch bản văn xuôi, 20 truyện, cột 3–7 chữ biến thiên; QN mức truyện (28 trang) | **đã chạy đủ 65/65** (`layout: prose`) |
| 2 — đối chứng ngoài có phiên âm độc lập | LucVanTien1916, TruyenKieu1872 (IHR-NomDB / Nôm Foundation) | mộc bản 35 px/chữ, câu↔câu 96,6 / 99,97 % | **GT độc lập** đo precision (§4) |
| 3 — chép tay chưa phiên âm | TruyenKieuPhongTinhCoLuc, KimVanKieu1894, TamTuKinhDienAm, CacThanhTruyen1646 | không QN cùng nguồn (dị bản sai ≥ 6 % âm) | không chạy (ngoài đề tài); 1871 LVĐ dùng làm tham chiếu B1' |
| loại | LyHangCaDao | 30 px/chữ | loại |

## 2. Những gì đã xây (thành phần → tệp → kiểm chứng)

| Thành phần | Tệp | Kiểm chứng |
|---|---|---|
| Engine tham số theo sách: `layout lithograph\|prose`, `n_columns` (auto), `det_xmargin`, `det_thr`, **`box_decoder`** | `pipeline/align_engine/book_layout.py`, `align_production.py`, `build_dataset.py`, `step0/1` | STT 3 trang `labels.csv` **byte-identical** HEAD 4e0a314bca ↔ mã mới, md5 **59e436d7641fa849bb6759868ac29259** (556 dòng; `summary.json` chỉ thêm khoá `box_decoder: legacy`); book_layout 79/79; phase1_engine 253/0 |
| **pitch_decode** (I5 phương án A): tầng-cột, ứng viên ≥ 0,05 + ô ảo chiếu mực, DP chọn đúng N theo bước; `box_source` detector \| detector_low \| ink_cut; `count_source` pitch \| pitch_ocr; **`n_det` giữ nghĩa hộp thô** | `pipeline/align_engine/char_detector/pitch_decode.py` (mới, 434 dòng) + `assign_boxes_pitch`, `pitch_target_count`, `expected_tier_counts` | selftest 22/22; `scripts/measure/box_ref_eval.py` (bước `box_ref` của `measure.py`): 27 trang/sách, 3.412 + 3.527 ô tham chiếu (§6) |
| Adapter ingest thạch bản / văn xuôi; B1' | `pipeline/tools/ingest_lithograph_book.py` (`--verse-map`, `--contrast`, `--verses`), `ingest_prose_book.py`, `scripts/measure/verses_ref_fix.py` | 61/61 · 33/33 · 19/19; LVT 105/105, KVK 163/163, Chresto 65/65 cache hợp lệ |
| Cổng cơ chế B4' + tầng `GOLD_text_only` + **luật (a') chế độ pitch** + cờ `n_det_mismatch` | `pipeline/remediation/mechanism_gates.py`, `export_final_dataset.py` (TRACE), `make_dataset_docs.py` | mechanism_gates_selftest **94/94** (62 cũ + 32 mới: nhận biết chế độ cli>config>summary>labels, ink_cut/detector_low → text_only, n_det≠N chỉ cờ, legacy không đổi schema); STT: cổng TẮT = sao byte |
| **`run_pipeline.sh --book`** (khối SÁCH MỚI, bash 3.2): B0 setup+bộ đo+B1' → B1 ingest → B2 build+enrich → B3 remediation (`--out`) → B4 auto_precision cross + gates → B5 export+docs+xlsx → B6 cross gated; `--dry-run --skip-ingest --no-api --no-auto-precision --suffix`; hồ sơ từ khối `run:`/`run_config:` của `config/pipeline_<Book>.yaml`; log `logs/run_<Book>_*.log`, sha256 `CHECKSUMS.txt` | `run_pipeline.sh` (chỉ 2 hunk thêm; hàm STT/MAIN không đổi), `config/pipeline_{LucVanTien1883,KimVanKieu1884,KimVanKieu1884_b1,Chrestomathie1872}.yaml` | `--suffix _rp` tái lập md5 bản chốt 3/3 sách (HUONG_DAN §2.1); `./run_pipeline.sh --dry-run` in đúng 6 bước STT; lần chạy chốt cuối `_pitch` EXIT 0, 0 API |
| Bộ đo offline 0 token: `measure.py --all` (layout, qn_ocr, chresto_map, detector_transfer, code_facts, **box_ref**) 118 invariants; `auto_precision.py` (IHR, cross, gates) | `scripts/measure/*`, `docs/PIPELINE_FACTS.json` | idempotent; FACTS 18/18 sau vòng 3 |
| I5 phương án B — huấn luyện detector v2 trên nhãn yếu | `train_crop/build_lithograph_manifest.py`, `train_v2_lithograph.py`, `eval_boxes_ref.py`, `train_crop/data_lithograph/` (7,3 MB manifest) | manifest 31.776 ô / 793 vùng ignore; smoke 1 epoch 3+3 trang chạy được; **chưa chạy dài** (§6) |

## 3. Kết quả chạy 3 sách — bảng chốt cuối (pitch + cổng a'), `dataset_out_<Book>[_b1]/{summary,mechanism_gates_report,remediation_report}.json`, `dataset_<Book>/`

| Chỉ số | LucVanTien1883 | KimVanKieu1884 (B1') | Chrestomathie1872 (prose) |
|---|---|---|---|
| Trang / page_ok · thời gian `--book` (0 API) | 105 / 103 · 169 s | 163 / 162 · 236 s | 65 / 65 · 137 s |
| Ô sinh ra · cột · M==N (kim = QN) | 14.476 · 1.041 · 83,8 % | 22.704 · 1.628 · 88,8 % | 8.303 · 417 · 69,5 % |
| **I5 thô** n_det==N (hộp thô @det_thr, KHÔNG đổi khi pitch) | **65,2 %** ❌ (< 75) | **59,2 %** ❌ | **70,7 %** ❌ |
| box_source (pitch): detector / detector_low / ink_cut | 14.240 / 64 / 172 | 21.825 / 449 / 430 | 8.090 / 4 / 209 |
| labels_final: GOLD / SYL / REVIEW / QUAR (F1 cross-col quarantine) | 9.667 / 1.684 / 3.111 / 14 | 15.931 / 3.206 / 3.465 / 102 | 5.892 / 998 / 1.413 / 0 |
| Cổng quyết: (c) crop / (d) dị bản gần hình / (b) cầu + sửa dấu / **(a') box_low_conf** | 14 / 194 / 569 + 112 / **130** | 183 / 692 / 601 + 58 / **510** | 119 / — / 238 + 71 / **126** |
| Cờ `n_det_mismatch` (dòng) · GOLD giữ ảnh dù cột n_det≠N | 4.978 · 2.910 | 9.238 · 5.430 | 2.408 · 1.502 |
| **Sau cổng: GOLD ảnh / GOLD_text_only / SYLLABLE / REVIEW** | **8.650 / 130** / 2.363 / 3.319 | **13.908 / 510** / 3.844 / 4.340 | **5.359 / 126** / 1.286 / 1.532 |
| **Ảnh export** (GOLD + SYL) · dòng `labels.csv` | **11.013** · 11.143 | **17.752** · 18.262 | **6.645** · 6.771 |
| IoU ≥ 0,5 với ô tham chiếu (27 trang, pitch; legacy@0,15) | 98,1 % (97,4) | 97,0 % (94,1) | không có tham chiếu (văn xuôi) |
| … riêng cột n_det≠N (pitch; legacy) | 95,3 % (94,1) | 95,3 % (90,1) | — |
| "Cắt vào thân chữ" hộp thô (mực chạm mép > 0,20): 27 trang pitch (legacy) · toàn sách bbox labels pitch (legacy) | 9,5 % (11,0) · 9,63 % (9,28)¹ | 2,6 % (3,8) · 3,06 % (3,15) | — (ảnh gốc nền xám, chỉ số không hiệu chuẩn) |
| Crop thật (sau pad/carve/tighten), mọi ô: bleed · blank · truncated (legacy → pitch) | 2.074 → 2.025 · 6 · 8 | 2.052 → 1.992 · 219 → 177 · 13 → 10 | 136 → 137 · 0 · 124 → 119 |
| Khớp dị bản GOLD-ảnh (bỏ ref PUA): trước cổng → sau (a)(b)(c) → B6 +(d) *(tự khẳng định)* | 72,6 % (n 2.796) → **73,4 %** (n 2.595; legacy 73,4 / n 1.866) → 79,1 % (legacy 79,5) | 1871: 80,6 % → **81,3 %** (n 11.461; legacy 81,1) → 85,8 % (85,7); **1872 độc lập**: 73,0 % → B6 **78,5 %** (78,4) | — |

¹ Toàn sách LVT tăng 0,35 điểm vì 1.136 ô `midpoint → detector`: hộp midpoint cũ được nới `BOX_OVERLAP_FRAC` (cao 184 px) nên mép ít mực **theo cấu tạo**, hộp detector sát chữ (169 px);
trên hộp cùng loại thì giảm: `detector → detector` 24,8 → 19,5 %, `detector → ink_cut` 33,7 → 5,6 % (LVT); KVK 13,2 → 6,4 / 22,7 → 1,4 %. Chỉ số cuối để so là **crop thật** (hàng trên): giảm ở cả 3 sách.

### 3.2 Vì sao chọn pitch làm bản chốt (so `_pitch` với bản legacy vòng 2, cùng `run_pipeline.sh --book`, cùng cache)

1. **Tier văn bản không giảm**: LVT GOLD 9.666 → 9.667, KVK 15.938 → 15.931 (−7 = 10 ô quarantine thêm do census F1 cross-col 94 → 108 dòng), Chresto y hệt — tier do DP văn bản quyết, không do hộp.
2. **Khớp dị bản không giảm** (proxy văn bản, mù lỗi hộp): sau (a)(b)(c) 73,4 = 73,4 (LVT), 81,1 → 81,3 (KVK); B6 trong CI (±1–2 điểm).
3. **Hộp đúng chữ hơn** (proxy ô tham chiếu, 27 trang/sách): IoU ≥ 0,5 +0,7 / +2,9 điểm; miss 0,5 → 0,1 / 3,0 → 0,4 %; extra/100 ô 2,0 → 0,1 / 2,3 → 0,4; midpoint+split (1.277 / 3.268 / 284 ô) → 0.
4. **Ảnh giao nộp tăng mà lỗi crop giảm**: +2.857 / +5.315 / +1.437 ảnh GOLD; bleed/blank/truncated trên ô đổi hộp: LVT bleed 292 → 243, KVK bleed 503 → 442, blank 89 → 47 (ô đổi hộp 1.607 / 3.999 / 613 = 11 / 18 / 7 %).
5. Chi phí: build +8 s (LVT) / +11 s (KVK); F1 cross-col KVK +14 dòng (đã quarantine); Chresto 321 ô đổi hộp IoU < 0,3 trong cột n_det≠N — **không kiểm được** (không tham chiếu), crop flag không đổi.

Rủi ro còn lại (§6): ô GOLD-ảnh trong cột n_det≠N (2.910 / 5.430 / 1.502) có ≈ 4,7 % hộp lệch theo proxy (IoU < 0,5) ≈ 137 / 255 / ? ô; người muốn bộ chặt hơn lọc `n_det_mismatch = 1` trong `labels_trace.csv`.

## 4. Độ đúng tự động thay người kiểm (`scripts/measure/auto_precision.py --all` → `measure_out/auto_precision/`; chi tiết cũ: PHUONG_AN_TU_DONG §1–§3)

| Phép đo | KVK1884 / Kiều | LVT1883 / Lục Vân Tiên | Ý nghĩa / caveat |
|---|---|---|---|
| (1) Precision **văn bản** luật GOLD trên **GT độc lập** IHR-NomDB (mộc bản, 200 patch/sách, kim ×3, cột cắt đúng) | **84,6 %** [81,7–87,1] (cov 49,5 %, n 689); bỏ GT PUA 86,6 % | **89,3 %** [86,9–91,3] (cov 55,1 %, n 768); bỏ PUA 90,5 % | khác miền in (mộc ≠ thạch), có điều kiện cắt đúng; kim thô chỉ 42–49 % → từ điển cộng ~40 điểm |
| (2) Khớp dị bản trên **chính thạch bản** (câu QN giống hệt / ≥ 75 %, vị trí âm giống hệt) | 1871: 80,6 % (n 11.463) → sau cổng 81,3 %; 1872: 73,0 → 78,5 %; **nền 1871↔1872 = 82,9 %** (9.961 vị trí) | 1916: 72,6 % (n 2.796) → 73,4 % (1916 xa 1883, chỉ 29 % câu khớp) | cận dưới của precision; ước lỗi văn bản GOLD KVK ≈ 1 − 80,4/82,9 ≈ 3 % |
| (3) Bản chất lỗi | 100 % lỗi GOLD trên GT (72 + 90 ô) là **đồng âm dị thể** hoặc GT PUA; **0** ca ngoài từ điển; 22,7–24,4 % bất đồng là cặp gần hình → trần lỗi kim ≈ 4,5 % (KVK) / 6,5 % (LVT) | | S3/ArcFace mù (AUC 0,57) nên không phân xử đồng âm |
| (4) Cổng theo proxy | mọi cổng dịch proxy ≤ +1,2 điểm (trong CI) → cổng chọn theo **cơ chế** (B4'), không theo proxy | | proxy văn bản mù với lỗi hộp — lý do ra đời box_ref (§6) |

## 5. Cổng máy B4' và B1' (số chốt cuối)

- **B4'** (`mechanism_gates.py`, tự bật khi `layout: lithograph`; prose khai `mechanism_gates: true`; STT tắt = sao byte): (c) blank/truncated → REVIEW (14 / 183 / 119); (d) `--cross` bất đồng dị bản gần hình → REVIEW (194 / 692 / —), không gần hình → cờ `di_ban_khac` (571 / 1.710); (b) cầu `s1_inter_s2_similar` + `am_sua_dau` → SYLLABLE (681 / 659 / 309; precision cầu trên GT chỉ 74,0 %); (a) legacy: cột n_det≠N \| midpoint/split → text_only (**đã loại 2.985 / 5.784 / 1.557 ô ở vòng 2**); **(a') pitch**: chỉ ô `ink_cut \| detector_low` → text_only (130 / 510 / 126), cột n_det≠N chỉ cờ. Ưu tiên (c) > (d) > (b) > (a/a').
- **B1' KVK** (`verses_ref_fix.py --fuzzy-min 0.9`, QN đầu vào = phiên âm 1871 LVĐ): 3.251 dòng → exact 465 · fuzzy 1.548 (0 dòng đổi số âm) · giữ OCR 1.238; GOLD +882 (+5,9 %, toàn bộ ở dòng fuzzy 66,7 → 74,7 %); khớp dị bản 1872 độc lập không đổi (72,3 → 73,0 %) → sửa QN không kéo nhãn sai. **Caveat**: cột `syllable` mang chính tả **Bắc** ở **676 dòng** (sanh→sinh, nhơn→nhân…; `qn_source` chỉ ghi trong `transcriptions/*.json`, chưa vào export); 50 âm (2,0 %) nghi xoá dị bản QN thật của 1884; 125 dòng lệch số âm không sửa được ở 0,9. **`--dict-boost` LOẠI** (đổi 556 chữ nhãn theo 1871 → khớp 1871 = 100 % theo định nghĩa; 59 ô đúng 1872 bị làm sai); ablation giữ ở `config/pipeline_KimVanKieu1884_b1_boost.yaml` (legacy).
- Sự cố đã khắc phục: `pipeline.remediation apply` không `--out` ghi đè `dataset_out/` STT (2 lần 22/09, `git checkout` khôi phục; `run_pipeline.sh --book` luôn truyền `--out`; invariant `dataset_out_tracked_clean`).

## 6. I5 — hộp chữ detector trên thạch bản (`docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md`, `measure_out/box_ref/summary.json`)

**Chẩn đoán** (27 trang/sách, 270 cột, `box_ref_eval.py`, 79 s): I5 cột 63,7 / 54,4 % nhưng ở mức ô chỉ 2–6 % ô lệch (IoU ≥ 0,5 legacy 97,4 / 94,1 %); lỗi là ±1 hộp trong tầng (bắt 2 lần: dup 7–12, stray 60–70 / 100 ô; bỏ sót chữ nhạt dưới ngưỡng: 17 / 90 ô), **không** phải số câu in lọt vào (0–0,4 % cột có hộp ngoài tầng); điểm tin cậy CenterNet (học STT) trên thạch bản thấp (trung vị 0,35). Ô tham chiếu = hộp tầng kim cắt tại N−1 khe mực yếu nhất (DP có phạt lệch bước), chỉ tầng verified (kim == N: 90,6 / 93,5 %).

**Phương án A — đã làm, đã đo, đã chốt** (`pitch_decode.py`, `box_decoder: pitch`): bảng §3; miss 0,1 / 0,4 %, extra 0,1 / 0,4, |dy| p90 16,3 / 15,4 % bước; ô `ink_cut` trùng tham chiếu **theo cấu tạo** → số "honest" chỉ ô nguồn detector: 98,3 / 97,3 % (vẫn hơn legacy 0,9 / 3,2 điểm). Trường hợp I5 mù: `page_0008` cột 7 LVT n_det = 14 = N nhưng legacy lệch 5 hộp một chữ, pitch đúng.

**Phương án B — kế hoạch + script, CHƯA chạy dài**: nhãn yếu `train_crop/data_lithograph/manifest_*.json` (268 trang thạch bản → **31.776 ô** ở 4.550/5.343 tầng + 793 vùng `ignore_boxes`; STT 445 trang + 14.172 bbox REVIEW làm ignore; chia page-disjoint theo sách, `--lobo`); trainer `train_v2_lithograph.py` (init v1, cân bằng miền 50/50, augment nền xám + kéo dọc ±10 %); eval mỗi epoch `eval_boxes_ref.py` (litho ok50, % tầng n == N của hộp **thô**, cắt thân chữ; STT F1 hồi quy — mốc v1: ok50 95,0 %, tầng n==N 77,0 %, STT F1 0,876). Thời gian: 20 epoch ≈ 5–8 h CPU Mac hoặc **≈ 1 h Kaggle T4** (đóng gói `pack_for_kaggle.py` + `data_lithograph/` + `prepared/{LVT,KVK}/pages`). Nghiệm thu: `detector_transfer.py` STT ≥ 90,1 % không giảm; `box_ref_eval.py` ok50 ≥ 98 %, cắt thân chữ giảm; build `--limit 10` `detector_low`/`ink_cut` giảm.

**Cách đo trung thực**: (i) không dùng `n_det == N` khi ép N (hằng đúng) — `n_det` giữ hộp thô; (ii) tách `*_honest(det_src_only)`; (iii) chỉ số không phụ thuộc tham chiếu duy nhất = mực chạm mép hộp thô (và crop flag thật, đã bão hoà); (iv) tham chiếu là proxy (kim tầng cụt/rộng, khe trong chữ ⿱), tầng không verified 7–9 % bị loại; (v) CI 27 trang ≈ ±3,5 điểm cột, ±0,7 điểm ô. **Không có hộp GT người → không đo được precision crop.**

## 7. Cách chạy và cách thêm sách (`docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` §2.0, §7)

```bash
./run_pipeline.sh --book all-new                    # 3 sách, B0→B6, ~9 phút, 0 API (cache kim_raw/); log logs/run_<Book>_*.log
./run_pipeline.sh --book KimVanKieu1884 --dry-run   # chỉ in lệnh; --suffix _x (bản so sánh, không ghi đè); --skip-ingest; --no-auto-precision
# KHÔNG dùng --no-api cho KVK: content -> formula, khác bản chốt.   ./run_pipeline.sh (không tham số) = đường STT cũ.
```
Sách thứ tư: `data/<BOOK>/pages/` + `SOURCE.md` → thêm vào `BOOKS` của `scripts/measure/` → `measure.py --book <BOOK>` → chọn `layout` (lithograph: cột = cặp 6⧺8, `ingest_lithograph_book --plan-only`; prose: `ingest_prose_book` + bảng truyện↔trang) → `config/pipeline_<BOOK>.yaml` (`books[]`: layout, n_columns, det_xmargin 0,05, quét `det_thr`, **`box_decoder: pitch`**; khối `run:`) → thêm vào `NEW_BOOKS_ALL` (run_pipeline.sh) và `CROSS_BOOKS` (auto_precision.py) nếu có dị bản → `./run_pipeline.sh --book <BOOK>` → viết `docs/CHAY_<BOOK>.md`. Selftest trước commit: §9.

## 8. Giới hạn thật và việc chưa làm

1. **Không có GT người** — mọi độ đúng là proxy: văn bản (mộc bản IHR, dị bản), hộp (ô tham chiếu tự động). Precision ảnh crop **không đo được**; GOLD-ảnh trong cột n_det≠N (2.910 / 5.430 / 1.502 ô) ước ≈ 4,7 % hộp lệch. Bộ mẫu mù `kiem_nguoi_*.py` để ngoài commit.
2. **I5 gốc chưa chữa**: I5 thô 65,2 / 59,2 / 70,7 % < 75 %; phương án B chưa chạy dài; detector thích ảnh nhị phân hoàn toàn (KVK otsu 65,6 vs prepared 54,4 %) nhưng engine chưa tách "ảnh cho detector" khỏi "ảnh để crop".
3. **Blank KVK** 219 → 177 ô (c) hạ REVIEW: ngưỡng `enrich_crop_quality` hiệu chuẩn trên STT; 40 ô đo có mực trên ảnh gốc → có thể hạ oan.
4. **QN Chresto** tesseract lỗi âm ≈ 13 % + mất dòng → 1.239 ô `no_context` REVIEW; không phiên âm chuẩn để B1'; không dị bản → không (d)/B6; hộp không có tham chiếu (321 ô đổi hộp IoU < 0,3 chưa kiểm).
5. **B1'**: chính tả Bắc 676 dòng chưa vào export/DATASHEET; luật theo từng âm với kim trọng tài chưa làm; KVK lệch 5 câu Nôm/QN chưa định vị (`KET_QUA_DO_CUOI` §4).
6. **Mã**: `step2_align.py` (CLI riêng) vẫn ghim 9 cột; `remediation apply` chưa chặn thiếu `--out` trong mã; `run_pipeline.sh` STT chưa gọi B4'; bản chốt là kết quả `--suffix _pitch` đổi tên (đường dẫn `_pitch` còn trong `CHECKSUMS.txt`/`mechanism_gates_report.json`); `tools.selftest` 3 fail có sẵn ở HEAD; SILVER = 0 trên thạch bản là đúng (S3 học chữ thảo).
7. Không làm và không nên làm: self-training từ chính hộp detector (vòng tròn), dùng khớp dị bản sau (d) làm bằng chứng cổng (tự khẳng định), trích 99 % GOLD của STT cho sách mới.

## 9. Chỉ mục tài liệu và commit

| Nội dung | Tệp |
|---|---|
| Hướng dẫn chạy (lệnh, cổng nghiệm thu 3 sách, thêm sách) | `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` |
| I5: chẩn đoán, pitch_decode, huấn luyện v2, cách đo | `docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md`; `measure_out/box_ref/` |
| Lần chạy từng sách (đầu tệp = chốt cuối pitch) | `docs/CHAY_LVT1883_2026-09-21.md`, `CHAY_KVK1884_B1_2026-09-22.md` (chính thức), `CHAY_KVK1884_2026-09-21.md`, `CHAY_CHRESTO1872_2026-09-22.md` |
| Lịch sử vòng 1–2 và phương án tự động (đã gộp vào đây) | `docs/BAO_CAO_TONG_THE_SACH_MOI_2026-09-22.md`, `docs/PHUONG_AN_TU_DONG_2026-09-22.md` |
| Số đo dữ liệu + đặc tả | `docs/KET_QUA_DO_CUOI_2026-09-21.md`, `docs/PIPELINE_SACH_MOI_2026-09-20.md`, `docs/PIPELINE_FACTS.json`, `measure_out/{SUMMARY.json,REPORT.md}` |
| Kế hoạch commit | vòng 1 `KE_HOACH_COMMIT_2026-09-21.md` (đã commit fb345a29b1…853cadfd9c), vòng 2 `KE_HOACH_COMMIT_VONG2_2026-09-22.md` (đã commit 6b8e215576, e06f32dce7, 4e0a314bca), **vòng 3 `KE_HOACH_COMMIT_VONG3_2026-09-22.md` (chưa commit)** |

Hồi quy/selftest lúc chốt (22/09 chiều): STT 3 trang md5 59e436d7… hai bên; `./run_pipeline.sh --dry-run` 6 bước; book_layout 79/79 · ingest_lithograph 61/61 · ingest_prose 33/33 · mechanism_gates 94/94 · pitch_decode 22/22 · verses_ref_fix 19/19 · phase1_engine 253/0 · tools 139/3 (có sẵn) · code_facts 18/18; `git status -- dataset_out data prepared/SachThanhTruyen*` trống.
