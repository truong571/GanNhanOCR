# BÁO CÁO TỔNG HỢP — gán nhãn tự động cho sách mới (LVT1883, KVK1884, Chrestomathie1872): vòng 1 + vòng 2 + I5 + run_pipeline

Ngày 2026-09-22 (chiều, vòng 3; **cập nhật tối, vòng 4: răng cưa INTER_AREA §3.3/§6**) · đã commit trên main: 59fde272c4 (engine pitch), 6575d4d764 (run_pipeline --book + measure + train_crop), 10e6fa1b3f + 8c08591e9a (docs) — kế hoạch vòng 3: `docs/KE_HOACH_COMMIT_VONG3_2026-09-22.md`; **vòng 4 (chưa commit: khoá `detector_ckpt`/`detector_resize`, lab/i5_detector_v2): `docs/KE_HOACH_COMMIT_VONG4_2026-09-22.md`**.
**Cập nhật 2026-09-23 (vòng 5)**: bản chốt 3 sách đổi sang kênh kim `lang_type = 2` (Nôm) + luật đếm QN 6/8 + rào tầng DP — **§3.4**; kế hoạch commit `docs/KE_HOACH_COMMIT_VONG5_2026-09-23.md`; cơ sở đo `docs/CHOT_KENH_OCR_VA_QUY_HOACH_GAN_2026-09-23.md` và `docs/CHAN_DOAN_3_SACH_MOI_2026-09-23.md`. Mọi số ở §3 (không kể §3.4) là bản vòng 4, nay giữ ở `dataset/<Book>_v5_lang1/`.
Báo cáo này **gộp và thay** `BAO_CAO_TONG_THE_SACH_MOI_2026-09-22.md` (vòng 1–2) và `PHUONG_AN_TU_DONG_2026-09-22.md` (hai tệp giữ làm lịch sử).
Mọi con số sinh bởi script (0 token LLM), nguồn ghi ngay cạnh số; **không có người kiểm** (ràng buộc đề tài).

## 0. Câu trả lời ngắn

| Câu hỏi | Trả lời |
|---|---|
| Chạy được sách mới bằng pipeline STT không? | **Có, 1 lệnh**: `./run_pipeline.sh --book LucVanTien1883 \| KimVanKieu1884 \| Chrestomathie1872 \| all-new` (B0→B6, 0 API khi có cache kim; 169 / 236 / 137 s). Đường STT (`./run_pipeline.sh` không tham số) **không đổi byte** (§2). |
| Kết quả chốt cuối (sau pitch + cổng a')? | **CẬP NHẬT 23/09 (vòng 5, §3.4)**: LVT **10.831** GOLD ảnh + 149 text_only → **12.038 ảnh**; KVK (B1') **19.303** + 727 → **19.921 ảnh**; Chresto **5.998** + 226 → **6.854 ảnh**. Bản vòng 4 (kim `lang_type = 1`) giữ ở `dataset/<Book>_v5_lang1/`: LVT 8.650 + 130 → 11.013; KVK 13.908 + 510 → 17.752; Chresto 5.359 + 126 → 6.645 (§3). |
| Nhãn đúng bao nhiêu (không người)? | Precision **văn bản** GOLD trên GT độc lập IHR-NomDB (mộc bản): **89,3 %** (LVT1916) / **84,6 %** (Kiều 1872); khớp dị bản trên chính thạch bản: KVK↔1871 **81,3 %** sau cổng (a)(b)(c), ↔1872 độc lập 78,5 % (B6), nền dị bản 1871↔1872 chỉ 82,9 % (§4). **Độ đúng hộp/ảnh không đo được** — chỉ có proxy ô tham chiếu tự động (§6). |
| I5 (detector đếm lệch ±1) đã chữa chưa? | Chữa phần **hoà giải** bằng `box_decoder: pitch` (hộp IoU ≥ 0,5 với ô tham chiếu 97,4 → 98,1 % LVT, 94,1 → 97,0 % KVK) và cổng (a') theo ô; **gốc mô hình chưa chữa trong bản chốt** (I5 thô 65,2 / 59,2 / 70,7 %). **Tối 22/09 tìm ra gốc**: phần lớn I5 là **răng cưa** khi thu ảnh 3.200 → 1.024 bằng INTER_LINEAR — chỉ đổi INTER_AREA (`detector_resize: area`, cùng ckpt v1) cho I5 thô **77,7 / 77,9** / 69,3 %, ok50 99,3 / 98,6 %, GOLD ảnh +83 / +257; nhưng LVT `bleed` ảnh export 18,0 → 21,7 % và Chresto không lợi → theo luật "không chỉ số nào giảm" **chưa lấy làm chốt** (§3.3). Phương án B (huấn luyện v2 trên Kaggle, mốc = v1+area) đã gói sẵn, chưa chạy (§6). |
| Kênh kim đã gọi chuẩn chưa? | **Chưa, đến 23/09**: body gửi `lang_type = 1` (Hán) cho ba cuốn chữ **Nôm**. Vòng 5 thêm khoá theo sách `books[].kim_lang_type` (mặc định 1 ⇒ STT không đổi byte) và chạy lại cả 3 sách với `2` (Nôm): **GOLD ảnh +2.181 / +5.395 / +639**, khớp dị bản **độc lập** +5,4 điểm (LVT↔1916) và +5,2 điểm (KVK↔1872) (§3.4). |
| Quyết định pitch? | **PITCH (linear) = bản chốt** cho cả 3 sách (tier không giảm, khớp dị bản không giảm, ảnh lỗi ở mức crop giảm; §3.2). Bản legacy giữ ở `*_v3_legacy/`; bản thử `area` giữ ở `*_area/` (§3.3), config ghi `detector_resize: linear` kèm số đo. |

## 1. Dữ liệu (`data/`, **9 thư mục còn trên đĩa** / 13 mô tả trong `data/README.md`, 3 nhóm — `data/*/SOURCE.md`, `data/KIEM_TRA_KHOP_1-1_2026-09-20.md`)

> **23/09**: 4 thư mục (`KimVanKieu1894`, `TamTuKinhDienAm`, `CacThanhTruyen1646`, `LyHangCaDao`, `IHR-NomDB_nlp`) đã bị xoá khỏi
> working tree từ 20–21/09 (`git status -- data` = `D …`), nên `data/` hiện còn **9 bộ**. **Cả 9 nay đã được xử lý: 6 chạy trọn
> pipeline (3 STT + LVT1883 + KVK1884 + Chresto), 2 chạy trọn với tư cách TẬP ĐÁNH GIÁ (LVT1916, TK1872), 1 mới chỉ ĐO BỐ CỤC
> (TruyenKieuPhongTinhCoLuc)** — xem §10 và `docs/CHAY_3_BO_CON_LAI_2026-09-23.md`.

| Nhóm | Thư mục | Loại · QN cùng nguồn | Trạng thái |
|---|---|---|---|
| 1 — cặp Nôm–QN cùng nguồn (đầu vào gán nhãn) | SachThanhTruyen2/4/11 | viết tay thật; QN trang đối diện | kho chính STT (không đổi) |
| | **LucVanTien1883** | thạch bản, 10 cột × 2 tầng (cột = cặp 6⧺8), **2.088** câu; 139 trang QN in cùng sách | **đã chạy đủ 105/105** |
| | **KimVanKieu1884** | thạch bản, `page = 167 − canvas`, **3.256** câu (QN 3.251 dòng, lệch 5 chưa định vị); 295 trang QN xen dịch Pháp | **đã chạy đủ 163/163** (B1') |
| | **Chrestomathie1872** | thạch bản văn xuôi, 20 truyện, cột 3–7 chữ biến thiên; QN mức truyện (28 trang) | **đã chạy đủ 65/65** (`layout: prose`) |
| 2 — đối chứng ngoài có phiên âm độc lập | **LucVanTien1916**, **TruyenKieu1872** (IHR-NomDB / Nôm Foundation) | mộc bản 35–40 px/chữ, **mỗi ô cột = 1 CẶP lục bát 14 chữ** (99,9 %), 10 cột/trang | **23/09: đã CHẠY TRỌN pipeline làm TẬP ĐÁNH GIÁ** (§10) — không trộn vào tập huấn luyện |
| 3 — chép tay chưa phiên âm | **TruyenKieuPhongTinhCoLuc** (4 thư mục kia đã xoá khỏi đĩa) | không QN cùng nguồn (nền dị bản 1871↔1872 chỉ **71,3 %** chữ trùng) | **23/09: ĐÃ ĐO BỐ CỤC, CHƯA CHẠY** — bố cục thật là 2 tầng (trên = lời bình chữ Hán, dưới = 1 cặp 14 chữ); lý do và đường chạy: §10.3 |
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
| **Vòng 4 (chưa commit)**: khoá theo sách `detector_ckpt` (ckpt riêng, fail fast) + `detector_resize` linear\|area (INTER_AREA khử răng cưa); gói Kaggle v2 | `book_layout.py` (`resolve_detector_ckpt`), `align_production.py` (cache `(ckpt, resize, thr)`, `detector_backend_name`), `build_dataset.py`, `char_detector/detector_infer.py`, `train_crop/infer_centernet.py` (`resize`), `scripts/measure/box_ref_eval.py` (`--ckpt --resize`), `lab/i5_detector_v2/` | book_layout **100/100** (+21); STT md5 59e436d7… hai bên (worktree 8c08591e9a); `_area` 3 sách EXIT 0 (§3.3); `measure_out/box_ref_area/` 24/24 |

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

### 3.3 Thử INTER_AREA (`--suffix _area`, 22/09 tối) — không đổi chốt, giữ làm bản so sánh (`dataset_out_<Book>[_b1]_area/`, `dataset_<Book>_area/`)

Gốc: `CenterNetDetector._preprocess` thu trang về 1.024 px bằng `cv2.resize` INTER_LINEAR; thạch bản 3.204 / 2.789 / 2.289 px bị thu 3,1× / 2,7× / 2,2× →
răng cưa rụng nét mảnh. Khoá mới `books[].detector_resize: area` (INTER_AREA; mặc định `linear` = STT không đổi byte, md5 59e436d7… hai bên), cùng ckpt v1,
chạy `./run_pipeline.sh --book all-new --suffix _area` (0 API, 174 / 237 / 138 s) và `box_ref_eval.py --resize area` (`measure_out/box_ref_area/`, 24/24 invariants):

| Chỉ số (chốt pitch-linear → area) | LucVanTien1883 | KimVanKieu1884 (B1') | Chrestomathie1872 |
|---|---|---|---|
| **I5 thô** n_det==N (toàn sách) | 65,2 → **77,7 %** ✅ | 59,2 → **77,9 %** ✅ | 70,7 → 69,3 % (nhiễu, thu 2,2×) |
| box_ref 27 trang: I5 cột / tầng n==N / pitch ok50 / miss legacy | 63,7 → 79,6 / 77,4 → 88,5 / 98,1 → 99,3 / 0,5 → 0,1 | 54,4 → 74,8 / 71,1 → 85,4 / 97,0 → 98,6 / 3,0 → 1,0 | — |
| box_ref "cắt thân chữ" legacy · pitch | 11,0 → 10,8 · 9,5 → **10,6** | 3,8 → 3,1 · 2,6 → 2,5 | — |
| hộp thô h/p trung vị · điểm tin cậy | 1,14 → 1,16 · 0,35 → 0,39 | 1,11 → 1,13 · 0,32 → 0,36 | — |
| tier thô (GOLD/SYL/REVIEW) · labels_final GOLD / QUAR | y hệt · 9.667 → 9.665 / 14 → 18 | y hệt · 15.931 → 15.927 / 102 = 102 | y hệt · y hệt |
| box_source detector_low / ink_cut | 64 / 172 → 22 / 75 | 449 / 430 → 175 / 175 | 4 / 209 → 3 / 220 |
| cờ `n_det_mismatch` (dòng) | 4.978 → 3.156 | 9.238 → 4.988 | 2.408 → 2.559 |
| **GOLD ảnh / text_only** · ảnh export | 8.650 / 130 → **8.733 / 43** · 11.013 → 11.097 | 13.908 / 510 → **14.165 / 201** · 17.752 → 17.999 | 5.359 / 126 → 5.344 / 134 · 6.645 → 6.636 |
| F1 cross-col (census dòng) | 14 → 18 | 108 → 106 | 0 |
| khớp dị bản GOLD-ảnh sau (a)(b)(c) → B6 | 73,4 → 73,4 (n 2.595 → 2.607) · 79,1 → 79,2 | 1871: 81,3 → 81,1 (n 11.461 → 11.661) · 85,8 → 85,7; 1872: 78,5 → 78,3 | — |
| crop mọi ô: bleed · blank · truncated | 2.025 → **2.459** · 6 · 8 → 7 | 1.992 → 1.990 · 177 → 217 · 10 → 24 (124 ô mới đều → REVIEW bởi (c)) | 137 → 135 · 0 · 119 → 120 |
| **bleed trên ảnh export** (GOLD + SYL) | 18,0 → **21,7 %** (+429) | 10,5 = 10,5 % | 2,0 = 2,0 % |

Đọc: area chữa đúng thứ nó nhắm (I5 thô ≥ 75 % lần đầu ở 2 thạch bản, miss/extra giảm 3–5 lần, text_only còn 1/3, +340 ảnh GOLD), văn bản không đổi;
nhưng hộp cao/rộng hơn ≈ 2 % (+3 px, dịch lên 2 px) nên crop LVT ngậm thêm mực hàng xóm: cờ `bleed` đổi dồn vào ô có crop cao thêm > 4 px (+607 / −204),
GOLD-ảnh LVT bleed 18,0 → 21,6 %; "cắt thân chữ" pitch LVT +1,1 điểm (đổi cờ 179 / 142 ô, ≈ CI ±0,7). Chresto không lợi (thu 2,2×). **Theo luật đã đặt
("không chỉ số then chốt nào giảm") `_area` chưa thay bản chốt**; KVK là sách duy nhất không có chỉ số nào giảm → nếu chấp nhận đổi bleed lấy I5 (hoặc chỉ bật
cho KVK), đổi `detector_resize: linear → area` trong config sách đó và đổi tên `_area` thành chốt. Với detector v2 (Kaggle), mốc so sánh là **v1+area**
(`docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md` §0, `lab/i5_detector_v2/README.md`); v2 phải giảm cả bleed lẫn cắt (học lại kích thước hộp thạch bản).

### 3.4 Vòng 5 (23/09) — kênh kim `lang_type = 2` (Nôm) + luật đếm 6/8 + rào tầng DP = **BẢN CHỐT MỚI**

Cơ sở đo: `docs/CHOT_KENH_OCR_VA_QUY_HOACH_GAN_2026-09-23.md` (§1 kênh kim, §5–§6 quy hoạch gán) và
`docs/CHAN_DOAN_3_SACH_MOI_2026-09-23.md` (§1 sổ kế toán, §5 trần lợi ích). Ba thay đổi đi **cùng một lần chạy**
(`./run_pipeline.sh --book all-new`, ingest lại 331 lượt kim, ~60 phút):

1. **`books[].kim_lang_type: 2`** — body `/image-ocr` gửi `lang_type = 2` (Nôm) thay 1 (Hán). Mặc định vẫn là 1 nên
   đường STT không đổi byte (md5 59e436d7…). Cache `kim_raw/` tách theo tham số: `page_XXXX.json` (lang 1) và
   `page_XXXX_lt2.json` (lang 2) cùng tồn tại ⇒ quay lại tốn **0 lượt API**.
2. **Luật đếm QN 6/8 trước DP** (`--qn-count-rule`, chỉ lithograph): ba luật **giữ nguyên vị trí** mọi âm đọc được —
   `restore_unreadable` (âm mờ bị tesseract trả token không chữ cái, đúng chỗ → giữ lại làm âm không đọc được),
   `drop_verse_number` (câu chia hết 5 mà bộ đọc số lề không tách được → số câu in còn dính đầu dòng), `merge_unreadable`
   (một âm bị tách đôi thành 2 token rác) + `resplit_6_8` (cột đủ 14 âm mà chia ≠ 6/8). Chạm **165 tầng LVT** (sửa 138)
   và **126 tầng KVK** (sửa 97 + 2 cột resplit).
3. **`books[].tier_dp: true`** — DP chữ↔âm chạy 6↔6 rồi 8↔8 thay 14↔14, và `expected_tier_counts` lấy **luật 6/8**
   thay `len_odd` của QN khi cột đủ 14 âm.

| Chỉ số (vòng 4 lang 1 → **vòng 5 lang 2 + 6/8 + rào tầng**) | LucVanTien1883 | KimVanKieu1884 (B1') | Chrestomathie1872 |
|---|---|---|---|
| Ô sinh ra | 14.476 → 14.474 | 22.704 → **22.750** | 8.303 → **8.016** (−287) |
| Cột QN đủ 14 âm (ingest) | 85,5 → **97,4 %** | 92,8 → **98,4 %** | — (văn xuôi) |
| Cột kim đọc đủ 14 chữ | 98,1 → **93,0 %** ↓ | 95,8 → **98,5 %** | — |
| **M==N** | 83,8 → **90,4 %** | 88,8 → **96,9 %** | 69,5 → **44,0 %** ↓↓ |
| I5 thô `n_det==N` | 65,2 → **70,7 %** | 59,2 → **62,3 %** | 70,7 → 66,7 % ↓ |
| tier thô GOLD | 9.667 → **11.431** | 15.931 → **20.636** | 5.892 → **6.569** |
| **GOLD ảnh / GOLD_text_only** | 8.650 / 130 → **10.831 / 149** | 13.908 / 510 → **19.303 / 727** | 5.359 / 126 → **5.998 / 226** |
| SYLLABLE / REVIEW | 2.363 / 3.319 → **1.207 / 2.259** | 3.844 / 4.340 → **618 / 1.971** | 1.286 / 1.532 → **856 / 936** |
| QUARANTINE (F1 cross-col) | 14 → 28 | 102 → 131 | 0 → 0 |
| **Ảnh export** · dòng `labels.csv` | 11.013 · 11.143 → **12.038 · 12.187** | 17.752 · 18.262 → **19.921 · 20.648** | 6.645 · 6.771 → **6.854 · 7.080** |
| `count_source = pitch_ocr` (cột QN lệch) | 1.319 → **319** | 1.088 → **256** | 1.902 → 1.109 |
| crop `bleed` / tổng dòng | 17,8 → 18,0 % | 10,4 → 10,4 % | 2,02 → 1,96 % |
| **Khớp dị bản GOLD (bỏ ref PUA) — trước cổng** | 1916: 72,6 (n 2.796) → **78,0 %** (n 3.293) | 1871: 80,6 (n 11.463) → **90,2 %** (n 14.635) · **1872 độc lập**: 73,0 → **78,2 %** (n 14.976) | không có dị bản |
| **…B6 (labels_gated)** | 79,1 → **82,1 %** (n 3.068) | 1871: 85,8 → **92,0 %** · 1872: 78,5 → **79,9 %** | — |

**Vì sao nhận**: cổng đặt trước khi chạy (`CHOT_KENH_OCR §6 #1`) là "khớp dị bản tăng ≥ 5 điểm, chỉ số hộp không giảm".
Đạt ở cả hai sách có dị bản (**+5,4** LVT↔1916 và **+5,2** KVK↔**1872 độc lập**; 1871 +9,6 điểm nhưng chiều đó có phần
tự khẳng định vì 1871 là nguồn QN của B1'), cờ `bleed` không xấu đi ở cả ba sách, I5 thô **tăng** ở hai sách thạch bản.
KVK khớp 1871 92,0 % nay **vượt nền dị bản 1871↔1872 = 82,9 %**.

**Nợ còn lại — Chrestomathie1872**: GOLD +639 và ảnh +209 nên theo luật quyết định (giữ lang 1 *chỉ khi* GOLD giảm) thì
nhận; **nhưng** M==N 69,5 → 44,0 % và số ô sinh ra −287, mà sách này **không có dị bản** để kiểm chéo ⇒ bằng chứng chỉ
một chiều ("hợp từ điển"). Quay lại bằng đúng một khoá: `books[].kim_lang_type: 1` trong `config/pipeline_Chrestomathie1872.yaml`
rồi `./run_pipeline.sh --book Chrestomathie1872` (0 API).

**Đường dẫn**: bản chốt mới `dataset/<Book>/` + `prepared*/<Book>/dataset_out*`; bản vòng 4 giữ nguyên ở
`dataset/<Book>_v5_lang1/` + `prepared*/<Book>/dataset_out*_v5_lang1/`. Kế hoạch commit: `docs/KE_HOACH_COMMIT_VONG5_2026-09-23.md`.

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

**Phát hiện răng cưa (22/09 tối, §3.3; `HUONG_DAN_HUAN_LUYEN_I5` §0)**: cùng ckpt v1, chỉ đổi phép thu ảnh trang INTER_LINEAR → INTER_AREA
(`books[].detector_resize: area`): 27 trang val ảnh gốc ok50 95,0 → 98,1 %, tầng n==N thô 77,0 → 91,9 %, cắt 7,1 → 6,4 %, STT F1 0,8767 → 0,8785;
box_ref 27 trang/sách I5 cột 63,7 → 79,6 / 54,4 → 74,8 %; toàn sách I5 thô 77,7 / 77,9 %. Tức "điểm tin cậy thấp trên thạch bản" phần lớn là răng cưa,
không phải model. Chưa chốt vì bleed LVT (§3.3). Engine: khoá `detector_ckpt` (ckpt riêng sách, fail fast) + `detector_resize`, cache `(ckpt, resize, thr)`,
`seg_backend` ghi `+area`; STT không khai → byte-identical.

**Phương án B — gói sẵn cho Kaggle, CHƯA chạy dài** (`lab/i5_detector_v2/README.md`: `make_bundle.py` → zip 243 MB → notebook T4 ≈ 1–1,5 h →
`apply_v2.sh best.pt` đo v1 · v1+area · v2+area + build `_v2`; `best.pt` chỉ ghi khi vượt **v1+area** và qua guard STT; mốc `v1_baseline_val.json`): nhãn yếu `train_crop/data_lithograph/manifest_*.json` (268 trang thạch bản → **31.776 ô** ở 4.550/5.343 tầng + 793 vùng `ignore_boxes`; STT 445 trang + 14.172 bbox REVIEW làm ignore; chia page-disjoint theo sách, `--lobo`); trainer `train_v2_lithograph.py` (init v1, cân bằng miền 50/50, augment nền xám + kéo dọc ±10 %); eval mỗi epoch `eval_boxes_ref.py` (litho ok50, % tầng n == N của hộp **thô**, cắt thân chữ; STT F1 hồi quy — mốc v1: ok50 95,0 %, tầng n==N 77,0 %, STT F1 0,876). Thời gian: 20 epoch ≈ 5–8 h CPU Mac hoặc **≈ 1 h Kaggle T4** (đóng gói `pack_for_kaggle.py` + `data_lithograph/` + `prepared/{LVT,KVK}/pages`). Nghiệm thu: `detector_transfer.py` STT ≥ 90,1 % không giảm; `box_ref_eval.py` ok50 ≥ 98 %, cắt thân chữ giảm; build `--limit 10` `detector_low`/`ink_cut` giảm.

**Cách đo trung thực**: (i) không dùng `n_det == N` khi ép N (hằng đúng) — `n_det` giữ hộp thô; (ii) tách `*_honest(det_src_only)`; (iii) chỉ số không phụ thuộc tham chiếu duy nhất = mực chạm mép hộp thô (và crop flag thật, đã bão hoà); (iv) tham chiếu là proxy (kim tầng cụt/rộng, khe trong chữ ⿱), tầng không verified 7–9 % bị loại; (v) CI 27 trang ≈ ±3,5 điểm cột, ±0,7 điểm ô. **Không có hộp GT người → không đo được precision crop.**

## 7. Cách chạy và cách thêm sách (`docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` §2.0, §7)

```bash
./run_pipeline.sh --book all-new                    # 3 sách, B0→B6, ~9 phút, 0 API (cache kim_raw/); log logs/run_<Book>_*.log
./run_pipeline.sh --book KimVanKieu1884 --dry-run   # chỉ in lệnh; --suffix _x (bản so sánh, không ghi đè); --skip-ingest; --no-auto-precision
# KHÔNG dùng --no-api cho KVK: content -> formula, khác bản chốt.   ./run_pipeline.sh (không tham số) = đường STT cũ.
```
Sách thứ tư: `data/<BOOK>/pages/` + `SOURCE.md` → thêm vào `BOOKS` của `scripts/measure/` → `measure.py --book <BOOK>` → chọn `layout` (lithograph: cột = cặp 6⧺8, `ingest_lithograph_book --plan-only`; prose: `ingest_prose_book` + bảng truyện↔trang) → `config/pipeline_<BOOK>.yaml` (`books[]`: layout, n_columns, det_xmargin 0,05, quét `det_thr`, **`box_decoder: pitch`**; khối `run:`) → thêm vào `NEW_BOOKS_ALL` (run_pipeline.sh) và `CROSS_BOOKS` (auto_precision.py) nếu có dị bản → `./run_pipeline.sh --book <BOOK>` → viết `docs/CHAY_<BOOK>.md`. Selftest trước commit: §9.

## 8. Giới hạn thật và việc chưa làm

0. **(vòng 5, 23/09)** Bản chốt đã đổi sang kim `lang_type = 2` (§3.4). Ba việc còn mở ngay từ lần chạy này:
   (a) **Chresto M==N 44,0 %** — cột kim và cột QN lệch số chữ ở hơn nửa số cột, không có dị bản để kiểm;
   (b) **27 tầng LVT + 29 tầng KVK** vẫn `qn_count_unfixed` (thiếu âm, không luật nào chữa được mà không dịch chỗ);
   (c) `GOLD_text_only` tăng (130 → 149, 510 → 727, 126 → 226) vì có nhiều ô GOLD hơn đi qua cổng (a') `box_low_conf` —
   phần này chờ detector v2 / INTER_AREA chứ không phải lỗi kênh kim.

1. **Không có GT người** — mọi độ đúng là proxy: văn bản (mộc bản IHR, dị bản), hộp (ô tham chiếu tự động). Precision ảnh crop **không đo được**; GOLD-ảnh trong cột n_det≠N (2.910 / 5.430 / 1.502 ô) ước ≈ 4,7 % hộp lệch. Bộ mẫu mù `kiem_nguoi_*.py` để ngoài commit.
2. **I5 gốc chưa chữa trong bản chốt**: I5 thô 65,2 / 59,2 / 70,7 % < 75 %; gốc đã định vị là răng cưa (INTER_AREA cho 77,7 / 77,9 %, §3.3) nhưng chưa chốt vì bleed LVT +3,7 điểm; phương án B (Kaggle) chưa chạy; detector thích ảnh nhị phân hoàn toàn (KVK otsu 65,6 vs prepared 54,4 %) nhưng engine chưa tách "ảnh cho detector" khỏi "ảnh để crop".
3. **Blank KVK** 219 → 177 ô (c) hạ REVIEW: ngưỡng `enrich_crop_quality` hiệu chuẩn trên STT; 40 ô đo có mực trên ảnh gốc → có thể hạ oan.
4. **QN Chresto** tesseract lỗi âm ≈ 13 % + mất dòng → 1.239 ô `no_context` REVIEW; không phiên âm chuẩn để B1'; không dị bản → không (d)/B6; hộp không có tham chiếu (321 ô đổi hộp IoU < 0,3 chưa kiểm).
5. **B1'**: chính tả Bắc 676 dòng chưa vào export/DATASHEET; luật theo từng âm với kim trọng tài chưa làm; KVK lệch 5 câu Nôm/QN chưa định vị (`KET_QUA_DO_CUOI` §4).
6. **Mã**: `step2_align.py` (CLI riêng) vẫn ghim 9 cột; `remediation apply` chưa chặn thiếu `--out` trong mã; `run_pipeline.sh` STT chưa gọi B4'; bản chốt là kết quả `--suffix _pitch` đổi tên (đường dẫn `_pitch` còn trong `CHECKSUMS.txt`/`mechanism_gates_report.json`); `tools.selftest` 3 fail có sẵn ở HEAD; SILVER = 0 trên thạch bản là đúng (S3 học chữ thảo).
7. Không làm và không nên làm: self-training từ chính hộp detector (vòng tròn), dùng khớp dị bản sau (d) làm bằng chứng cổng (tự khẳng định), trích 99 % GOLD của STT cho sách mới.


## 10. Ba bộ còn lại của `data/` (23/09) — LucVanTien1916, TruyenKieu1872, TruyenKieuPhongTinhCoLuc

Chi tiết đầy đủ (đo bố cục, adapter, rủi ro, khuyến nghị): **`docs/CHAY_3_BO_CON_LAI_2026-09-23.md`**.
Tái lập: `./run_pipeline.sh --book all-ihr`; `.venv/bin/python scripts/measure/ihr_endtoend_eval.py --book all`;
`.venv/bin/python scripts/measure/ptcl_layout.py --kim-pages 6`.

### 10.1 Bảng đầy đủ 9 bộ trong `data/`

| # | Bộ | Loại chữ | Nguồn QN dùng để gán | Vai trò | Ô sinh | GOLD ảnh | Ảnh export | Độ đúng đo được |
|---|---|---|---|---|---|---|---|---|
| 1 | SachThanhTruyen2 | viết tay | QN trang đối diện (VietOCR) | huấn luyện (kho chính) | — | — | — | không có nhãn người |
| 2 | SachThanhTruyen4 | viết tay | như trên | huấn luyện | — | — | — | không có nhãn người |
| 3 | SachThanhTruyen11 | viết tay | như trên | huấn luyện | — | — | — | không có nhãn người |
| 4 | LucVanTien1883 | thạch bản | QN in cùng sách (tesseract) | **giao nộp** | 14.474 | **10.831** | 12.038 | khớp dị bản 1916: 78,0 % |
| 5 | KimVanKieu1884 (B1') | thạch bản | 1871 + QN in (tesseract) | **giao nộp** | 22.750 | **19.303** | 19.921 | khớp dị bản 1871/1872: 90,2 / 78,2 % |
| 6 | Chrestomathie1872 | thạch bản văn xuôi | QN truyện (tesseract) | **giao nộp** | 8.016 | **5.998** | 6.854 | không có dị bản |
| 7 | **LucVanTien1916** | mộc bản | **phiên âm NGƯỜI** (Nôm Foundation) | **TẬP ĐÁNH GIÁ** | **13.761** | **8.783** | **8.872** | **nhãn người: GOLD 97,2 %** |
| 8 | **TruyenKieu1872** | mộc bản | **phiên âm NGƯỜI** | **TẬP ĐÁNH GIÁ** | **22.499** | **13.465** | **13.513** | **nhãn người: GOLD 98,5 %** |
| 9 | **TruyenKieuPhongTinhCoLuc** | chép tay | chỉ có **dị bản** 1871/1872 | chưa dùng | — | — | — | nền dị bản 1871↔1872 = 71,3 % |

Số của bộ 4–6 là **bản chốt vòng 5** (§3.4). Bộ 7–8 dùng `prepared_ihr/`, `dataset/<Book>/`, config riêng,
`manifest.json` mang `evaluation_only: true`, và **không** nằm trong `--book all-new`.

### 10.2 Điều quan trọng nhất học được: **mức ảnh gửi kim, chứ không phải chất lượng bản in, quyết định kim đọc đúng hay sai**

`auto_precision/ihr` (22/09) đo kim trên **patch từng CÂU** của hai bộ IHR và kết luận "kim thô chỉ đúng **49,2 %**
(LVT1916) / **42,3 %** (Kiều1872)". Gửi **cả trang** cùng 10 trang ấy, cùng `lang_type = 2`: **97,2 % / 98,2 %**
(`measure_out/<Book>/ihr_layout/summary.json`). Điều này ăn khớp với `CHOT_KENH §2` (cắt xuống mức tầng-cột làm kim
rụng gần nửa số chữ) và **hạ bệ mốc "kim kém trên mộc bản"** — mốc ấy phải gỡ khỏi các bảng đã viết.
Kéo theo: bảng xếp hạng nút thắt của `CHAN_DOAN_3_SACH_MOI §5` (kim = 77–81 % mất mát) đúng **cho thạch bản với QN
OCR**, nhưng khi QN đúng và ô cột đúng thì kim chỉ còn gây ~3 % lỗi. **Suy luận này bắc qua hai miền in** (mộc bản ↔ thạch
bản) nên là **giả thuyết có số đỡ, chưa phải kết luận**: muốn chắc thì đo lại LVT1883/KVK1884 với QN thay bằng phiên âm
chuẩn và ô cột thay bằng ô vẽ tay trên một mẫu trang — nếu precision ở đó cũng nhảy lên ~97 % thì ngân sách lỗi của
3 sách giao nộp đúng là nằm ở **QN (tesseract) + dò cột** chứ không ở kênh OCR Hán-Nôm.

### 10.3 Vì sao TruyenKieuPhongTinhCoLuc chưa chạy

Bố cục thật (đo `ptcl_layout`, xác nhận bằng chính chuỗi chữ kim): tờ đôi 2000×1820, ~14 cột có chữ (7 cột × 2 nửa tờ,
đúng như `data/README.md`), **ranh giới tầng là một đường ngang chung cả tờ ở y ≈ 490–555**; tầng trên = **lời bình chữ
Hán cỡ nhỏ** (cao/chữ ≈ 61 px, thường 2 cột con), tầng dưới = **ĐÚNG MỘT CẶP LỤC BÁT 14 chữ** (cao/chữ ≈ 77 px; kim
trung vị đúng 14,0 và 85,2 % cột nằm trong [13,15]). ⇒ `ingest_lithograph_book` (tầng trên = câu lục 6) **không dùng
được**; cần adapter riêng. Thêm ba chặn đã định lượng: không có số câu neo; kim đọc chép tay chỉ đạt thu hồi LCS
**35–44 %** so bản 1871 trong khi **trần của phép so ấy là 71,3 %**; và QN dị bản làm cận trên lớp "nhãn sai hệ thống"
lên tới ~29 % vị trí mà **không đo được trên chính C** (không có phiên âm của R.987). Đường chạy đã định lượng
(~1 ngày công + ~130 lượt kim) ở `CHAY_3_BO_CON_LAI §3.4`.

## 9. Chỉ mục tài liệu và commit

| Nội dung | Tệp |
|---|---|
| Hướng dẫn chạy (lệnh, cổng nghiệm thu 3 sách, thêm sách) | `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` |
| I5: răng cưa INTER_AREA (§0), chẩn đoán, pitch_decode, huấn luyện v2 Kaggle, cách đo | `docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md`; `lab/i5_detector_v2/README.md`; `measure_out/box_ref/`, `measure_out/box_ref_area/` |
| Lần chạy từng sách (đầu tệp = chốt cuối pitch) | `docs/CHAY_LVT1883_2026-09-21.md`, `CHAY_KVK1884_B1_2026-09-22.md` (chính thức), `CHAY_KVK1884_2026-09-21.md`, `CHAY_CHRESTO1872_2026-09-22.md` |
| Lịch sử vòng 1–2 và phương án tự động (đã gộp vào đây) | `docs/BAO_CAO_TONG_THE_SACH_MOI_2026-09-22.md`, `docs/PHUONG_AN_TU_DONG_2026-09-22.md` |
| Số đo dữ liệu + đặc tả | `docs/KET_QUA_DO_CUOI_2026-09-21.md`, `docs/PIPELINE_SACH_MOI_2026-09-20.md`, `docs/PIPELINE_FACTS.json`, `measure_out/{SUMMARY.json,REPORT.md}` |
| **Ba bộ còn lại (23/09): đo bố cục, chạy A/B làm tập đánh giá, độ đúng theo nhãn người, vì sao chưa chạy C** | **`docs/CHAY_3_BO_CON_LAI_2026-09-23.md`** |
| Kế hoạch commit | vòng 1 `KE_HOACH_COMMIT_2026-09-21.md` (đã commit fb345a29b1…853cadfd9c), vòng 2 `KE_HOACH_COMMIT_VONG2_2026-09-22.md` (đã commit 6b8e215576, e06f32dce7, 4e0a314bca), vòng 3 `KE_HOACH_COMMIT_VONG3_2026-09-22.md` (đã commit 59fde272c4, 6575d4d764, 10e6fa1b3f, 8c08591e9a), vòng 4 `KE_HOACH_COMMIT_VONG4_2026-09-22.md`, vòng 5 `KE_HOACH_COMMIT_VONG5_2026-09-23.md`, **vòng 6 `KE_HOACH_COMMIT_VONG6_2026-09-23.md` (chưa commit: adapter IHR + 3 bộ đo mới + 2 config)** |

Hồi quy/selftest lúc chốt (22/09 chiều): STT 3 trang md5 59e436d7… hai bên; `./run_pipeline.sh --dry-run` 6 bước; book_layout 79/79 · ingest_lithograph 61/61 · ingest_prose 33/33 · mechanism_gates 94/94 · pitch_decode 22/22 · verses_ref_fix 19/19 · phase1_engine 253/0 · tools 139/3 (có sẵn) · code_facts 18/18; `git status -- dataset_out data prepared/SachThanhTruyen*` trống.
Vòng 4 (22/09 tối, worktree HEAD 8c08591e9a ↔ mã mới): STT 3 trang md5 **59e436d7641fa849bb6759868ac29259** hai bên (344/346 tệp giống byte, 2 tệp chỉ khác đường dẫn REPO); `--dry-run` 6 bước; book_layout **100/100** · phase1_engine 253/0 · pitch_decode 22/22 · mechanism_gates 94/94 · tools 139/3 · code_facts 18/18 (pin book_layout.py 50 → 61).
