# GOLD chính xác — bước `gold_exact` trong pipeline (27/09/2026)

> Tài liệu cho tác giả luận văn. Mọi con số dưới đây do script sinh (`python -m pipeline.gold_exact`,
> `scripts/measure/gold_exact_eval.py`); bản chạy chốt: `./run_pipeline.sh --book all --yes` ngày 27/09, **cập nhật
> 28/09 sau rà soát độc lập** (`./run_pipeline.sh --merge --yes --verify`, policy `2026-09-28`, §7, §10).
> Tự động hoàn toàn, **0 gọi API**, không người chấm. Nguồn thiết kế: `lab/thu_nghiem_anh_chu/TN1…TN4` (§9).

## 1. Định nghĩa

Một ô `tier == "GOLD"` là **GOLD chính xác** (`gold_exact = ok`) khi máy chứng được CẢ HAI vế:

1. **ảnh** là đúng MỘT chữ, của đúng khe (ô) — kiểm trên **crop chuẩn** (§4) + hình học cột/hộp;
2. **chữ** trong ảnh đúng là nhãn — kiểm bằng bộ kiểm ảnh↔chữ (ngưỡng TN1) và, ở sách có dị bản người, bằng văn bản người.

Bước này **chỉ gắn trạng thái**: không sửa nhãn, không đổi `labels.csv` (sha256 kiểm trước/sau và so với
`CHECKSUMS.txt` của bước gộp), không đụng `pipeline/align_engine/`, `core/`, `data/`. Nó chạy **sau** bước gộp
`dataset/_ALL/` và chỉ thêm tệp.

## 2. Bốn trạng thái và `evidence_level`

| `gold_exact` | nghĩa | ảnh giao |
|---|---|---|
| `ok` | qua mọi cổng tự động | **crop chuẩn** `crops_chuan/…` (+ `crops_chuan_128/`); cột `image` gốc vẫn nằm trong `labels.csv` |
| `text_only` | nhãn giữ, nhưng ảnh không chắc là đúng một chữ của đúng khe | không (dùng ảnh gốc thì tự chịu) |
| `uncertified` | không có bằng chứng ảnh↔chữ đủ mạnh, hoặc dị bản người không chứng nhãn | không |
| `review` | luật A (ảnh của ô khác, nhãn rescue, cầu tự dạng, văn bản yếu) hoặc M-OCR t50 | không |

`evidence_level` = mức chắc của **độ chính xác** ô ok (không phải của trạng thái):

| mức | bộ | nghĩa |
|---|---|---|
| `do_tren_nhan_nguoi` | L16 (LucVanTien1916), TK (TruyenKieu1872) | ĐO trên nhãn người IHR-NomDB (tập đánh giá) |
| `uoc_luong` | KVK (KimVanKieu1884), L83 (LucVanTien1883) | ƯỚC LƯỢNG qua văn bản người của dị bản (bộ ước lượng TN3) |
| `suy_doan` | stt2, stt4, stt11, Chr (Chrestomathie1872) | SUY ĐOÁN — không có sự thật nào trên máy |

## 3. Luật (luật đầu tiên khớp thắng) — `pipeline/gold_exact/policy.py`

| thứ tự | trạng thái | luật (`reason`) | nguồn |
|---|---|---|---|
| 1 | review | `A0a_anh_cua_o_khac` (ô rescue dùng tệp crop của ô khác) · `A1_rescue_nhan_khong_do_OCR_doc` · `A2a_cau_tu_dang` (s1_inter_s2_similar) · `A2b_van_ban_yeu` (am_sua_dau, lowp, corpus_reading*) | luật A, chính sách v3 |
| 1 | review | `M_ocr_t50` (sách in, bộ có M-OCR: L83, KVK) · `M_ocr_t50_STT` (STT; v3 xếp human_check → ở đây review) | M-OCR t50, gate_v2 |
| 2 | text_only | crop chuẩn trắng / cắt nét / hai chữ / mực bất thường so với nhãn / bleed–truncated trên crop chặt mới / quá cao; một hộp hai cột; hai hộp chồng nặng | A0_new ∪ B0_new (TN4) |
| 2 | text_only | `CNT` số chữ OCR của cột ≠ số âm QN · `BC` trượt so với hộp chữ kim / vis (L16: bc_k06v1, bộ khác bc_k05dxv05, cấu hình chọn NGOÀI sách) | TN3 |
| 3 | uncertified | dị bản người chống nhãn (TK, KVK, L83) | TA, r5 |
| 3 | uncertified | bộ kiểm ảnh↔chữ dưới ngưỡng TN1 (τ = 0,995, LOBO; sách in chấm **crop cũ**) · STT: bộ kiểm viết tay LOBO-sách không chứng nhận / < 3 nguyên mẫu người | TN1, TN4 "lai" |
| 3 | uncertified | TK/KVK/L83: nhãn không được văn bản người chứng (TA chỉ dùng văn bản người của DỊ BẢN; KHÔNG nhìn GT IHR — sửa N6, §10) | TA_OK |
| 4 | ok | còn lại | = "lai" TN4 − luật A − M-OCR |

Bất biến từng ô: **ok = lai ∧ ¬luật A ∧ ¬M-OCR** (`decomposition_ok`, `ok_bang_lai_tru_luatA_tru_MOCR`).
Ngưỡng/cờ ở `config/gold_exact.yaml` (`version` = `policy_version` ghi vào từng dòng, sha256 16 hex = `config_sha16`).

**core_loss bị bác (quyết định 27/09).** Chốt "crop chuẩn cắt vào lõi chữ" (> 25 % mực lõi ±0,3 bước bị bỏ) được
**đăng ký trước** (2026-09-27 00:39:49, sha mã `de81acb40e83`) rồi đo một lần trên nhãn người: nó làm mất **46 ô ok đều
đúng hai vế** (L16 20 + TK 26, 0 ô sai) và chỉ bắt **6/61** ô 'damaged' TN4 ở L16 (TK 1/2; 0/8 ô damaged nằm trong "lai").
Lợi ích âm → bỏ khỏi cổng (`core_loss.gate: false`), giữ `core_loss`/`core_loss_flag` làm cột thông tin. Hệ quả:
ok tăng **+55** so với lượt run1 (stt2 +2, stt4 +4, KVK +3, L16 +20, TK +26) — đúng bằng số ô "lai ngoài luật A/M-OCR"
mang cờ core_loss (dự kiến ban đầu ~60; số đo là 55).

## 4. Crop chuẩn (ảnh giao của ô ok)

Thuật toán `pipeline/gold_exact/crop_chuan.py` = bản chép nguyên `cclib_v2` của TN4 (PARAMS đóng băng, md5 kiểm
trong selftest; crop khớp md5 byte TN4 v2 trên 400 ô/bộ): ranh giới dọc = trung điểm tâm hai hộp kề, dải ngang
xc ± 0,75·w, Otsu cục bộ, bóc nét kẻ, gán thành phần mực, **khung VUÔNG** quanh hộp mực (lề 10 %), điểm ảnh lấy từ
**trang gốc** (5 sách `crop_source: original`; STT: trang đã xử lý như pipeline), mực ngoại lai tô màu giấy; kèm bản
128×128. Tệp: `dataset/_ALL/crops_chuan/<book_set>/<book>/<page>/c<cột>_n<nom_idx>_s<syl_idx>.png`.
`gold_exact.csv` ghi cả hai đường (`image` gốc + `crop_chuan`, `crop_chuan_128`) và md5 (`image_file_md5`,
`crop_chuan_md5`, `crop_chuan_128_md5`). v2 là **hậu kiểm** (TN4 §1, §8).

## 5. Kết quả theo bộ (bản chạy 28/09, policy `2026-09-28` — sau sửa N6)

| Bộ | evidence_level | GOLD | **ok** | text_only | uncertified | review | ok trước sửa (27/09) |
|---|---|---:|---:|---:|---:|---:|---:|
| stt2 | suy_doan | 17.678 | **2.316** | 3.910 | 9.074 | 2.378 | 2.316 |
| stt4 | suy_doan | 17.440 | **1.495** | 4.544 | 9.142 | 2.259 | 1.495 |
| stt11 | suy_doan | 17.589 | **2.646** | 5.881 | 7.210 | 1.852 | 2.646 |
| Chr | suy_doan | 4.238 | **325** | 2.425 | 1.488 | 0 | 325 |
| L83 | uoc_luong | 10.910 | **642** | 838 | 9.276 | 154 | 642 |
| KVK | uoc_luong | 19.958 | **3.628** | 645 | 15.491 | 194 | 3.628 |
| L16 | do_tren_nhan_nguoi | 11.587 | **2.758** | 639 | 8.190 | 0 | 2.758 |
| TK | do_tren_nhan_nguoi | 19.554 | **10.322** | 1.114 | 8.118 | 0 | 9.746 |
| **Tổng** | | **118.954** | **24.132** (20,3 %) | 19.996 | 67.989 | 6.837 | 23.556 |

Chỉ TK đổi (sửa N6, §10): 976 ô TK không có GT IHR trước bị `ta = na` nay được chấm bằng Kiều 1871 LVD
(746 attested · 131 contradicted · 99 unattestable; 28 ô trúng cổng (d) vẫn `na`) → **+576 ô ok** (uncertified 8.694 → 8.118).

Lý do lớn nhất: `uncertified` do bộ kiểm ảnh dưới ngưỡng (sách in 34.337; STT viết tay 20.811 + thiếu nguyên mẫu 4.615),
`text_only` do CNT (15.118), `review` do rescue (2.818) và cầu tự dạng (2.962) — bảng đủ ở `dataset/_ALL/GOLD_EXACT.md` §5.

## 6. Độ chính xác trên nhãn người IHR (CHỈ đo được ở L16/TK)

Đúng hai vế = V1+(nhãn, chữ người) ∧ khe của CROP CHUẨN = 1 (hai mô hình khe của harness); CI 95 % bootstrap cụm trang
(B = 2000). Chỉ BÁO (không là cổng PASS/FAIL).

| Bộ | ô ok | có GT | đúng hai vế V1+ | CI 95 % | strict (trùng hẳn) | lỗi (nhãn/ảnh) |
|---|---:|---:|---:|---|---:|---|
| L16 | 2.758 | 2.757 | **99,46 %** | [99,09–99,74] | 99,24 % | 15 (13/8) |
| TK | 10.322 | 9.746 | **99,87 %** | [99,80–99,93] | 99,81 % | 13 (13/0) |

Trước/sau sửa N6: số đo trên ô CÓ GT **không đổi** (TK 9.746 ô có GT, 99,87 % [99,80–99,93] cả hai lượt; L16 y hệt) — đúng
như phải thế, vì trước sửa mọi ô TK không GT đều bị loại. **576 ô ok mới của TK KHÔNG đo được** (không có GT người); bằng
chứng của chúng là văn bản người Kiều 1871 LVD chứng nhãn + cùng các cổng ảnh như ô có GT — độ chính xác ở đây là ngoại suy.

Các bộ khác KHÔNG đo được; bộ ước lượng TN3 cho tập mẹ "lai" (TN4 §5, trước khi trừ luật A/M-OCR):
KVK 99,81 % [98,02–99,92], L83 99,81 % [94,95–99,92] (ƯỚC LƯỢNG); stt2 98,39 % [96,40–99,24], stt4 98,26 %
[96,19–99,19], stt11 98,47 % [96,48–99,28], Chr 99,10 % [97,13–99,77] (SUY ĐOÁN — cận dưới đều < 97,2 %).

## 7. Chạy và nghiệm thu

```bash
./run_pipeline.sh --book all --yes           # 8 bộ -> gộp -> gold_exact (--publish) -> nghiệm thu (0 API)
./run_pipeline.sh --merge --yes --verify     # tối thiểu: gộp lại + gold_exact + nghiệm thu (bản chạy 28/09)
./run_pipeline.sh --book all --yes --gold-exact off   # bỏ bước (bộ gộp sẽ KHÔNG có gold_exact.csv)
./run_pipeline.sh --summary-only             # bảng 8 bộ + 4 cột GX (ok/text_only/uncert./review)
.venv/bin/python -m pipeline.gold_exact --all-dir dataset/_ALL --out measure_out/_gold_exact --publish   # chạy riêng
.venv/bin/python scripts/measure/gold_exact_eval.py   # 25 bất biến -> measure_out/gold_exact/summary.json
.venv/bin/python -m pipeline.gold_exact --selftest    # selftest gói (62 phép)
.venv/bin/python -m pipeline.gold_exact.export_assets --out /tmp/x --check-against models/gold_exact   # dựng lại tài sản từ kho bền, so sha
```

Đầu ra: `dataset/_ALL/{gold_exact.csv, GOLD_EXACT.md, crops_chuan/, crops_chuan_128/}` (+ `CHECKSUMS.txt` sinh lại,
dòng `labels.csv` giữ nguyên) và `dataset/<Bộ>/gold_exact.csv` (bản lọc; `crop_chuan` trỏ `../_ALL/…`). Bước gộp
dọn đầu ra gold_exact cũ (không bao giờ để `gold_exact.csv` lệch `labels.csv`). Thư mục làm việc
`measure_out/_gold_exact/` (summary.json, pkl); cache `prepared/_gold_exact/` (≈ 3,8 GB; có cache ≈ 3 phút, 0 API).
Nghiệm thu (`--verify`) thêm `gold_exact_eval` (FAIL cứng) trước `measure.py --all --report-only`.
gold_exact CHỈ chạy khi bước gộp chạy cùng lượt (`--book all` hoặc `--merge`); `--book all --no-merge` và `--book <Bộ>` không
chạy nó. Sau `--book <Bộ>`, `dataset/<Bộ>/gold_exact.csv` bị bước export xoá → `gold_exact_eval` chỉ CẢNH BÁO (SKIP
`ban_theo_bo_du_mat`) và gợi ý `./run_pipeline.sh --merge`.

**Bản chạy 28/09** (`./run_pipeline.sh --merge --yes --verify`, 0 API): gộp 140.867 dòng (bất biến 20/20);
`labels.csv` sha256 `6eed2c83…` **không đổi**; bước gold_exact 458 s (khoá cache mới có sha mã align_engine → dựng lại crop
chuẩn + hộp kim một lần; md5 crop ↔ TN4 v2 400/400 mỗi bộ); `gold_exact.csv` sha256 `370bcd07…`, thiết bị `mps`; nghiệm thu
7/7 PASS (`gold_exact_eval` 25/25, 0 SKIP); `measure.py --all --report-only` 195 PASS / 0 FAIL / 2 mềm; kiểm độc lập của người
rà soát (`indep_check`, `indep_ihr`, thêm `indep_ta` tính lại TA từ bảng căn gốc không qua gói) 0 lệch.

**Bản chạy chốt 27/09** (`./run_pipeline.sh --book all --yes`, 28 phút, 0 API): 16/16 tệp nhãn (8 bộ + gated/final)
trùng byte bản trước; `dataset/_ALL/labels.csv` sha256 `6eed2c83…` (không đổi); hồi quy STT 3 trang md5
`59e436d7641fa849bb6759868ac29259`; bước gold_exact 248 s (cache); `gold_exact.csv` sha256 `84d3f291…` trùng
lượt chạy riêng trước đó (tất định); nghiệm thu 7/7 PASS (thêm `gold_exact_eval` 22/22); `measure.py --all --report-only`
192 PASS / 0 FAIL / 2 mềm (trước: 170 / 0 / 2). Dung lượng thêm: `dataset/_ALL` +≈ 0,63 GB (crop 582 MB + csv 51 MB),
các `dataset/<Bộ>/gold_exact.csv` ≈ 49 MB.

## 8. Giới hạn

1. **Không có sự thật cho STT/Chr** (suy đoán) và KVK/L83 chỉ ước lượng qua dị bản — đừng trích "ok" ở các bộ này như
   tập đã kiểm chứng. TN3 §0: chỉ TK đạt mục tiêu 99,5 % một cách chắc chắn; L16 sát ngưỡng (cận dưới 99,09 %).
2. **Số ô ok dao động mạnh theo chọn ngưỡng TN1** (TN4 §5.1, bootstrap trang tune B = 300): số ô giữ L16 1.343–6.670,
   TK 9.200–12.137 (5 %–95 %) — con số ok là của MỘT điểm vận hành (τ = 0,995 đăng ký trước), không phải hằng số.
3. **core_loss bị bác** (§3); ~0,5 % ô L16 bị crop chuẩn cắt vào lõi nét (TN4 §0.2/§3: 61 ô 'damaged') vẫn **chưa có cờ
   bắt được** — 8 ô trong số đó nằm trong ok của L16 (đã tính trong 99,46 % ở §6). "Một chữ" không có sự thật người ở bộ
   nào; khe người IHR chỉ là thước đo gián tiếp.
4. **Đọc lại STT bằng kim lt2 (H7) chưa làm** vì cần gọi API; TN3 §6 cho thấy kể cả xoá hết lỗi riêng của STT, cận bi
   quan vẫn ≤ 97,31 % → không đủ để nâng STT lên "đo được".
5. Crop chuẩn v2 và cấu hình "lai" là **hậu kiểm** (chọn sau khi thấy v1); bộ kiểm ảnh vẫn học trên crop cũ.
6. Tái lập tín hiệu (nhúng lại trên MPS) lệch "lai" TN4 ≤ 2 ô/bộ (`compare_ref`, GOLD_EXACT run1 §4). Tất định trên
   **MPS** (config `device: mps`, cột `device`); chạy CPU có thể lật vài cờ sát ngưỡng.
7. **576 ô ok mới của TK (sửa N6) không có GT người** → không đo được; con số 99,87 % chỉ nói về 9.746 ô có GT.
8. Độ phủ tham chiếu dị bản (invariant mềm `ref_coverage`, sàn ở config): TK 0,900 · KVK 0,813 · L83 0,358 (TA);
   M-OCR KVK 0,813 · L83 0,291 — ô không có tham chiếu thì không bao giờ ok ở TK/KVK/L83.

## 9. Tham chiếu

- `lab/thu_nghiem_anh_chu/TN1_cong_kiem/KET_QUA.md` — bộ kiểm ảnh↔chữ, ngưỡng τ = 0,995 LOBO (C5/C2).
- `lab/thu_nghiem_anh_chu/TN2_vong_sua_hop/KET_QUA.md` — vòng sửa hộp (kết luận: tắt mặc định).
- `lab/thu_nghiem_anh_chu/TN3_tat_ca_bo/KET_QUA.md` — H1…H7 trên 8 bộ, bộ ước lượng, CNT/BC/TA.
- `lab/thu_nghiem_anh_chu/TN4_crop_chuan/KET_QUA.md` — crop chuẩn v1/v2, cấu hình "lai" (§5, §7), độ nhạy (§5.1).
- Mã: `pipeline/gold_exact/` (README trong `__init__.py`), `config/gold_exact.yaml`, `models/gold_exact/MANIFEST.json`,
  `scripts/measure/gold_exact_eval.py`; lượt run1 (có core_loss): `measure_out/_gold_exact_run1/GOLD_EXACT.md`.

## 10. Sửa sau rà soát độc lập (28/09) — không đổi nhãn, không đổi `labels.csv`

| mục | vấn đề | sửa | kiểm |
|---|---|---|---|
| **N6** (rò rỉ) | `signals_text.text_attested` bỏ qua ô TK không có GT người IHR (`has_gt`) → 1.004 ô GOLD TK bị `ta = na` chỉ vì thiếu nhãn ĐO | TA của TK tính cho MỌI ô căn được với Kiều 1871 LVD (bảng căn `t01_build` dùng trang/cột/syl_idx/âm của bộ dữ liệu, không dùng GT); tài sản `ta_refs.pt` dựng lại KHÔNG còn cột GT; policy `2026-09-28` | selftest `ta_tk_no_gt_dependency`, `ta_refs_asset_has_no_ihr_gt`; invariant `ta_refs_no_ihr_gt`; `indep_ta` 0 lệch; KVK/L83 không có phụ thuộc tương tự (tham chiếu = căn câu thạch bản) |
| **N1** (tài sản) | nguồn dựng lại tài sản chỉ nằm ở /private/tmp | kho bền `measure_out/_gold_exact_assets_src/` (55 tệp, 1,39 GB, `SHA256SUMS` + `SOURCE.json`, gitignored); `export_assets` mặc định đọc kho + kiểm sha từng tệp; MANIFEST ghi `source_store` + hướng dẫn sao lưu; gói kiểm sha TOÀN BỘ MANIFEST lúc khởi động | dựng lại từ kho: 13/13 tài sản trùng sha MANIFEST (`--check-against`) |
| **N2** | gộp lại / publish không xoá `dataset/<Bộ>/gold_exact.csv` cũ (trỏ crop đã xoá) | `per_book.clean_per_book` gọi từ `merge_datasets` (lúc dọn) và `publish.clean`; khối README/DATASHEET của bộ đổi theo tệp có/không (marker) | selftest merge 27/27, gói `pub_clean_per_book_N2` |
| **N3** | gold_exact chạy cả khi `--book all --no-merge` | chỉ chạy khi bước gộp chạy cùng lượt (`MERGED_THIS_RUN`); invariant tập GOLD `dataset/<Bộ>/labels.csv` == dòng `dataset/<Bộ>/gold_exact.csv` (publish chặn cứng, eval kiểm) | dry-run 5 biến thể; `ban_theo_bo_trung_tap_GOLD_labels_bo` |
| **N4** | sau `--book X` thiếu bản lọc → eval FAIL cứng | cảnh báo mềm + gợi ý `--merge` (ít rủi ro hơn để `--book` tự gộp: gộp đụng MỌI bộ) | SKIP `ban_theo_bo_du_mat` |
| **N5** | thiếu ảnh → nhúng ảnh trắng | thiếu/rỗng/không giải mã ảnh ô GOLD hoặc trang (crop chuẩn, view B) → dừng lỗi | invariant `all_gold_images_present`, `crop_no_page = 0`; selftest `n5_*` |
| nhỏ | — | cột `device` + config `device: mps`; CSV `%.17g` (tái lập cờ đúng bit, eval dung sai 0); chặn `--out` theo thành phần đường dẫn (không chặn nhầm `dataset_out/`); cache hỏng → tính lại, ghi nguyên tử; sha mã align_engine vào khoá cache crop chuẩn + kim; invariant độ phủ `ref_coverage`; thay hằng `crop_md5_eq_cache_record` bằng kiểm thật (`crop_md5_cache_eq_build`, `crop_md5_dest_verify` đọc lại 48.264 tệp); tài liệu nêu lt2 STT chưa làm + biên độ theo ngưỡng + bản lọc theo bộ chỉ khi có tệp | selftest gói 62/62 |

