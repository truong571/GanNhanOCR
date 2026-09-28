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
| 2 | text_only | `CNT` số chữ OCR của cột ≠ số âm QN (**trừ profile handwriting** — §11) · `BC` trượt so với hộp chữ kim / vis (L16: bc_k06v1, bộ khác bc_k05dxv05, cấu hình chọn NGOÀI sách) | TN3 |
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


## 11. Profile `handwriting` — chính sách riêng cho CHỮ VIẾT TAY (28/09, policy `2026-09-28.2`)

**Vì sao.** Trên hai bản chép tay Borg có nhãn người (tập đánh giá), bước gold_exact cũ chỉ cho ok 112 ô (B18 106 + B34 6):
≈ 80 % ô GOLD bị hạ `text_only` bởi **CNT** (số chữ kim của cột ≠ số âm QN), mà CNT gắn 95–97 % ô GOLD Borg và KHÔNG phân biệt
đúng/sai. Profile là một khối config (`profiles.handwriting`) áp cho bộ viết tay (stt2/stt4/stt11 + B18/B34); **6 bộ in/khắc
còn lại = profile printed = hành vi cũ từng ô** (kiểm: dưới). Nhãn người chỉ dùng để ĐO/hiệu chuẩn, không vào quyết định.
Mã/số: `lab/thu_nghiem_anh_chu/TN5_viet_tay/h01…h05`, đầu ra `measure_out/_thu_nghiem_anh_chu/TN5/` (0 API, CPU, không mở ảnh).

### 11.1 Sự thật hai vế trên Borg (h01)

Ô GOLD tự động → chỉ số âm của trang (khoảng âm cột của adapter + syl_idx, như `borg_endtoend_eval`) → (câu, vị trí) → chữ người
(vế NHÃN, V1+) → chỉ số chữ người của trang → **hộp người** của `dataset/_BORG_NHAN_NGUOI` (vế ẢNH: tâm hộp mực CROP CHUẨN ∈ hộp
người; chỉ tin hộp mức keep_high trở lên, trượt ±1 ≤ 1,9 %). Kiểm: chữ người theo hộp == chữ người theo câu ở 100 % ô ghép được.
Kết quả: GOLD tự động B18 đúng NHÃN 91,2 % nhưng crop đúng VỊ TRÍ chỉ **74,3 %** (B34 80,6 %) — lệch ±1 ô đối xứng (786/778
ô), như nhau dù nhãn đúng hay sai (74,4 % vs 72,6 %): nhãn chữ viết tay đúng nhờ căn văn bản, còn hộp gán theo bước cột nên
trượt khi số hộp ≠ số âm. **Đúng hai vế của GOLD tự động Borg ≈ 69 %** (n 7.072 ô đo được).

### 11.2 Tín hiệu nào phân biệt trên chữ viết tay (h03, Borg gộp, AUC hướng "điểm cao = đúng"; < 0,5 = giá trị cao đi với SAI)

| tín hiệu (có ở cả STT) | AUC nhãn | AUC khe | AUC hai vế | trong ô đã chứng (simg) | ghi chú |
|---|---:|---:|---:|---:|---|
| **CNT** | 0,493 | 0,494 | 0,495 | 0,486 | gắn 95–97 % ô; hai vế 69,0 % (cờ) vs 77,0 % (n 183 không cờ) → KHÔNG phân biệt |
| BC (hộp kim/vis) | 0,485 | 0,377 | 0,398 | 0,465 | phân biệt khe (61,5 % vs 79,2 %), trong simg gần như không (99,1 vs 99,4 %) |
| ady | 0,476 | 0,367 | 0,390 | 0,403 | |
| **vis_z** | 0,582 | **0,955** | **0,911** | 0,386 | mạnh cho khe, không thêm gì sau simg |
| **lobo_pT / lobo_pL** (bộ kiểm viết tay LOBO) | 0,727 / 0,731 | 0,940 / 0,934 | **0,949 / 0,943** | 0,49 / 0,55 | |
| simg (lobo_cert q 0,00015 ∧ ≥ 3 nguyên mẫu) | cờ | | | | hai vế **99,25 %** khi chứng nhận (n 1.855) |
| \|n_det − n_qn\| | 0,488 | 0,244 | 0,289 | 0,437 | phân biệt khe (ngược) |
| f_two / tall_new / f_ink / … (cờ crop chuẩn) | ≈ 0,50 | ≈ 0,50 | ≈ 0,50 | | không phân biệt nhãn/khe, NHƯNG phân biệt "một chữ": hộp mực phủ ≥ 50 % một hộp người khác ở 52,5 % ô f_two vs 7,9 % (tall 56 vs 10 %, f_ink 67 vs 12 %) → GIỮ |
| dp_ratio / n_match (chỉ adapter Borg) | 0,527 / 0,491 | 0,540 / 0,596 | 0,538 / 0,579 | 0,742 / 0,725 | STT không có → không dùng |

### 11.3 Thiết kế + chọn ngưỡng ĐĂNG KÝ TRƯỚC (h04, `prereg_handwriting.json`, 2026-09-28 15:17:45, sha mã `158708e6…`)

Profile: luật A + M-OCR giữ; cờ ảnh/hộp "một chữ" (A0/B0) giữ; **CNT bỏ**; cổng khe ∈ {none, bc, ad_dn0, vis0}; bộ kiểm viết
tay ở mức q ∈ thang 14 mức của `hand_tables` (FAR hiệu chuẩn sẵn theo biến thể), ≥ 3 nguyên mẫu người. Quy tắc chọn trên phần
HỌC: khả thi ⇔ hai vế ≥ 99,5 % ∧ nhãn ≥ 99,0 % ∧ đủ ô đo; lấy cấu hình nhiều ô ok nhất; không khả thi → hai vế cao nhất.
Phần kiểm: LOBO Kinh→DungLy, LOBO DungLy→Kinh, CV 5 khối trang trong Kinh. Khai báo trung thực: phép AUC gộp hai sách (11.2) và
xem nhanh simg cũ đã làm TRƯỚC đăng ký (nên việc bỏ CNT không hoàn toàn "giữ ngoài"); cơ chế h04 thử trên nhãn xáo.

**Kết quả chọn:** trên toàn Kinh KHÔNG cấu hình nào khả thi (tốt nhất: q 0,00015 + bc 99,48 %, q 0,00015 + none 99,46 %) → dự
phòng "hai vế cao nhất" → **`slot_gate: bc`, `hand_q: 0.00015`, `n_hum_min: 3`** (= BC + mức q cũ; khác cũ DUY NHẤT ở bỏ CNT).
Chọn trên DungLy (354 GOLD): q 0,001 + ad_dn0 (khả thi trên 73 ô đo được, 0 lỗi — rất nhiễu).

### 11.4 Kết quả Borg (đo trên nhãn người; hai vế = V1+ ∧ tâm crop chuẩn ∈ hộp người keep_high+; CI 95 % bootstrap cụm trang)

| | ô ok | hai vế | CI | nhãn (mọi ô có gt) | CI | ghi chú |
|---|---:|---:|---|---:|---|---|
| Hiện tại B18 (trước profile) | 106 | 100 % (42/42) | — | 100 % (70/70) | — | quá ít ô |
| Hiện tại B34 | 6 | 1/1 | — | 1/1 | — | |
| **Profile B18** (trong mẫu — cấu hình chọn trên Kinh) | **1.120** | 99,48 % (4 lỗi/765) | [98,92–99,88] | 99,18 % (8/970) | [98,53–99,70] | "một chữ" 96,3 % |
| **Profile B18 — CV 5 khối (GIỮ NGOÀI, quy tắc chọn)** | 1.847 | **99,11 %** (12/1.353) | [98,57–99,61] | **98,68 %** (22/1.663) | [98,13–99,20] | hộp 'khong' (nhiễu) 93,8 % |
| **Profile B34 — LOBO Kinh→DungLy (GIỮ NGOÀI)** | **88** | 97,56 % (1/41) | [90,63–100] | 100 % (65/65) | — | n quá nhỏ |
| LOBO DungLy→Kinh (tham khảo) | 2.862 | 98,93 % (21/1.969) | [98,41–99,41] | 97,79 % | [97,17–98,38] | chọn trên 73 ô |

Mục tiêu "≥ 99 % hai vế trên phần giữ ngoài": **đạt ở điểm ước lượng CV Kinh (99,11 %) nhưng CHƯA chứng được** — cận dưới
98,57 %, vế nhãn trên mọi ô có gt 98,68 %, B34 97,56 % (n 41). Tập hộp keep lạc quan (hộp người chọn nhờ encoder v1/v2 — ô dễ).
Nói gọn: ô ok chữ viết tay ≈ **98,5–99,5 %** đúng hai vế, thấp hơn sách in IHR (L16 99,46 %, TK 99,87 %).

### 11.5 STT (SUY ĐOÁN — không có sự thật)

| bộ | GOLD | ok trước | **ok sau** | thêm (đều từ lý do CNT) |
|---|---:|---:|---:|---:|
| stt2 | 17.678 | 2.316 | **2.864** | +548 |
| stt4 | 17.440 | 1.495 | **1.872** | +377 |
| stt11 | 17.589 | 2.646 | **3.750** | +1.104 |
| tổng | 52.707 | 6.457 | **8.486** | +2.029 (0 ô ok bị mất) |

Ước lượng chuyển từ Borg (SUY ĐOÁN): STT dùng biến thể bộ kiểm **Kinh** (như B34), nên số gần nhất là B34 LOBO 97,56 %
[90,6–100] (n 41) và CV Kinh 99,11 % [98,57–99,61] → đọc **≈ 97,5–99 %**, không trích như số đo. Khác biệt làm số chuyển kém
chắc: STT đọc kim `lang_type = 1` (Borg lt2); người chép khác (bộ kiểm học trên nét Borg Kinh); STT là văn vần 9 cột có QN OCR
riêng từng cột (Borg văn xuôi, cột ↔ âm do adapter DP) → cơ chế trượt hộp khác; STT có luật A/M-OCR (Borg không có ô nào dính);
CNT trên STT chỉ gắn 22–32 % ô GOLD (Borg 95–97 %) nên tín hiệu CNT trên STT có thể khác Borg — việc bỏ CNT ở STT dựa trên lập
luận "bộ kiểm ảnh↔chữ đã chặn ô trượt" (Borg: simg ⇒ đúng khe 99,4 %), chưa đo được trên STT.

### 11.6 Kiểm

- 5 bộ in/khắc (Chr, L83, KVK, L16, TK; 66.247 ô): so từng `cell_uid` với bản trước profile (sha256 `da5bdc9a…`) — `gold_exact`,
  `reason`, `crop_chuan`, `crop_chuan_md5`, `crop_chuan_128_md5`, `crop_status`: **0 ô lệch**; md5 crop chuẩn của 24.244 ô ok ở cả
  hai bản: 0 lệch (h05_compare.py → `measure_out/_thu_nghiem_anh_chu/TN5/compare.json`). Bộ viết tay: 0 ô ok bị mất.
- Số ô ok Borg của pipeline == số h04 dự đoán (1.120 / 88).
- selftest gói 85/85 (thêm `_profile_tests`: kiểm config, CNT chỉ bỏ ở ô viết tay, cổng khe bc/none/vis0, n_hum_min/hand_q theo
  profile, bộ in KHÔNG đổi quyết định với tín hiệu ngẫu nhiên 3.000 ô × 3 cổng khe, config thật + sha bản đăng ký, md5 ↔ TN4 N/A).
- `gold_exact_eval` thêm: `profile_hw_lobo_cert_tai_lap_tai_hand_q`, `profile_hw_cong_bo_khong_ha_o`, `printed_khong_mang_ly_do_profile_hw`.
- **Bản chạy 28/09** `./run_pipeline.sh --merge --yes --verify` (0 API): gộp 173.973 dòng (bất biến 24/24); gold_exact policy
  `2026-09-28.2`, ok **27.369** / GOLD 130.895 (trước 24.244; +3.125 = STT 2.029 + Borg 1.096); `gold_exact.csv` sha256 `3a391bf6…`; nghiệm thu **8/8 PASS**
  (`gold_exact_eval` 28/28, `borg_endtoend_eval`, `measure.py --all --report-only`, `--check` bộ gộp); `labels.csv` của 13 tệp
  (`dataset/*/labels.csv`) trùng sha256 trước lượt chạy; IHR ô ok không đổi: L16 **99,46 %** [99,09–99,74] (2.758 ô), TK
  **99,87 %** [99,80–99,93] (10.322 ô).
- Sửa hiển thị: invariant `md5_crop_chuan_vs_TN4` — bộ không có tham chiếu TN4 (B18, B34: n = 0) là **N/A**, không làm
  PASS = False; log in `md5 ↔ TN4 v2: PASS (8 bộ, 400 ô mỗi bộ; N/A không có tham chiếu TN4: B18, B34)`.

### 11.7 Giới hạn

1. Mục tiêu 99 % chưa chứng được (11.4); cấu hình giao là dự phòng của quy tắc đăng ký trước. Nếu muốn nhiều ô hơn: `slot_gate:
   none` cho B18 2.025 ô ok ở 99,46 % trong mẫu (gần như bằng bc) — muốn đổi phải ĐĂNG KÝ LẠI (không dò trên cùng dữ liệu).
2. Vế ảnh đo bằng hộp người do máy gióng (keep_high+: trượt ≤ 1,9 %), trên tập hộp chọn nhờ encoder (thiên dễ); vế "một chữ" chỉ
   gián tiếp (hộp mực vs hộp người).
3. DungLy nhỏ (354 GOLD) → LOBO K→D và mọi số B34 rất rộng CI; LOBO D→K chọn trên 73 ô.
4. STT hoàn toàn suy đoán (11.5); Borg vẫn là tập ĐÁNH GIÁ (không vào tập huấn luyện).
5. Cổng khe `ad_dn0` có trong lưới đăng ký nhưng không được chọn nên chưa cài vào gói (`policy.SLOT_GATES` = bc/none/vis0).
