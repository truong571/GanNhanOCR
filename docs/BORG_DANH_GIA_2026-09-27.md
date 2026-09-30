# Borg chép tay — TẬP ĐÁNH GIÁ chữ viết tay của pipeline tự động (27/09/2026)

> Hai bản chép tay chữ Nôm Công giáo của Thư viện Vatican (Borgiano Tonchinese) đưa vào **pipeline tự động chính** với
> vai trò giống hai bộ IHR-NomDB: sách **có nhãn người** (Excel phiên âm), pipeline chạy lên chỉ để **ĐO** — không giao nộp,
> không trộn vào tập huấn luyện. Mọi con số dưới đây do script sinh (`scripts/measure/borg_endtoend_eval.py`,
> `python -m pipeline.gold_exact`, `scripts/measure/gold_exact_eval.py`).

| mã | sách | Borg | trang chữ | âm QN người | câu | câu "đếm bằng" (#chữ = #âm) | chữ người ∈ R(âm) |
|---|---|---|---:|---:|---:|---:|---:|
| B18 | `SachKinhThayCaBinh` (Philiphê Bỉnh) | Borg.Tonch.18 | 529 | 119.133 | 1.584 | 1.252 (87,8 % âm) | 96,8 % |
| B34 | `SachDungLyHoThan` | Borg.Tonch.34 | 112 | 24.061 | 329 | 277 (93,4 % âm) | 98,3 % |

"chữ người ∈ R(âm)" = tỉ lệ cặp (chữ Nôm người, âm người) của câu đếm bằng mà chữ nằm trong tập chữ của âm theo
`Dict/QuocNgu_SinoNom.csv` — **trần** của luật GOLD (`kim ∈ R(âm)`) nếu kim đọc đúng 100 %.

## 0. TRẠNG THÁI (cập nhật 28/09) — ĐÃ CÓ SỐ ĐO NHÃN MÁY (§3, §6)

Cache kim đủ cả hai sách (`ingest_borg_book --status`: B18 **529/529** trang, sổ 531 lượt / 529 ok / 1 lỗi tài khoản; B34
**112/112**, sổ 112 lượt / 112 ok). Hai bộ đã chạy trọn B1–B6, vào bộ gộp (`evaluation_only`), qua `gold_exact` và
`borg_endtoend_eval` (7/7 bất biến PASS mỗi sách, `api_calls = 0`). Từ nay `./run_pipeline.sh --book all --yes --borg require`
chạy Borg TỪ CACHE; `scripts/clean_rebuild_all.sh` chặn cứng nếu cache thiếu. Đoạn dưới là **lịch sử 27/09** (trước khi có tài khoản).

Lượt thử API duy nhất (27/09, trang `page_0001` của B18, tham số `lang_type 2`): đăng nhập tự động **thất bại** (máy chủ không
trả cookie `token`), token dự phòng `SN_OCR_TOKEN` trong `.env` **đã hết hạn**, và bước tải ảnh bị từ chối
`t_access_denied — "Truy cập bị từ chối!"`. Theo quy tắc của nhiệm vụ (lỗi tài khoản → DỪNG, không đoán, không thử đường vòng
như Guest), **không gọi thêm lượt nào**. Tổng lượt: **1 POST đăng nhập + 1 POST tải ảnh, 0 lượt nhận dạng (image-ocr)**; ghi ở
`prepared/_auto/SachKinhThayCaBinh/kim_calls.json`. Cache kim: **0/529** (B18), **0/112** (B34).

Hệ quả: chưa có nhãn máy cho Borg ⇒ chưa có độ đúng nhãn GOLD, độ đúng `gold_exact = ok`, độ đúng kim trên chữ viết tay. Toàn bộ
mã đã sẵn và đã chạy thông trên **fixture khói** (§5); khi API dùng lại được, chỉ cần các lệnh ở §2 (không sửa mã).

**Việc của người:** cấp lại tài khoản/tokens HCMUS (`SN_OCR_USERNAME`/`SN_OCR_PASSWORD` đăng nhập được, hoặc `SN_OCR_TOKEN` còn hạn
trong `.env`), rồi chạy §2.

## 1. Kiến trúc (đúng khuôn IHR + sách mới)

| khâu | tệp | ghi chú |
|---|---|---|
| ingest | `pipeline/tools/ingest_borg_book.py` | ảnh gốc 720 px (`data/<Sách>/*.jpg`) → `prepared/_auto/<Sách>/{pages,pages_denoised,detected,transcriptions,kim_raw,manifest.json,kim_calls.json}` |
| cấu hình | `config/pipeline_SachKinhThayCaBinh.yaml`, `config/pipeline_SachDungLyHoThan.yaml` | `run.ingest: borg`, `paths.data_dir: prepared/_auto`, `run.dataset_out: prepared/_auto/<Sách>/dataset_out`, `kim_lang_type: 2`, `box_decoder: pitch`, `mechanism_gates`, `qn_count_gate: review`, `crop_source: original` |
| bí danh | `config/pipeline_MSS_Borg_tonch_{18,34}.yaml` | giữ tệp (đã commit) nhưng thêm `run.alias_of` → `run_pipeline.sh --book MSS_Borg_tonch_18` **từ chối** (không chạy trùng) |
| chạy | `run_pipeline.sh` | nhánh `ingest=borg` (không đi `ingest_prose_book` — adapter đó viết riêng cho Chrestomathie); `--book all-borg`; `--borg auto\|require\|off` |
| đóng dấu | `pipeline/tools/mark_eval_dataset.py --kind borg` | `dataset/<Sách>/{evaluation_only.json,TAP_DANH_GIA.md}` → bộ gộp `evaluation_only = 1` (không chia tập) |
| gộp | `run_merge` | Borg nối SAU 8 bộ cũ, CHỈ khi bộ đã xuất + đóng dấu + `manifest.gates.kim_cache_complete` |
| GOLD chính xác | `pipeline/gold_exact/` + `config/gold_exact.yaml` | B18/B34, LOBO-sách (§4), `evidence_level = do_tren_nhan_nguoi` |
| đo | `scripts/measure/borg_endtoend_eval.py` | đăng ký trong `measure.py` (bước `borg_endtoend`) + `run_pipeline.sh --verify` + B6 |

**Adapter (không đọc nhãn Nôm người).** Từ Excel chỉ đọc `img_id`, `sentence_id`, `ChuQN_txt` (kiểm tiêu đề theo tên; cột
`SinoNom_Char` không được đọc; không mở `prepared/<Sách>/nom_transcriptions/`). Thứ tự và đánh số `page_XXXX` y hệt
`ingest_borg_tonch` (bất biến `page_map_eq_ingest_borg_tonch` PASS cả hai sách). Kim **toàn trang ×1** gửi JPG gốc, tham số từ
`books[].kim_*` (lang_type 2), cache `kim_raw/page_XXXX_lt2.json` theo (md5 ảnh gửi, tham số) → chạy lại 0 lượt. Ngân sách
`--budget` (số trang), sổ `kim_calls.json` (nối thêm, không ghi đè), thử lại 2 lần giãn 5 s/15 s (ngoài phần thử lại HTTP của
`core/ocr/ocr_api.py`), dừng ngay khi: từ chối truy cập / đăng nhập hỏng / token hết hạn / tài khoản không hoạt động / rơi về
Guest, hoặc 3 trang liên tiếp hỏng. `--ocr cache` = không bao giờ gọi API (thiếu → mã 3). Cột: hộp kim → chữ chia đều theo
chiều cao → gom theo TÂM x (dung sai 0,6 × bề ngang hộp trung vị) → gộp đoạn bị tách (chồng x ≥ 50 %, không chồng y) và mảnh
≤ 2 chữ sát cột kề; đọc PHẢI→TRÁI, trên→dưới. QN = âm tiết NGƯỜI của trang; DP đơn điệu của `ingest_prose_book`
(khớp +3 khi chữ ∈ R(âm), chèn/xoá −1, hai đầu QN tự do); âm thừa đầu/cuối trang gán cột đầu/cuối nếu ≤ max(2, 30 % số chữ);
cột < 3 khớp hoặc tỉ lệ < 0,25 → dòng giữ chỗ `khongkhop` (chỉ REVIEW). `dp_ratio` từng cột → cổng (e) `qn_count_gate: review`.
Trang không có kim KHÔNG được ghi (engine bỏ qua) và có cờ.

## 2. Cách chạy

```bash
.venv/bin/python -m pipeline.tools.ingest_borg_book --status                 # độ phủ cache kim 2 sách (0 API)
./run_pipeline.sh --book all-borg --dry-run                                   # in chuỗi lệnh (ngân sách = số trang thiếu)
./run_pipeline.sh --book all-borg --yes     # thiếu cache -> GỌI kim đúng số trang thiếu (641 lần đầu), rồi B2–B6 (0 API)
./run_pipeline.sh --merge --yes --verify    # gộp 8 bộ + Borg (evaluation_only) -> gold_exact (LOBO) -> nghiệm thu
./run_pipeline.sh --book all --yes          # từ đó về sau: Borg chạy TỪ CACHE (0 API) trong --book all
.venv/bin/python scripts/measure/borg_endtoend_eval.py --book all            # số đo (0 API) -> measure_out/<Sách>/borg_endtoend/
```

`--book all` **không bao giờ tự gọi API cho Borg**: `--borg auto` (mặc định) — cache đủ → chạy từ cache + gộp; chưa có trang nào
(như hiện nay) → bỏ Borg kèm cảnh báo, 8 bộ cũ chạy như cũ; thiếu MỘT PHẦN → dừng cứng (như cache STT). `--borg require` → thiếu là
dừng; `--borg off` → không bao giờ đụng Borg. `--book <Borg>` không `--yes` thì hỏi `GOI_KIM` trước khi gửi; `--no-api` = chỉ cache.

## 3. Phép đo (`scripts/measure/borg_endtoend_eval.py`, 0 API)

Ghép ô ↔ chữ người qua **vị trí âm tiết**: ô (trang, cột, `syl_idx`) → chỉ số âm của trang J = `syl_span[0]` của cột (adapter
ghi) + `syl_idx` → (câu, vị trí) theo số âm từng câu (làm sạch y hệt adapter) → chữ Nôm người cùng vị trí; chỉ câu đếm bằng.
Thước: **strict** (trùng hẳn) · **V1+** (biến thể Unihan/OpenCC/kJapanese — `gold_exact.common.var_eq_plus`) · **V1+ + 4 cặp** quy
ước người phiên đã biết (người phiên gõ 𠸜 cho hình 先, 𢧚/年, 𠀧/巴, 𧘇/意 — `measure_out/_audit_2026-09-26/r5/borg_quality`) ·
**+ 11 cặp** (bảng `PAIRS` của q14, độ nhạy) — báo cả số trước/sau khi coi các cặp là tương đương. Theo tầng và cho ô
`gold_exact = ok`; CI 95 % bootstrap **cụm trang** (B = 2000) + Wilson. Kim thô: chuỗi chữ kim cả trang (cùng cách gom cột
của adapter) căn **LCS** với chuỗi chữ người cả trang → precision = khớp/chữ kim, recall = khớp/chữ người (strict, V1+, +4 cặp),
độc lập với phép ghép QN của pipeline. Bất biến: ánh xạ trang == ingest_borg_tonch · adapter không đọc cột Nôm ·
manifest `evaluation_only` · ô khoá duy nhất · âm ghép == âm người (≥ 99 %) · ≥ 100 ô GOLD có chữ người · cache kim đủ trang.
Thiếu nhãn máy / cache → SKIP. **Hiện (28/09): 7 PASS / 0 FAIL / 0 SKIP mỗi sách.**

Số đo 28/09 (`measure_out/<Sách>/borg_endtoend/summary.json`, labels = `prepared/_auto/<Sách>/dataset_out/labels_gated.csv`;
dòng CNT do `scripts/bao_cao_tong_hop.py` tính từ `borg_endtoend/cells.csv` ⋈ `dataset/_ALL/gold_exact.csv`). CI = bootstrap cụm
trang trừ khi ghi Wilson. Mức chắc: **ĐO** (so chữ Nôm người từng vị trí).

| đại lượng | B18 `SachKinhThayCaBinh` | B34 `SachDungLyHoThan` |
|---|---|---|
| ô pipeline (mọi tầng) · GOLD · GOLD có chữ người | 90.747 · 11.587 (12,8 %) · 10.294 | 19.348 · 354 (1,8 %) · 278 |
| kim thô cả trang vs người (LCS): precision strict / V1+ / +4 cặp | 49,2 / 52,2 / 52,5 % | 32,7 / 34,7 / 35,0 % |
| kim thô cả trang: recall strict / V1+ | 37,7 / 39,9 % | 26,4 / 28,0 % |
| **kim ở ô** (chữ kim của ô vs chữ người cùng vị trí): strict [CI] · V1+ | **50,3 % [49,4–51,1]** · 53,4 % (n 73.415) | **37,7 % [36,2–39,2]** · 39,7 % (n 13.274) |
| GOLD strict [CI] | 85,7 % [84,9–86,5] | 84,9 % [81,0–89,7] |
| **GOLD V1+ [CI]** | **91,2 % [90,5–91,8]** | **91,7 % [88,5–95,1]** |
| GOLD +4 cặp / +11 cặp | 91,4 / 91,5 % | 92,1 / 92,4 % |
| GOLD_text_only V1+ | 94,0 % (n 268) | 100 % (n 19) |
| REVIEW V1+ (không giao) | 63,7 % (n 47.170) | 51,5 % (n 10.132) |
| `gold_exact`: ok / text_only / uncertified | 106 / 11.149 / 332 | 6 / 342 / 6 |
| **`gold_exact = ok` có chữ người: đúng V1+ (= strict)** | **70/70** [94,8–100] Wilson | 1/1 [20,7–100] Wilson |
| GOLD V1+ theo cờ CNT của gold_exact: CNT = 1 · CNT = 0 (Wilson) | 91,1 % [90,5–91,6] (n 9.942) · 94,6 % [91,7–96,5] (n 352) | 91,5 % [87,6–94,3] (n 272) · 100 % (n 6) |
| trần luật GOLD (chữ người ∈ R(âm)) | 96,8 % | 98,3 % |

Đọc bảng: (1) kim trên chữ viết tay đúng chỉ ~50 % (B18) / ~38 % (B34) ở ô, nhưng luật GOLD (`kim ∈ R(âm)` + căn QN) lọc
còn 12,8 % / 1,8 % số ô và nâng độ đúng lên ~91 % (V1+) — ~6 điểm dưới V1+ là quy ước người phiên (strict 85,7 / 84,9 %).
(2) `gold_exact = ok` ĐÚNG 70/70 ở B18 (cận dưới Wilson 94,8 %) nhưng chỉ phủ 106/11.587 ô GOLD (0,9 %); ở B34 chỉ 1 ô ok có chữ
người ⇒ chưa kết luận được. (3) Cờ **CNT không phân biệt** ô sai/đúng trên chữ viết tay: nó gắn cho 96,6 % (B18) / 97,8 % (B34)
ô GOLD, nhóm CNT = 1 đúng 91,1 % ≈ toàn GOLD 91,2 %; nhóm CNT = 0 cao hơn (94,6 %) nhưng nhỏ (352 ô, CI chạm 91,7 %) — tác dụng
thực của CNT trên Borg là dồn gần hết GOLD sang `text_only` (chữ đúng ~91 %, ảnh chưa chứng nhận).

## 4. GOLD chính xác cho Borg — bộ kiểm chữ viết tay LOBO THEO SÁCH

Bộ kiểm chữ viết tay của STT (`r5/hand_lobo`) được HỌC TRÊN CHÍNH CROP BORG, nên với Borg phải giữ ngoài theo sách:

| bộ chấm | biến thể | mô hình học trên | nguyên mẫu người | hiệu chuẩn (ngưỡng q = 0,00015) |
|---|---|---|---|---|
| STT (như cũ) | Kinh | Borg Kinh fold 0–3 (+ IHR tune phần A) | Borg Kinh | Borg Kinh fold 4 |
| **B34** DungLy | Kinh | Borg **Kinh** | Borg **Kinh** | Borg **Kinh** |
| **B18** Kinh | **DungLy** (tài sản mới) | Borg **DungLy** fold 0–3 (4.436 crop) | Borg **DungLy** | Borg **DungLy** fold 4 |

Tài sản mới `models/gold_exact/hand_{model,tables}_{T,L}_DungLy.pt` dựng bằng `export_assets` từ kho bền (thêm 13 tệp nguồn
`r5/hand_lobo/out/*DungLy*` + `stt_cert_DungLy.pkl`, kho 68 tệp); `export_assets` **dừng** nếu nguyên mẫu hoặc tập hiệu chuẩn lấy
từ sách khác sách học (MANIFEST: `hand_{T,L}_DungLy_proto_books = hand_…_cal_books = [SachDungLyHoThan]`); 13 tài sản cũ trùng
sha256; tái lập cert h03 = 1,0. Borg dùng CÙNG công thức STT (`lobo_cert ∧ n_hum ≥ 3`), lý do riêng
`U_Borg_bo_kiem_viet_tay_LOBO_khong_chung_nhan` / `U_Borg_thieu_nguyen_mau_nguoi_LOBO`. **Không** dùng `p_wood`/`viss` (nguyên
mẫu verifier_ft chứa crop Borg cả hai sách → để trống cho Borg), không M-OCR, không TA (không có dị bản). `vis_z` (chỉ vào cờ BC
trượt) dùng **hiệu chuẩn STT** (chữ viết tay, chọn ngoài sách — Borg chưa có hiệu chuẩn riêng). Trạng thái `evidence_level =
do_tren_nhan_nguoi`. Kỳ vọng từ phòng thí nghiệm (h03, crop người Borg, KHÔNG phải ô pipeline): Kinh→DungLy FRR 17,4 %, FAR
đồng âm 0,032 %, FAR trượt 0; DungLy→Kinh FRR **64,8 %** (mô hình DungLy học ít crop → phủ thấp), FAR đồng âm 0,009 %, trượt
0,013 % ⇒ ô ok của B18 sẽ ít, của B34 nhiều hơn.

## 5. Kiểm đã chạy (0 API)

- `ingest_borg_book --selftest` 25/25 (gom cột, DP, QN, Excel 3 cột, ngân sách, thử lại, dừng khi từ chối truy cập/Guest,
  sổ nối thêm, cache theo md5 + tham số); `borg_endtoend_eval --selftest` 10/10 (LCS = LCS kinh điển trên chuỗi ngẫu nhiên);
  `mark_eval_dataset --selftest` 18/18; `gold_exact --selftest` 70/70 (+8 phép Borg: LOBO khác sách, tài sản, `_auto`, mức chắc,
  lý do, 8 bộ không đổi).
- **Fixture khói** (thư mục tạm, KHÔNG phải số đo — "kim giả" = hộp chiếu mực chia đều + chữ người): B34 trang 5–6 + B18 trang
  10–11 chạy trọn adapter → `build_dataset` (crop từ ảnh gốc) → remediation → cổng → export → `mark_eval --kind borg` → gộp
  cùng 8 bộ (22/22 bất biến gộp) → `gold_exact` (B34 → biến thể Kinh, B18 → DungLy; `crop_src_original_fallback = 0`,
  `decomposition_ok`) → `borg_endtoend_eval`: nhãn giả = chữ người cho **100 % strict** ⇒ phép ghép âm → chữ người đúng.
- Hồi quy 8 bộ: dry-run `--book all` chỉ thêm cảnh báo Borg + 1 bước nghiệm thu; `dataset/_ALL/labels.csv` sha256 `6eed2c83…`
  **không đổi**, 6 book_set trùng sha từng dòng, 16/16 tệp nhãn bộ/trung gian trùng sha; khi CÓ Borg trong bộ gộp (fixture) 8 bộ
  cũ trùng từng dòng cả ở `labels.csv` lẫn `gold_exact.csv`. `gold_exact.csv` xuất lại (vì `config/gold_exact.yaml` thêm mục Borg):
  **chỉ cột `config_sha16` đổi** (118.954 dòng, mọi trạng thái/lý do/điểm trùng; ok 24.132). STT 3 trang md5
  `59e436d7641fa849bb6759868ac29259` (556 dòng) không đổi. `./run_pipeline.sh --verify` **8/8 PASS** (thêm `borg_endtoend_eval`),
  `gold_exact_eval` 25/25, `measure.py --all --report-only` 219 PASS / 0 FAIL / 2 mềm. `scripts/run_all_selftests.sh`: 9 lỗi có
  sẵn như mốc 24/09 (1.192/9); `pipeline.gold_exact.selftest` trước đây "KHÔNG CHẠY ĐƯỢC" qua `-m` (khối `__main__` đặt giữa tệp,
  trước `_ta_tests`) — đã dời xuống cuối tệp → 70 passed.
- `--no-api` với sách Borg chưa có cache: ingest dừng mã 3, 0 lượt API (đã chạy thật với B34).

## 6. So với mộc bản IHR và với STT

**Mộc bản (ĐO, `measure_out/<IHR>/ihr_endtoend/summary.json`) vs viết tay (ĐO, §3)** — cùng pipeline, cùng luật GOLD:

| đại lượng | L16 mộc bản | TK mộc bản | B18 viết tay | B34 viết tay |
|---|---|---|---|---|
| kim ở ô (mọi ô có chữ người) | 95,2 % [94,9–95,6] | 97,5 % [97,2–97,7] | 50,3 % (strict) · 53,4 % (V1+) | 37,7 % · 39,7 % |
| GOLD / ô pipeline | 84 % (11.587 / 13.760) | 87 % (19.554 / 22.499) | 12,8 % | 1,8 % |
| GOLD đúng — trùng hẳn | 98,05 % [97,78–98,29] | 98,61 % [98,44–98,77] | 85,7 % [84,9–86,5] | 84,9 % [81,0–89,7] |
| GOLD đúng — tính cả dị thể (IHR: bỏ PUA + dị thể · Borg: V1+) | 99,68 % | 99,99 % | 91,2 % [90,5–91,8] | 91,7 % [88,5–95,1] |
| `gold_exact = ok`: đúng (IHR: hai vế V1+ ∧ crop · Borg: V1+) | 99,46 % (n 2.757) | 99,87 % (n 9.746) | 70/70 | 1/1 |

(Hai thước "tính cả dị thể" không hoàn toàn trùng định nghĩa: IHR loại PUA + dị thể theo bảng của `ihr_endtoend_eval`, Borg dùng
`var_eq_plus`; so theo hàng "trùng hẳn" là chặt nhất.) Trên chữ viết tay kim yếu hẳn (−45…−60 điểm), luật GOLD giữ được ít ô
hơn nhiều và độ đúng GOLD thấp hơn mộc bản ~7–13 điểm; `gold_exact = ok` là bộ lọc duy nhất đạt mức mộc bản nhưng phủ rất ít.

**STT** chỉ có số **SUY ĐOÁN** (không có sự thật người): bộ ước lượng TN3 cho tập "lai" stt2 98,39 % [96,40–99,24], stt4 98,26 %
[96,19–99,19], stt11 98,47 % [96,48–99,28] (`docs/GOLD_CHINH_XAC_2026-09-27.md` §6). Borg là chữ viết tay Công giáo cùng loại có
nhãn người từng chữ và là **bằng chứng ĐO ĐƯỢC đầu tiên** trên chữ viết tay: GOLD Borg đúng 91,2 / 91,7 % (V1+), thấp hơn rõ so với
số suy đoán của STT. Hai con số KHÔNG so thẳng được (tập "lai" của STT ≠ toàn GOLD; kim trên STT có thể khác Borg vì nét/khổ ảnh;
QN của Borg là phiên âm người nên Borg còn là **cận trên**), nhưng đủ để coi số STT ~98 % là lạc quan cho tới khi có nhãn người STT.
Bảng tổng hợp 10 bộ (mức chắc ĐO / ƯỚC LƯỢNG / SUY ĐOÁN): `docs/BAO_CAO_TONG_HOP_<ngày>.md` (`scripts/bao_cao_tong_hop.py`).

## 7. Giới hạn

1. (28/09: đã có số đo, §3.) Luôn báo cả strict và V1+ ± cặp quy ước; khác biệt giữa hai con số (~6 điểm ở GOLD) phần lớn là quy
   ước người phiên (A3), không phải lỗi máy. B34 chỉ có 278 ô GOLD và 1 ô `gold_exact = ok` có chữ người ⇒ CI rộng.
2. QN là phiên âm NGƯỜI theo trang ⇒ số đo là cận trên của phương pháp; chỉ câu đếm bằng (87,8 % / 93,4 % âm) được chấm.
3. Chưa đo tham số kim riêng cho chữ viết tay (`font_type 2 = Viết tay` chưa thử; dùng `lang_type 2` như config). Borg có 1–2
   bản duy nhất 720 px trên máy (≈ 60 px/cột) — thấp hơn IHR.
4. `vis_z` của Borg dùng hiệu chuẩn STT; mô hình DungLy (chấm B18) học ít dữ liệu → ô ok B18 thấp (§4).
5. Thêm Borg vào bộ gộp đổi khoá cache hộp kim của gold_exact (khoá gồm mọi config có mặt) ⇒ lần đầu dựng lại `_detect` cho
   mọi trang (vài phút, 0 API, kết quả không đổi).
6. `prepared/_auto/` nằm trong `prepared/` (gitignore) — `kim_raw/` là DỮ LIỆU GỐC tốn tiền, không xoá.
