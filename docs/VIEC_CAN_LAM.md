# DANH SÁCH VIỆC — TỔNG HỢP SAU TOÀN BỘ NGHIÊN CỨU

**Chốt ngày**: 2026-08-22 · **Đây là tài liệu điều hành duy nhất.** Bốn tài liệu nghiên cứu bên
dưới là căn cứ, không phải danh sách việc:

| tài liệu | nội dung |
|---|---|
| `docs/PHAN_HOI_GOP_Y_2026-08-22.md` | phản hồi bản góp ý MI3, đã qua kiểm định 199 khẳng định |
| `docs/KE_HOACH_TONG_THE_2026-08-22.md` | §0 chứng minh vì sao mọi số precision cũ bị huỷ |
| `docs/CHUONG_TRINH_THI_NGHIEM_2026-08-22.md` | chi tiết lưới cấu hình từng chương trình T1–T6 |
| `docs/NGHIEN_CUU_UNIHAN_KDEFINITION_2026-08-22.md` | tín hiệu nghĩa Hán, một chiều |

---

## BỐI CẢNH ĐIỀU HÀNH — ba sự thật chi phối mọi việc

1. **Đề tài có 0 phán quyết người.** Toàn bộ "chấm tay" trước đây là máy chấm; verdict thô đã mất.
   Mọi con số precision / error-AUC / κ đã bị huỷ.
2. **Chấm tay đang tạm gác.** Sẽ rút ô chưa chấm từ bộ dữ liệu, chấm, nạp lại sau.
3. ~~Chưa có gì được commit.~~ ✅ Đã chốt mốc `17289d145c` (2026-08-23): `HEAD` giờ khớp đĩa.

---

## ĐƯỜNG GĂNG

```
KHỐI 0 ✅       →  KHỐI 1 (vá lỗi)  →  KHỐI 3 ✅  →  KHỐI 4 (T1→T6)
                            ↓                                              ↓
                     KHỐI 2 ✅                          KHỐI 5 (tín hiệu nghĩa)
                                                                           ↓
                                                    KHỐI 6 (dựng hạ tầng chấm — làm SONG SONG)
                                                                           ↓
                                                    KHỐI 7 (chấm tay)  →  KHỐI 8 (đóng gói)
```

**KHỐI 6 xây được ngay bây giờ, không phải chờ.** Xây sớm thì lúc chấm không phải dừng lại làm công cụ.

---

# KHỐI 0 — CỨU DỮ LIỆU ✅ HOÀN THÀNH 2026-08-23 · commit `17289d145c`

- [x] **0.1 Sao lưu primary data + sha256** → `~/backup_ocr_cache_2026-08-22/` (**1.783 tệp, 67 MB**)
  Checklist ban đầu ghi `cp -r prepared/*/detected/` — **thiếu một nửa**: cache QN nằm ở
  `transcriptions/`, không phải `detected/`. Bản sao thực tế gồm:
  445 cache Nôm (tốn tiền API) · 445 cache QN · 445 cột đã bóc · **445 `pages/*.png`** (ảnh mà
  `image_hash` neo vào — thiếu nó thì không kiểm chứng được chuỗi) · 3 manifest.
  Kèm `SHA256SUMS.txt` + `README.md` (cách kiểm & khôi phục). Đối chiếu từng tệp với bản gốc:
  **khớp tuyệt đối**; `shasum -a 256 -c` → 0 dòng lỗi.

- [x] **0.2 Commit mốc SHA** — commit `17289d145c`, 36 tệp.
  `.gitignore` đã loại `*.png`/`prepared/`/`dataset/` nên chỉ 36 tệp vào commit, không phải
  145.000 ảnh. Unihan được add theo **đúng casing `Dict/`** mà git đang dùng → index **không**
  sinh thư mục trùng `dict/`. Xác minh: `HEAD:dataset_out/labels_final.csv` = 82.269 dòng
  (GOLD 50.063 / SILVER_uncalibrated 10.890 / SYLLABLE 6.761 / REVIEW 14.555) — **khớp đĩa**.
  Commit vào `main` theo quy ước repo (`push.sh:10`: chỉ dùng nhánh main). **Chưa push.**

- [x] **0.3 Chốt đối chiếu cache OCR** — `core/ocr/ocr_api.py`
  Không phải "5 dòng" như dự tính. Nghiên cứu cho thấy `extract_nom_image` lưu ảnh qua
  `PIL.Image.save(..., "PNG")`, tức **mã hoá lại** → byte tệp phụ thuộc phiên bản Pillow/zlib,
  còn pixel thì không. Chốt md5 thô sẽ **báo động giả** khi đổi môi trường. Thiết kế thực hiện:

  | tình huống | xử lý |
  |---|---|
  | md5 tệp khớp | `ok` (đường nhanh, không mở ảnh) |
  | byte lệch, **pixel y hệt** | `healed` — cập nhật `image_hash`, cache vẫn dùng |
  | byte lệch, **pixel đổi** | **ném `StaleOCRCacheError`** |
  | cache chưa có `pixel_hash` | ném lỗi kèm hướng dẫn chạy backfill |
  | cache đời cũ không có `image_hash` | `skipped` |
  | `SN_OCR_SKIP_CACHE_VERIFY=1` | `skipped` (cửa thoát hiểm) |

  **KHÔNG tự gọi lại API khi lệch** — OCR lại tốn tiền, phải do người quyết định.
  `backfill_pixel_hash()` đã vá **445/445** cache Nôm; verify lại toàn bộ → **445 `ok`**.
  445 cache QN được bỏ qua an toàn (không có khoá `image`).
  Đường QN (`qn_ocr.py:129`) **vốn đã** đối chiếu md5 → không cần sửa.
  **+13 assertion**, mốc selftest **414 → 427**, không hồi quy.

## Phát hiện thêm trong lúc làm KHỐI 0

- [x] **0.4 `nom-embed` submodule "bẩn"** — ✅ **ĐÓNG 2026-08-24: BÁO ĐỘNG GIẢ, tôi đọc sai.**
  Đo thật trên HuggingFace API: `mdnt571/nom-embed` @ revision **`7ff74f57c4be`** — *đúng con trỏ
  submodule mà repo cha ghi* — và **sha256 bản công bố KHỚP TỪNG BYTE bản trên đĩa**:

  | tệp | LFS oid trên HF | sha256 trên đĩa | |
  |---|---|---|---|
  | `best.pt` | `eee2f3e706b08622…` | `eee2f3e706b08622…` | ✅ |
  | `last.pt` | `c05dd1723c751059…` | `c05dd1723c751059…` | ✅ |

  **Vì sao git báo bẩn:** git HEAD lưu **con trỏ LFS 134 byte**; `huggingface_hub` tải tệp THẬT
  (140.727.509 byte) ghi đè lên con trỏ, nên git so 134 byte với 140 MB → `M best.pt`. Nhưng
  `oid sha256:` **trong** con trỏ chính là băm tệp trên đĩa. **Chuỗi tái lập chưa từng bị đứt** —
  checkpoint sinh ra cột `s3_cosine` là công khai, cố định, và đã được con trỏ submodule ghim.

  🔴 **ĐỪNG `git checkout nom-embed/best.pt`** — sẽ thay tệp 140 MB bằng con trỏ 134 byte và làm
  sập S3. Cảnh báo này nay in ra trong khối bằng chứng mỗi lần chạy pipeline.

- [x] **0.5 `font_diffusion` LỆCH THẬT** (tìm ra khi kiểm 0.4) — repo cha ghi `61cbf1ba4ac9`, đĩa
  ở **`fc1874150f8c`**, cây làm việc sạch: submodule đã được cập nhật mà **con trỏ chưa commit**.
  Đây đúng là lớp lỗi mà 0.4 bị *nghi oan*. Không ảnh hưởng bộ giao nộp (`font_diffusion` chỉ
  *sinh* glyph, và 89.898 glyph đã sinh sẵn ở `gannhanocr-fd/`, `skip_local_fd_gen: true`), nhưng
  vẫn commit con trỏ để lịch sử khớp đĩa.

  **Bài học:** hai submodule cùng báo "bẩn", một cái vô hại một cái thật. *"Git báo modified"
  không phải bằng chứng dữ liệu lệch — phải băm nội dung ra mà so.*

# KHỐI 1 — VÁ 9 LỖI CHẶN ✅ HOÀN THÀNH 2026-08-23

| # | lỗi | đã vá thế nào | kiểm chứng |
|---|---|---|---|
| 1.1 | 48 ô 㝵/"người" ở GOLD | `s3_unwind.py`: chốt chặn lớp confusion — **CHỮA** (demote) chứ không chỉ chặn readmit, nên chạy trên tệp hỏng sẵn cũng ra sạch. Bất biến mới ở cuối `unwind()`. | bước 5 demote **1.972** ô (trước 1.924, +48); 㝵/người ở GOLD = **0** |
| 1.2 | `--measure` trả `null` im lặng | `confusion_fix.normalize_image_key()` chuẩn hoá `yen*`→`stt*` trước join; thêm cờ `provenance` + cảnh báo in ra | join **816/825** (trước **0/825**) |
| 1.3 | Rơi ngầm về midpoint | `DetectorUnavailableError` + `preflight_detector()` fail-fast **trước** khi duyệt trang; `build_dataset` **re-raise** thay vì nuốt thành warning (nếu không sẽ bỏ qua cả 445 trang); ghi cột `seg_backend` | thử giấu checkpoint → ném lỗi ✓; có checkpoint → `detector_centernet_v1` |
| 1.4 | `CHECKSUMS.txt` chưa từng sinh; hash hiện hành không có trong index | `evidence()` thêm khối `<!-- HIEN_HANH -->` **ghi đè mỗi lần chạy** (nhật ký "## Lần chạy" vẫn cộng dồn); thêm `scripts/check_evidence.sh` | `check_evidence.sh` → **khớp 4 · lệch 0 · thiếu 0** |
| 1.5 | `banner()` in "BƯỚC 7/6" | đổi toàn bộ đánh số về `/7` (8 dòng) | không còn `/6` nào trong tệp |
| 1.6 | `Dict/` (git) vs `dict/` (đĩa) | `core.text.dictionary.dict_dir()` — dò tên có thật lúc chạy, **không** ghim cứng lối viết nào. Áp cho 5 tệp | 5/5 module import OK; `dict_dir()` → `Dict/` |
| 1.7 | `index.csv` lệch thế hệ | preflight kiểm **thế hệ** chứ không chỉ "tệp tồn tại" | đo được **51.195** dòng `yen*` / **0** dòng `stt*` → sẽ cảnh báo |
| 1.8 | `pipeline_today.yaml` còn `auto` | xoá tệp | — |
| 1.9 | 73 ô sửa dấu mồ côi | ghi rõ trạng thái mồ côi trong docstring + trỏ sang T1 (chỗ đúng là chuẩn hoá **trước** build) | — |

## Nghiệm thu

| tiêu chí | kết quả |
|---|---|
| Chạy lại bước 4→7 | **56.776 dòng** (GOLD 50.015 + SYLLABLE 6.761), 0 ảnh thiếu |
| 㝵/"người" ở GOLD | **0** (toàn bộ 1.972 ô ở REVIEW) |
| `CHECKSUMS.txt` | đã sinh, 3 mốc |
| Chuỗi bằng chứng | `check_evidence.sh` → **4/4 khớp** |
| Selftest | **448 passed, 0 failed** (mốc 427 → 448, +21) |

**Còn lại**: cột `seg_backend` chỉ xuất hiện sau lần **build** tới (bước 3) — bước 4→7 không sinh
lại cột. Không chặn gì.

# KHỐI 2 — DỌN SỐ LIỆU ✅ HOÀN THÀNH 2026-08-23

- [x] **2.1 Đánh dấu CHƯA ĐO / ĐÃ HUỶ** — quét toàn bộ `docs/`; các số huỷ tập trung ở 4 tệp, cả 4
  đã gắn cờ: `CODE_FREEZE.md` · `EVIDENCE_INDEX.md` · `FLOW_CAP_NHAT_2026-08-19.md` ·
  `KE_HOACH_CHAM_TAY.md` (gắn banner **ĐÃ BỊ THAY THẾ** — mẻ chấm nó mô tả chính là mẻ máy chấm).

- [x] **2.2 Viết lại `BANG_SO_LIEU_CHINH_THUC.md`** — bản cũ chứa **đồng thời hai bộ số mâu thuẫn**,
  một bộ không khớp tệp nào trên đĩa, và ghi selftest 414. Bản mới có 3 nhãn nguồn kiểm định
  (🔢 MÁY ĐẾM / ⚪ CHƯA ĐO / 🔴 ĐÃ HUỶ), §4 liệt kê từng số bị rút kèm lý do, §6 quy tắc trích dẫn.
  Sửa câu sai ở `EVIDENCE_INDEX.md` ("66.589 khớp đúng dataset/labels.csv" — đĩa khi đó 66.529,
  nay **56.776**).

- [x] **2.3 Sửa cách phát biểu Bước 6** — trong chính `s3_unwind.py`: "CHƯA CHỨNG MINH ĐƯỢC",
  không phải "ĐÃ BÁC BỎ", kèm ba lẽ (lớp âm 24 ca → AUC nhỏ nhất phát hiện được 0,607 · đo trên
  GOLD nơi S3 không gán nhãn, chỉ 7,7% ô có điểm · 74,7% crop rò rỉ vào TRAIN). Báo cáo JSON thêm
  `provenance` + `how_to_state_it`. Gỡ nốt hai chỗ docstring còn viện 40/41 và 737/752.

- [x] **2.4 Datasheet khai đủ giới hạn** — `pipeline/publish/datasheet.py`: bỏ hẳn câu
  "measured estimate on a stratified **human** audit sample" (nay sai sự thật), thay bằng 6 giới hạn
  bắt buộc, mở đầu bằng **"NO HUMAN-VERIFIED PRECISION IS AVAILABLE"** kèm bằng chứng 846/846
  item_id và 47/846 verdict. Có cả giới hạn DPI STT4 và tư cách tier SYLLABLE/SILVER.

**Kiểm chứng**: selftest **448 passed, 0 failed** (khớp mốc, không hồi quy) · quét lại toàn `docs/`:
0 tệp còn số huỷ mà thiếu cờ.

# KHỐI 3 — DỰNG BÀN THÍ NGHIỆM ✅ HOÀN THÀNH 2026-08-23

- [x] **3.1 `pipeline/lab/runner.py`** — chạy 1 cấu hình YAML → 1 dòng `lab/results.csv`, không đụng
  `dataset_out/`. `run_id` = 12 ký tự đầu sha256 của cấu hình đã chuẩn hoá (**bỏ qua `name`** vì đó
  chỉ là nhãn người đọc) → đổi tham số là thành dòng mới, không ghi đè nhầm. Có `--quick` để thử tay.
- [x] **3.2 `pipeline/lab/metrics.py`** — 4 nhóm thước đo họ D: hình học crop (**lần đầu tiên
  `crop_quality.py` được dùng thật**), chất lượng âm QN, cấu trúc cột + tỉ lệ ô neo, thành phần
  tier/lớp + mâu thuẫn md5.
- [x] **3.3 `pipeline/lab/perturb.py`** — chuẩn nhiễu loạn có đáp án: 7 loại hỏng × 3 mức × N seed,
  chỉ số `anchor_retention`. Mốc không hỏng = **1,0 chính xác** (chuẩn không thiên lệch).
- [x] **3.4 `pipeline/lab/synth.py`** — trang tổng hợp 9 cột từ **89.898 glyph** FontDiffusion, đáp
  án hộp chính xác 100%, nhiễu (đứt nét/nhoè/vân gỗ) **hiệu chuẩn theo `ink_pct` đo trên ảnh thật**
  qua `calibrate()` đọc `results.csv`. Giới hạn khai thẳng trong docstring: kiểm **hành vi thuật
  toán**, KHÔNG dùng để công bố số chất lượng của bộ thật.
- [x] **3.5 `lab/results.csv`** — 62 cột/dòng; `append_row` ghi đè theo `run_id` và tự mở rộng header.

## MỐC XUẤT PHÁT (`baseline`, run_id `d004238795fe`)

### Chất lượng hình học 56.776 crop — **lần đo đầu tiên của đề tài**

| cờ | số ô | % |
|---|---|---|
| `ok` | 52.787 | **92,97%** |
| `bleed` (dính mực hàng xóm) | 3.560 | **6,27%** |
| `truncated` (cắt vào nét) | 389 | 0,69% |
| `blank` | 40 | 0,07% |

Ngoại lai tỉ lệ khung 1,98% · `stray_ink` p95 0,136 · `border_ink` p95 0,100 · `ink_pct` p50 0,175.
⇒ **7,03% bộ giao nộp có khuyết tật hình học đo được** — chiều CROP nay do **máy** đo, không phải mắt người.

### Độ bền căn chỉnh — 🔴 **BẢNG CŨ ĐÃ SAI, ĐÃ RÚT LẠI**

Bảng công bố ở KHỐI 3 (`split_char` 0,9112 và `swap_syl` 0,9438 là "hai chỗ giòn nhất")
là **GIẢ TẠO DO CÁCH CHẤM**, không phải tính chất của thuật toán. Đợt phản biện độc lập
2026-08-24 phát hiện và tôi đã tự xác minh lại:

| loại hỏng | ĐÃ CÔNG BỐ (sai) | ĐÚNG |
|---|---|---|
| `split_char` | **0,9112** ← *"yếu nhất"* | **0,9997** |
| `drop_char` · `drop_syl` · `ins_syl` | 0,999 | 0,999 |
| `swap_syl` | 0,9438 | 0,9423 |
| **`subst_char`** | 0,9997 | **0,9132** ← yếu nhất thật |
| **`tone_syl`** | 0,9987 | **0,9160** |

**Kết luận đúng, ngược hẳn kết luận cũ**: căn chỉnh **BỀN trước hỏng CẤU TRÚC** (mất hộp,
thêm hộp, tách đôi hộp) nhưng **GIÒN trước hỏng NỘI DUNG** (OCR đọc nhầm chữ, sai dấu
thanh). Hợp lý, vì phép tra từ điển phụ thuộc nội dung chứ không phụ thuộc vị trí.

⇒ Định hướng cho T4 cũng đổi theo: **over-segmentation KHÔNG phải vấn đề của align**
(0,9997). Nếu crop hỏng thì hỏng ở khâu cắt ảnh, không ở khâu ghép.

## Một lỗi tự bắt được

Phép kiểm tất định của chính bàn thí nghiệm **bắt được lỗi trong bàn thí nghiệm**: seed dựng bằng
`tuple.__hash__()` chứa chuỗi, mà hash chuỗi bị **ngẫu nhiên hoá theo `PYTHONHASHSEED`** → hai lần
chạy ra hai con số khác nhau (78.866 vs 78.841). Thay bằng md5 ổn định; 3/3 tiến trình nay ra số y
hệt. Đã thêm assertion chạy `_seed` dưới `PYTHONHASHSEED` 0/1/random để không tái diễn.

**Kiểm chứng**: selftest **495 passed, 0 failed** (mốc 448 → 495, +47) · baseline chạy lại ra số y hệt.

# KHỐI 4 — SÁU CHƯƠNG TRÌNH THÍ NGHIỆM (4 tuần)

Thứ tự **bắt buộc từ thượng nguồn xuống** — chỉnh crop trên hộp xấu là công cốc.

| # | chương trình | cấu hình | thước đo chính | baseline |
|---|---|---|---|---|
| **T1** ✅ | Đường Quốc ngữ — chuẩn hoá dấu phụ + rác marker | — | âm **ngoài từ điển** | **0,946%**, xem kết quả bên dưới |
| **T2** ✅ | Hình học trang Nôm — dò cột + bóc dòng QN | — | trang đủ **9 cột** | **439/445 → 443/445** |
| **T3** ✅ | Căn chỉnh — chuẩn nhiễu loạn + quét chi phí | 78 (từ 144) | `anchor_retention` (đã sửa) | **GIỮ MỐC** — xem dưới |
| **T4** | Tách ký tự & chất lượng crop | 144 | 6 thước đo hình học | **chưa đo bao giờ** |
| **T5** | Độ giòn từ điển + `syllable_gate` + top-K cầu tự dạng | ~30 | độ giòn, nhất quán liên sách | — |
| **T6** | Tái lập & bằng chứng | — | byte-identical ×2, clone sạch | — |

## T1 — ĐƯỜNG QUỐC NGỮ · KẾT QUẢ 2026-08-23

### Phân loại chính xác 778 ô ngoài từ điển (0,946%)

| nhóm | ô | % | ví dụ | sửa máy được? |
|---|---|---|---|---|
| Âm hợp lệ nhưng từ điển không có | 508 | 65,3% | `樞/giu` `傳/truyen` `衣/ay` | một phần — xem dưới |
| **Rác marker** (chữ số lọt vào nội dung) | **103** | 13,2% | `1` `0` `19` `2017` `290` | gắn cờ, không sửa |
| Sửa thanh được (nhóm A) | 88 | 11,3% | `礼/trấy` `孛/but` | ✅ |
| Âm không hợp lệ (hỏng nặng) | 79 | 10,2% | `mortthay` `038struyen` `rút2%` | ❌ |

Bóc tiếp nhóm 508 ô: **35** có ứng viên duy nhất là đọc âm của chính chữ (`衣/ay→ấy`,
`丑/xau→xấu`, `門/muon→muôn`) · **29** chốt được bằng bằng chứng corpus (`旦/den→đến` 526×,
`各/cac→các` 309×) · 316 có ứng viên nhưng KHÔNG phải đọc âm của chữ (nhóm B mở rộng — hai
giả thuyết ngang nhau, cấm sửa máy) · 125 không có ứng viên nào kể cả bỏ hết dấu.

### Đã làm

`fix_tone` mở rộng từ "chỉ dấu THANH" sang "mọi dấu phụ": thêm `strip_all` (bỏ cả dấu tạo
chữ và đ→d), thử **tầng 2 chỉ khi tầng 1 không ra ứng viên**, giữ nguyên hai chốt an toàn
(ứng viên phải là đọc âm của chính chữ; nhập nhằng chỉ corpus mới chốt). Thêm phát hiện
rác marker (gắn cờ, không sửa). **+23 assertion** — `tools/` trước nay không có test nào.

    73 ô  ->  115 ô   (tone_unique 66 · diacritic_unique 35 · tone_corpus 7 · diacritic_corpus 7)

### Kết quả quyết định: VỊ TRÍ quan trọng hơn cấu hình

| | |
|---|---|
| Sau khi sửa, đủ điều kiện `s1_inter_s2_direct` (=GOLD) | **115/115** |
| Tier hiện tại của 115 ô | 106 SILVER_uncalibrated + 9 REVIEW — **0 ô trong bộ giao nộp** |
| Áp **SAU** build (như hiện nay) | **+0** ô vào bộ giao nộp |
| Áp **TRƯỚC** build | **+115 ô GOLD** → 56.776 → **56.891** |

Đây là lần đầu định lượng được điều mà `fix_tone` tự viết trong docstring từ đầu. Lưới 54
cấu hình của T1 là thứ yếu: đòn bẩy thật nằm ở **chỗ đặt phép chuẩn hoá**, không ở tham số.

⚠️ Mục tiêu "~0,6%" tôi nêu lúc lập kế hoạch là **lạc quan**. Con số trung thực sau khi
phân loại: **0,946% → 0,636%** (152 ô sửa + 103 ô rác gắn cờ), và chỉ đạt được nếu chuẩn
hoá chạy TRƯỚC build.

### ĐÃ THỰC HIỆN 2026-08-24 — chuẩn hoá TRƯỚC align + build lại

`pipeline/align_engine/syllable_normalize.py` (mới) chạy ngay trước `realign_column`,
dùng đọc âm của **mọi chữ trong cột** (cặp ghép chưa có — đó là thứ NW sắp tính). Giữ
nguyên ba chốt an toàn của `fix_tone`. `strip_tone`/`strip_all` chuyển xuống
`core/text/text_utils.py` để engine không import ngược từ `tools/`.

| chỉ số | trước T1 | sau T1 |
|---|---|---|
| **Bộ giao nộp** | 56.776 | **56.882** (+106) |
| GOLD | 50.015 | **50.120** (+105) |
| SYLLABLE | 6.761 | 6.762 |
| Lớp ký tự trong bộ giao nộp | 1.582 | **1.585** |
| **Âm ngoài từ điển** | 778 = **0,946%** | 643 = **0,782%** |
| Hình học crop `ok` | 92,974% | 92,98% |
| `anchor_retention` | 0,9773 | 0,9773 |

Dự đoán +100 ô GOLD từ phép đo cấp-cột, thực tế **+105**. `check_evidence.sh` 4/4 khớp.

⚠️ Không đạt mốc 0,636% đã nêu: mốc đó giả định loại luôn 103 ô rác marker, mà chúng
**đã ở REVIEW** nên loại thêm không đổi gì. Số thật là **0,782%**.

### T1.y (rác marker) — BỎ, có căn cứ đo được

| kiểm chứng | kết quả |
|---|---|
| 103 ô rác nằm ở tier nào | REVIEW cả 103 |
| Ô âm-không-hợp-lệ (195) lọt vào bộ giao nộp | **0** |
| `is_plausible_qn_syllable` chặn được | 195/195 |

Chẩn đoán "rò rỉ marker" **sai tên**: thực tế là 4 cơ chế — 63 ô nhiễu OCR thuần
(`0000000000`, `100,0,`) · 22 ô không truy được nguồn · **16 ô VietOCR đọc I→1 / Ô→0
trong từ mượn** (`1-na-xu`, `0-sa-ka`) · **2 ô** chú thích thật (`Phô-li-ca-phô?3`).
Trong 11 ô từ mượn đầu-số, sửa xong chỉ **4 ô** đủ điều kiện GOLD. Thêm một luật để lấy
4 ô là thêm một luật phải bảo vệ trước hội đồng — không đáng.

### Hai lỗi tự bắt được khi làm T1

1. **`seg_backend` không vào CSV** — thêm vào `records` nhưng chỗ ghi tệp dùng danh sách
   `labels` với `fields` cố định. Log in "20 cột" thay vì 21. Nghiệm thu rẻ bằng
   `--limit 3 --no-crops` (vài giây thay vì 20 phút), rồi **build lại** để tệp trên đĩa
   khớp đúng mã đã commit — không tự miễn trừ khỏi chính quy tắc tái lập đang đi vá.
2. **`run_id` của bàn thí nghiệm chỉ băm cấu hình**, nên chạy cùng cấu hình trên dữ liệu
   trước/sau T1 cho cùng id và dòng sau **đè** dòng trước — đã làm mất mốc baseline
   trước T1. Nay `run_id` gồm cả vân tay bộ nhãn (`labels_sha`).

Ngoài ra một assertion phụ thuộc dữ liệu phải cập nhật: `similar_bridge` 4.098 → **4.100**.

## T2 — HÌNH HỌC TRANG NÔM · KẾT QUẢ 2026-08-24

Lưới 48 cấu hình **không chạy**: mỗi cấu hình phải build lại 445 trang (~20 phút) ⇒ ~16 giờ.
Thay bằng **chẩn đoán trực tiếp 6 trang hỏng** — rẻ hơn và cho gốc rễ thay vì thứ hạng.

### 6 trang tách thành hai lớp khác hẳn nhau

| lớp | trang | triệu chứng | gốc rễ |
|---|---|---|---|
| **A** | `stt4` 0110, 0146, 0252 | cột Nôm hoàn hảo, QN chỉ ra 8 dòng | `parse_v5` bỏ dòng khi marker hỏng |
| **B** | `stt11` 0042, `stt4` 0028, 0122 | cột 1–2 **rỗng**, cột khác **gộp** (45/44/31 chữ) | `nom_cols_hybrid(min_len=4)` |

### Lớp B — gốc rễ và phép sửa

`stt11 page_0042`: tâm-x thật của 9 cột cách đều ~145px, nhưng bộ dò ra một cột giả
**(1631–1690)** ở lề phải — rộng **59px, 0 chữ**: đó là **mực viền**. Cột giả chiếm một suất
trong 9 ⇒ suất khác phải gộp hai cột thật (47 chữ thay vì 23) ⇒ mất nguyên một cột nhãn.

Nguyên nhân sâu hơn: `min_len=4` **loại cột thật chỉ 2–3 chữ** (dòng cuối đoạn) ⇒ còn 8 cột
⇒ rơi xuống nhánh projection ⇒ chính nhánh đó bắt nhầm viền.

Đo trước khi sửa trên 445 trang: nới `min_len` sửa **12 trang**, **hỏng 0 trang** (433 giữ
nguyên). Sửa an toàn nhất — chỉ nới khi `min_len=4` cho THIẾU cột. Kết quả: **445/445 trang
ra đúng 9 cột**, nhánh projection mong manh **không còn được dùng lần nào**.

### Lớp A — hai bộ bóc giỏi ở những trang khác nhau

`_get_qn_lines` bóc lại bằng `parse_v5`, **không dùng** `transcriptions/page_*.json` (do
`parse_numbered_lines` sinh ở bước 1) — hai hàm khác nhau và bất đồng.

| bộ bóc | trang ra đủ 9 dòng |
|---|---|
| `parse_v5` (hiện dùng) | 442/445 |
| `parse_numbered_lines` | 415/445 |
| **ưu tiên bộ nào ra 9** | **443/445** |

Chỉ thêm 1 trang, nhưng rẻ và **chỉ kích hoạt khi `parse_v5` đã hỏng**. Hai trang còn lại
(`0110`, `0252`) không cứu được — text OCR không có dòng thứ 9.

### Kết quả

| chỉ số | trước T2 | sau T2 |
|---|---|---|
| **Trang đủ 9 cột** | 439/445 | **443/445** |
| Nhánh projection được dùng | 12 trang | **0** |
| Bộ giao nộp | 56.882 | **56.909** (+27) |
| GOLD | 50.120 | **50.156** (+36) |
| Âm ngoài từ điển | 643 (0,782%) | 661 (0,804%) |
| Hình học crop `ok` | 92,98% | 92,991% |
| `anchor_retention` | 0,9773 | 0,9773 |

⚠️ Âm ngoài từ điển **tăng nhẹ** — không phải hồi quy: 4 cột trước đây bị mất nay quay lại,
mang theo cả âm ngoài từ điển của chúng. Đây là **thêm dữ liệu**, không phải giảm chất lượng.

### Ghi chú về test giòn

`similar_bridge` là assertion **phụ thuộc dữ liệu**, hỏng sau mỗi lần đổi bộ nhãn (4098 →
4100 ở T1 → 4102 ở T2). Nay tách làm hai: **băng rộng 3500–4700** bắt sụp thật (bền, không
phải sửa) và **mốc chính xác** làm canary bắt "đổi dữ liệu mà quên cập nhật".

---

## T3 — CĂN CHỈNH · KẾT QUẢ 2026-08-24

### Bản đầu bị bác: THIẾT KẾ HỎNG

Đợt phản biện 6 góc độc lập bác bản đầu và **bác cả kết quả KHỐI 3 tôi đã báo cáo**. Ba lỗi chặn:

| # | lỗi | bằng chứng |
|---|---|---|
| 1 | **Chấm theo chỉ số thay vì nội dung** — 88,7% "mất neo" là giả tạo | `split_char` nhân đôi chữ, DP chọn bản sao thứ hai, neo ở chỉ số cũ bị đếm là mất **dù nhãn đúng**. Tự kiểm: 0,9140 (chỉ số) → **0,9982** (nội dung) |
| 2 | **Mốc được ưu ái tuyệt đối** | neo sinh bởi mốc ⇒ `ret@nhiễu=0` của mốc = 1,00000 theo định nghĩa. Biên độ giữa các cấu hình **khi chưa có nhiễu** (0,622pp) **lớn hơn** toàn bộ biên độ chỉ số sau nhiễu (0,577pp) = 108%. Bỏ đặc quyền → mốc rơi **1/17 → 13/17** |
| 3 | **Trục sản lượng vòng tròn và ngược dấu** | `confirmed` không phụ thuộc chi phí ⇒ `yield_confirmed` đo "chịu bẻ đường ghép bao nhiêu để nhặt thêm cặp dict-confirm" = chính luật gán nhãn. Cấu hình "thắng" phát ra **ít hơn 352 cặp ghép**; op `del` biến mất hoàn toàn khỏi `labels.csv` |

Thêm: hai trục tương quan **Pearson +0,905** (không phải đánh đổi → Pareto là trang trí);
biên Pareto **ghim vào cạnh dưới lưới**; `seeds=1` có nhiễu **lớn hơn** khoảng cách giữa
nhiều cặp cấu hình.

### Đính chính bảng KHỐI 3 (xem mục KHỐI 3 ở trên)

`split_char` **0,9112 → 0,9997**. Kết luận cũ *"giòn nhất trước over-segmentation"* **sai**.
Đúng: **bền trước hỏng CẤU TRÚC, giòn trước hỏng NỘI DUNG** (`subst_char` 0,9078 ·
`tone_syl` 0,9123 mới là hai chỗ yếu nhất).

### Bảy phép sửa

neo **độc lập cấu hình** (34.785 cặp đường chéo dict-confirm trên cột m=n, không chạy
align) · chấm theo **nội dung** (đa tập) · thêm `ret_noise0` + `ret_drop` · trục sản lượng
→ `yield_match` + tách `confirmed`/`unconfirmed`, trừ cặp `confusion_fix` hạ cấp ·
`seeds` 1→3 (vòng A) / 5 (vòng B) · lưới **nới để bao cực trị** và bỏ `BAND_SLACK` khỏi
lưới chính (144 → **78**) · luật chọn **một trục + ràng buộc**, viết trước.

### Ba trong bốn núm KHÔNG nhận dạng được

| núm | kết luận |
|---|---|
| **`BAND_SLACK`** | **bằng 0 tuyệt đối** — band2 / band3 / band4 / band2-thích-nghi cho `ret 0,96510` và `match 24.608` **giống hệt đến 5 chữ số** |
| `COST_NODICT` | Δ ≈ 0,0003 ≈ nhiễu |
| `COST_SIMILAR` | thật nhưng nhỏ, và **thứ hạng đảo khi đổi cách cắt** (gộp theo mọi tham số khác thì chính mốc 0,30 cao nhất) |
| **`COST_DEL=COST_INS`** | **núm thật duy nhất**, có vách hai đầu |

⇒ **Giải quyết dứt điểm đề xuất §3.3 của bản góp ý MI3** (*"nới băng `|i−j| ≤ |m−n|+3`"*):
đo được **không đổi một ô nào**. Không phải "khó đánh giá" — là **bằng 0**.

### Lưới nới rộng đã bao được cực trị

```
COST_DEL=INS  0,45 → match 16.203   sụp 34% sản lượng (đúng cấu hình bệnh hoạn phản biện dự đoán)
              0,60 → ret 0,96538  ┐
              0,70 → ret 0,96529  ├ cao nguyên phẳng, chênh ≈ nhiễu   ← MỐC nằm giữa
              0,85 → ret 0,96503  ┘
              1,00 → ret 0,95151   sụp độ bền
```

### Quyết định: GIỮ NGUYÊN ma trận chi phí

Vòng B (toàn bộ 4.003 cột, seeds=5): cấu hình tốt nhất qua ràng buộc là
`sim0,60_nod0,90_di0,70`, hơn mốc **+0,00072** — vượt nhiễu 0,00023. Nhưng:

| | mốc | thắng | chênh |
|---|---|---|---|
| `yield_match` | 82.249 | 82.422 | +173 |
| **`yield_confirmed`** | **53.453** | **53.266** | **−187** |
| retention (cả 7 loại) | 0,96596 | 0,96666 | **+0,00070** |
| **retention (bỏ `swap_syl`)** | **0,96950** | **0,96959** | **+0,00009** ← dưới nhiễu |

**`swap_syl` góp 87% toàn bộ mức tăng.** Mà `swap_syl` đổi chỗ hai âm kề → tạo phép ghép
**chéo**, thứ căn chỉnh đơn điệu **về nguyên tắc không thể sinh ra**. Một cải thiện chỉ tồn
tại trên một phép thử bất khả thi thì không phải cải thiện — và nó còn phải trả **187 cặp
dict-confirmed**, mà **không có ground truth** để biết 187 ô đó đúng hay sai.

**Luật quyết định của tôi có lỗ hổng**: nó chỉ so mức tăng GỘP với nhiễu, không đòi mức
tăng phải **bền qua các lớp hỏng**. Đã bổ sung điều kiện đó vào `decide()` thay vì lặng lẽ
bỏ qua luật; chạy lại thì luật tự trả về **GIỮ MỐC**.

### Sản phẩm của T3

Không phải một cấu hình mới, mà là **chính cái chuẩn đo** — một phép đo độ bền tái lập được
mà đề tài trước đây không có — cộng **ba câu trả lời phủ định có bằng chứng** (băng, nodict,
similar) và **bằng chứng bao cực trị** cho giá trị `del/ins = 0,70` đang dùng. Kết quả âm,
nhưng là kết quả âm *đo được*, viết vào luận văn được.

---

### Ba cảnh báo bắt buộc

1. **T3 — không tối ưu theo "số ô GOLD" hay tỉ lệ dict-confirm.** Đó chính là luật gán nhãn; cấu
   hình nới lỏng nhất sẽ luôn thắng trong khi sinh nhiều nhãn sai nhất. Dùng **biên Pareto**
   `anchor_retention` × sản lượng.
2. **T3 — chuẩn đo neo vào Ô NEO** (`s1_inter_s2_direct`, trung vị 58%/cột), **không** phải "cột
   sạch tuyệt đối" — toàn corpus chỉ có **6 cột** đạt 100% dict-confirmed.
3. **T2 — không dùng công thức `pad_px = pad_frac × pitch × (300/DPI)`.** `pitch` đo bằng pixel của
   chính trang đó nên `pad_frac × pitch` **vốn đã tự chuẩn hoá**. Cách đúng: **resample crop về kích
   thước chuẩn**, hoặc chuẩn hoá theo `pitch`.

### Sản phẩm đi kèm

- [ ] **T3.4** Thêm **≥ 10 assertion** cho `anchor_align` (hiện **0** test phủ ma trận chi phí)
- [ ] **T4.1** Bảng chất lượng hình học của **toàn bộ 66.589 crop** — chưa ai đo bao giờ; tự nó là
  một mục trong luận văn, và thay được chiều CROP vốn phải nhờ mắt người
- [ ] **T5** Cột **`fragility`** trong `labels.csv` = số mục từ điển hậu thuẫn cho nhãn này. Ô
  `fragility = 1` là ứng viên chấm tay số một → **làm mẻ chấm sau này rẻ đi nhiều lần**

---

# RÀ SOÁT TOÀN BỘ — 2026-08-24

Kiểm lại từng khẳng định đã tuyên bố hoàn thành. **Ba lỗ hổng thật**, đã vá.

| mục | tuyên bố | thực tế |
|---|---|---|
| 0.1 sao lưu | khớp đĩa | ✅ 0 dòng lỗi |
| 0.3 chốt cache | đã cài | ✅ chạy trong mọi build |
| **0.4 `nom-embed` bẩn** | *"lỗ hổng tái lập"* | 🔴 **TÔI ĐỌC SAI** — khớp HF từng byte |
| **0.5 `font_diffusion`** | *(chưa ai thấy)* | ⚠️ **lệch THẬT** — con trỏ ≠ đĩa, đã commit |
| 1.1 㝵 ở GOLD | 0 | ✅ 0 |
| 1.4 bằng chứng | 4/4 khớp | ✅ nay **6/6** |
| 1.6 công cụ từ điển | chạy được | ✅ 6/6 |
| **1.7 index.csv** | *"đã thêm cảnh báo"* | 🔴 **chỉ cảnh báo, chưa sửa** — nay đã sinh lại |
| 1.8 pipeline_today | đã xoá | ✅ |
| **KHỐI 2 bảng số liệu** | *"đã viết lại, khớp đĩa"* | 🔴 **LỆCH LẠI** sau T1+T2 — nay TỰ SINH |

## Lỗ hổng 1 (nặng nhất) — bảng số liệu lệch lại ngay sau khi viết

`BANG_SO_LIEU_CHINH_THUC.md` ghi 56.776 / GOLD 50.015 / hash `dbad35e9…`, đĩa là
56.909 / 50.156 / `236cbc4f…`. **Chính lớp lệch mà KHỐI 2 viết ra để diệt** — và do tôi
gây ra khi đổi dữ liệu hai lần mà không cập nhật.

Gốc rễ: **không có gì ÉP cập nhật**. Gõ tay thì sẽ quên. Nên nay **7 khối phụ thuộc dữ
liệu tự sinh** giữa mốc `<!-- AUTO:… -->` (header · nguồn gốc+hash · phân hạng · luật ·
phạm vi · vá lỗi · đo khác), cùng cơ chế với `evidence()`:

```bash
python -m pipeline.tools.update_bang_so_lieu          # ghi lại
python -m pipeline.tools.update_bang_so_lieu --check  # exit 1 nếu lệch
```

## Lỗ hổng 2 — `index.csv` mới chỉ được *cảnh báo*, chưa *sửa*

KHỐI 1.7 thêm phép kiểm thế hệ vào preflight nhưng **không sinh lại chỉ mục**: nó vẫn
**51.195/51.195 dòng trỏ `yen*`**, giao với bộ nhãn hiện hành = **0**. Các tệp `yen*` còn
trên đĩa nên preflight cũ báo xanh — "tệp tồn tại" không có nghĩa là "đúng thế hệ".

`pipeline/tools/rebuild_proto_index.py` (mới) sinh lại: **41.835 crop GOLD/train, 100%
`stt*`, 1.571 lớp**, 300/300 mẫu có ảnh thật. Ảnh hưởng **nằm ngoài bộ giao nộp** (S3 chỉ
quyết SILVER; GOLD = S1∩S2 trả về trước mọi lần đọc S3) và có hiệu lực ở lần build kế tiếp.

## Lỗ hổng 3 — ĐÃ BÁC BỎ: checkpoint S3 **vẫn** truy nguyên được (0.4)

Tôi ghi ở trên rằng checkpoint S3 "không truy nguyên được". **Sai — xem 0.4.** Đo trên HF API:
bản công bố tại `mdnt571/nom-embed` @ `7ff74f57c4be` khớp **từng byte** bản trên đĩa; git báo bẩn
chỉ vì so con trỏ LFS 134 byte với tệp thật 140 MB. Không có lỗ hổng tái lập ở đây.

Cái lệch **thật** nằm ở submodule *khác*: `font_diffusion` (0.5) — con trỏ repo cha `61cbf1ba4ac9`
≠ HEAD trên đĩa `fc1874150f8c`. Đã commit con trỏ.

## Bài học đã đóng thành lệnh

Mỗi lớp lệch phát hiện được phải để lại **một lệnh bắt được nó**, nếu không lần sau vẫn
lệch y hệt. Nay gộp thành một:

```bash
bash scripts/check_consistency.sh     # 3/3: bằng chứng · bảng số liệu · thế hệ chỉ mục
```

`MoTaCode.txt` được gắn banner **ảnh chụp có ngày** — nó đã gửi ra ngoài làm nguồn `[1]`
của bản góp ý nên giữ nguyên văn, không sửa lén, chỉ ghi rõ số đã cũ và trỏ về tệp hiện hành.

---

# KHỐI 5 — TÍN HIỆU NGHĨA HÁN (2 ngày)

Công cụ đã có: `pipeline/tools/sem_score.py` (chạy được, `--bench` tái lập số).

- [ ] **5.1** Ghi 2 cột `sem_score` + `sem_confirmed` vào `labels.csv`.
  **CHỈ PHONG, KHÔNG BAO GIỜ HẠ** — 47,7% cặp đúng cũng cho điểm 0; 㝵/"người" (sai) và 主/"chúa"
  (đúng) **cùng điểm 0,0**. Ngưỡng đã hiệu chuẩn: > 0,05 chính xác 100%; > 0,02 chính xác 98,9%.
  Sản lượng: **451 ô** xác nhận trên luật GOLD yếu nhất `s1_inter_s2_similar`.
- [ ] **5.2** Xuất bảng **"mượn nghĩa vs mượn âm"** cho toàn corpus — đóng góp ngôn ngữ học, chưa
  từ điển nào của ngữ liệu Công giáo cổ này có.
- [ ] **5.3** Tải CHISE `ids.txt`, kiểm giả thuyết **mô hình lỗi OCR**: OCR quy chữ Nôm lạ về chữ
  Hán **chia sẻ thành phần** (韋→喡 rụng bộ 口 · 等→寺 thêm bộ 竹 · 忝→𡗶 cùng 天 · 洋→群 cùng 羊).
  *Đây mới là chỗ IDS có ích thật — để mô hình hoá lỗi, KHÔNG phải để huấn luyện thị giác.*

---

# KHỐI 6 — DỰNG HẠ TẦNG CHẤM TAY (làm SONG SONG với KHỐI 4, chưa chấm)

- [ ] **6.1 `audit_grid` bổ sung 4 cơ chế kiểm soát chất lượng**:
  - **ô mồi 10%** — nửa là ô chắc chắn đúng, nửa **cố ý gán sai**; nếu độ chính xác trên ô mồi
    < 90% thì **huỷ cả buổi chấm**. *Đây là cơ chế trực tiếp phát hiện chấm ẩu hoặc chấm bằng máy —
    đúng vấn đề đã xảy ra.*
  - **ô lặp ẩn 8%** — đo κ nội tại thật
  - **làm mù**: không hiện tier / rule / `s3_cosine`
  - ghi `session_id` + `dwell_ms`; ô nào `dwell_ms < 800ms` gắn cờ xem lại
- [ ] **6.2 Gắn nghĩa Hán vào HTML chấm** (dùng `sem_score.load_meanings`) — 2 giờ, **giá trị cao
  nhất**: người chấm thấy ngay *"㝵 = (dạng cổ của 得) to get / to obstruct"* khi đang chấm ô âm
  "người".
- [ ] **6.3 Tách đôi nhiệm vụ**: chiều **NHÃN** cho người; chiều **CROP** đo bằng máy
  (`crop_quality.py` từ T4). Trộn hai chiều vào một nút bấm là lỗi thiết kế của mẻ cũ.
- [ ] **6.4 Rút mẻ, niêm phong TEST**:
  - **DEV 650** (GOLD 300 · SILVER 200 · SYLLABLE 150) — chỉnh tham số thoải mái
  - **TEST 1.050** (GOLD 600 · SILVER 250 · SYLLABLE 200) — **chỉ mở MỘT LẦN** ở KHỐI 8
  - loại mọi trang từng dùng để phát hiện lớp lỗi 㝵 khỏi TEST (chống post-hoc)
  - ưu tiên ô `fragility = 1` và ô `sem_score` thấp
- [ ] **6.5 Chấm thử 50 ô** để kiểm hướng dẫn chấm **trước khi** cam kết 2.000 ô.

---

# KHỐI 7 — CHẤM TAY & ĐO LẠI (sau KHỐI 4–6)

- [ ] **7.1** Chấm DEV (~650 ô + mồi + lặp), 4–6 buổi ≤ 45 phút/buổi.
  *Chia nhỏ vì mẻ cũ trôi 4,2% → 16% → 35% qua ba buổi dài.*
  **Cổng**: ô mồi ≥ 95%, κ chiều NHÃN ≥ 0,6. Nếu κ < 0,4 → **dừng, sửa hướng dẫn, làm lại**.
- [ ] **7.2** Đo lại toàn bộ: precision từng **tier**, từng **rule**, từng **sách** (bắt buộc vì
  STT4 lệch DPI), kèm Clopper-Pearson + Horvitz-Thompson.
- [ ] **7.3** Khai thác lại lớp nhiễu hệ thống, **có hiệu chỉnh Benjamini-Hochberg** (mẻ cũ không
  hiệu chỉnh đa so sánh). Xác nhận hoặc **hoàn 1.924 ô 㝵 về GOLD**.
- [ ] **7.4** Đo S3 **trên SILVER**, không trên GOLD — tỉ lệ lỗi SILVER 14–28% nên n = 300 cho
  50–80 ca lỗi; GOLD chỉ 2–3% nên cần 2.000–6.000 ô. **Rẻ hơn 10 lần cho cùng kết luận.**
  Bắt buộc holdout page-disjoint sạch (bộ cũ rò rỉ 74,7% vào train).
- [ ] **7.5** Chấm TEST (~1.050 ô), **mở một lần duy nhất**, không chỉnh gì sau khi mở.

---

# KHỐI 8 — ĐÓNG GÓI & VIẾT

- [ ] **8.1** Chốt chặn **Pre-Export**: Bước 7 **từ chối chạy** nếu SHA256 của `labels_final.csv`
  khác hash ghi trong **manifest của mẻ audit**. *(Không phải so hash Bước 6 với Bước 7 — đó chỉ là
  toàn vẹn trong một lần chạy, không phải "bộ đem đo = bộ đem nộp".)*
- [ ] **8.2** Chạy Bước 7 + `pipeline/publish/` (Frictionless + Croissant + datasheet Gebru + HF
  Parquet, CI gate 11/11 — đã viết xong, chỉ chạy ngoài `run_pipeline.sh`).
- [ ] **8.3** Datasheet khai đủ **5 giới hạn**: 3 sách 1 nét chữ · **STT4 ~201 DPI vs 302 DPI** kèm
  precision tách theo sách · SYLLABLE là nhãn cấp âm tiết, 316 cặp không nguồn nào xác nhận ·
  SILVER công bố riêng gắn nhãn "uncalibrated" · κ đo được + lịch sử mẻ phán quyết bị huỷ.
- [ ] **8.4** Viết luận văn theo ba đóng góp ở dưới.

---

# CÁI GÌ DỨT KHOÁT **KHÔNG** LÀM

| hạng mục | lý do (đã đo) |
|---|---|
| **SVTR** | Là mô hình nhận dạng **chuỗi text-line**; tác vụ là phân lớp **một ký tự đã cắt sẵn**. Bộ xếp hạng thật hiện đã là **đầu ArcFace 1.591 lớp**, không phải backbone |
| **IDS cho mô hình thị giác** | Tiền đề sai: **1.582/1.582 lớp đều có codepoint**; ví dụ ⿰口巴 xuất hiện **0 lần**. (IDS **có** ích cho mô hình hoá lỗi OCR — mục 5.3) |
| **Mở từ điển lên 120.000 cặp** | `kDefinition` sinh **0 cặp**; `kVietnamese` chỉ +109 ròng. Đổ `dict_gap` vào từ điển đẩy **20.144 ô** lên GOLD (+40,2%) mà không ô nào được xác nhận |
| **Sửa dấu thanh nhóm B** | Chạm **0 ô** bộ nhãn công bố; áp dụng sẽ **phá 170 cặp đang đúng** |
| **Hợp phiếu có VLM** | Vi phạm ràng buộc gốc: phán đoán của AI không bao giờ trở thành nhãn |
| **Đặt KPI Error-AUC "0,850" hay "0,720–0,750"** | Cả hai đều là số tự chế, không dựa trên công trình nào |
| **Dùng `sem_score` để hạ cấp** | 㝵/"người" (sai) và 主/"chúa" (đúng) cùng điểm 0,0 |

---

# BỐN RÀNG BUỘC BẤT DI BẤT DỊCH

1. **Phán đoán của AI không bao giờ trở thành nhãn** — chỉ để xếp hạng việc cho người.
2. **KPI phải là đại lượng đo được kèm khoảng tin cậy**, không phải số lượng.
3. **Mọi con số phải có lệnh tái sinh**, và phải nói rõ đo trên commit nào.
4. **Nếu một thí nghiệm đã chạy, phải nêu kết quả cũ trước khi đề xuất chạy lại.**

---

# LUẬN ĐỀ CUỐI — ba đóng góp

1. **Phương pháp gán nhãn cấp ký tự bằng cầu song ngữ**: NW có băng + ma trận chi phí từ điển +
   tier hoá theo giao tín hiệu, và đặc biệt **quy trình khai thác lớp nhiễu hệ thống bằng kiểm
   định Fisher có hiệu chỉnh FDR** — chuyển giao được sang bất kỳ corpus nào gán nhãn bằng cầu từ
   điển.
2. **Kết quả âm có đo đạc về tín hiệu thị giác**, kèm phân tích lực thống kê và phát hiện rò rỉ
   74,7% — cộng đồng vẫn mặc định so khớp glyph hoạt động trên chữ Nôm.
3. **Bộ dữ liệu ~57.000 ô có precision đo được kèm CI**, chuẩn quốc tế, chuỗi bằng chứng SHA256
   khép kín, và **datasheet khai cả những gì đã hỏng**.

Điều thứ ba phân biệt luận văn tốt với luận văn trung bình trong lĩnh vực này: phần lớn bộ dữ liệu
di sản được công bố kèm những con số chất lượng không ai tái lập được. Đề tài này đã tự phát hiện
mình ở trong tình trạng đó và sửa — **viết thẳng ra thì đó là chương hay nhất của luận văn**.

---

# T4 — CẮT ẢNH & CHẤT LƯỢNG CROP ✅ HOÀN THÀNH 2026-08-24

Công cụ mới: `pipeline/lab/crop_grid.py` (lưới 48 cấu hình + luật tiền đăng ký),
`pipeline/lab/crop_iou.py` (ngữ liệu tổng hợp), `pipeline/lab/crop_purity.py` (thước đo
liên thông). Luật quyết định commit ở `fd7da32cd9` **trước** khi chạy; bản siết ở `d6229a9c8a`.

## Kết luận: GIỮ MỐC (pad 0,12 · ngưỡng 128 · carve BẬT · resolve TẮT)

Luật tiền đăng ký tự trả `GIỮ MỐC`: trong 12 cấu hình của nhóm pad-khoá, **mốc chính là
cực đại** `flag_ok` (0,9201). Không có điều kiện nào trong (2)(3)(4) được thoả.

## 4 lỗi cấu hình / mã tìm ra

| # | phát hiện | bằng chứng |
|---|---|---|
| T4.a | **`crop_pad_frac: 0.18` là dòng cấu hình CHẾT** | `build_dataset` không đọc nó; `--pad` default **0,12**; `run_pipeline.sh:333` không truyền → **toàn bộ 66k crop đã giao cắt ở pad 0,12**, không phải 0,18 như `config/pipeline.yaml:70` khai. Đặc tả T4.2 của tôi in đậm 0,18 là "hiện tại" — sai mốc. |
| T4.b | **`resolve_overlap` là MÃ CHẾT** | chỉ định nghĩa trong `crop_quality.py`; đường build chưa từng gọi. Bật lên thì **xấu đi**: `flag_ok` 0,9352 → 0,9317. Giữ tắt, nay có số để biện minh. |
| T4.c | **Trục "ngưỡng siết hộp" KHÔNG THỂ có tác dụng** | ảnh trang **đã nhị phân sẵn**: 8,4% pixel < 64 · 91,6% > 192 · **0,0% trong khoảng 64–192**. Nên Otsu/Sauvola cho **đúng cùng một mặt nạ** với ngưỡng 128: **0/893 hộp** cho hộp siết khác nhau, và cả 48 cấu hình giống nhau tới 6 chữ số. Trục này đóng **vĩnh viễn**, kèm lý do — mọi kỹ thuật nhị phân hoá thích nghi đều vô nghĩa trên `pages/`. |
| T4.d | **`flag_ok` KHÔNG có thẩm quyền chọn `pad`** | tăng đơn điệu tới pad 0,60 (0,9224 → 0,9738), không cực đại nội. Lý do cấu trúc: `border_ink` = mực chạm mép nên pad lớn thì giảm theo định nghĩa; `stray_ink` chỉ tính dải giữ < 35% mực **và** nằm trong 30% trên/dưới → chữ láng giềng **lọt trọn** (giữ ~33%, trải quá 30%) **không bị tính**. Nó bắt mảnh vụn, không bắt láng giềng nguyên chữ. |

## Phát hiện cấu trúc: mọi hộp ký tự cao 1,20 × BƯỚC LẶP

`align_production.py:155-163` dựng hộp từ trung điểm với `m = pitch * 0.10`, nên
cao hộp = `pitch + 2m` = **pitch × 1,20** *theo thiết kế* (chú thích mã: "so tall glyphs
keep their tails"). Đo trên 537 cột / 60 trang: **trung vị 1,2069** (p10 1,183 · p90 1,247),
**99,44% cột** có hộp cao hơn bước lặp, giống nhau ở cả 3 sách (1,2015 / 1,2070 / 1,2164).

Hệ quả trực tiếp: **84,46% cặp chữ liền kề trong cùng cột CHỒNG nhau** (khe hở trung vị
**−0,171** chiều cao hộp; 4.344 cặp / 30 trang). Cộng pad 0,12 thì **cửa sổ crop = 1,49 ×
bước lặp**, tức luôn trùm sang ~24,5% chữ trên và chữ dưới.

**Bộ dò CenterNet không tách khỏi quy ước đó.** Nó cấp **44,62%** hộp (không phải "gần như
không dùng" như tôi đoán ban đầu — 55,38% là midpoint, trộn lẫn *trong cùng* một cột, trung
vị 52,6%/cột). Nhưng tỷ lệ cao/bước-lặp của hộp CenterNet là **1,2063**, gần trùng midpoint
**1,2069**: bộ dò — huấn luyện trên GT do chính pipeline này sinh — **đã học lại đúng mức nới
1,2×**. Đây là một vòng tuần hoàn, khớp với "CenterNet detector ceiling" đã ghi trước đó.

## `carve_neighbor_ink` là bộ phận CHỊU LỰC, không phải tô điểm

Lần đầu định lượng, gộp mọi pad/ngưỡng/resolve (48 cấu hình, 9.404 hộp/60 trang):

| carve | `flag_ok` | `flag_truncated` |
|---|---|---|
| TẮT | 0,3167 | **0,6635** |
| BẬT | **0,9335** | 0,0163 |

**+61,7 điểm phần trăm.** Không có carve thì 2/3 crop bị cờ "cắt thiếu" — đúng hệ quả phải
có của việc hộp cao 1,2× bước lặp. Đây là lời biện minh định lượng đầu tiên cho `bbox_fix.py`.

## Vì sao pad 0,12 là điểm làm việc đúng (đo bằng thước đo thay thế)

`flag_ok` không quyết được `pad`, nên dựng thước đo dựa trên **liên thông** (`crop_purity.py`):
mực "của mình" = thành phần liên thông giao với ô bước lặp; đo *bị cắt* theo đúng ngưỡng của
`border_ink` (> 0,20 hàng biên), không phải "có pixel nào chạm mép".

| pad | bị cắt thật | mực của mình ngoài ô (tb) | crop có > 20% ngoài ô |
|---|---|---|---|
| 0,00 | **51,7%** | 13,0% | 24,3% |
| **0,12 (MỐC)** | **0,4%** | 19,1% | 51,1% |
| 0,45 | 0,2% | 28,1% | 69,0% |

pad 0,12 **đã xử lý xong việc cắt thiếu** (0,4%, khớp `flag_truncated` 0,69% của production).
Tăng lên 0,45 mua thêm **0,2 điểm phần trăm** nhưng đẩy dính-láng-giềng từ 51% lên 69% crop.
Nên GIỮ MỐC không phải vì thiếu số liệu, mà vì đã đo và mốc thắng.

## Ba lỗi CỦA TÔI trong lúc làm T4

1. **Đề xuất "đệm bất đẳng hướng" bị chính phép đo của tôi bác.** Tôi thấy chỗ dư dọc ≈ 0
   (trung vị 0,000; p90 0,013–0,025) và kết luận "hộp sát mực theo chiều dọc → nên đệm dọc
   nhiều hơn". Sai: dư ≈ 0 là vì **mực láng giềng đã lấp kín phần dư**, nên `tighten_box`
   không co được. Trục dọc **đã** nới 1,2× rồi; đệm thêm là đi ngược. Đã cài `cut()` nhận
   `(pad_x, pad_y)` và giữ lại như công cụ, nhưng **không dùng để đổi cấu hình**.
2. **Phép kiểm "bị cắt" đầu tiên sai ngưỡng.** `own_mask[:2,:].any()` bật khi *một* pixel
   chạm mép → báo 50,36% ở pad 0,12, chọi với `flag_truncated` 0,69%. Đặt lại theo đúng
   ngưỡng của `border_ink` thì ra 0,4%.
3. **Ngưỡng phân loại "dính chữ > 0,50 pitch" NẰM NGOÀI dải hình học.** Cửa sổ chỉ vươn
   0,244 pitch quá mép ô, nên "> 0,50" là **bất khả**; con số "0,00% dính chữ" là hằng đúng,
   **không phải phát hiện**, và tôi rút lại. Điều đo được: mức vươn trung vị **0,2069** trên
   tối đa **0,244** → blob mực lấp ~85% khoảng còn lại của cửa sổ. Câu "bao nhiêu % crop dính
   chữ thật" **vẫn CHƯA đo được**, cần dụng cụ khác.
4. **T4-B (ngữ liệu tổng hợp) KHÔNG dùng được để chọn pad.** `synth` vẽ glyph vừa khít ô nên
   `recall` = 1,0000 ở *mọi* pad — mất hẳn cánh "pad nhỏ thì cắt mất nét". Thêm nữa, hiệu
   chuẩn sai số hộp của tôi **một chiều theo cấu trúc**: nó đo bằng `tighten_box`, mà hàm đó
   chỉ co được *trong* hộp cho sẵn → **không bao giờ quan sát được hộp THIẾU**. Giữ lại tệp
   kèm ghi chú, không dùng làm căn cứ.

## Việc T4 mở ra (chưa làm)

### T4.e ✅ ĐÃ ĐO — `m = pitch * 0.10` nên hạ về **0**, lợi **+3,8 điểm** độ tinh khiết

Dựng lại hộp theo **đúng công thức production** (trung điểm ± `m`) với `m = pitch × M` thay
đổi, cắt bằng cấu hình đã giao (pad 0,12 + carve), đo bằng thước đo liên thông
(`crop_purity.sweep_m`). **9.404 crop / 60 trang.**

| M | cao hộp | độ tinh khiết mực | bị cắt | F1 |
|---|---|---|---|---|
| −0,10 | 0,80×pitch | 0,9570 | 0,0248 | 0,9660 |
| −0,05 | 0,90×pitch | 0,9527 | 0,0200 | **0,9661** ← cực đại |
| **0,00** | **1,00×pitch** | **0,9453** | **0,0179** | **0,9634** ← đề xuất |
| **+0,10** | **1,20×pitch** | **0,9072** | **0,0165** | **0,9438** ← PRODUCTION |

**Thước đo bấu vào trục theo CẢ HAI chiều** (M âm thì `bị cắt` tăng 0,0179 → 0,0248 → 0,0345
→ 0,0508), nên khác ba lần trước, đây là cực đại nội THẬT, không phải nghiệm biên.

**Kết quả bác lý do ghi trong chính mã.** `align_production.py:158` chú thích
`m = pitch * 0.10  # small overlap so tall glyphs keep their tails`. Nhưng **`bị cắt` gần như
KHÔNG đổi theo M** (0,0165 ở M=+0,10 vs 0,0179 ở M=0 — chênh **0,0014, KHÔNG vượt khoảng tin
cậy 95%**). Mức nới dọc **không cứu được đuôi nét nào**; `pad 0,12` đã lo xong việc đó. Nó chỉ
đổi lấy **−3,81 điểm** mực của chính chữ (0,9453 → 0,9072).

Chọn **M = 0** chứ không phải cực đại −0,05: chênh F1 chỉ 0,0027, mà M = 0 có nghĩa hình học
bảo vệ được — **hộp = đúng một ô bước lặp = đúng phần của một chữ trong cột** — còn M âm là
hộp NHỎ hơn ô, rủi ro khi ước lượng bước lặp sai.

🔴 **CHƯA ÁP DỤNG, và không được áp dụng lén.** Đổi `m` là cắt lại toàn bộ 66k ảnh → đổi
`image_md5`, `crop_w/h`, `ink_pct` và mọi hash trong chuỗi bằng chứng. Đúng cảnh báo T4.4 của
đặc tả: *"Đổi crop = đổi ảnh dưới chân bộ nhãn. Chốt cấu hình MỘT LẦN, trước khi rút mẻ chấm tay."*
Phạm vi rủi ro **hẹp hơn tưởng**: GOLD = S1∩S2 không đọc crop nên **thành phần bộ giao nộp
không đổi**; chỉ ảnh đổi, cộng khả năng đổi thành phần SILVER (S3 đọc crop). Quyết định cắt lại
là của bạn, và phải làm **một lần duy nhất** trước KHỐI 6.
- [ ] **T4.f** Vòng tuần hoàn của bộ dò: GT huấn luyện CenterNet do chính pipeline sinh nên nó
  học lại mức nới 1,2×. Muốn thoát phải có hộp do người vẽ (thuộc KHỐI 6).
- [ ] **T4.g** Đo tỷ lệ dính chữ thật — cần dụng cụ khác (xem lỗi 3).

---

# T5 — LUẬT TIER & ĐỘ GIÒN CỦA TỪ ĐIỂN ✅ HOÀN THÀNH 2026-08-24

Luật quyết định tiền đăng ký ở `pipeline/lab/t5_rules.py`, commit `b7b1c39536` **trước** khi
chạy bất cứ phép đo nào. Thi hành bằng 4 trinh sát song song, mỗi báo cáo qua 1 phản biện đối
kháng được lệnh *cố bác bỏ* và tự chạy lại. **Cả 4 phản biện đều trả `MOT_PHAN`** — không báo
cáo nào đứng nguyên vẹn.

## Kết luận 1 — cột `fragility` của đặc tả: BÁC BỎ, không được xây

`dict/QuocNgu_SinoNom.csv` có **104.177 dòng và đúng 104.177 cặp (âm, chữ) phân biệt — không
một cặp nào lặp lại**. Nên "số mục từ điển hậu thuẫn cho nhãn" chỉ nhận 3 giá trị trên bộ giao
nộp 56.909 ô:

| giá trị | số ô | là gì |
|---|---|---|
| 1 | 50.130 (88,09%) | toàn bộ GOLD |
| 0 | 6.755 (11,87%) | **trùng khít tầng SYLLABLE** |
| 2 | 24 (0,04%) | tạo tác gộp chính tả cũ/mới (`choè`/`chòe`), không phải bằng chứng độc lập |

Cột này là **bản diễn đạt lại cột `tier`**, mang **0 bit thông tin mới**. Vi phạm cổng
tiền-đăng-ký **G1.2** (>95% cùng giá trị → loại). *Đầu ra chính mà đặc tả T5 đặt hàng là một
cột không tồn tại được.*

## Kết luận 2 — mọi định nghĩa "độ giòn" thay thế đều hỏng ở chỗ quan trọng nhất

Đã đo 8 định nghĩa (số lần trong ngữ liệu · số sách · số trang · số cột · tỉ phần trong âm ·
số chữ cùng âm · va chạm chữ-giống · …). Phép thử quyết định: **lớp lỗi hệ thống DUY NHẤT dự
án từng xác định được — `㝵`/"người" — đứng ở cực AN TOÀN của mọi định nghĩa đó**:

- hạng **3/2.370** theo tần suất (phân vị 99,92), sau `麻`/mà 1.625 và `朱`/cho 1.488
- có mặt ở **3/3 sách**, **406 trang**, chiếm **90,9%** số ô của âm "người"

Quét bộ giao nộp theo tần suất tăng dần thì **phải chấm tay 94,7% số ô mới chạm tới nó**. Nói
cách khác: **chiến lược "chấm đuôi hiếm" bỏ sót đúng loại lỗi nguy hiểm nhất.** Đây là lý do
định lượng để KHÔNG rút mẻ chấm tay theo độ hiếm.

## Kết luận 3 — phản thực từ điển: con số THẬT, và nó có hai chiều

Phản biện phát hiện `align_page` **chạy lại được READ-ONLY** từ `prepared/*/detected/*_ocr_cache.json`
+ `train_crop/detector_r34.best.pt`: **445 trang / 280 giây, 0 lần gọi API, 0 dòng mã sửa**, tái
lập `labels.csv` **chính xác tuyệt đối** (82.246 cặp, 4.003 cột, 0 lệch). Nên phản thực không cần
xấp xỉ bằng cận — đo thẳng được.

**Bỏ ngẫu nhiên 10% mục từ điển (104.053 → 93.648): GOLD 51.601 → 48.923 = −2.678 ô (−5,19%).**

Điều dễ bỏ sót và trinh sát đầu đã bỏ sót: **bỏ mục từ điển cũng TẠO ra GOLD**. Khi số cầu nối
tụt từ ≥2 xuống đúng 1, luật bắc cầu kích hoạt — GOLD-similar **tăng** 4.102 → 4.645. Vì chỉ
đếm một chiều nên "dải cận 47.184–47.264" của trinh sát **không chứa giá trị thật 47.268**.

Ba bất biến đo được kèm theo: `column_count_matched` **bất biến** với từ điển (0/4.003 cột lật);
bóc tách QN `parse_v5` **không đổi** (0/445 trang); nhưng ghép đôi crop↔âm tiết **KHÔNG bất biến**
(212/4.003 cột = 5,30% đổi chuỗi).

## Kết luận 4 — thước đo chất lượng mà CHÍNH TÔI tiền-đăng-ký là SUY BIẾN

Trong `t5_rules.py` tôi viết rằng thước đo chất lượng *duy nhất được phép* cho `syllable_gate` là
"tỷ lệ cặp qua cổng mà từ điển cũng công nhận", với lập luận rằng cổng không đọc từ điển nên từ
điển là trọng tài độc lập.

**Lập luận đó sai, và sai theo cách vòng tròn kinh điển.** Hồ chưa-xác-nhận mà cổng ăn vào được
`decide_label` **định nghĩa** chính là "ocr_char KHÔNG nằm trong `qn_to_nom[âm]`". Nên tỷ lệ ấy
**bằng 0,0% ở CẢ 100 tổ hợp ngưỡng** — và phản biện còn siết chặt hơn: **0/17.714 dòng** trong hồ
có thể thoả, *bất khả theo cấu trúc*. Đo nó là đo lại chính định nghĩa.

Ghi lại đây vì nó đúng là cạm bẫy mà tệp luật ấy được viết ra để chặn, và tôi vẫn rơi vào.

## Kết luận 5 — `syllable_gate`: GIỮ MỐC 5/3/0,6, và một no-op

Cổng **tái lập hoàn hảo ngoại tuyến**: 310 cặp → đúng **6.753 ô**, 0 sai-dương / 0 sai-âm, giống
hệt trên `labels.csv` lẫn `labels_final.csv` → **các bước 4/5/6 không đụng vào tầng SYLLABLE**
(giả thuyết ngược của tôi bị bác). Cổng G2.1 qua.

- **`min_pages = 3` là NO-OP**: ở `min_occ = 5`, đặt 1/2/3 cho kết quả **giống từng byte**. Cổng
  thực chất chỉ có **hai** tham số. (Nó chỉ cắt thật từ `min_pages = 5`: 310 → 301 cặp.)
- **Không tham số nào có cực đại nội.** `min_purity` từng có vẻ đỉnh ở 0,65, nhưng chạy 8 seed
  cho thấy 0,60/0,65/0,70/0,80 chỉ chênh 0,1–0,6 điểm trong khi nhiễu Monte-Carlo là 0,35–1,03 —
  chỉ là **bình nguyên phẳng 0,6–0,8**. Toàn bộ phép quét suy biến → **GIỮ MỐC** theo đúng luật.
- Lập luận "nằm trên Pareto front" là **vô nghĩa**: 53/100 tổ hợp cũng nằm trên đó. Con số phải
  nêu là **hạng 52/100** — tức mức trung vị.

## Kết luận 6 🔴 — cổng SẠCH lớp `㝵` chỉ nhờ TAI NẠN THỨ TỰ BƯỚC

Đo được: nếu `confusion_fix` (bước 6) chạy **TRƯỚC** cổng — một cách sắp xếp lại hoàn toàn tự
nhiên — cổng cho **321 cặp / 8.534 ô**, và **12 cặp MỚI đều là âm "người"**, gồm `(㝵, người)`:

> `(㝵,人)(冐)(冒)(哥)(子)(尋)(早)(旱)(景)(畢)(耳)(𭘾)` — tất cả với âm "người"

Tức cổng sẽ **nuốt trọn lớp lỗi hệ thống đã được chứng minh và tẩy trắng 1.781 ô thành SYLLABLE**.
Cổng hiện sạch `㝵` **không phải vì nó có cơ chế bác lỗi OCR lặp đều — nó không có cơ chế đó** —
mà vì thứ tự bước hiện thời cộng với việc từ điển tình cờ công nhận `㝵 → người` (nên lớp ấy lên
GOLD chứ không rơi vào hồ của cổng). Đây là **ràng buộc thứ tự bước phải ghi vào tài liệu và canh**.

## Kết luận 7 — cầu nối top-K: cổng hợp lệ CHỈ QUA MỘT NỬA

`SinoNom_Similar.csv` **không có cột điểm hay thứ hạng**. Nhưng kiểm gián tiếp bằng **tính đáp lễ**
cho thấy 20 vị trí đầu **thật sự được sắp** theo độ giống — tỷ lệ được đáp lễ giảm đơn điệu
**94,14% (hạng 1) → 25,29% (hạng 20)** — còn **từ hạng 21 trở đi là phụ lục đối xứng VÔ THỨ TỰ**
(đáp lễ đúng **100,00%** ở mọi hạng). Phản biện tấn công bằng nhiễu độ dài và null xáo trộn: tín
hiệu sắp thứ tự **sống sót**.

Nên **G3.1 chỉ qua với K ≤ 20**; cắt ở K > 20 là cắt ngẫu nhiên. Và **548/3.829 ô (14,3%) GOLD-similar
đang lấy cầu từ chính phần vô thứ tự đó**, với hạng NGƯỢC trung vị 23 (chỉ 7,5% nằm trong top-20 có
thứ tự của chữ cầu) — tức **không có chỗ dựa thứ hạng ở CẢ HAI chiều**.

**Đường cong sản lượng KHÔNG đơn điệu**, và điều đó còn tệ hơn: quét đủ K = 1…89 thì cực đại là
**NỘI TẠI ở K = 28** (3.840 ô) > K = full (3.827). Nhưng K = 28 **nằm giữa phụ lục vô thứ tự**, nên
đó là một **cực đại GIẢ** — tối ưu theo sản lượng sẽ ra một con số trông như phát hiện mà thực chất
vô nghĩa. Đây là biến thể tinh vi hơn của bẫy T3/T4: không phải nghiệm biên, mà là **nghiệm nội tại
giả**.

## Kết luận 8 — phép đo ĐỘC LẬP đầu tiên về 548 ô đáng ngờ: KHÔNG có dấu hiệu xấu

Phản biện đề xuất một kênh **không vòng tròn với ArcFace**: Unihan kDefinition (`sem_score.py`).
Tôi chạy:

| nhóm | n chấm được | > τ_confirm |
|---|---|---|
| vùng có thứ tự (hạng ≤ 20) | 2.483 | 12,6% |
| **phụ lục vô thứ tự (hạng ≥ 21)** | 528 | **26,1%** |

Nhóm phụ lục **cao hơn**, và khác biệt **sống sót qua phân tầng** theo độ giàu nghĩa (tầng 1–3:
10,4% vs 4,3%, p = 0,005; tầng 4–7: 35,3% vs 21,2%, p = 1,4e-7).

**Diễn giải phải rất dè dặt**, theo đúng tính chất đã biết của kênh này: Unihan **chỉ phong, không
bao giờ hạ** (47,7% cặp đúng cũng cho điểm 0). Nên đây là **"không có bằng chứng 548 ô đó xấu hơn"**,
**KHÔNG phải "bằng chứng chúng tốt"**. Còn một nhiễu chưa khử hết: nhãn cầu ở nhóm phụ lục **98,0%
là chữ Hán chuẩn** so với 79,4% ở nhóm kia, nên hai nhóm khác nhau về **loại chữ** (Hán chuẩn vs Nôm
tự tạo), và điểm cao hơn có thể chỉ phản ánh "mượn nghĩa nhiều hơn" chứ không phải "đúng nhiều hơn".

## Lỗi mã CÒN SỐNG tìm ra và ĐÃ VÁ: `nan` là một ÂM TIẾNG VIỆT

`nan` (難) là âm Quốc ngữ thật. pandas mặc định đọc chuỗi `nan`/`NA`/`NULL`/`None`/`null`/`NaN`
thành `NaN`, và **không một chỗ nào trong toàn bộ mã dùng `keep_default_na=False`**. Hậu quả đo được:

- **3 ô mất hẳn âm** khi đi qua `remediation/cli.py`, trong đó **2 ô là GOLD `s1_inter_s2_similar`
  NẰM TRONG bộ giao nộp**: `gold/stt4_page_0040_c02_023.png` và `gold/stt4_page_0108_c08_178.png`
  (ocr = 准, label = 难)
- **8 mục từ điển biến mất** mỗi lần `QuocNgu_SinoNom.csv` được đọc bằng pandas
- 81 ô trong `labels.csv` có âm nằm trong danh sách NA của pandas

Vá **9 chỗ** bằng `keep_default_na=False, na_values=[""]` — ô RỖNG vẫn thành NaN (`s3_cosine` cần
thế: 49.156 NaN trước và sau, không hồi quy) nhưng mọi chuỗi có nội dung được giữ. **+8 test hồi quy**,
mốc selftest **553 → 561**. May mắn: `core.text.dictionary.load_qn_to_nom` dùng `csv` thuần nên
đường nạp từ điển CHÍNH luôn an toàn.

## Đính chính số đã công bố

`BANG_SO_LIEU_CHINH_THUC.md` ghi lớp `㝵`/"người" bị hạ **1.977** ô. Con số đó **đúng** về tổng số
dòng bị hạ, nhưng luật khớp trên **(syllable, label)** nên 1.977 = **GOLD 1.445 + SILVER 532**.
**Phần chạm bộ GIAO NỘP chỉ là 1.445** — 532 ô SILVER chưa bao giờ nằm trong bộ giao nộp. Cần nêu
tách bạch, nếu không người đọc sẽ hiểu là bộ giao nộp mất 1.977 ô.

## Việc T5 mở ra (chưa làm)

- [ ] **T5.a** Vá 2 ô GOLD mất âm — gộp vào lần chạy lại MỘT LẦN trước KHỐI 6, cùng với T4.e.
      Không vá riêng lúc này (ràng buộc tiền-đăng-ký: không đổi bộ nhãn đã công bố trong lượt này).
- [ ] **T5.b** Canh ràng buộc THỨ TỰ BƯỚC: `syllable_gate` phải chạy TRƯỚC `confusion_fix`. Viết
      thành phép kiểm chạy được, không phải ghi chú — nếu không, một lần sắp xếp lại vô hại sẽ tẩy
      trắng 1.781 ô lớp lỗi đã chứng minh.
- [ ] **T5.c** 548 ô cầu-phụ-lục: đưa vào mẻ chấm tay KHỐI 6 như một tầng riêng. Đây là nhóm duy
      nhất T5 xác định được bằng lý do CẤU TRÚC (không có chỗ dựa thứ hạng ở cả hai chiều) thay vì
      bằng tần suất — và nó chỉ 548 ô, rẻ.
- [ ] **T5.d** `min_pages` bỏ khỏi cổng hoặc đặt ≥ 5 nếu muốn nó có tác dụng; hiện là no-op gây
      hiểu nhầm rằng cổng có 3 lớp bảo vệ trong khi chỉ có 2.

---

# T6 — TÁI LẬP & BẰNG CHỨNG ✅ PHẦN TẤT ĐỊNH HOÀN THÀNH 2026-08-24

## Kết quả chính: pipeline TẤT ĐỊNH TỚI TỪNG BYTE, tới tận pixel

Chạy lại **bước 3** (`build_dataset --use-s3 --reseg detector`) hai lần trên cùng đầu vào, ghi
vào hai thư mục nháp tách biệt (bộ đã công bố **không hề bị đụng**):

| đầu ra | lần A | lần B | |
|---|---|---|---|
| `labels.csv` | `9c11f8b940d1d637…` | `9c11f8b940d1d637…` | ✅ trùng từng byte |
| `summary.json` | `549ed52b1f9f28ca` | `549ed52b1f9f28ca` | ✅ trùng |
| **51.601 ảnh `gold/`** | — | — | ✅ **trùng hết** |
| **11.090 ảnh `silver/`** | — | — | ✅ **trùng hết** |
| **6.911 ảnh `syllable/`** | — | — | ✅ **trùng hết** |

**69.602 ảnh crop trùng nhau từng byte.** Điều này trả lời rủi ro sắc nhất tôi đã khoanh trước
khi đo: **bộ dò CenterNet chạy trên MPS (Apple GPU) LÀ tất định** giữa các lần chạy — nó cấp
44,62% toàn bộ hộp ký tự, nên nếu nó trôi thì mọi tiêu chí T6 đều sụp.

**Toàn pipeline bước 3→7 chạy hai lần, nối chuỗi đầy đủ** (build → remediate → confusion_fix →
s3_unwind → export), mỗi lần từ đầu ra của chính nó:

| đầu ra | lần A | lần B | |
|---|---|---|---|
| `labels_remediated.csv` | `3050ff8f5b43ff7323` | `3050ff8f5b43ff7323` | ✅ |
| `labels_final.csv` | `0eec0e119c2867627e` | `0eec0e119c2867627e` | ✅ |
| `dataset/labels.csv` | `9525aeb480b1b6a9c0` | `9525aeb480b1b6a9c0` | ✅ |

Bộ giao nộp tái lập được: **GOLD 50.156 — y hệt bản công bố** — cộng SYLLABLE 6.911, tổng
**57.067** (công bố 56.909). Chênh +158 đúng bằng phần SYLLABLE đã giải thích ở dưới.

Các bước hạ nguồn đo riêng, chạy hai lần trên cùng đầu vào:

| bước | tất định | tái lập bản công bố |
|---|---|---|
| 4–5 (remediate → confusion_fix) | ✅ trùng từng byte | — |
| 4–5–6 (thêm s3_unwind) | ✅ trùng từng byte | **lệch ĐÚNG 3 dòng** |
| 7 (export) | ✅ trùng từng byte | ✅ **trùng tuyệt đối** (`236cbc4f…`) |

**3 dòng lệch đó chính xác là 3 ô `syllable='nan'`** — tức **phép vá pandas của T5 đang hoạt
động đúng như thiết kế**, và **không có bất kỳ sai lệch nào khác**: 0 ô đổi tier, 82.247/82.247
dòng còn lại trùng khít. Đây là một phép tự kiểm chứng chéo rất chặt: thứ duy nhất khác là thứ
tôi cố ý sửa.

## Đính chính đặc tả: tiêu chí "đổi thứ tự sách → byte-identical" KHÔNG đạt được như câu chữ

Đo bằng cách đảo ngược danh sách `books` trong config (6 trang/sách):

| | kết quả |
|---|---|
| trùng từng byte | 🔴 **KHÔNG** |
| nội dung sau khi sắp | ✅ **giống hệt** (3.398 dòng, cùng tập) |
| thứ tự dòng | khác |

`build_dataset.py:231` duyệt `for b in config["books"]` **theo thứ tự cấu hình, không sắp**, rồi
ghi dòng theo thứ tự đó. Nên tiêu chí đúng phải là **bất biến theo thứ tự** (đạt), không phải
byte-identical (không đạt, và không thể đạt trừ khi sắp dòng trước khi ghi).

## Đính chính một tuyên bố của tôi ở KHỐI 1

Tôi đã viết rằng sinh lại `index.csv` **"không đụng bộ giao nộp"**. Đo được nay cho thấy **đúng
một nửa**. So lần A với bản công bố:

| tier | công bố | lần A | chênh |
|---|---|---|---|
| GOLD | 51.601 | 51.601 | **+0** ✅ |
| SILVER | 11.332 | 11.090 | −242 |
| REVIEW | 12.560 | 12.644 | +84 |
| **SYLLABLE** | **6.753** | **6.911** | **+158** 🔴 |

GOLD đứng yên **tuyệt đối**, đúng bất biến trong mã (GOLD = S1∩S2 trả về **trước** mọi lần đọc
S3). Nhưng **SYLLABLE nằm TRONG bộ giao nộp** và nó tăng 158 ô — vì S3 đổi quyết định SILVER,
làm đổi hồ chưa-xác-nhận mà `syllable_gate` ăn vào. Vậy lần dựng lại tới, **bộ giao nộp sẽ tăng
+158 ô**. Đó là hệ quả BIẾT TRƯỚC của một đầu vào tôi đã cố ý sửa, không phải bất tất định —
phép thử A≡B chứng minh điều đó.

## Ba khoảng trống bằng chứng đã vá

| # | khoảng trống | vá thế nào |
|---|---|---|
| E1 | `detector_r34.best.pt` (82 MB, **cấp 44,62% hộp**) chưa từng được băm — chỉ được *nhắc tên* trong một câu văn xuôi ở `EVIDENCE_INDEX.md:220` | đưa vào `evidence()`, cùng `config/pipeline.yaml`. Chuỗi **8 → 10 tệp**, `check_evidence` 10/10 |
| E2 | `evidence()` ghi commit SHA nhưng **không ghi cây làm việc bẩn hay sạch** — commit SHA một mình không định danh được lần chạy | in rõ SẠCH/BẨN kèm danh sách tệp; bỏ qua submodule vì `nom-embed` luôn bẩn do artefact con trỏ LFS |
| E3 | cache nguyên mẫu S3 **ký bằng mtime chứ không bằng băm nội dung** (`visual_signal.py:195`) | thêm phép kiểm `R2`; và **bỏ theo dõi git** tệp cache 7 MB đó |

Về E3, hai hệ quả đo được: (a) bản `s3_proto_cache.pkl` commit trong git có chữ ký **không bao
giờ khớp sau một lần clone** (mtime là lúc checkout) nên là 7 MB chết; (b) nó bị **ghi đè mỗi
lần build**, nên sau bất kỳ lần chạy nào cây làm việc cũng bẩn — làm hỏng vĩnh viễn phép kiểm E2.
Đã `git rm --cached` + `.gitignore` (tệp vẫn nguyên trên đĩa và tự tái sinh).

## Chốt chặn mới: `pipeline/tools/repro_check.py`

Theo đúng nếp "mỗi lớp lệch phát hiện được phải để lại một lệnh bắt được nó". Gắn vào
`check_consistency.sh`, nay **4 phép kiểm** thay vì 3:

- **R1** cây làm việc sạch — nếu bẩn thì commit trong chuỗi bằng chứng không định danh được mã
- **R2** chữ ký cache nguyên mẫu khớp mtime hiện tại của `index.csv` + checkpoint
- **R3** **không có RNG nào tới được đường build** — tính tất định hiện nay dựa **hoàn toàn** vào
  bất biến này, mà trước đó không ai canh; thêm một `random.shuffle` là mất tất định trong im lặng

R3 bắt được `torch.randn` ở `nom_classifier/model.py:50`. Kiểm ra là **dương tính giả**:
`ArcMargin` là đầu ArcFace *chỉ dùng khi huấn luyện*, `grep 'ArcMargin('` chỉ khớp đúng dòng định
nghĩa lớp (không nơi nào khởi tạo), `infer.py:31` chỉ nạp `ck['backbone']`, và giá trị ngẫu nhiên
còn bị `xavier_uniform_` ghi đè ngay dòng sau. Đưa vào danh sách miễn trừ **có ghi lý do đã kiểm
chứng**, để nếu sau này ai làm nó thành đường sống thì phải gỡ miễn trừ một cách có ý thức.

## Điều kiện phụ đã đo

- **Không tệp đầu ra nào nhúng dấu thời gian hay đường dẫn tuyệt đối** (7 tệp `.json` + `labels.csv`)
  → tiêu chí byte-identical là đạt được **về nguyên tắc cho mọi tệp**, không riêng `labels_final.csv`.
- **Selftest 561** — vượt xa mốc T6 yêu cầu (≥ 450).
- `evidence()` **có** ghi commit SHA của lần chạy (không phải khoảng trống như tôi đã ngờ).

## Còn khuyết (nói thẳng)

- [ ] **T6.a Chưa kiểm được trên Linux/Docker.** Máy này là macOS arm64 và **không có Docker**.
  Tất cả kết quả tất định ở trên là **cùng-máy, cùng-thiết-bị**. Câu hỏi "MPS và CPU có cho cùng
  hộp không" và "Linux có ra cùng sha256 không" **vẫn chưa trả lời được** — mà đó chính là câu
  quyết định cho người thứ ba tái lập.
- [ ] **T6.b** Sắp dòng trước khi ghi `labels.csv` (theo `book, page, column, chỉ-số-ký-tự`) thì
  tiêu chí "đổi thứ tự sách → byte-identical" sẽ đạt được thật. Rẻ, nhưng đổi thứ tự dòng của bản
  đã công bố → gộp vào lần chạy lại MỘT LẦN trước KHỐI 6, cùng T4.e và T5.a.
