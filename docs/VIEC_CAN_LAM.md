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
