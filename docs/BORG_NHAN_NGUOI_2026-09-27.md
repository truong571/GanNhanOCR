# Bộ crop NHÃN NGƯỜI Borg.tonch 18 + 34 (`dataset/_BORG_NHAN_NGUOI/`) — 27/09/2026

**Là gì.** Crop chữ Nôm viết tay có nhãn = chữ **người phiên** gõ (bản phiên Excel đi kèm 2 bản chép tay Vatican DigiVatLib:
Borg.tonch.18 = *Sách kinh Thầy cả Bỉnh*, 529 trang; Borg.tonch.34 = *Sách Dũng Lý Hộ Thần*, 112 trang). Máy chỉ **gióng**
chuỗi chữ người của mỗi trang với hộp chữ trên ảnh — không đặt nhãn. Bộ TÁCH khỏi GOLD tự động (`dataset/<Bộ>/`, `dataset/_ALL/`):
thư mục riêng, `cell_uid` riêng tiền tố `BORG/`, không vào bước gộp. 0 API, không web, không tải model.

## 1. Mã và cách chạy

| tệp | vai trò |
|---|---|
| `pipeline/borg_human/params.py` | mọi tham số ĐÓNG BĂNG + nguồn (script vòng 4/5) + số gốc để kiểm tái lập |
| `pipeline/borg_human/geom.py` | cột phải→trái, chuỗi chữ người, đơn vị (hộp thật + ô ảo) — port `blib.py`, `a04_units.py` |
| `pipeline/borg_human/encoders.py` | encoder v1 (`nom-embed`) / v2 (`ArcFace`) + glyph font/FD — port `score_visual.py` |
| `pipeline/borg_human/align.py` | DP gióng Viterbi + forward-backward, phát xạ font → nguyên mẫu khối khác (2 vòng) — port `a05_align.py` |
| `pipeline/borg_human/paddle.py` | lọc vòng 5 trên hồ sơ Paddle đóng băng: keep_paddle_ok, chuẩn hoá, keep_v5, θ — port `qlib/q06/q07/q14` |
| `pipeline/borg_human/crops.py` | crop `save_crop` (như `a08_crops.py`) + crop chuẩn v2 (`gold_exact/crop_chuan.canon`) + `one_char_ok` |
| `pipeline/borg_human/export.py`, `__main__.py` | labels.csv, README, DATASHEET, BUILD_INFO.json, CHECKSUMS.txt; CLI theo bước |
| `scripts/measure/borg_human_eval.py` | nghiệm thu 20 invariant; đăng ký bước `borg_human` trong `measure.py` |

```bash
.venv/bin/python -m pipeline.borg_human --stage all       # ≈ 11 phút MPS (detect 4,5′ · units 3,5′ · align 44″ · crop 66″ · validate 1′), ≤ 3 worker
.venv/bin/python scripts/measure/borg_human_eval.py       # hoặc: measure.py --all --steps borg_human
```
Trung gian ở `measure_out/_borg_human/` (hộp, đơn vị, nhúng, 6 lượt căn, `repro.json`). Đầu vào đóng băng duy nhất:
`measure_out/_borg_human_src/prof_borgx.pkl` (hồ sơ thời gian Paddle PP-OCRv6 của vòng 5, sha256 `53d3c64e…`, kèm mã sinh +
SOURCE.json) — venv Paddle không còn trên máy, cài lại = tải gói (cấm), nên bước Paddle KHÔNG chạy lại mà đọc hồ sơ; mã kiểm
TRÙNG 159.291/159.291 đơn vị (uid, y1, y2, ảo) trước khi dùng. **Hai thư mục `measure_out/_borg_human_src/` và kho lưu
`measure_out/_audit_2026-09-26/` cần sao lưu** (measure_out không commit).

## 2. Kiểm tái lập so với vòng 4/5 — TRÙNG TỪNG BIT

| đại lượng | vòng 4/5 | chạy lại | lệch |
|---|---:|---:|---:|
| hộp detector (641 trang) | r4 `boxes_v1.json` | 641/641 trang trùng | 0 |
| đơn vị / nhúng / glyph | r4 `units_v1.pkl`, `emb_v1.npy`, `glyph_emb.pkl` | 159.291 đơn vị, max\|Δemb\| = 0 | 0 |
| 6 lượt căn f/q1/q2 × v1/v2 | r4 `align/*.csv` | 143.043/143.043 ô cùng (unit, kind), Δpost = 0 | 0 |
| keep / keep_high | 56.261 / 71.951 | 56.261 / 71.951 | 0 ô |
| keep_paddle_ok / keep_v5 | 54.946 / 54.884 | 54.946 / 54.884 | 0 ô |
| bỏ vì chuẩn hoá | 𠸜 57, 𢚸 3, 𠀧 2 | như cũ | 0 |
| θ Paddle keep (q06 mid+delay) | 1,32 % [0,94–1,69], n = 15.383 | 1,32 % [0,94–1,69], n = 15.383 | 0 |
| θ encoder T1_v2 / T2_v2 keep | 1,27 % / 0,12 % (n 50.155 / 43.357) | như cũ | 0 |
| crop save_crop 56.261 ô | r4 `crops/` | md5 trùng 56.261/56.261 | 0 |

## 3. Kết quả

| sách | trang | chữ người | keep_high | keep | **keep_v5** |
|---|---:|---:|---:|---:|---:|
| SachKinhThayCaBinh (Borg.tonch.18) | 529 | 118.935 | 64.258 | 50.740 | **49.484** |
| SachDungLyHoThan (Borg.tonch.34) | 112 | 24.108 | 7.693 | 5.521 | **5.400** |
| tổng | 641 | 143.043 | 71.951 | 56.261 | **54.884** |

Luỹ kế (keep_v5 ⊂ keep ⊂ keep_high). Có ảnh (crops + crops_chuan): 56.261 ô mức keep trở lên (crop chuẩn ok 56.171;
`one_char_ok` = 1: 55.376, trong keep_v5 54.030). keep_high chỉ ghi hộp (thêm ảnh ≈ +16 k ô, không được yêu cầu).
Dung lượng 465 MB (byte; ≈ 580 MB theo `du`). Không chia tập (30/09 bỏ cột `split_hint`, quyết định A-10).

**Tỉ lệ trượt ±1 ô (Paddle, bộ đọc độc lập):** keep **1,32 %** [0,94–1,69]; keep_high 1,88 % [1,47–2,28]
(riêng keep_high − keep 3,99 %); ô thật ngoài keep_high 19,4 %. Trên keep_v5 phép đo Paddle bị chọn lệch (ô lệch đã bỏ,
θ = 0) → đọc "≤ 1,32 %". θ bằng encoder v2 lệch lạc quan (v2 tham gia chọn keep), chỉ để đối chiếu.

**Chuẩn hoá tự dạng của người phiên** (cột `chuan_hoa_nguoi_phien`): gõ 𠸜 cho hình 先 ≥ 72 %, 𢧚/年 ≥ 18 %, 𠀧/巴 ≥ 6 %,
𧘇/意 ≥ 5 % (cận dưới π_LB vòng 5 q13) → mọi ô gõ 4 chữ này (2.675 ô) mang cờ: `N~L|paddle` khi Paddle thấy L ngay trong hộp
(306 ô tính cả 11 cặp thăm dò — không ô nào ở keep_v5), còn lại `N~L` (2.397 ô).

## 4. Nghiệm thu (`borg_human_eval.py`, 20/20 PASS)
Số dòng = số chữ Lo của bản phiên (đếm lại độc lập); nhãn dòng (page, idx) == chữ/âm người tại idx; khoá duy nhất, idx liền;
luật keep TÍNH LẠI từ cột == keep/keep_high; keep_v5 == keep − Paddle lệch − chuẩn hoá; ảnh đúng mức, 112.432 tệp tồn tại + md5
khớp; θ tính lại từ `paddle_test` == BUILD_INFO và ∈ [0,94; 1,69] %; không có cột chia tập; CHECKSUMS; không lọt vào
`dataset/_ALL`; ≤ 1 GB; (mềm) tái lập từng ô r4/r5.

## 5. Giới hạn / việc mở
- Nhãn = (chữ người phiên, hộp máy chọn): rủi ro chính là trượt ±1 (mục 3); lỗi gõ của người phiên không đo được.
- Không dùng bộ này để đánh giá bộ kiểm chữ viết tay LOBO-sách của `pipeline/gold_exact` (`models/gold_exact/hand_*`) trên
  chính sách nó học; encoder v1/v2 tham gia chọn keep.
- Bước Paddle phụ thuộc hồ sơ đóng băng; muốn đổi đơn vị (detector/tham số) phải dựng lại hồ sơ bằng venv Paddle
  (`measure_out/_borg_human_src/q02_paddle_profiles.py`) — mã sẽ tự chặn (units_match ≠ 0 ⇒ trang lệch không được keep_v5).
