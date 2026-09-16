# Khối C · C1 — mẻ chấm mù hai câu đã dựng (16/09/2026)

> **ĐÍNH CHÍNH C3 (16/09, cùng ngày, sau phê bình toàn khối — `docs/BAO_CAO_KHOI_C_CHUAN_BI_2026-09-16.md`).**
> Mẻ mô tả dưới đây đã được **dựng lại** (chưa ai chấm; verdict = 0). Ba lỗi chặn đã vá trong mã và các con số
> đã đổi: (1) giao diện **hai pha** — MÃ/glyph chỉ lộ sau Q1 (bản C1 in "chưa có mã (chỉ hỏi Q1)" ngay từ đầu
> → lộ nhóm SYL/REVIEW/T1 = 425/1.064 ô, và glyph là gợi ý riêng cho ô CHAR ở Q1); (2) mồi dương = SRS phân
> tầng theo tier_v3 trên **cả 2.012 ô QĐ-01, có trọng số** (bản C1: 45 ô CHAR_A theo sách, NaN) và đạt/rớt chỉ
> theo Q1 vì mã 𠊚 nhận ra được; thêm T6 40/283 ô 'người' chưa khoá (bản C1: 84/89 ô 'người' trong mẻ là
> QĐ-01 → 'người' ⇒ QĐ-01; nay 87/132 = 66 %); (3) lặp ẩn **100 ô phân tầng** (bản C1: 75 SRS mẻ chính → KTC κ
> ≈ ±0,28; nay ≈ ±0,13). Ngoài ra T1 150 → **300** (150 ô chỉ "≤ 3 %" khi 0 lỗi, P(đạt | lỗi 1 %) = 0,22; 300 ô
> cho phép ≤ 3 lỗi, P = 0,65). Tổng **1.279 lượt = 1.179 ô + 100 lặp, 5 phiên × 256**; `KHOA.jsonl` sha256
> `7cb21bafb604d4c9…` (tất định, hai lần dựng byte-identical); selftest ground_truth **232/0**, toàn repo
> **1.140/0**. Các đoạn dưới đây giữ nguyên làm lịch sử C1; số nào khác đoạn này thì đoạn này đúng.

Mã: `pipeline/ground_truth/make_khoi_c_batch.py` (986 dòng). Chạy: `.venv/bin/python -m pipeline.ground_truth.make_khoi_c_batch` (≈22 s, tất định — hai lần chạy cho `KHOA.jsonl` sha256 `d7877906c26e347b…` (giá trị đang nằm trong `plan.json` + `manifest.jsonl`; bản in `22f64bed…` trước đó là của lần dựng chưa chốt — đính chính 16/09 khi viết `estimate_khoi_c`) và `phien_1.html` byte-identical).
Đầu ra: `dataset_out/human_audit/khoi_c_2026-09-16/` (sidecar, KHÔNG đụng `labels*.csv`, `re-dataset/`, `prepared/`).

## 1. Nguồn và khoá

| | |
|---|---|
| Bộ nhãn | `dataset_out/labels_final.csv` 83.239 dòng, sha256 `8da7da81fc9c1da2…` = `CHECKSUMS.txt:59,61` = `BANG_SO_LIEU_CHINH_THUC.md:43` (md5 `1a2d8e0c…`) |
| Seed | 20260916 (`make_khoi_c_batch.py:77`); mọi SRS con dẫn xuất sha1(seed, tag) `:127` |
| Khoá | `_khoa/KHOA.jsonl` (1.064 dòng, 54 trường: tầng, stratum, N_h, design_weight, book/page/column/nom_idx/syl_idx, tier/tier_v3/rule/box_source/count_source/n_*, qd01_locked, image/md5/bbox, âm-mã THẬT, âm-mã HIỂN THỊ, đáp án mồi, repeat_of, chain_id/shift/argmax, crop_source…) + `_khoa/manifest.jsonl` (item_id, session, audit_order, stratum, design_weight, `labels_sha256`, `batch_sha256`, seed) |
| Tệp người chấm | `phien_1..4.html` (6,6–7,1 MB, offline, ảnh base64), `README.md`, `plan.json` |

## 2. Tầng và cỡ mẫu (in từ `plan.json.strata`)

Mẻ chính 600 = rút từ 70.368 ô có crop **trừ** 2.012 QĐ-01 và 42 QUARANTINE; REVIEW không có tệp crop nên **cắt lại từ scan** bằng đúng công thức `build_dataset.save_crop` (pad 0,12 + carve + tighten, `:414`) — kiểm 4/4 ô byte-identical với tệp đã giao.

| tier_v3 | tầng phụ (lớp cột × box) | N | n | w = N/n |
|---|---|---:|---:|---:|
| CHAR_A | eq·detector / eq·legacy / eq·mid / ne·detector / ne·legacy / ne·mid | 19.535 / 14.685 / 491 / 7.149 / 4.390 / 493 | 125 / 93 / 5 / 45 / 27 / 5 | 156 / 158 / 98 / 159 / 163 / 99 |
| CHAR_B | (cùng 6 tầng) | 1.242 / 939 / 29 / 468 / 303 / 36 | 39 / 29 / 5 / 14 / 8 / 5 | 32 / 32 / 6 / 33 / 38 / 7 |
| SYL | (cùng 6 tầng) | 7.752 / 5.699 / 211 / 2.965 / 1.704 / 223 | 61 / 45 / 5 / 22 / 12 / 5 | 127 / 127 / 42 / 135 / 142 / 45 |
| REVIEW | (cùng 6 tầng) | 5.196 / 3.678 / 197 / 2.295 / 1.262 / 233 | 17 / 12 / 5 / 6 / 5 / 5 | 306 / 307 / 39 / 383 / 252 / 47 |

Σw theo tier = dân số (CHAR_A 46.743 · CHAR_B 3.017 · SYL 18.554 · REVIEW 12.861) → ước lượng Horvitz–Thompson khép kín.

| Tầng bổ sung | n | Nguồn | design_weight |
|---|---:|---|---|
| T1 qua cổng B-3 (gate_09) | 150 (50/sách) | `KhoiB/v3/visual_syl_candidates.csv` (1.025; 2 ô đã vào REVIEW chính) | 5,3 / 7,1 / 8,0 |
| T2 chuỗi trượt | 100 = 24 chuỗi nguyên (run 3:4 · 4:13 · 5:6 · 6:1), **8/8 ô QĐ-01** | `k5_register_shift_thr0.8_len3.csv` (114 chuỗi tái lập bằng `chain_ids` `:209`) | NaN (mẫu cụm) |
| T3 B-2 đổi âm | 19 (census) | `b2_changed_cells.json` | 1,0 |
| T3 B-5 hộp 3 nhánh lệch | 30/117 | `b5/qd01_3nhanh_lech_117.csv` | 3,9 |
| T4 mồi dương | 45 (15/sách) QĐ-01 CHAR_A 𠊚 | đáp án Q1 `dung`, Q2 `dung` | NaN |
| T4 mồi âm | 45 CHAR_A `crop_quality_flag=ok`, hiển thị ÂM/MÃ ô kề syl_idx±1 (khác âm, khác mã, không QĐ-01) | đáp án Q1 `sai_am`, Q2 `sai` | NaN |
| T5 lặp ẩn | 75 từ mẻ chính, id mới, gap ≥ 100 (đo: min 103), 66/75 rơi vào phiên khác | `repeat_of` | NaN |

Tổng **1.064 lượt = 989 ô + 75 lặp**, 4 phiên × 266. Crop: 847 từ tệp, 217 cắt lại (50 REVIEW chính + 150 T1 + 9 REVIEW trong chuỗi + 8 lặp ẩn của ô REVIEW). Glyph tham chiếu: 0 ô thiếu; ngữ cảnh: 0 ô thiếu.

## 3. Mù tuyệt đối — kiểm được

- id = UUID4 rút gọn 12 hex từ RNG có seed (`assign_ids` `:372`), **không** dẫn xuất từ image/md5 (khác `sampling.py:95` sha1(image) suy ngược được).
- HTML chỉ có: crop 3× (base64), ngữ cảnh cột ±2,2 ô khung đỏ (`_column_context` `:438`), ÂM hiển thị, MÃ + glyph tham chiếu (`audit_grid._reference_glyph`, chuỗi 6 font). Không `cands`, không `canh_bao`, không tên tệp.
- `check_blind` `:769` grep 22 mẫu (`gold/ syllable/ review/ tier rule stt2 stt4 stt11 page_ CHAR_A CHAR_B REVIEW box_source design_weight stratum qd01 moi_ repeat gate chain decoy KHOA`) **sau khi bỏ payload base64** → 0/4 phiên; grep shell độc lập cũng 0.
- Ô mồi/lặp không có dấu hiệu nào khác ô thường; SYL/REVIEW hiện "chưa có mã Unicode (chỉ hỏi Q1)" — điều này lộ nhóm "không mã" (C0 đã ghi: không tránh được vì Q2 không áp dụng). 2/150 ô SYL chính vẫn có mã vì bộ giao nộp (tier GOLD, tier_v3 SYL — 665 ô toàn bộ) mang `label`; Q2 vẫn hỏi để đo đúng cái đem nộp (`plan.json.note_665_syl_co_ma`).

## 4. Giao diện (`_HTML` `:519`) — kiểm chạy thật bằng Chrome headless

Một ô/màn; Q1 phím 1·2·3·4 (`dung / sai_crop / sai_am / khong_ro`, định nghĩa + ngưỡng 1/3 chữ trên nút), Q2 phím 5·6·7 (`dung / sai / khong_ro`), ô nhập tự do "âm đúng là…" khi `sai_am` và "mã đúng là…" khi `sai`; tự chuyển ô khi đủ câu.
Mỗi ô ghi: `dwell_ms` (thời gian hiển thị tích luỹ tới lúc trả lời đủ, trừ lúc tab ẩn), `visits`, `t_first_shown`, `t_answer`, `order`, `session_id`, `sitting_id` (mỗi lần mở tab), `hist[]` mọi lần đổi; Xuất JSONL → **khoá** phiên (mở khoá được nhưng ghi sự kiện). Thử máy trên `phien_1.html`: 1→5 rồi sửa 2 → `q1=sai_crop, q2=dung, dwell=308 ms, visits=2, hist=3`; ô không mã nhận `3` + nhập "thử", bỏ qua phím 6; Xuất ra 266 dòng, `locked=true`, phím sau khoá không đổi verdict, localStorage giữ.

Định dạng verdict (khác `estimate.load_verdicts` cũ — C-3 cần `estimate_khoi_c.py`): `{item_id, session_id, order, q1, q1_am, q2|null, q2_code, has_code, t_first_shown, t_answer, dwell_ms, visits, sitting_id, hist, exported_at, source:"human"}`.

## 5. Ghi chú cho C-3 (ước lượng)

- Precision Q1/Q2 theo tầng: Wilson 95 % trên các ô `tang=MAIN` (và T1, T3) có `design_weight`; tổng hợp HT với `stratum_N`. `khong_ro` loại khỏi mẫu số và báo số.
- Mồi âm: đạt = **không** chấm `dung` (Q1 ∈ {sai_am, sai_crop} đều tính đạt, vì chữ thật có thể chứa thành phần của chữ kề; xem ảnh trong scratchpad, không ghi vị trí ở đây để giữ mù). Mồi dương: đạt = Q1 `dung` ∧ Q2 `dung`.
- κ nội tại: 75 cặp `repeat_of` (Q1 4 mức; Q2 3 mức trên cặp có mã).
- T2: tỉ lệ `sai_am` theo chuỗi (≥ 50 % ô/chuỗi → hạ REVIEW theo chuỗi, BAO_CAO_KHOI_B §4); T3-B2 so `q1_am` với `argmax` (âm thị giác) ghi sẵn trong KHOA.

Selftest: `pipeline/ground_truth/selftest.py::test_khoi_c_batch` +14 (189/0); tổng `scripts/run_all_selftests.sh` 1.097/0 (mốc cập nhật 1083 → 1097).
