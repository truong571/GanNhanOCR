# Điều hướng pipeline — kết quả nghiên cứu → đã áp dụng ở đâu (28/09/2026)

Tài liệu này trả lời một câu: **mỗi kết quả nghiên cứu đã chứng minh (hoặc đã bác) đang nằm ở đâu trong pipeline tổng thể**,
và `run_pipeline.sh` mặc định chạy "đường tốt nhất" nào cho từng loại sách. Bài toán TỰ ĐỘNG hoàn toàn: mọi bước mới dưới đây
**chỉ hạ / gắn trạng thái**, không sửa nhãn, không đổi `labels.csv`. 0 API (trừ công cụ OCR lt2 do người dùng chạy).

## 0. Một lệnh

```bash
./run_pipeline.sh --book all --yes --publish     # 10 bộ -> gộp -> gold_exact (kèm lt2 STT khi đủ cache) -> tập công bố -> nghiệm thu
./run_pipeline.sh --book all --dry-run --publish # in bảng ĐƯỜNG CHẠY + toàn bộ chuỗi lệnh, không chạy gì
./run_pipeline.sh --merge --yes [--publish] --verify   # chỉ gộp lại + gold_exact (+ công bố) + nghiệm thu (0 API)
bash scripts/clean_rebuild_all.sh --yes --run    # dọn dẫn xuất rồi chạy lại đủ chuỗi trên (bước 1 đã kèm --publish)
```

Đầu MỖI lượt, `run_pipeline.sh` in bảng **ĐƯỜNG CHẠY** (`pipeline/tools/duong_chay.py`, gọi từ `run_pipeline.sh:1014`
`print_route`): đọc config hiện hành qua `book_layout` (giá trị đã giải mặc định) + `config/gold_exact.yaml` + cache lt2 (cùng
hàm bước gold_exact dùng để bật/tắt). Bảng lúc viết tài liệu (28/09, config HEAD + thay đổi của tài liệu này):

| bộ | box_decoder | kim lang_type | crop_source | gold_exact profile | second_read (lt2) |
|---|---|---|---|---|---|
| SachThanhTruyen2 / 4 / 11 | legacy | 1 (Hán) | processed | handwriting | stt2 BẬT (160/160 trang) · stt4 TẮT (một phần) · stt11 TẮT (0/143) |
| LucVanTien1883, KimVanKieu1884 (`_b1`), LucVanTien1916, TruyenKieu1872 | pitch | 2 (Nôm) | original | printed | — |
| Chrestomathie1872 | pitch | 2 | original | printed | — |
| SachKinhThayCaBinh, SachDungLyHoThan (Borg) | pitch | 2 | original | handwriting | — |

Chuỗi: các bộ → gộp `dataset/_ALL` (chặn cứng ảnh dùng chung khác nhãn) → gold_exact (policy `2026-09-28.3`; ô ok → crop chuẩn v2;
lt2 `auto`) → [`--publish`] tập công bố `dataset/_ALL/cong_bo/` → nghiệm thu.

## 1. Bảng kết quả → áp dụng

| # | Kết quả nghiên cứu | Số đo (nguồn) | Trạng thái | Ở đâu (tệp:dòng) / vì sao không |
|---|---|---|---|---|
| 1 | GOLD chính xác cấu hình **"lai"** (crop chuẩn v2 + cổng H1 cờ một chữ + bộ kiểm chấm crop CŨ, ngưỡng TN1 τ 0,995 LOBO) + luật A / M-OCR | L16 99,46 %, TK 99,87 % V1+ hai vế (TN4 §5, `lab/thu_nghiem_anh_chu/TN4_crop_chuan/KET_QUA.md`) | **ÁP DỤNG** (mặc định) | `pipeline/gold_exact/policy.py:145` `decide`; `pipeline/gold_exact/__main__.py`; `run_pipeline.sh:962` `run_gold_exact` (ngay sau gộp) |
| 2 | **Profile handwriting** (STT + Borg): bỏ CNT, cổng khe `bc`, `hand_q` 0,00015, ≥ 3 nguyên mẫu | chọn theo đăng ký trước TN5; Kinh CV hai vế 99,11 % [98,57–99,61]; LOBO Kinh→DungLy 97,56 % (n 41) (`docs/GOLD_CHINH_XAC_2026-09-27.md` §11) | **ÁP DỤNG** | `config/gold_exact.yaml:23` `profiles.handwriting`; `policy.py:36` `profile_hw`, `policy.py:78` `slot_gate_masks` |
| 3 | **CNT** (số chữ OCR cột ≠ số âm QN) cho chữ viết tay | AUC nhãn 0,493 / khe 0,494 / hai vế 0,495 trên Borg; cờ gắn 95–97 % ô GOLD Borg (GOLD_CHINH_XAC §11.2) | **BỎ** cho viết tay (giữ cho in/khắc) | `config/gold_exact.yaml:26` `drop_gates: [cnt]` |
| 4 | **Crop chuẩn v2** (khung vuông một chữ, ảnh trang gốc) | mực ngoài khe người L16 13,45 → 7,13 %, TK 5,16 → 3,45 % (TN4 §0) | **CHỈ Ô ok** (ảnh giao của ô ok = `crops_chuan/`; `labels.csv` giữ `image` gốc) | `pipeline/gold_exact/crop_chuan.py:24` `PARAMS` (md5 đóng băng, selftest); `__main__.py:400` chỉ chép crop ô ok. **Không thay toàn bộ ảnh vì**: ~0,54 % ô L16 bị crop chuẩn cắt vào nét (TN4: 61 ô "damaged"); cho bộ kiểm chấm crop chuẩn **không lợi** (mộc bản AUC giảm nhẹ; Borg nhận dương −14,1 điểm ở cùng FAR — TN4 §7.3); chốt core_loss đăng ký trước cho kết quả âm (mất 46 ô đều đúng, bắt 6/61 damaged) → bỏ khỏi cổng. **Đề xuất**: giữ nguyên. |
| 5 | **Vòng sửa hộp TN2** (dời hộp theo prev/next/kim + bộ kiểm chọn) | P2 − P1 (sửa so với chỉ hạ) = 0,00 [−0,02; 0,01] điểm % L16, 0,00 TK; hộp kim không bộ kiểm làm hỏng 16 (L16) / 43 (TK) ô đúng (`lab/thu_nghiem_anh_chu/TN2_vong_sua_hop/KET_QUA.md` §2) | **BÁC** | không lợi so với "chỉ hạ" (gold_exact đã hạ ô trượt qua cờ BC) |
| 6 | **Vòng lặp OCR / TTA kim** (đọc nhiều lượt, bỏ phiếu) | lỗi trùng giữa các lượt 71–81 % (IHR), 95–96 % (thạch bản); trần "một trong hai đúng" +0,5–1,0 điểm; luật hợp nhất +0,19 / −0,3 điểm; TTA k-way kém chỉ-lt2 (`measure_out/_audit_2026-09-26/r23/r3_digest.md` C3, K7) | **BÁC** | chỉ giữ lt2 của STT làm cờ CHỈ HẠ (mục 9) |
| 7 | **Engine OCR khác** (tesseract / Apple Vision / PaddleOCR / GLM-OCR / NomNaOCR / ROVER) | tesseract+Vision đúng 12–14 % trên IHR, cứu 0–4/60 lỗi kim; Paddle/GLM phủ quyết kim precision 47–59 %; NomNaOCR đã học trên IHR (rò dữ liệu) (r3_digest C4–C6) | **BÁC** | "không engine cục bộ nào bổ trợ được kim" (r3_digest §ROVER) |
| 8 | **LM / tiên nghiệm ngữ cảnh** P(x \| âm, câu) | cứu ô kim ∉ R đúng 37–41 %, không ngưỡng nào đạt 90 % LOBO (r3_digest L6) | **BÁC** | `tier_v3.py` không đổi |
| 9 | **Lần đọc thứ hai STT (kim lt2)** — mục A | hiệu chuẩn thạch bản (ước lượng cận dưới): P(nhãn đúng \| lt1 = lt2) 89,8 % [89,3; 90,3] vs 11,3 % khi lệch, nền 81,8 % (`docs/STT_LT2_2026-09-28.md` §4) | **ÁP DỤNG** (`second_read: auto`, CHỈ HẠ) | `pipeline/gold_exact/signals_lt2.py` (`:106` `activation`, `:253` `signal`); `policy.py:194` luật; `config/gold_exact.yaml:39`; `__main__.py:276` tín hiệu, `:352` đối chứng + invariant `lt2_chi_ha` — §2 |
| 10 | **Tập công bố tách ẢNH / VĂN BẢN** — mục B | 173.973 dòng → tập ảnh 27.369 ô ok (14.288 thuộc bộ đánh giá; **30/09: bỏ chia tập**) · tập văn bản 146.604; 10/10 bất biến — §3 | **ÁP DỤNG** (cờ `--publish`, tuỳ chọn) | `pipeline/publish/gold_exact_release.py` (`:71` `build`, `:117` `verify`, `:163` `run`); `pipeline/publish/cli.py:185`; `run_pipeline.sh:1029` `run_publish` |
| 11 | **Rescue va tên tệp** — mục C | 22 ô rescue mang ảnh của ô khác; 21 đường ảnh dùng chung giữa hai nhãn khác nhau (42 dòng) → 0 / 0 — §4 | **SỬA GỐC + chặn cứng** | `pipeline/remediation/self_training_rescue.py:49` `rescue_name`, `:58` `write_rescue_png`, `:297` bỏ ô không crop; `pipeline/tools/merge_datasets.py:306/344` bất biến `anh_khong_dung_chung_giua_nhan_khac` (gộp + `--check`), `:324` |
| 12 | **Hộp ảnh theo thị giác (TN6)** — DP trang âm QN ↔ đơn vị detector, phát xạ R(âm) + nguyên mẫu tự học; biến thể lai (ô không tự tin lệch pitch → hộp pitch) | đúng vị trí so hộp người Borg 61,5 → 99,4 % (Kinh), 58,6 → 99,8 % (DungLy); khe người IHR TK 98,69 → 98,91 %, L16 (lai) 95,77 → 96,65 %; Chr chỉ gián tiếp (`docs/HOP_ANH_TN6_2026-09-28.md` §1, §6) | **ÁP DỤNG theo bộ** | `box_decoder: visual_dp` Kinh/DungLy/TK/Chr, `visual_dp_hybrid` L16; GIỮ pitch L83 + KVK_b1 (proxy không chứng được lợi), legacy STT (chưa thắng rõ theo đăng ký trước ⇒ md5 `59e436d7…` giữ). Mã `pipeline/align_engine/visual_dp.py` (`apply_book`), nối ở `align_production.align_page` + `build_dataset` (sau PASS 1); cổng (a') `mechanism_gates.BOX_LOW_CONF` |

## 2. Mục A — lần đọc thứ hai STT trong gold_exact

- **Tín hiệu** (`signals_lt2.py`): hộp chữ lt1 của ô = cột kim dựng lại (`signals_geom.kim_vs_box`, nay trả thêm `kim_char`/`kim_box`)
  → ghép chữ lt2 bằng phương pháp `line` của `scripts/measure/stt_lt2_eval.py` (bản chép; selftest `lt2_match_eq_stt_lt2_eval`
  so từng kết quả với bản đo trên 60 trang ngẫu nhiên). Cột mới trong `gold_exact.csv`: `lt2_char`, `lt2_match`, `lt2_dis`, `lt2_unm`.
- **Luật** (CHỈ HẠ, cuối nhóm uncertified): ô sẽ-là-ok mà nhãn ≠ chữ lt2 (V1+) → `U_STT_lt2_khac_lt1`; trang có lt2 mà không ghép
  được chữ (`unmatched: demote`, bảo thủ như đề xuất cổng của bản đo) → `U_STT_lt2_khong_ghep_duoc`. Không đổi nhãn.
- **Kích hoạt**: `auto` = bật cho MỘT bộ chỉ khi MỌI trang lt1 có cache lt2 hợp lệ (lang_type 2, cùng image_hash, toạ độ
  fullpage); thiếu → tắt cả bộ, ghi rõ trong `GOLD_EXACT.md`. `--second-read partial` chỉ để đo (cấm với `--publish`).
- **Đo trước/sau** (bộ gộp hiện tại, cache gold_exact ấm, `measure_out/_gold_exact_lt2/`):
  - `--second-read off`: so với `dataset/_ALL/gold_exact.csv` đã giao → **0 / 130.895** ô khác `gold_exact` và `reason`
    (invariant "thiếu cache = y hệt"; trong lượt chạy còn kiểm `lt2_chi_ha.identical_when_off`).
  - `--second-read partial` (lúc đo: stt2 160/160 trang, stt4 17/145): **132 ô ok → uncertified** (71 nhãn ≠ lt2, 61 không ghép);
    stt2 ok 2.864 → 2.737, stt4 1.872 → 1.867 (phần trang có lt2); ô ok trên trang lt2: 3.066, ghép 3.005, trùng lt2 97,6 %.
    Mọi ô đổi đều ok → uncertified (`only_ok_to_uncertified` true); quyết định tái lập từ cột CSV + config: 0 lệch; đối chứng
    tự ghép lt1 19.806/19.806.
  - Độ chính xác của ô còn lại trên STT vẫn là **SUY ĐOÁN** (hiệu chuẩn ở thạch bản, tham chiếu dị bản; lt1/lt2 cùng engine nên
    lỗi tương quan).
- `scripts/measure/gold_exact_eval.py`: đọc cột lt2 (tệp cũ vắng → 0), phân rã `ok = lai − A − M-OCR − lt2`, invariant mới
  `lt2_chi_ha_o_STT_co_co`.

## 3. Mục B — tập công bố

`python -m pipeline.publish gold-exact` (hoặc `--publish`): `images.csv` = ô `gold_exact = ok`, `image` = crop chuẩn, `image_goc` =
đường gốc (tham khảo); `text.csv` = mọi dòng khác (không cột ảnh) + `gold_exact`, `gold_exact_reason`, `ly_do_khong_anh`;
`EXCLUSIONS.json`, `RELEASE.md`, `CHECKSUMS.txt`. In từng lý do loại ra màn hình ("loại trừ phải ồn ào"). **30/09: KHÔNG chia
tập** (quyết định A-10 ngày 16/09 — bỏ cột `split`/`lobo_group` của tập công bố và `split_hint` của bộ gộp); bộ đánh giá chỉ mang
cờ `evaluation_only`. Bất biến (fail loud, không ghi gì nếu FAIL): tập ảnh ⊂ GOLD ∧ ok, hai tập rời + phủ đủ, không có cột chia
tập, mọi crop tồn tại + md5 khớp `gold_exact.csv`. Thư mục `cong_bo/` bị bước gộp và `gold_exact --publish`
dọn (phụ thuộc `gold_exact.csv`).

Đo trên bộ gộp hiện tại (ghi ra thư mục tạm): **173.973 dòng → ảnh 27.369 / văn bản 146.604**; lý do loại lớn nhất:
`tier:SYLLABLE` 42.237, `uncertified:U_bo_kiem_anh_duoi_nguong_TN1` 34.292, `uncertified:U_STT_bo_kiem_viet_tay_khong_chung_nhan`
27.363, `text_only:BC_truot_theo_hop_kim_vis` 7.550…; (bản 28/09 còn chia train/val/test — đã bỏ 30/09); bất
biến PASS. Trước: người dùng lấy `labels.csv` của bộ gộp sẽ nhận 173.132 dòng có ảnh (mọi GOLD/SYLLABLE) — không tách ô được
chứng nhận ảnh + chữ. Đường cũ `pipeline.publish all` (dataset_out/ STT theo tầng) giữ nguyên.

## 4. Mục C — sửa gốc va tên tệp của rescue

- **Gốc**: `self_training_rescue` đặt tên `<book>_<page>_c<cột>_<nom_idx>.png` — trùng khuôn tên `<book>_<page>_c<cột>_<chỉ số>.png`
  của build — và **tái dùng tệp có sẵn** → 22 ô rescue mang ảnh của ô khác.
- **Sửa**: tên theo khoá riêng `rescue_<book>_<page>_c<cột>_n<nom_idx>_s<syl_idx>.png`; tệp có sẵn khác byte → `RescueFileConflict`
  (dừng), trùng byte → dùng; ô không có crop của chính nó → không giải cứu. `merge_datasets` chặn cứng khi một đường ảnh bị các
  dòng **khác nhãn** dùng chung (cùng nhãn: vẫn chỉ gắn cờ `image_dup`), cả khi gộp (trước khi ghi) lẫn `--check`.
- **Đo trước/sau** (mô phỏng STT bước 4→5→6 trong thư mục tạm; mã cũ tái lập `dataset_out/labels_final.csv` TRÙNG TỪNG BYTE và
  `dataset/SachThanhTruyen/labels.csv` trùng từng byte): rescue 2.840 ô cả hai bên, nhãn không đổi; khác duy nhất `image` (2.840
  dòng đổi tên) và `image_md5` (22 dòng = đúng các ô từng mang ảnh ô khác; 2.818 ô còn lại cùng điểm ảnh); ảnh dùng chung khác
  nhãn 21 đường / 42 dòng → **0**; gộp thử 8 bộ: mã cũ bị chặn, mã mới 173.973 dòng, `image_trung` 0, bất biến 25/25.
- **Mốc md5**: hồi quy build STT 3 trang (`build_dataset … --pages pages3.csv`) **59e436d7641fa849bb6759868ac29259 (556 dòng)
  KHÔNG đổi** — build không gọi rescue/merge (kiểm import) và đã chạy lại xác nhận. Cái ĐỔI là `labels.csv` bản giao STT sau khi
  chạy lại: `fb34973c…` → `90ea8a60…` (dự kiến, từ mô phỏng) vì đường ảnh + md5 của dòng rescue.
- **Hệ quả tạm thời**: `dataset/SachThanhTruyen/` + `dataset/_ALL/` hiện có là bản dựng TRƯỚC sửa → `./run_pipeline.sh --merge`
  bị chặn (không ghi gì) và `merge_datasets --check` FAIL `anh_khong_dung_chung_giua_nhan_khac` cho tới khi chạy
  `./run_pipeline.sh --book all --yes` (STT chạy trước bước gộp).

## 5. Mục D — run_pipeline.sh / clean_rebuild_all.sh

- Bảng ĐƯỜNG CHẠY ở đầu mọi lượt (`--book all`, `--book <Bộ>`, `--merge`, STT không tham số, mọi `--dry-run`).
- `--publish` (B9, tuỳ chọn, 0 API): chỉ chạy khi gold_exact xong CÙNG lượt; `--dry-run` in đúng lệnh.
- `--book all` = 10 bộ (Borg khi đủ cache) → gộp → gold_exact (lt2 auto) → [công bố] → nghiệm thu.
- `scripts/clean_rebuild_all.sh`: bước 1 thêm `--publish`; cache lt2 STT (`kim_raw_lt2/`, `kim_calls_lt2.json`) vào nhóm "cache gốc
  từ API" (khoá sha256 trước/sau, cấm tệp mới sau `--run`).

## 6. Kiểm (0 API)

| Kiểm | Trước | Sau |
|---|---|---|
| `python -m pipeline.gold_exact --selftest` | 85 | **97** passed (+12 lt2: config, chỉ hạ, phân rã, ghép ≡ bản đo, tín hiệu tổng hợp, kích hoạt) |
| `python -m pipeline.publish.selftest` | 69 | **81** passed (+12 tập công bố) |
| `python -m pipeline.tools.merge_datasets --selftest` | 27/27 | **30/30** (+ cùng nhãn chỉ gắn cờ, khác nhãn chặn cứng, `--check` FAIL) |
| `python -m pipeline.remediation.self_training_rescue --selftest` | — | **7/7** (móc vào `pipeline.remediation.selftest`: 175 → 176 passed; 4 FAIL dữ liệu có sẵn ở HEAD, không đổi) |
| `python -m pipeline.tools.duong_chay --selftest` | — | **6/6** |
| `scripts/measure/code_facts.py --check` | 18/18 | 18/18 |
| hồi quy STT 3 trang | 59e436d7… | 59e436d7… (556 dòng) |

## 7. Việc chờ người dùng

1. Chạy đủ lần đọc lt2 (API kim): `.venv/bin/python -m pipeline.tools.stt_reocr_lt2 --book all` (đứt thì chạy lại đúng lệnh),
   rồi `scripts/measure/stt_lt2_eval.py --require-full`.
2. `./run_pipeline.sh --book all --yes --publish` (0 API): dựng lại STT với rescue đã sửa → gộp qua được chặn cứng → gold_exact
   policy `2026-09-28.3` (lt2 bật cho bộ đủ cache) → tập công bố → nghiệm thu. Trước lượt này, `gold_exact_eval` trên bản giao cũ
   FAIL `policy_version`/`config_sha` (config đã đổi) — đúng ý đồ.
3. Cập nhật `CLAUDE.md` (người dùng tự làm); dòng 12 (TN6) đã có quyết định cuối — xem `docs/HOP_ANH_TN6_2026-09-28.md`.

## 8. Giới hạn

- lt2: hiệu chuẩn trên thạch bản so dị bản (ước lượng cận dưới); STT chép tay KHÔNG có sự thật người → số ô ok mới của STT vẫn
  là suy đoán. lt1/lt2 cùng engine, cùng ảnh, cùng bộ dò dòng — "đồng ý" không phải hai phiếu độc lập. Ghép vị trí là xấp xỉ
  (hộp chữ kim = chia đều hộp dòng); chọn `unmatched: demote` là bảo thủ, đổi sang `keep` bằng config.
- Tập công bố: tập ảnh chỉ đo được độ chính xác ở L16/TK (và Borg) — đều là bộ đánh giá (`evaluation_only = 1`); phần giao
  nộp (STT, Chr, L83, KVK) là ước lượng/suy đoán theo `evidence_level`.
- Sau `--book <Bộ>` đơn lẻ, bộ gộp + gold_exact + công bố cũ bị dọn khi gộp lại; luôn kết thúc bằng `--merge`.
