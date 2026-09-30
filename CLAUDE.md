# GanNhanOCR — hướng dẫn cho Claude Code

## Quy ước đo đạc (2026-09-21)
- **KHÔNG chạy lại phép đo bằng LLM.** Mọi số liệu về dữ liệu mới (LVT1883, KVK1884, Chrestomathie1872) và về mã pipeline
  lấy từ bộ đo tái lập được: `.venv/bin/python scripts/measure/measure.py --all` (≈10 phút CPU, 0 token).
- Chỉ đọc `measure_out/SUMMARY.json` và `measure_out/REPORT.md` (mỗi phép đo có `invariants` PASS/FAIL thay cho agent phản biện).
- Nghi ngờ một con số → đổi tham số script (`--limit`, `--workers`, `--stt-pages`…) hoặc **thêm invariant** vào mô-đun,
  rồi chạy lại; **không mở ảnh/CSV thô bằng LLM**.
- Đọc mã pipeline qua `docs/PIPELINE_FACTS.json` (sinh bởi `scripts/measure/code_facts.py`) trước; chỉ mở tệp nguồn khi FACTS thiếu.
- Mỗi phép đo ghi vào `measure_out/<book>/<phép đo>/`; `measure_out/` không commit. Cách thêm phép đo/sách mới: `scripts/measure/README.md`.
- Mã đo chỉ phụ thuộc `.venv` + tesseract, không sửa `pipeline/`, `core/`, `data/`.
- Số liệu chốt + việc còn mở: `docs/KET_QUA_DO_CUOI_2026-09-21.md`; đối chiếu số cũ↔mới: `docs/PIPELINE_SACH_MOI_2026-09-20.md` §11.
- Sách thạch bản mới (LVT1883/KVK1884): lệnh chạy, cổng nghiệm thu, việc còn lại, cách thêm sách thứ tư: `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md`.
- **Báo cáo gộp duy nhất** cho sách mới (dữ liệu, mã đã xây, kết quả 3 sách sau `box_decoder: pitch` + cổng (a'), độ đúng tự động, I5, cách chạy, giới hạn):
  `docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` (thay `BAO_CAO_TONG_THE_*` và `PHUONG_AN_TU_DONG_*`, hai tệp đó chỉ còn lịch sử). Chạy 1 lệnh:
  `./run_pipeline.sh --book <Book>|all-new` (không `--no-api` cho KVK). Kế hoạch commit vòng 3: `docs/KE_HOACH_COMMIT_VONG3_2026-09-22.md`.

## Chạy lại toàn bộ (2026-09-23)
- **Một lệnh duy nhất cho cả 8 bộ**: `./run_pipeline.sh --book all --yes`
  (3 STT theo `config/pipeline.yaml` + `all-new` 3 sách giao nộp + `all-ihr` 2 bộ đánh giá, rồi **nghiệm thu tự động**).
  `--yes` bắt buộc khi chạy thật (bỏ prompt STT, dùng cache OCR → **0 gọi API**); script chặn cứng nếu thiếu cache.
- `--book all --dry-run` in đủ chuỗi lệnh 8 bộ · `--summary-only` in lại bảng 8 bộ · `--verify` chỉ chạy nghiệm thu
  (`align_audit --book all`, `ihr_endtoend_eval --book all`, `auto_precision --steps cross`, `measure.py --all --report-only`).
- Log riêng mỗi bộ trong `logs/`; mã thoát ≠ 0 nếu bất kỳ bộ nào lỗi hoặc có FAIL cứng khi nghiệm thu.
- **Tài liệu chốt cuối** (mọi quyết định + bảng 8 bộ + giới hạn): `docs/CHOT_CUOI_2026-09-23.md`;
  cách chạy chi tiết: `docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md` §2.0.

## GOLD chính xác — ảnh + chữ (2026-09-27)
- Bước tự động SAU gộp trong `run_pipeline.sh` (`run_gold_exact`, cờ `--gold-exact on|off`, mặc định on; chỉ chạy khi có
  gộp trong cùng lượt; 0 API). Ghi tệp phụ `dataset/_ALL/gold_exact.csv` (ok / text_only / uncertified / review +
  `evidence_level`) và `crops_chuan/` cho ô ok; **KHÔNG đổi `labels.csv`** (mốc md5 STT giữ nguyên).
- Mã `pipeline/gold_exact/`, ngưỡng `config/gold_exact.yaml`, tài sản `models/gold_exact/` (.pt không commit; nguồn dựng
  lại bền ở `measure_out/_gold_exact_assets_src/`). Nghiệm thu `scripts/measure/gold_exact_eval.py` (trong `measure.py --all`).
- Độ chính xác chỉ ĐO được ở LVT1916/TK1872 (nhãn người IHR); KVK/L83 ước lượng; STT/Chr suy đoán. Chi tiết + giới hạn:
  `docs/GOLD_CHINH_XAC_2026-09-27.md`; thử nghiệm nền `lab/thu_nghiem_anh_chu/TN1–TN4`.

## Đường chạy tốt nhất theo nghiên cứu (2026-09-28)
- `./run_pipeline.sh --book all --yes [--publish]` in bảng ĐƯỜNG CHẠY (pipeline/tools/duong_chay.py) — đường lấy từ config:
  hộp ảnh `box_decoder` theo TN6 (docs/HOP_ANH_TN6_2026-09-28.md): visual_dp cho 2 Borg + TK + Chr, visual_dp_hybrid cho L16,
  pitch cho L83/KVK(_b1), legacy cho STT; gold_exact profile handwriting (STT+Borg) + lần đọc thứ hai STT lt2
  (`pipeline/tools/stt_reocr_lt2.py`, cache `kim_raw_lt2/`, tự bật khi đủ trang); công bố `--publish` → `dataset/_ALL/cong_bo/`
  (ảnh chỉ ô ok). Bảng kết quả nghiên cứu → đã/không áp dụng: `docs/DIEU_HUONG_PIPELINE_2026-09-28.md`.
- Dọn + dựng lại toàn bộ đúng thứ tự: `bash scripts/clean_rebuild_all.sh --yes --run` (giữ cache OCR/lt2, chặn API, đo, nghiệm thu, báo cáo).

## KHÔNG chia tập + web chỉ tầng nhãn (2026-09-30)
- Quyết định A-10 (16/09): **không** cột/tập chia (`split`, `split_hint`, `lobo_group`, train/val/test, LOBO) ở bất kỳ đầu ra
  nào (`dataset/<Bộ>/`, `dataset/_ALL/`, `cong_bo/`, `_BORG_NHAN_NGUOI/`); bộ có nhãn người chỉ mang cờ `evaluation_only`.
  Bất biến `khong_cot_chia_tap` ở merge_datasets, publish gold-exact, borg_human_eval.
- `web/` chỉ hiện tầng nhãn GOLD / SYLLABLE / REVIEW / QUARANTINE (như bản đầu); gold_exact vẫn sinh tệp phụ nhưng không hiện.

## Borg.Tonch.18/34 — chữ viết tay Công giáo có nhãn người (2026-09-27)
- Sách `SachKinhThayCaBinh` (Borg.18) và `SachDungLyHoThan` (Borg.34); `data/MSS_Borg.tonch.*` là symlink, config
  `pipeline_MSS_Borg_tonch_*` bị `run_pipeline.sh` từ chối (dùng tên Sach*).
- Tập ĐÁNH GIÁ tự động: `./run_pipeline.sh --book all-borg --yes` (adapter `pipeline/tools/ingest_borg_book.py`, QN người,
  Nôm người GIẤU, kim lt2; cần API kim — 641 lượt, cache `prepared/_auto/`), rồi `--merge --yes --verify`; đo
  `scripts/measure/borg_endtoend_eval.py`; gold_exact trên Borg LOBO theo sách. Cờ `--borg auto|require|off`.
- Bộ crop NHÃN NGƯỜI (0 API, tách khỏi GOLD tự động): `dataset/_BORG_NHAN_NGUOI/` (`python -m pipeline.borg_human
  --stage all`; đo `scripts/measure/borg_human_eval.py`). Tài liệu: `docs/BORG_DANH_GIA_2026-09-27.md`,
  `docs/BORG_NHAN_NGUOI_2026-09-27.md`.
