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
