# lab/ — thí nghiệm (KHÔNG nằm trên đường chạy sản xuất)

Quy ước (xem `docs/CAU_TRUC_MA_NGUON_2026-09-23.md`): một thí nghiệm = **một thư mục** có tên riêng
hoặc có ngày, kèm `README.md`. Không để tệp `.py` rời ở gốc `lab/`.

| Thư mục | Nội dung | Còn dùng? |
|---------|----------|-----------|
| `i5_detector_v2/` | Huấn luyện detector v2 (Kaggle). `make_bundle.py`, `train_kaggle.py`, `i5v2/` | **ĐIỂM VÀO** — còn dùng. `last.pt` là nguồn duy nhất còn lại của ckpt v2 (HF riêng tư 401) ⇒ GIỮ. `bundle/` + `i5v2_bundle.zip` đã xoá 23/09 (dựng lại: `make_bundle.py`). |
| `gan_nhan_2026-09-13/` | Mẻ gán nhãn 13/09; `crops.npz` là **nhánh dự phòng** của `run_pipeline.sh:364`; `thuc_nghiem.py rebuild` dựng `cols.pkl` cho 7 test `tier_v3_selftest` | Còn dùng (phụ trợ) |
| `self_training_v2/`, `enhanced_self_training_v3/` | Tự huấn luyện giải cứu REVIEW (bước 5). `best_nom_ocr.pt` / `best_enhanced_nom_ocr.pt` được `run_pipeline.sh:249-252, 349-352` nạp | Ckpt còn dùng; `.zip` gói Kaggle đã xoá 23/09 |
| `kaggle_diffusion/` | Sinh `fd_cache` phổ quát bằng FontDiffusion. **Dời từ gốc kho về đây 23/09** (nhánh step-3 DINOv2 đã nghỉ); đã sửa `parents[1]` → `parents[2]` trong 5 tệp | Lịch sử / đối chứng luận văn |
| `ocr_compare/` | So sánh engine OCR (GLM, mlx-vlm, PaddleOCR) | Lịch sử |
| `tham_dinh_2026-09-16/` | Mẻ thẩm định MI3 ngày 16/09; được chú thích trong `pipeline/align_engine/{visual_emission,anchor_align}.py` dẫn lại | Lịch sử (bằng chứng) |
| `configs/` | Cấu hình cho `pipeline.lab.runner` | Còn dùng khi chạy tay |

Ghi chú: `pipeline/lab/` **không** dời về đây vì nó là gói Python chạy bằng `python -m pipeline.lab.*`
(`scripts/run_all_selftests.sh:161`) và được `docs/PIPELINE_FACTS.json` lập chỉ mục.
`khoiB/` cũng **không** dời (là đầu vào sống `KhoiB/v3/crops_v3.npz` của `run_pipeline.sh:363`).
