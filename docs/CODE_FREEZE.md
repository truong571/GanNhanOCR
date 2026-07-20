# TUYÊN BỐ CODE-FREEZE TÍNH NĂNG

**Hiệu lực từ**: 2026-07-20, commit `3c93615346`, sau khi Giai đoạn 1 hoàn tất
**Hết hiệu lực**: sau khi bảo vệ luận văn

---

## Nguyên tắc

> Từ thời điểm này, repo **chỉ nhận 3 loại thay đổi**. Mọi thứ khác bị từ chối, kể cả khi "chỉ tốt hơn một chút".

| Được phép | Ví dụ |
|---|---|
| **1. Sửa lỗi chặn** | Vá tiền điều kiện làm pipeline không chạy được; sửa assertion đang đỏ cho khớp số liệu đã chốt |
| **2. Nối dây (wiring)** | Nối 8 bước FLOW §5 vào `run_pipeline.sh`; thêm preflight; truyền `--strict` |
| **3. Tài liệu và số liệu** | `BANG_SO_LIEU_CHINH_THUC.md`; sửa README/docstring ghi số đã bị bác; kết quả audit, ablation, downstream |
| **4. Dọn rác không đổi hành vi** | Dời/xoá file **đã gitignore** và **0 tham chiếu** (phải grep xác nhận tại thời điểm hành động); `git mv` không đổi nội dung file |
| **5. Vá lỗi làm kiểm định mù** | Sửa chỗ khiến suite selftest **crash** thay vì fail có số — mất khả năng đo còn nguy hiểm hơn một assertion đỏ |

> **Điều kiện bắt buộc cho loại 4 và 5**: phải chứng minh bằng `bash scripts/run_all_selftests.sh` ra **đúng 212 passed / 11 failed**, không test nào chuyển từ pass sang skip.
>
> *Ghi chú lịch sử*: GĐ2 (`27ba0558d1`) được thực hiện SAU mốc đóng băng `77bb46d31d`. Về hình thức là vi phạm, về thực chất hợp lệ vì chỉ dời dữ liệu đã gitignore + `git mv`, selftest không đổi. Mục 4–5 được bổ sung để lần sau phân biệt được ngay.

Lý do: quỹ thời gian còn ~2–3 tháng, và **cả 4 blocker của luận văn đều nằm ở tầng chất lượng/kiểm định, không phải tầng tính năng**. Mỗi giờ đổ vào tính năng mới là một giờ rút thẳng khỏi Chương 3–4.

---

## Bị cấm cho tới khi bảo vệ xong

Mỗi mục đều có bằng chứng đo được, không phải phỏng đoán.

| Việc | Vì sao cấm |
|---|---|
| **Thêm kênh/mô hình/tính năng mới** | Fusion Path B, kênh nặng (kraken, Qwen-235B full-corpus, `nna_lobo` runtime), Kish n_eff, blind-MCQ, PPI — đã đo train-AUC **0,54** với 25 negative không held-out, và PPI không chạy được vì coverage S3 chỉ 30%. Không đổi được điểm bảo vệ. |
| **Tái cấu trúc thư mục** | src-layout / `paths.py` / dời 25 hằng `parents[N]`: đụng ~250 file, và lỗi biểu hiện là **ghi số sai chứ không crash**. `align_engine/data/index.csv` chứa 52.786 đường dẫn hard-code `dataset_out/gold/*.png` — dời là crop-protos = 0 → **SILVER −32% âm thầm**. |
| **Dời `dataset_out`, `prepared/`, `Data/`, `dict/`** | `Data`/`data` cùng inode trên APFS; index git ghi `Dict/` còn đĩa ghi `dict/`. |
| **Gộp file** | 42 đề xuất đã phân tích, **39 bị phản biện bác**. Riêng "gom 7 selftest về `tests/`" làm **66 assertion skip âm thầm** mà `main()` vẫn `return 0` — biến lỗi thật thành màu xanh giả ngay trước bảo vệ. Xem `KIEM_KE_FILE_VA_LO_TRINH_2026-07-20.html`. |
| **Thêm sách mới / mua 5 quyển còn thiếu** | Số trang bản in không khớp PDF hiện có (T2: 362 in vs 320 PDF; khổ 17×24 vs 16×24) → **bản in khác**, chưa biết còn giữ bố cục 9+9. Và thêm trang nào cũng buộc **đo lại toàn bộ**: mẫu SRS mới, audit mới, ablation mới, downstream mới. |
| **`git filter-repo` rút `.git` 1,8 GB** | Viết lại mọi SHA → hỏng commit-hash đã trích trong luận văn và trong `docs/EVIDENCE_INDEX.md`. |
| **Chạy `--fresh` (xoá cache OCR)** | Cache OCR là *primary data*: OCR Nôm phụ thuộc API Kimhannom bên ngoài, không version-pin. Xoá là mất tái lập, tốn tiền, và **không tất định**. |

---

## Sau khi bảo vệ

Làm trên **nhánh riêng**, theo đúng thứ tự rủi ro tăng dần:

1. `pyproject.toml` + `paths.py` — rủi ro 0, chưa dời gì
2. Gom selftest về `tests/` — kèm sửa hằng `parents[N]` và test khẳng định đường dẫn
3. Mới cân nhắc src-layout và tách `data/` 3 tầng

---

## Trạng thái tại thời điểm đóng băng

| Chỉ số | Giá trị |
|---|---|
| Commit | `3c93615346` |
| Tag | `freeze-pre-thesis-2026-07-20` |
| Selftest | **212 passed, 11 failed** (`bash scripts/run_all_selftests.sh`) |
| Dataset (`labels_final.csv`) | GOLD 48.969 · SILVER 10.856 · SYLLABLE 6.751 · REVIEW 15.690 · QUARANTINE 8 |
| Precision GOLD | 97,08% → **98,00%** sau demote 1.923 crop 㝵/người |
| Sao lưu | 3 gói ở `~/ThS_archive/backup_2026-07-20/`, đã verify sha256 |

**11 assertion đỏ không phải bug code** — chúng hard-code census của thế hệ `labels.csv` cũ. Đây là biểu hiện của blocker "số liệu bất nhất", phải xử lý ở Giai đoạn 3 chứ không phải bằng cách sửa code.
