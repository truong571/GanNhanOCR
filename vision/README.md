# vision/ — Google Cloud Vision OCR cho 10 sách Hán-Nôm

Mã và kết quả chạy Vision nằm hết trong thư mục này. **Đã OCR toàn bộ 1.682 trang** của 10 sách, **ghép 2 trang thành 1 ảnh** (845 yêu cầu, 175.029 ký hiệu, 0 lỗi).
Hạn mức tháng 10/2026 (giờ Pacific): **876/950** (miễn phí 1.000/tháng). **Chạy lại sẽ gọi 0 yêu cầu**: phần nào đã có trong `cache/` thì bỏ qua.

## Dùng cache để nghiên cứu (không gọi mạng, không tốn hạn mức)

```python
import sys; sys.path.insert(0, "/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/vision")
import vision_data as vd
vd.status()                                  # số trang + số ký hiệu mỗi cuốn
syms = vd.load_page("KVK", "page_0075")      # [{ch, conf, x0, y0, x1, y1, ang}] — toạ độ TRANG GỐC (px của prepared/<Bộ>/pages_denoised)
df = vd.symbols_df(["TK"])                   # pandas.DataFrame mọi ký hiệu
raw = vd.load_raw("KVK", "page_0075")        # phản hồi THÔ của Vision + bố cục ghép (ảnh ghép chứa trang này)
```
Khoá sách: `KVK L83 TK L16 Chr stt11 stt2 stt4 B34 B18`. Cấu hình chính (cố định): `n2-grid-zhHant-h80-d9e1aa` — ghép 2 trang, languageHints `zh-Hant`, cao ô mục tiêu 80 px, trần 10,5 MP/ảnh.

## Cấu trúc

| đường dẫn | nội dung |
| --- | --- |
| `cache/n2-grid-zhHant-h80-d9e1aa/` | **KẾT QUẢ VISION**: `raw/*.json.gz` (phản hồi thô, 845 tệp), `layout/` (vị trí/hệ số co giãn từng trang trong ảnh ghép), `pages/<sách>/<trang>.json` (ký hiệu đã quy về toạ độ trang gốc), `scales.json` (hệ số đóng băng) |
| `cache/_pilot/`, `cache/legacy-n2-…/` | 6 cấu hình pilot (10 trang) và 5 phản hồi cũ đã nhập |
| `vision_harvest.py` | thu hoạch: `plan` · `run --yes` · `pilot` · `status` · `import-legacy` (sổ cái theo tháng, khoá tệp, tiếp tục được) |
| `vision_data.py` | đọc cache (ở trên) |
| `vision_crosscheck.py` | đối chứng ký hiệu Vision ↔ ô/nhãn pipeline (khớp 1–1, dị thể, lớp mã, "hại hình học", bỏ-một-ô) → `ket_qua/<cfg>/` |
| `vision_vs_human.py` | Vision so với nhãn NGƯỜI (IHR: L16, TK; Borg: B18, B34) |
| `prereg_q1_eval.py` | chấm tiêu chí Q1 (chọn cấu hình ghép) của prereg |
| `prereg/` | tiêu chí quyết định đăng ký TRƯỚC pilot (+ sha256) và ghi chú 2 lệch (D1, D2) |
| `ket_qua/` | đầu ra phân tích: `n2-grid-…/{summary.json, cells.csv.gz, disagree_candidates.csv, drift.csv, offsets.csv, column_model.csv}`, `vs_human_*.json`, `q1_stage1_*.json` |
| `tai_lieu/VISION_DOI_CHUNG_2026-10-04.md` | báo cáo: điều đã đo, kết luận, giới hạn |
| `tests/` | `test_vision_harvest.py` (27 phép) · `test_vision_crosscheck.py` (14 phép) — offline, không gọi mạng |
| `ledger/` | sổ cái yêu cầu (bản chính; bản sao ở `~/.cache/gannhanocr/`, nạp lấy MAX từng tháng) |
| `nhat_ky/run_logs/` | log từng đợt chạy thật |
| `phan_bien/`, `legacy/` | 3 phản biện độc lập (+ kết quả) và kịch bản cũ đã vô hiệu hoá + 5 phản hồi cache đầu tiên |

## Chạy thêm / làm mới (chỉ phần thiếu)

```bash
# cần interpreter có google-auth (venv riêng ngoài repo); khoá nằm NGOÀI repo
~/.venvs/vision/bin/python vision/vision_harvest.py plan                  # xem trước: 845 nhóm, 0 cần gọi
~/.venvs/vision/bin/python vision/vision_harvest.py run --n 2 --layout grid --hints zh-Hant --cap-mp 10.5 --max-requests 0 --yes
```
- Thêm trang/sách mới rồi chạy đúng lệnh trên: **chỉ nhóm mới** được gọi. Trang có ảnh nhưng không có ô trong `labels.csv` được xếp nhóm riêng ở cuối cuốn (không làm lệch cặp đã chạy).
- **ĐỪNG đổi** `--n/--layout/--hints/--cell-h/--cap-mp`: mỗi bộ tham số là một cấu hình khác nên sẽ gọi lại TẤT CẢ. Mỗi trang lưu `src_md5`: ảnh trang đổi thì chỉ nhóm đó chạy lại.
- Khoá: `~/.config/gcloud/vision-ocr-*.json` (quyền 600). **Không đặt khoá trong repo.** Hạn mức miễn phí tính theo tài khoản thanh toán; kiểm Console → Vision API → Metrics nếu có dùng nơi khác.
- Phân tích (offline) chạy bằng `.venv` của dự án; gọi mạng chạy bằng `~/.venvs/vision`.

## Kết quả đã có (chi tiết ở `tai_lieu/VISION_DOI_CHUNG_2026-10-04.md`)

- Vision đọc được 35 % ô GOLD toàn bộ (KVK 84 %, L83 81 %, Chr 75 %, TK 57 %, L16 44 %, stt11 33 %, stt2 20 %, B18 18 %, stt4 17 %, B34 12,5 %).
- So nhãn người IHR (L16, TK): ô Vision xác nhận (conf ≥ 0,8) chính xác 99 % nhưng chỉ 5,1 % / 14,2 % số ô GOLD; **bất đồng Vision≠nhãn không giàu lỗi hơn nền** (Vision đúng 4/2.723 ô) ⇒ không giúp bắt kim sai hay kiểm lại nhãn; không đưa vào pipeline.
- Crop: phần chữ nằm ngoài cửa sổ crop (bbox ± 0,12) ở ô Vision đọc được — STT 21–25 %, Chr 20 %, L16 20 %, B18 17 %, B34 15 %, KVK/L83/TK 0–2 %. Viết tay do lệch tâm từng ô (không phải lệch hằng theo cột); L16 do hộp quá thấp. Hiệu chỉnh theo cột không đạt tiêu chí.
- Còn mở để nghiên cứu: kiểm độc lập (cosine crop↔glyph nhãn, `pipeline/tools/eval_crops_v2.py`) việc nới cửa sổ L16 / dời tâm ô lệch; chữ Latin Vision đọc ở lề (B18/B34, KVK) như nguồn chữ quốc ngữ thứ ba; so Vision với kim ở cấp trang/dòng.

## Việc nên làm / không nên

- **Sao lưu `cache/`** (đã trả hạn mức; `.gitignore` không commit nó): ví dụ `tar czf vision_cache_$(date +%F).tgz vision/cache` hoặc đẩy lên HF private.
- Commit được: mã (`*.py`, `tests/`), `prereg/`, `tai_lieu/`, `README.md`; **commit `.gitignore` trước khi `git add`** để chắc chắn khoá không bị bắt.
