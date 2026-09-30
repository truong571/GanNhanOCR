# `pipeline/publish/` — tập công bố theo GOLD chính xác

Một lệnh duy nhất, chạy tự động ở bước B9 của `./run_pipeline.sh … --publish` (sau gộp + gold_exact), 0 API:

```bash
.venv/bin/python -m pipeline.publish gold-exact [--all-dir dataset/_ALL] [--out <dir>] [--no-files]
.venv/bin/python -m pipeline.publish.selftest
```

Từ `dataset/_ALL/labels.csv` + `gold_exact.csv` sinh `dataset/_ALL/cong_bo/`:

| tệp | nội dung |
|---|---|
| `images.csv` | tập ẢNH: chỉ ô `gold_exact = ok`, `image` = crop chuẩn (`crops_chuan/…`) |
| `text.csv` | tập VĂN BẢN/NHÃN: mọi dòng còn lại, không cột ảnh, kèm lý do không vào tập ảnh |
| `EXCLUSIONS.json`, `RELEASE.md`, `CHECKSUMS.txt` | số đếm theo lý do loại, theo bộ; bất biến; sha256 |

**Không sửa nhãn** (`labels.csv` giữ nguyên). **KHÔNG chia tập** (quyết định A-10, 16/09): không có cột train/val/test hay
LOBO; bộ có nhãn người chỉ mang cờ `evaluation_only`. Bất biến fail-loud: tập ảnh ⊂ GOLD ∧ ok, hai tập rời + phủ đủ,
không cột chia tập, crop tồn tại + md5 khớp `gold_exact.csv`.

30/09: đã xoá công cụ công bố cũ tháng 7 (`splits.py`, `metadata.py`, `datasheet.py`, `export.py`, `validate.py`,
`hashing.py`; lệnh `split`/`metadata`/`datasheet`/`export`/`validate`/`all`) vì toàn bộ dựng trên chia tập. Xem lại trong
lịch sử git nếu cần.
