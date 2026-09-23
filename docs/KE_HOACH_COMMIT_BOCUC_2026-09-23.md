# Kế hoạch commit — bố cục đầu ra + bộ gộp chung + dọn rác (23/09/2026)

Trên HEAD `11001b7225` (đã push `origin/main`). **Agent KHÔNG commit** — bảng dưới là để người chạy
`git add` theo từng commit. `dataset/`, `prepared/`, `measure_out/`, `logs/`, `archive/` đều
**gitignore**, nên mọi thay đổi trên đĩa ở §"đã làm trên đĩa" **không vào git**.

## 1. Bốn commit đề nghị

### C1 — `feat(dataset): bộ gộp chung dataset/_ALL + công cụ merge_datasets`

```
pipeline/tools/merge_datasets.py        (MỚI, ~540 dòng, selftest 23/23)
```

Gộp 6 thư mục bộ thành một `labels.csv` (17 cột), `crops/` copy hẳn, `labels.xlsx`, `README`,
`DATASHEET`, `CHECKSUMS.txt`, `SOURCES.json`. Cột mới `cell_uid` (khoá chính duy nhất 100 %),
`book_set`, `evaluation_only`, `split_hint`, `image_dup`. Bất biến kiểm cả lúc gộp lẫn `--check`.

### C2 — `fix(publish): chặn rò rỉ TẬP ĐÁNH GIÁ vào train/val/test`

```
pipeline/publish/splits.py              (+ eval_only_books/eval_only_mask/EVAL_SPLIT, _verify bắt rò rỉ)
pipeline/publish/cli.py                 (truyền cờ + in số dòng giữ ngoài)
pipeline/publish/selftest.py            (+ test_eval_only_gate: 13 phép kiểm → 69/0)
```

### C3 — `refactor(bố cục): dataset/ chỉ chứa thư mục; bộ STT xuống dataset/SachThanhTruyen/`

```
run_pipeline.sh                         (FINAL_DIR, B7 merge, --merge/--no-merge/--merge-mode,
                                         --prune/--keep-old/--clean, --summary-only + --verify bộ gộp)
pipeline/tools/update_bang_so_lieu.py
pipeline/tools/selftest.py              (_ds_dir)
pipeline/remediation/selftest.py        (2 vòng lặp thư mục giao nộp)
pipeline/remediation/confusion_fix.py   (chỉ THÊM CHÚ THÍCH: cố ý KHÔNG đổi NGUON_VERDICT)
pipeline/tools/variant_table.py
pipeline/export_review_excel.py
.gitignore                              (+ /archive/)
README.md                               (cây thư mục)
```

Mọi chỗ **giữ đường dẫn cũ làm dự phòng** ⇒ chạy được trên cả cây thư mục đời trước.

### C4 — `docs: bố cục đầu ra, bộ gộp, quy tắc dọn + khuyết tật 21 ảnh dùng chung`

```
docs/BO_CUC_DAU_RA_2026-09-23.md        (MỚI — bố cục, bộ gộp, dọn rác, giới hạn)
docs/CHOT_CUOI_2026-09-23.md            (§4b bộ gộp + bảng lệnh)
docs/KE_HOACH_COMMIT_BOCUC_2026-09-23.md (tệp này)
docs/PIPELINE_FACTS.json                 (sinh lại bằng code_facts.py)
```

## 2. KHÔNG commit trong vòng này

| tệp | lý do |
|---|---|
| `nom-embed` (submodule) | đã `modified` từ TRƯỚC vòng này, không phải việc của vòng này |
| `web/{app.js,style.css,index.html,sample_data.json,server.py}` | **KHÔNG do vòng này sửa.** Lúc bắt đầu phiên `git status` sạch; 2 tệp đầu đổi ngay sau đó, 3 tệp sau đổi TRONG lúc chạy ⇒ có người/tiến trình khác đang làm việc trong kho. Cũng là lời giải thích khả dĩ nhất cho `dataset/SachThanhTruyen{2,4,11}/` (BO_CUC §8.7) |
| `dataset_out/CHECKSUMS.txt`, `docs/EVIDENCE_INDEX.md` | bị lần chạy **ghi thêm** (hành vi cũ của `tick`/`evidence`) — commit hay hoàn nguyên là quyết định của người |
| `dataset_out/*` | **đã `git checkout` về HEAD sau lần chạy** (`summary.json` + `CHECKSUMS.txt`); `labels*.csv` vốn không đổi một byte. `git status -- dataset_out` nay **trống**, `code_facts` **18/18** |
| `docs/BANG_SO_LIEU_CHINH_THUC.md` | `update_bang_so_lieu` (bước export STT) tự cập nhật 7 khối — nay trỏ `dataset/SachThanhTruyen/labels.csv`; nên commit CÙNG C3 |
| `dataset/SachThanhTruyen{2,4,11}/` | xuất hiện 19:29 **không do pipeline** (BO_CUC §8.7) — gitignore, đã GIỮ NGUYÊN, không commit |

## 3. Đã làm TRÊN ĐĨA (không vào git, nhưng phải biết)

| việc | kết quả |
|---|---|
| dời bộ STT `dataset/*` → `dataset/SachThanhTruyen/` | `mv`; `shasum -c` **36/36 OK** trước và sau khi dọn |
| dọn 25 bản dựng cũ (`*_v5_lang1`, `*_v7`, `*_v8`, `*_v9`, `*probe`, `prepared/*/dataset_out_*`) | **3.722 MB** → `archive/` rồi xoá; `dataset/` 3,8 G → 1,1 G |
| dựng `dataset/_ALL/` | 140.733 dòng · 140.231 tệp crop · 31.509 dòng `evaluation_only` |

## 4. Cổng nghiệm thu trước khi commit

```bash
.venv/bin/python -m pipeline.tools.merge_datasets --selftest     # 23/23
.venv/bin/python -m pipeline.publish.selftest                    # 69 PASS / 0 FAIL
.venv/bin/python -m pipeline.tools.mark_eval_dataset --selftest  # 15/15
.venv/bin/python scripts/measure/code_facts.py --check           # 18/18
./run_pipeline.sh --summary-only                                 # bảng 8 bộ TRÙNG CHOT_CUOI §4
./run_pipeline.sh --verify                                       # 6/6 PASS, 0 FAIL cứng
git status --porcelain -- data                                   # TRỐNG
```

`pipeline/tools/selftest.py` còn **3 FAIL có từ trước** (2 khuyết tật dữ liệu ở
`BO_CUC_DAU_RA` §3 + 1 phép kiểm mẫu mã của khối preflight đã bị gỡ) — **không phải hồi quy của
vòng này**; đã đối chiếu `labels.csv` byte-identical với bản công bố.

## 5. Việc còn mở

1. **21 đường dẫn ảnh dùng chung** trong bộ STT (`self_training_rescue` cấp lại chỉ số ô) — sửa sẽ
   đổi bộ đã công bố + mọi chuỗi băm; chưa làm, đã gắn cờ.
2. `pipeline/publish/` vẫn chỉ chạy trên `dataset_out/` (bộ STT). Muốn công bố cả 8 bộ thì trỏ
   `--labels dataset/_ALL/labels.csv` — cổng rò rỉ đã đọc được cột `evaluation_only` của bộ gộp.
3. `--prune` nhận diện theo **tên thư mục**; bản dựng đặt tên khác quy ước sẽ không được dọn.
4. `dataset/SachThanhTruyen{2,4,11}/` chưa rõ nguồn gốc (BO_CUC §8.7) — nên hỏi lại ai/cái gì tạo,
   rồi hoặc xoá hoặc khai chính thức; `merge_datasets` đã cảnh báo nhưng không tự quyết.
