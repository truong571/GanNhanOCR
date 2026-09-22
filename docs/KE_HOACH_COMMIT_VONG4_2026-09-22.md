# Kế hoạch commit vòng 4 (22/09 tối) — trên HEAD 8c08591e9a, nhánh main

Trạng thái lúc chốt: `git status --short -- dataset_out data prepared/SachThanhTruyen*` **trống**; hồi quy STT 3 trang (`build_dataset --config config/pipeline.yaml
--qd01-cells none --force --pages pages3.csv`, stt2/0024 · stt4/0050 · stt11/0100) `labels.csv` md5 **59e436d7641fa849bb6759868ac29259** (556 dòng) worktree HEAD 8c08591e9a
== mã mới (346 tệp, 344 giống byte; `summary.json`/`decisions_report.json` chỉ khác đường dẫn REPO; 0 dòng "detector riêng sách"); `./run_pipeline.sh --dry-run` in đúng 6 bước STT;
selftest **book_layout 100/100** (+21: `detector_ckpt`/`detector_resize`, `resolve_detector_ckpt`, cache `(ckpt, resize, thr)`) · phase1_engine 253/0 · pitch_decode 22/22 ·
mechanism_gates 94/94 · tools 139 pass / 3 fail (có sẵn ở HEAD, `KE_HOACH_COMMIT_2026-09-21.md` §4) · `scripts/measure/code_facts.py` → FACTS **18/18** (pin `book_layout.py`
`DEFAULT_N_COLUMNS` 50 → 61). `step0_setup` 4 config "Validation passed". `git add` **đích danh** (không `-A`; submodule `nom-embed` ` m` không đụng).

**Quyết định vòng 4**: bản chốt 3 sách **vẫn pitch + `detector_resize: linear`**; bản `_area` (INTER_AREA) giữ làm bản so sánh
(`docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` §3.3): I5 thô 77,7 / 77,9 / 69,3 % nhưng LVT bleed ảnh export 18,0 → 21,7 %, Chresto không lợi.

## Commit 1 — engine: detector_ckpt / detector_resize theo sách + phép thu ảnh `resize` trong CenterNet; STT byte-identical

```
pipeline/align_engine/book_layout.py                  (M)  khoá books[].detector_ckpt (str|None) + detector_resize linear|area; resolve_detector_ckpt (fail fast); docstring
pipeline/align_engine/book_layout_selftest.py         (M)  +21 phép → 100/100
pipeline/align_engine/align_production.py             (M)  DETECTOR_CKPT / DETECTOR_RESIZE, _get_detector cache theo (ckpt, resize, thr), detector_backend_name (+area, +<ckpt>)
pipeline/align_engine/build_dataset.py                (M)  resolve + dựng thử detector theo sách TRƯỚC align (fail fast); gán/khôi phục theo sách; summary.detector_params_by_book[].detector_ckpt/resize
pipeline/align_engine/char_detector/detector_infer.py (M)  tham số resize (mặc định linear)
train_crop/infer_centernet.py                         (M)  CenterNetDetector(resize=) → RESIZE_INTERP {linear: INTER_LINEAR, area: INTER_AREA}; mặc định linear
```
Thông điệp gợi ý: `feat(engine): detector_ckpt/detector_resize theo sách (INTER_AREA khử răng cưa khi thu 3.200→1.024; ckpt riêng sách, fail fast); mặc định linear/v1 = STT byte-identical`.
Bằng chứng kèm: md5 59e436d7… hai bên; book_layout 100/100; `measure_out/box_ref_area/` (v1+area 27 trang/sách: I5 cột 63,7 → 79,6 / 54,4 → 74,8 %, ok50 pitch 98,1 → 99,3 / 97,0 → 98,6 %).

## Commit 2 — lab/i5_detector_v2 (gói Kaggle v2) + config 3 sách (khoá resize có số đo) + box_ref_eval --ckpt/--resize + pin FACTS

```
lab/i5_detector_v2/.gitignore                          (??) bundle/, __pycache__/ (zip/.pt đã ignore ở gốc)
lab/i5_detector_v2/README.md                           (??) 6 bước Kaggle, phát hiện răng cưa, mục tiêu số, sự cố
lab/i5_detector_v2/make_bundle.py                      (??) bundle 777 ảnh 1536 px + manifest nhãn yếu + v1 + mã + notebook → zip (tự kiểm import)
lab/i5_detector_v2/build_notebook.py                   (??) sinh kaggle_train_i5_v2.ipynb
lab/i5_detector_v2/kaggle_train_i5_v2.ipynb            (??) notebook T4 (8 KB, không output)
lab/i5_detector_v2/train_kaggle.py                     (??) train / --smoke / --eval-only (mốc v1 trên bundle)
lab/i5_detector_v2/i5v2/{__init__,data,decode,evalref,hub,loss,model,pitch_decode,trainer}.py   (??) gói mã (model = bản sao train_crop/model_centernet.py)
lab/i5_detector_v2/apply_v2.sh                         (??) cài best.pt + ghi 4 config + box_ref_eval ×3 + build _v2 + compare
lab/i5_detector_v2/set_book_keys.py                    (??) ghi/xoá detector_ckpt/detector_resize trong config (giữ comment)
lab/i5_detector_v2/box_ref_compare.py                  (??) bảng v1 · v1+area · v2
lab/i5_detector_v2/compare_builds.py                   (??) so 2 build (tier, GOLD ảnh, I5, khớp dị bản)
lab/i5_detector_v2/v1_baseline_val.json, v1_baseline_val_linear.json   (??) mốc v1 trên bundle (area / linear)
config/pipeline_LucVanTien1883.yaml                    (M)  detector_resize: linear + số đo _area (I5 77,7 %, bleed 18,0 → 21,7 %) và lý do chưa chốt
config/pipeline_KimVanKieu1884_b1.yaml                 (M)  detector_resize: linear + số đo _area (I5 77,9 %, GOLD ảnh 14.165, không chỉ số giảm)
config/pipeline_Chrestomathie1872.yaml                 (M)  detector_resize: linear + số đo _area (thu 2,2× ≈ linear)
scripts/measure/box_ref_eval.py                        (M)  --ckpt, --resize; summary.detector {ckpt, resize, img}
scripts/measure/code_facts.py                          (M)  pin book_layout.py DEFAULT_N_COLUMNS 50 → 61
docs/PIPELINE_FACTS.json                               (M)  18/18 (sinh lại sau commit 1 nếu muốn git_head khớp)
```
Thông điệp gợi ý: `feat(lab,measure,config): gói huấn luyện detector v2 trên Kaggle (lab/i5_detector_v2, mốc v1+area) + box_ref_eval --ckpt/--resize + config 3 sách khoá detector_resize (linear, số đo INTER_AREA)`.
Bằng chứng kèm: `make_bundle.py` zip 243 MB tự kiểm import; `train_kaggle.py --smoke` 10 s; `--eval-only` v1 bundle 55 s; `_area` 3 sách EXIT 0, 0 API, 174 / 237 / 138 s.

## Commit 3 — docs

```
docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md            (M)  §0 phát hiện răng cưa (bảng linear ↔ area, quyết định, hệ quả v2), §3 mốc/nghiệm thu = v1+area, §5, §6 Kaggle, §7 tệp
docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md          (M)  §0, §2 (dòng vòng 4), §3.3 bảng _area 3 sách + quyết định, §6, §8.2, §9, hồi quy vòng 4
docs/HUONG_DAN_CHAY_SACH_MOI_2026-09-21.md            (M)  ghi chú vòng 4 (đầu tệp), §6.1
docs/CHAY_LVT1883_2026-09-21.md, CHAY_KVK1884_B1_2026-09-22.md, CHAY_CHRESTO1872_2026-09-22.md   (M)  khối đầu "22/09 tối — thử area, KHÔNG đổi chốt" + số
docs/KE_HOACH_COMMIT_VONG4_2026-09-22.md              (??) tệp này
```
Thông điệp gợi ý: `docs: phát hiện răng cưa INTER_AREA (I5 thô 77,7/77,9 % nhưng bleed LVT tăng → chốt vẫn pitch-linear), hướng dẫn v2 Kaggle, kế hoạch commit vòng 4`.

## KHÔNG commit

| Tệp / thư mục | Lý do |
|---|---|
| `lab/i5_detector_v2/bundle/` (161 MB), `i5v2_bundle.zip` (243 MB), `__pycache__/` | đã ignore (`lab/i5_detector_v2/.gitignore`, `*.zip` gốc); dựng lại 3 phút bằng `make_bundle.py` |
| `train_crop/*.pt`, `train_crop/data_lithograph/` | ignore sẵn (`*.pt`, `/train_crop/data_lithograph/`) |
| `dataset_*/`, `dataset_out_*/` (kể cả `*_area`, `*_v3_legacy`, `*_rp`), `prepared*/`, `measure_out/` (kể cả `box_ref_area/`), `logs/*.log` | đã ignore; tái sinh `./run_pipeline.sh --book all-new [--suffix _area]` (~9 phút, 0 API), `box_ref_eval.py --resize area` (78 s) |
| `pipeline/tools/kiem_nguoi_grid.py`, `kiem_nguoi_score.py` (??) | chưa selftest/review; đề tài không có người kiểm (như vòng 2–3) |
| `docs/KVK1884_TRANG_CAN_XAC_NHAN.csv` (??) | lỗi thời (như vòng 3) — xoá hoặc để ngoài |
| `nom-embed` (` m`) | submodule có best.pt/last.pt chưa track — không đụng |

## Kiểm trước khi bấm commit

1. `git status --short -- dataset_out data prepared/SachThanhTruyen*` trống; `git diff --stat` chỉ gồm tệp trong 3 nhóm trên.
2. Sau commit 1: `book_layout_selftest` 100/100, `phase1_engine_selftest` 253/0, `pitch_decode --selftest` 22/22; hồi quy STT 3 trang md5 59e436d7… (worktree tạm đã gỡ, dựng lại nếu cần).
3. Sau commit 2: `step0_setup` 4 config "Validation passed" (book_layout ghi `detector_resize=linear`, `detector_ckpt=None`); `./run_pipeline.sh --book all-new --dry-run` in B0→B6;
   `./run_pipeline.sh --dry-run` in 6 bước STT; `.venv/bin/python -c "import sys; sys.path.insert(0,'lab/i5_detector_v2'); import i5v2"`.
4. Sau commit 3 (tuỳ chọn): `scripts/measure/code_facts.py` → 18/18 rồi amend FACTS.

## Nếu muốn lấy `_area` làm chốt (quyết định người, 1 phút)

Đổi `detector_resize: linear → area` trong config sách muốn (KVK là sách không giảm chỉ số nào; LVT đổi bleed 18,0 → 21,7 % lấy I5 77,7 % + 83 ảnh GOLD),
đổi tên `dataset_out_<Book>[_b1]/` → `*_v4_linear/`, `dataset_<Book>/` → `*_v4_linear/`, rồi `_area` → không hậu tố (hoặc chạy lại `./run_pipeline.sh --book <Book>`
để có `CHECKSUMS.txt` + log trực tiếp); cập nhật khối đầu `CHAY_*` và bảng §3 BAO_CAO_TONG_HOP theo số ở §3.3.
