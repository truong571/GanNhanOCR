# lab/i5_detector_v2 — huấn luyện detector v2 (thạch bản + STT) trên Kaggle, 0 sửa tay

Mục tiêu (docs/HUONG_DAN_HUAN_LUYEN_I5_2026-09-22.md §3, phương án B): fine-tune CenterNet r34 v1 trên nhãn yếu thạch bản
(LVT1883 + KVK1884) + STT, đo trung thực trên val page-disjoint, rồi đưa vào pipeline bằng 1 lệnh. **Không commit; không sửa
run_pipeline.sh; không đụng dataset_out/, data/, STT** (STT 3 trang `labels.csv` md5 `59e436d7641fa849bb6759868ac29259` giữ nguyên).

## 🔴 KẾT QUẢ LẦN CHẠY 22/09 + SỬA CỦA VÒNG 7 (23/09) — đọc trước tiên

Kaggle T4, 349 s, **dừng sớm ở epoch 6/20**. Trên thạch bản v2 **tốt hơn hẳn**; nhưng guard STT chặn nên
**KHÔNG ckpt nào được ghi**, và chỉ `last.pt` (346 MB) được đẩy lên HF repo **riêng tư** `mdnt571/nom-char-det-v2`
— token trong `.env` đã hết hiệu lực (`Invalid user token`) nên **chưa lấy về được** (chi tiết + 3 cách lấy:
`docs/VONG7_DETECTOR_V2_VA_CROP_GOC_2026-09-23.md` §1.2). Output đầy đủ giữ ở `kaggle_run_2026-09-22_ketqua.ipynb`.

| epoch | litho ok50 | n==N @0,15 | cắt thân chữ | STT F1@0,2 | Chrestomathie n==N (held-out) |
|---|---|---|---|---|---|
| 0 = v1 | 98,71 | 94,7 | 11,32 | **0,8772** | **83,8** |
| 3 (tốt nhất) | **100,0** | **98,7** | **0,22** | 0,8584 | 74,7 |
| 6 (= `last.pt`) | 100,0 | 98,0 | 0,22 | 0,8625 | 73,7 |

- **Quên miền, không phải v2 kém**: STT tụt ~0,013. Vì repo đã có `books[].detector_ckpt` **THEO SÁCH**, ckpt thạch
  bản không bao giờ chạy trên STT → guard chỉ cần khi MỘT mô hình phục vụ cả hai miền.
- 🔴 **Chrestomathie (KHÔNG train) cũng TỤT ở mọi epoch** (83,8 → 64,6–78,8). Nên **chưa có bằng chứng** áp v2 cho
  Chresto; hai bộ IHR không có trong bundle nên cũng chưa có số. Khi có ckpt: khai v2 cho **LVT1883 + KVK1884
  trước**, mở rộng chỉ khi `box_ref_eval --ckpt` trên ảnh gốc của chính sách đó tốt hơn v1.

**Sửa vòng 7** (chạy lại 1 lần ~1 h là có ckpt): `--guard-stt-f1 <số|none>` (mặc định giữ 0,01) và trainer **LUÔN**
ghi `best_litho.pt` = epoch tốt nhất theo riêng tiêu chí thạch bản, kèm `warning` **trong chính ckpt** + mục cảnh báo
trong `report.md`; `apply_v2.sh` nhận ckpt đó (không bị guard chặn, chỉ cảnh báo) và ghi `detector_ckpt` cho **6
config** (thêm LVT1916, TK1872). Notebook có `GUARD_STT = "none"`.
**Chỉ sửa mã thì KHÔNG phải đẩy lại 243 MB**: `make_bundle.py` sinh thêm `i5v2_code.zip` (39 KB) — tạo 1 Kaggle
dataset từ nó, Add Input **cả hai** (bundle cũ + code mới); ô (2) ưu tiên input có `train_kaggle.py` mà không có
`bundle_stats.json`.

## Phát hiện khi dựng mốc v1 (22/09) — đọc trước khi train
Cùng mã đo, cùng 27 trang thạch bản val gốc, chỉ đổi phép thu ảnh trang về 1024 trong `CenterNetDetector._preprocess`:

| v1, 27 trang litho val (ảnh gốc) | ok50 | miss | extra/100 | tầng n==N @0,15 | cắt thân chữ | STT F1@0,2 (45 trang) |
|---|---|---|---|---|---|---|
| `cv2.resize` mặc định INTER_LINEAR (pipeline hiện tại) | 95,0 | 2,1 | 2,6 | **77,0** | 7,1 | 0,8767 |
| INTER_AREA (khử răng cưa khi thu ~3×) | **98,1** | 0,8 | 0,9 | **91,9** | 6,4 | 0,8785 |

Phần lớn "điểm tin cậy thấp / bỏ sót chữ nhạt / bắt 2 lần" của I5 là **răng cưa khi thu 3.200 → 1.024 px** làm rụng nét mảnh, không
phải model. Vì vậy: (i) khoá `books[].detector_resize: area` (mặc định `linear` = STT không đổi byte) đã thêm vào engine và là
**mốc so sánh thật** của v2; (ii) bundle đo bằng `area`; v2 phải thắng **v1+area**, không phải v1 cũ. `apply_v2.sh` đo cả ba.

Build trọn 3 sách với `area` (`./run_pipeline.sh --book all-new --suffix _area`, 0 API; `dataset_out_<Book>[_b1]_area/`, 22/09 tối): I5 thô
65,2 / 59,2 / 70,7 → **77,7 / 77,9 / 69,3 %**, GOLD ảnh 8.650 / 13.908 / 5.359 → 8.733 / 14.165 / 5.344, tier văn bản + khớp dị bản không đổi;
**nhưng** hộp cao hơn ≈ 2 % (h/p 1,14 → 1,16) → LVT `bleed` trên ảnh export 18,0 → 21,7 %, cắt thân chữ pitch 9,5 → 10,6 % (KVK 10,5 = 10,5 %,
Chresto thu 2,2× ≈ linear). Theo luật "không chỉ số nào giảm" **bản chốt vẫn `linear`** (config ghi `detector_resize: linear` kèm số;
`docs/BAO_CAO_TONG_HOP_SACH_MOI_2026-09-22.md` §3.3). Vì thế nghiệm thu v2 (bước 6) thêm điều kiện **bleed ảnh export ≤ 18,0 / 10,5 %**
và cắt thân chữ ≤ v1+area — v2 phải học lại kích thước hộp thạch bản, không chỉ điểm tin cậy.

## 6 bước
1. **Dựng bundle** (≈ 3 phút, không cần GPU): `.venv/bin/python lab/i5_detector_v2/make_bundle.py`
   → `bundle/` (777 ảnh long side 1536 PNG xám 161 MB: LVT 105 + KVK 163 + STT 445 + Chrestomathie 64 chỉ để đo; manifest nhãn
   yếu quy đổi theo `scale_table.csv`; `v1/detector_r34.best.pt`; gói mã `i5v2/`; notebook) + `i5v2_bundle.zip` **243 MB**.
   Script tự kiểm `import i5v2` với PYTHONPATH chỉ bundle; > 1,5 GB sẽ tự hạ 1280 (không xảy ra).
2. **Upload** kaggle.com → Datasets → New Dataset → kéo `i5v2_bundle.zip` (Kaggle tự giải nén) → Create. (`bundle/README.txt`)
3. **Notebook**: New Notebook → File → Import `kaggle_train_i5_v2.ipynb` → Add Input: dataset vừa tạo → Settings: GPU **T4**
   (P100 có thể lỗi với torch mới), Persistence **Files only**; Internet chỉ cần nếu đẩy HF (`HF_REPO` + Secret `HF_TOKEN`).
4. **Run All** (≈ 1–1,5 giờ: 20 epoch × 430 mẫu ≈ 2–3 phút/epoch T4 img 1024 + eval 92 trang ≈ 20 s). Cell cấu hình: EPOCHS 20,
   BATCH 4, IMG 1024 (hoặc 1280), LR 2e-4 cosine + warmup 1 epoch, init v1, EMA 0,995, AMP, seed 0. Cell cuối in bảng v1 ↔ v2.
5. **Tải về** `/kaggle/working/best.pt` (+ `metrics.csv`, `report.md`) — tab Output.
6. **Áp dụng + đo lại**: `lab/i5_detector_v2/apply_v2.sh ~/Downloads/best.pt` → copy `train_crop/detector_r34_v2_litho.pt`, ghi
   `detector_ckpt` + `detector_resize: area` vào 4 config sách mới (LVT, KVK chính + `_b1` run_config, Chresto), chạy
   `box_ref_eval` 3 lần (v1 · v1+area · v2+area, 27 trang/sách, ~80 s/lần) → bảng, rồi `./run_pipeline.sh --book all-new --suffix _v2`
   → `compare_builds.py` (tier, GOLD ảnh, I5, khớp dị bản). `--no-build` bỏ bước dài; `--ckpt-only` chỉ cài. Hoàn nguyên: in ở cuối.

## Mục tiêu số (val page-disjoint, đọc `metrics.csv` / `report.md`)
| chỉ số | pipeline hôm nay (ảnh gốc, v1 linear) | v1 area (ảnh gốc) | **mốc trong Kaggle** = v1 area trên bundle (`epoch 0`) | mục tiêu v2 |
|---|---|---|---|---|
| `litho_ok50` — ô tham chiếu IoU ≥ 0,5 (%) | 95,0 | 98,1 | 98,71 | **≥ 98 và > mốc** (tiêu chí chọn best) |
| `litho_tiers_eq_015` — % tầng hộp thô n == N @0,15 | 77,0 | 91,9 | 96,1 (@0,2: 89,7) | **≥ 90**, tăng so mốc |
| `litho_cut` — cắt vào thân chữ (%) | 7,1 | 6,4 | 11,4 (*) | **giảm** so mốc |
| `stt_F1_020` — STT F1 @0,2 (production, 45 trang) | 0,8767 | 0,8785 | 0,8791 (@0,15: 0,8778) | **≥ mốc − 0,01** (guard cứng) |
| `chresto_tiers_eq_015` — Chrestomathie (không train) % cột n == N | — | — | 90,9 (20 trang) | tăng (tổng quát hoá liên sách) |

(*) "cắt" đo 2 hàng mép ở độ phân giải bundle (≈ 4 hàng ảnh gốc) nên cao hơn số ảnh gốc; chỉ so trong cùng cột. Mốc bundle =
`train_kaggle.py --eval-only` (27 litho + 45 STT + 20 Chresto), lưu `v1_baseline_val.json` (area) / `v1_baseline_val_linear.json`
(linear trên bundle: 98,43 / 93,4 — bundle đã thu 1536 bằng INTER_AREA nên ít răng cưa hơn pipeline). **best.pt chỉ được ghi khi
v2 vượt mốc v1 area** theo (ok50, n==N, −cắt) và qua guard STT; không vượt → không có best.pt → giữ v1 + `detector_resize: area`.
Dừng sớm: 4 epoch không tăng ≥ 0,2 điểm ok50, nhưng không trước epoch 6. **Mọi số là proxy** (ô tham chiếu = kim + chiếu mực,
không GT người): page-disjoint + STT hồi quy + cắt-thân-chữ là ba hàng rào; nghiệm thu cuối là `box_ref_eval` trên ảnh gốc +
build `_v2` (bước 6), không phải số trong Kaggle.

## Cách đọc `metrics.csv`
Hàng `epoch 0` = v1 (cùng mã, cùng val). Mỗi epoch: `loss`, `litho_*` (ok50 / miss / extra / |dy| trung vị & p90 % bước / IoU /
n==N @0,15 & @0,2 / % tầng thiếu / thừa / cắt), `stt_F1_020`, `stt_F1_015`, `chresto_*`, `is_best`, `stt_ok`. `report.md` = bảng
v1 ↔ v2(best) + Δ + mục tiêu + theo epoch + kết luận ĐẠT/chưa. `best.pt` / `best_litho.pt` (định dạng pipeline: `model`, `arch`, `img`, `use_dcn`,
`val`, `base_v1`) nạp thẳng vào `CenterNetDetector`; `last.pt` = toàn trạng thái để resume (không dùng cho pipeline).

## Sự cố
- **Hết RAM GPU**: BATCH 4 → 2, hoặc IMG 1280 → 1024. **Chậm** (> 5 phút/epoch): kiểm GPU đã bật (cell 2 in "GPU : Tesla T4").
- **Reset / hết 12 h**: Run All lại → cell 3 tự resume `last.pt` (Persistence Files only) hoặc kéo từ HF nếu khai `HF_REPO`.
- **Không có best.pt** ("KHÔNG có best.pt (guard STT)"): v2 làm STT tụt → giữ v1; thử `EXTRA="--lr 1e-4 --p-gray 0.3"`.
- **Chỉ muốn thử đường ống**: `SMOKE = True` (4 trang/miền, 1 epoch, ~2 phút) — không phải bằng chứng.
- **P100 lỗi CUDA (sm_60)**: đổi T4. **DCN**: v1 dùng DeformConv2d; Kaggle torchvision có sẵn; máy Mac MPS không có backward → trainer tự chọn CPU.

## Chứng minh chạy được (máy, 22/09)
- `train_kaggle.py --data bundle --smoke --threads 4` (4 trang/miền, 1 epoch, img 512, CPU): **≈ 10 s**, ra `best.pt`/`last.pt`/
  `metrics.csv`/`report.md`; `--eval-only` v1 trên bundle val (92 trang, CPU 6 luồng) ≈ 55 s.
- Pipeline: `book_layout_selftest` 100/100 (+21 phép: `detector_ckpt`/`detector_resize`, `resolve_detector_ckpt`, cache `(ckpt, resize, thr)`),
  `phase1_engine_selftest` 253/0, `pitch_decode` 22/22; STT 3 trang `labels.csv` md5 **59e436d7641fa849bb6759868ac29259** (556 dòng);
  build LVT `--limit 2` với `detector_ckpt` smoke + `area` → `seg_backend detector_centernet_best+area+pitch`, summary ghi `detector_ckpt`.

## Tệp
`make_bundle.py` (bundle + zip + kiểm import) · `build_notebook.py` → `kaggle_train_i5_v2.ipynb` · `train_kaggle.py` (train / `--smoke` /
`--eval-only`) · `i5v2/` (model = bản sao `train_crop/model_centernet.py`; data/loss/decode/evalref/trainer/hub/pitch_decode) ·
`apply_v2.sh` · `set_book_keys.py` · `box_ref_compare.py` · `compare_builds.py` · `v1_baseline_val*.json`. Engine: `book_layout.py`
(khoá `detector_ckpt`, `detector_resize`, `resolve_detector_ckpt`), `align_production.py` (`DETECTOR_CKPT/RESIZE`, cache theo
`(ckpt, resize, thr)`, `detector_backend_name`), `build_dataset.py` (fail fast + gán/khôi phục theo sách, `summary.detector_params_by_book`),
`char_detector/detector_infer.py` + `train_crop/infer_centernet.py` (tham số `resize`, mặc định `linear`), `scripts/measure/box_ref_eval.py`
(`--ckpt`, `--resize`).
