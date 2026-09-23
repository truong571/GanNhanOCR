# Vòng 7 — detector v2 (chưa lấy được ckpt), crop giao nộp giữ ẢNH GỐC, gỡ PTCL (23/09/2026)

HEAD `014a0733fa`, nhánh `main`. **Chưa commit gì.** `data/` chỉ ĐỌC. STT vẫn **byte-identical**
(`labels.csv` md5 `59e436d7641fa849bb6759868ac29259`, 556 dòng, `diff -rq` 0 tệp khác — §5).

> **Ba dòng kết luận.**
> 1. **KHÔNG lấy được checkpoint v2.** Lần chạy Kaggle 22/09 chỉ đẩy `last.pt` lên HF repo **riêng tư**
>    `mdnt571/nom-char-det-v2`; token `HF_TOKEN` trong `.env` trả **"Invalid user token"** (401), máy không có
>    `~/.cache/huggingface/token` cũng không có `~/.kaggle/kaggle.json`. Đã làm nhánh (a): guard STT thành **tuỳ chọn**
>    (`--guard-stt-f1 <số|none>`) và trainer **LUÔN** ghi thêm `best_litho.pt` → chạy lại 1 lần (~1 h T4) là có ckpt.
> 2. **Crop giao nộp nay cắt từ ẢNH QUÉT GỐC** (`books[].crop_source: original`, bật cho 5 sách). Hình học KHÔNG đổi
>    một chữ số: so hai bản dựng cùng 2 trang LVT1883, `labels.csv` chỉ khác **đúng 1 cột `image_md5`** (126/126 ô).
> 3. **PTCL loại khỏi phạm vi**: mọi tham chiếu CHẠY ĐƯỢC đã gỡ (§4).

---

## 0. Tái lập

```bash
PY=.venv/bin/python
# (1) selftest
$PY -m pipeline.align_engine.book_layout_selftest          # 133/0  (+33: [10] crop_source)
$PY -m pipeline.phase1_engine_selftest                     # 270/0  (+17: crop_source: original)
$PY -m pipeline.tools.ingest_lithograph_selftest           # 82/82
$PY -m pipeline.tools.ingest_prose_selftest                # 33
$PY -m pipeline.tools.ingest_ihr_selftest                  # 64/64
$PY -m pipeline.align_engine.char_detector.pitch_decode --selftest   # 22/0
$PY scripts/measure/verses_ref_fix.py --selftest           # 19/19
$PY scripts/measure/code_facts.py --check                  # 18/18
# (2) hồi quy STT 3 trang (phải byte-identical)
printf 'book,page\nstt2,page_0024\nstt4,page_0050\nstt11,page_0100\n' > /tmp/pages3.csv
$PY -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --qd01-cells none \
    --force --pages /tmp/pages3.csv --out <scratch>/stt ; md5 <scratch>/stt/labels.csv
# (3) ảnh mẫu TRƯỚC/SAU (đã sinh: measure_out/crop_source/, 10 ảnh + samples.csv + summary.json)
$PY -m pipeline.align_engine.build_dataset --config config/pipeline_LucVanTien1883.yaml --reseg detector \
    --qd01-cells none --decisions none --use-s3 --two-pass --box-rule syl_index --force --limit 2 --out <scratch>/lvt
$PY scripts/measure/crop_source_samples.py --build <scratch>/lvt [--build …] --per-build 2
```

---

## 1. NHIỆM VỤ 1 — detector v2: **KHÔNG lấy được ckpt**, đã sửa để lần chạy sau không mất trắng

### 1.1 Kết quả huấn luyện 22/09 (đọc từ output trong notebook, nay lưu ở `lab/i5_detector_v2/kaggle_run_2026-09-22_ketqua.ipynb`)

Chạy 349 s, dừng sớm ở **epoch 6/20** (4 epoch không cải thiện). Val page-disjoint, cùng mã, cùng tập.

| epoch | litho ok50 | n==N @0,15 | cắt thân chữ | STT F1@0,2 | **Chrestomathie n==N @0,15** |
|---|---|---|---|---|---|
| 0 = **v1** | 98,71 | 94,7 | 11,32 | **0,8772** | **83,8** |
| 1 | 100,0 | 88,4 | 0,85 | 0,8597 | 64,6 |
| 2 | 100,0 | 97,8 | 0,37 | 0,8639 | 73,7 |
| **3** | **100,0** | **98,7** | **0,22** | 0,8584 | 74,7 |
| 4 | 100,0 | 98,7 | 0,34 | 0,8598 | 71,7 |
| 5 | 100,0 | 98,5 | 0,31 | 0,8625 | 78,8 |
| 6 | 100,0 | 98,0 | 0,22 | 0,8625 | 73,7 |

**Hai điều đề bài chưa nêu, phải nói rõ trước khi áp v2 cho 5 sách:**

- ✅ Trên **LVT1883 + KVK1884** (miền ĐÃ train) v2 tốt hơn hẳn, đúng như đề bài mô tả.
- 🔴 Trên **Chrestomathie1872** (held-out, **không** có trang nào trong train) v2 **KÉM HƠN v1 ở MỌI epoch**:
  83,8 % → 64,6–78,8 %. Đây cũng là quên miền, cùng gốc với STT. Vì vậy **không có bằng chứng** để khai
  `detector_ckpt` v2 cho Chrestomathie; hai bộ IHR (LVT1916, TK1872) **không có trong bundle** nên cũng chưa có số.
  → Khi có ckpt: khai v2 cho **2 sách thạch bản trước**, và chỉ mở rộng sang 3 sách còn lại nếu
  `box_ref_eval --ckpt` trên ảnh gốc của CHÍNH sách đó tốt hơn v1.

### 1.2 Vì sao không lấy được ckpt

| Đường | Kết quả |
|---|---|
| HF `mdnt571/nom-char-det-v2` (notebook đã đẩy `last.pt` 346 MB + `metrics.csv` + `report.md`) | HTTP **401** ẩn danh; `HF_TOKEN` trong `.env` → `whoami` trả **"Invalid user token"** |
| `~/.cache/huggingface/token` | không có |
| `~/Downloads`, `lab/i5_detector_v2/`, repo | không có `.pt` nào của v2 |
| Kaggle API (`~/.kaggle/kaggle.json`) | không có |

`last.pt` **dùng được ngay** nếu lấy về: trainer ghi nó ở **đúng định dạng pipeline** (`model` = trọng số EMA,
`arch`, `img`, `use_dcn`) kèm trạng thái resume. Đó là epoch 6 (ok50 100,0 · n==N 98,0 · cắt 0,22).

**Việc người dùng cần làm — chọn 1:**
1. cấp `HF_TOKEN` còn hạn (repo riêng tư) → tải `last.pt`, đổi tên `train_crop/detector_r34_v2_litho.pt`, chạy
   `lab/i5_detector_v2/apply_v2.sh <ckpt>`; **hoặc**
2. mở notebook Kaggle cũ → tab Output → tải `last.pt`; **hoặc**
3. **chạy lại 1 lần (~1 h T4)** với mã đã sửa dưới đây → có `best_litho.pt` sạch (chọn theo epoch tốt nhất).

### 1.3 Đã sửa (nhánh (a)) — lần chạy sau không bao giờ về tay trắng

| Tệp | Sửa |
|---|---|
| `lab/i5_detector_v2/train_kaggle.py` | `--guard-stt-f1 <số\|none>`; mặc định **giữ như cũ** (0,01). `none` = tắt guard. |
| `lab/i5_detector_v2/i5v2/trainer.py` | `_stt_ok(tol=None)` → guard tắt; **LUÔN ghi `best_litho.pt`** = epoch tốt nhất theo riêng (ok50 → n==N → −cắt), kèm khoá `selected_by="litho_only"`, `stt_guard_passed`, `warning` **trong chính ckpt**; đẩy HF; resume giữ `best_litho_epoch`. `report.md` thêm mục `best_litho.pt` + CẢNH BÁO. |
| `lab/i5_detector_v2/build_notebook.py` → `kaggle_train_i5_v2.ipynb` | ô cấu hình thêm `GUARD_STT = "none"` (kèm lý do), `HF_REPO` mặc định `mdnt571/nom-char-det-v2`, ô kết quả in cả 2 ckpt + cảnh báo. |
| `lab/i5_detector_v2/make_bundle.py` | sinh thêm **`i5v2_code.zip` (39 KB)** — khi CHỈ mã đổi thì không phải đẩy lại 243 MB ảnh; ô (2) notebook ưu tiên input chứa `train_kaggle.py` mà **không** có `bundle_stats.json`. |
| `lab/i5_detector_v2/apply_v2.sh` | nhận `best_litho.pt` (ckpt `selected_by=litho_only` không bị guard chặn, chỉ CẢNH BÁO), thêm `--allow-stt-drop`, ghi `detector_ckpt` cho **6 config** (thêm LVT1916 + TK1872), bước 4 chạy thêm `--book all-ihr` + `ihr_endtoend_eval`. |

**Lập luận của đề bài là đúng và đã được mã hoá**: repo có `books[].detector_ckpt` THEO SÁCH, nên ckpt thạch bản
không bao giờ chạy trên STT (`config/pipeline.yaml` không khai khoá này) → STT vẫn byte-identical. Cảnh báo đi kèm
ckpt + report + `apply_v2.sh` nhắc **không** đặt ckpt này vào `NOM_DETECTOR_CKPT` (detector toàn cục).

**Chạy lại (1 lần, ~1 h T4):** `.venv/bin/python lab/i5_detector_v2/make_bundle.py --no-zip` (bundle/ đã đồng bộ mã) →
tạo Kaggle dataset mới từ `lab/i5_detector_v2/i5v2_code.zip` (39 KB) → Add Input **cả hai** (bundle cũ + code mới) →
Run All. Kết quả: `/kaggle/working/{best.pt, best_litho.pt, last.pt, metrics.csv, report.md}`.

**DỪNG Ở ĐÂY cho nhiệm vụ 1** — không có ckpt thì không đo được `box_ref_eval` v1↔v2, không build `_v2`,
không so precision IHR trước–sau.

---

## 2. NHIỆM VỤ 2 — crop giao nộp phải giữ ẢNH GỐC ✅ XONG

### 2.1 Gốc của vấn đề (đo, không đoán)

`prepared/<Book>/pages/page_XXXX.png` do adapter ingest sinh: `PIL.convert("L")` + `--contrast stretch|otsu`
(+ `×scale` với IHR). Với LVT1883, trang 1: ảnh gốc `p2 = 0`, **`p90 = 128`**; sau `stretch_gray` thành `p90 = 255`,
|Δ| xám trung bình **117,6** trên toàn trang. (Ghi chú cũ trong `config/pipeline_LucVanTien1883.yaml` nói stretch là
"ánh xạ đồng nhất" — **đo lại thì không phải**; nền bản quét bị kéo trắng bệt.) KVK1884 dùng `otsu` còn ép thẳng
mọi điểm ảnh > ngưỡng về 255.

### 2.2 Thiết kế — hình học tính trên ảnh ĐÃ XỬ LÝ, điểm ảnh lấy từ ảnh GỐC

Khoá mới `books[].crop_source: processed | original` (`pipeline/align_engine/book_layout.py`), **mặc định
`processed`** = hành vi cũ. Khi `original`:

1. cửa sổ pad, **seam carve** mực hàng xóm và `tighten_box` **vẫn** tính trên ảnh đã xử lý / `gray_full`;
2. **đúng** khung hình đó cắt từ ảnh quét gốc `data/<Book>/…` (tra qua `manifest.json`:
   `pages[].source_file` + `scale`; sai kích thước hoặc thiếu ảnh → **lỗi ngay**, không rơi ngầm);
3. carve trên ảnh gốc tô **nền giấy** (p90 mỗi kênh) thay vì 255, để vết xoá không thành mảng trắng;
4. `ink` / `crop_w` / `crop_h` / `seg_flag` vẫn đo trên bản đã xử lý → **không đổi một chữ số**; `image_md5` đổi vì
   nó là băm của **tệp thật đã giao**;
5. bản đã xử lý ghi song song ở **`$out/crops_bin/<tier>/<cùng tên>.png`**; `enrich_crop_quality` **tự ưu tiên**
   thư mục này → `crop_quality_flag` / `stray_ink` / `border_ink` **không đổi** (cổng cơ chế B4' đọc cột này).
   `crops_bin/` **KHÔNG** đi vào bộ giao nộp (`export_final_dataset` chỉ copy đường `image`).

**Ai đọc crop (đã grep)**: chỉ `pipeline/tools/enrich_crop_quality.py`. S3 (`visual_signal.py`) đọc **trang**
(`_page(page_png).crop(...)`) chứ không đọc tệp crop; nguyên mẫu S3 lấy từ `align_engine/data/index.csv` (bộ STT).
`visual_emission` đọc trang. ArcFace / `train_crop` đọc `dataset_out/` (STT) — không đụng. `loss_ledger` đọc trang.
→ chỉ cần `crops_bin/` cho một nơi duy nhất, và đã nối.

### 2.3 Đo TRƯỚC/SAU (build 2 trang/sách, `crops_bin/` = "trước", crop giao nộp = "sau")

| sách | `--contrast` | n ô | số mức xám trước→sau | nền p90 trước→sau | \|Δ\| xám | r | % ô có MÀU thật |
|---|---|---|---|---|---|---|---|
| LucVanTien1883 | stretch | 126 | 57 → **85** | 255 → **129** | 110,1 | 0,998 | 0 % |
| KimVanKieu1884 | otsu | 192 | 73 → **160** | 255 → **131** | 113,4 | 0,974 | 0 % |
| Chrestomathie1872 | stretch | 194 | 129 → **157** | 255 → **130** | 96,2 | 0,999 | 0 % |
| LucVanTien1916 | none | 135 | 115 → 114 | 198 → 189 | 4,0 | 0,874 | **100 %** |
| TruyenKieu1872 | none | 280 | 153 → 153 | 255 → 255 | **0,0** | 1,000 | 0 % |

Đọc bảng: 3 sách `stretch`/`otsu` được lợi nhiều nhất (nền giấy quay lại, số mức xám tăng 1,2–2,2 lần).
**LucVanTien1916 là sách duy nhất có bản quét MÀU** → nay crop giữ màu thật. **TruyenKieu1872 KHÔNG đổi gì**
(`contrast: none` + ảnh gốc vốn đã xám, r = 1,000, |Δ| = 0,0) — bật khoá cho nó là vô hại, ghi ở đây để không ai
tưởng có cải thiện.

### 2.4 Bằng chứng "cùng vùng chữ, khác nền"

- `labels.csv` hai bản dựng cùng 2 trang LVT1883 (`processed` ↔ `original`): **chỉ cột `image_md5` khác**
  (126/126 ô). `ink_pct`, `crop_w`, `crop_h`, `seg_flag`, `tier`, `bbox`, `rule`… **giống hệt**.
- Sau `enrich_crop_quality`: vẫn **chỉ `image_md5`** khác (`ok` 102 / `bleed` 24 ở cả hai bản).
- 126/126 tệp `crops_bin/*.png` **TRÙNG BYTE** với crop của bản `processed`.
- 10 ảnh so sánh cạnh nhau + `samples.csv` + `summary.json` (2 bất biến, **0 FAIL**):
  **`measure_out/crop_source/`** — vd
  `lvt_orig__lucvantien1883_page_0001_c08_097.png`, `o_KimVanKieu1884__kimvankieu1884_page_0001_c01_005.png`.
  Trái = cũ (nền 255 bệt), phải = mới (nền giấy). Script: `scripts/measure/crop_source_samples.py`.

### 2.5 Còn phải làm (KHÔNG chạy trong vòng này)

Bộ giao nộp trong `dataset/<Book>/` **vẫn là crop cũ**. Sinh lại (≈ 3 h CPU, 0 gọi API) — để người dùng xem
10 ảnh mẫu trước đã:

```bash
./run_pipeline.sh --book all-new --suffix _croporig --skip-ingest
./run_pipeline.sh --book all-ihr --suffix _croporig --skip-ingest
# thăng cấp: bản cũ -> dataset/<Book>_v6_crop_processed/, bản mới -> dataset/<Book>/
```

---

## 3. Việc KHÔNG làm và lý do

| Việc trong đề bài | Vì sao không |
|---|---|
| Đặt `train_crop/detector_r34_v2_litho.pt`, khai `detector_ckpt` cho 5 sách | **Không có ckpt** (§1.2). `apply_v2.sh` làm việc này bằng 1 lệnh khi có. |
| `box_ref_eval --ckpt` v1↔v2, build `--book all-new/all-ihr` `_v2`, so GOLD/M==N/I5/precision IHR | cùng lý do |
| Rebuild bộ giao nộp với crop gốc | 3 h CPU và thay toàn bộ ảnh đã giao — chờ người dùng xem `measure_out/crop_source/` (§2.5) |

---

## 4. NHIỆM VỤ 3 — PTCL loại khỏi phạm vi ✅ XONG

**`data/TruyenKieuPhongTinhCoLuc/` đã bị xoá khỏi đĩa** (5 tệp còn trong git HEAD, trạng thái ` D`).
**Lý do loại (đã đo, `docs/CHAY_3_BO_CON_LAI_2026-09-23.md` §3):** mỗi cột có **2 tầng** mà **tầng trên là lời bình
chữ Hán**, tầng dưới mới là 1 cặp lục bát → adapter `lithograph` hiện có không dùng được; **không có số câu neo**;
QN chỉ có **dị bản** 1871/1872 chứ không phải bản phiên âm của chính bản chép tay R.987. Các phân tích cũ
(`scripts/measure/ptcl_layout.py`, `measure_out/TruyenKieuPhongTinhCoLuc/`, các mục trong báo cáo 22–23/09)
**giữ nguyên làm hồ sơ lịch sử**.

Tham chiếu **chạy được** đã gỡ:

| Nơi | Trước | Sau |
|---|---|---|
| `scripts/measure/measure.py` | `ALL_BOOKS` có PTCL, `ALL_STEPS` có `ptcl_layout` | bỏ cả hai; nhánh dispatch còn nhưng đòi `data/<PTCL>/` tồn tại |
| `config/pipeline_KimVanKieu1884_b1.yaml` | `run.verses_ref_fix.ref: data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871…json` | **comment lại** + lệnh `git checkout` để bật lại |
| `run_pipeline.sh` | gọi `verses_ref_fix --ref <tệp không có>` → chết | thiếu tệp → **log cảnh báo + bỏ B1'** |
| `scripts/measure/verses_ref_fix.py` | `--ref` mặc định trỏ thư mục PTCL | mặc định rỗng; thiếu/không có tệp → `ap.error` chỉ rõ cách khôi phục |
| `scripts/measure/auto_precision.py` | `CROSS_BOOKS[KVK].refs` có `Kieu1871_LVD` (PTCL) | bỏ; **ref thiếu tệp nay bị bỏ qua có cảnh báo** thay vì crash; hết ref → bỏ cross sách đó |
| `scripts/measure/qn_engine_compare.py` | `REFS[KimVanKieu1884]` trỏ thư mục PTCL | bỏ; `REFS` lọc theo tệp tồn tại |

> ✅ **ĐÃ SỬA Ở VÒNG 8 (23/09).** Người dùng khôi phục `thamchieu_kieu_1871_LieuVanDuong_phienam.json`; tệp này
> **GIỮ LẠI** vì là dị bản của KimVanKieu1884 chứ không phải dữ liệu của bản chép tay PTCL. Khối `verses_ref_fix`
> (config `_b1`), `CROSS_BOOKS[KimVanKieu1884].refs` và `REFS[KimVanKieu1884]` đã bật lại; số trôi KVK trùng khít
> mốc công bố (n 31.866 · 80,58 % · lệch vị trí 16). Xem `docs/VONG8_CAN_CHINH_VA_CROP_2026-09-23.md` §6.
> Bốn tệp CÒN LẠI của thư mục PTCL vẫn ở trạng thái xoá.

🔴 **Hệ quả phải biết:** bản phiên âm **Kiều 1871 (Liễu Văn Đường)** nằm *trong* thư mục PTCL nhưng là tham chiếu
của **KimVanKieu1884**, không phải của PTCL. Bỏ nó thì KVK còn **một** tham chiếu (`Kieu1872_DMT`,
`data/TruyenKieu1872/`), nên **"nền dị bản"** (hai bản 1871↔1872 khác nhau bao nhiêu ở cùng vị trí — số 82,9 % ở
`CHOT_KENH_OCR…` §1) **không còn tính được**, và bước **B1'** của bản `_b1` không chạy. Khôi phục:

```bash
git checkout -- data/TruyenKieuPhongTinhCoLuc/thamchieu_kieu_1871_LieuVanDuong_phienam.json
# rồi bỏ # của 4 dòng verses_ref_fix trong config/pipeline_KimVanKieu1884_b1.yaml
# và thêm lại ("Kieu1871_LVD", …) LÊN TRƯỚC trong CROSS_BOOKS["KimVanKieu1884"]["refs"]
```

---

## 5. Hồi quy + selftest

| Phép | Kết quả |
|---|---|
| STT 3 trang (`stt2/0024`, `stt4/0050`, `stt11/0100`), `config/pipeline.yaml --qd01-cells none` | `labels.csv` md5 **`59e436d7641fa849bb6759868ac29259`**, 556 dòng; `diff -rq` **0/346 tệp khác** (kể cả `summary.json`) |
| `book_layout_selftest` | **133 / 0** (thêm mục `[10] crop_source`, 33 phép) |
| `phase1_engine_selftest` | **270 / 0** (thêm `test_crop_source_original`, **17 phép** ≥ 10 yêu cầu) |
| `ingest_lithograph` / `ingest_prose` / `ingest_ihr` / `pitch_decode` / `verses_ref_fix` | 82/82 · 33 · 64/64 · 22/0 · 19/19 |
| `code_facts.py --check` | **18/18** (cập nhật known pin `book_layout.py` 83 → 98 vì docstring thêm khoá `crop_source`) |
| `crop_source_samples.py` | 10 ảnh, 2 bất biến, **0 FAIL** |

`summary.json` chỉ thêm khoá `crop_source_by_book` **khi có sách khai khác mặc định** → summary của STT không đổi.
