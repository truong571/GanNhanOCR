# Bố cục đầu ra + bộ gộp chung + quy tắc dọn rác (23/09/2026)

Trên HEAD `11001b7225`. **Vòng này chưa commit gì**; `data/` chỉ ĐỌC. Kế hoạch commit:
`docs/KE_HOACH_COMMIT_BOCUC_2026-09-23.md`.

> **Năm dòng kết luận.**
> 1. `dataset/` nay **chỉ chứa thư mục**: một thư mục mỗi **bộ** (6 bộ) + `dataset/_ALL/` là **bộ gộp chung**.
>    Bộ STT dời từ `dataset/{labels.csv,gold/,syllable/}` xuống `dataset/SachThanhTruyen/` bằng `mv`
>    (36/36 tệp sha256 y hệt). Chạy lại **cả 8 bộ** rồi so từng byte: `labels.csv`/`labels_trace.csv`/
>    `columns.csv` của **cả 6 bộ TRÙNG HẾT** (§1).
> 2. **Bộ gộp** `dataset/_ALL/`: **140.733 dòng** (= đúng tổng 6 bộ) · **140.252 dòng có ảnh** · **140.231 tệp
>    crop** · **31.509 dòng `evaluation_only = 1`** (2 bộ IHR). Khoá chính là **`cell_uid`** (duy nhất 100 %),
>    không phải `image` — lý do ở §3.
> 3. **Chặn rò rỉ**: `pipeline/publish/` nay gọi `mark_eval_dataset.is_marked()`; dòng thuộc bộ đã đóng dấu
>    nhận split riêng `eval_only`, **không bao giờ** vào train/val/test, và `_verify` FAIL nếu có một dòng lọt.
> 4. **Dọn rác**: `--prune [--keep-old N]` chuyển bản dựng cũ vào `archive/`, `--clean` xoá hẳn
>    `archive/ + measure_out/ + logs/`. Đã dọn **3.722 MB** (25 thư mục): `dataset/` 3,8 G → **1,1 G**
>    (rồi lên **2,2 G** vì bộ gộp copy thêm 1,1 G), `prepared/` 3,1 G → **2,2 G**.
> 5. **Nghiệm thu đã chạy thật** (§6): bảng 8 bộ trùng `CHOT_CUOI` từng ô · `--verify` 6/6 PASS ·
>    md5 STT `59e436d7…` · `code_facts` 18/18 · `git status -- data dataset_out prepared` trống.

---

## 1. Bố cục cuối

```
dataset/                       CHỈ chứa thư mục — không còn tệp nào ở gốc
├── SachThanhTruyen/           bộ STT: 3 quyển stt2 + stt4 + stt11 (một lần chạy = một bộ)
├── LucVanTien1883/            thạch bản
├── KimVanKieu1884/            thạch bản
├── Chrestomathie1872/         văn xuôi
├── LucVanTien1916/            ⚠️ TẬP ĐÁNH GIÁ  (evaluation_only.json + TAP_DANH_GIA.md)
├── TruyenKieu1872/            ⚠️ TẬP ĐÁNH GIÁ
└── _ALL/                      BỘ GỘP CHUNG của 6 bộ trên
archive/                       bản dựng CŨ (`--prune` chuyển vào; `--clean` xoá) — gitignore
```

Mỗi thư mục bộ có đúng bộ tệp mà `export_final_dataset.py` + `make_dataset_docs.py` sinh ra:
`labels.csv` (12 cột giao nộp) · `labels_trace.csv` · `columns.csv` · `labels.xlsx` ·
`gold/` + `syllable/` (ảnh crop) · `README.md` · `DATASHEET.md` · `LICENSE.md` ·
`NGUON_THU_TICH.md` (+ `TAP_DANH_GIA.md` · `evaluation_only.json` cho 2 bộ IHR).

### Vì sao 3 quyển STT nằm CHUNG một thư mục, không tách thành 3

Bộ STT là **một lần chạy pipeline**: một `labels.csv`, một `README`/`DATASHEET` mô tả cả ba quyển,
một chuỗi băm. Tách thành `dataset/SachThanhTruyen2|4|11/` sẽ sinh ra **hiện vật mà pipeline không
dựng lại được** (phải cắt tay `labels.csv` và viết tay 3 bộ tài liệu) — đúng loại rác mà vòng này
đi dọn. Cột `book` (`stt2`/`stt4`/`stt11`) tách ba quyển ở mức dòng, và bảng tóm tắt của
`run_pipeline.sh` vẫn in **8 bộ** như bảng chốt. Quy tắc chung: **một lần chạy = một thư mục bộ.**

### Đã đổi chỗ những gì (và bằng chứng không phá bộ đã công bố)

| | trước 23/09 | từ 23/09 |
|---|---|---|
| bộ STT | `dataset/labels.csv`, `dataset/gold/`, `dataset/syllable/`… | `dataset/SachThanhTruyen/…` |
| bộ gộp | *(chưa có)* | `dataset/_ALL/` |
| bản dựng cũ | rải trong `dataset/` (`*_v5_lang1`, `*_v7`, `*_v8`, `*_v9`, `*probe`) | `archive/` rồi xoá |

Dời bằng `mv` (không copy, không ghi lại), rồi `shasum -a 256 -c`: **36/36 tệp OK** (8 tệp STT +
25 tệp của 5 sách + 3 tệp `dataset_out/`), kiểm lại **sau** khi dọn `archive/`: vẫn nguyên.

**Bằng chứng mạnh hơn: chạy lại TOÀN BỘ 8 bộ rồi so từng byte.** `./run_pipeline.sh --book all --yes`
(1.517 s, 0 gọi API) dựng lại cả 6 thư mục bộ; đối chiếu sha256 với bản trước khi dời:

| | kết quả |
|---|---|
| `labels.csv` · `labels_trace.csv` · `columns.csv` của **cả 6 bộ** | **trùng từng byte** (18/18 tệp) |
| `README.md` · `DATASHEET.md` của **5 sách** | **trùng từng byte** (10/10 tệp) |
| `dataset_out/labels.csv` · `labels_remediated.csv` · `labels_final.csv` (có trong git) | **trùng từng byte** |
| `dataset/SachThanhTruyen/{README,DATASHEET}.md` | khác **đúng một từ** (xem dưới) |
| `dataset/SachThanhTruyen/labels.xlsx` | khác — openpyxl nhúng thời điểm tạo, **không bao giờ** tái lập từng byte |

Chỗ khác đúng một từ: `make_dataset_docs` in chân trang `commit \`<git log -1 -- <labels.csv>>\``.
Ở đường dẫn CŨ, `git log -1 -- dataset/labels.csv` trả `88adfbbc60` (một commit đời xưa từng theo dõi
tệp đó); ở đường dẫn mới không có lịch sử nên in `?` — **giống hệt 5 bộ kia**. Đã chứng minh bằng thay
thế: đổi `commit \`?\`` thành `commit \`88adfbbc60\`` trong bản mới cho **đúng sha256 của bản cũ**
(`9185…13ff` / `464e…a2b2`). Mọi con số trong hai tệp giữ nguyên; và `88adfbbc60` vốn là **xuất xứ sai**
(commit đó không sinh ra `labels.csv` hiện tại), nên đổi thành `?` là chính xác hơn.

Mã đọc đường dẫn cũ đã sửa theo, **giữ đường cũ làm dự phòng** để công cụ không chết trên cây thư
mục đời trước: `pipeline/tools/update_bang_so_lieu.py`, `pipeline/tools/selftest.py` (`_ds_dir`),
`pipeline/remediation/selftest.py`, `pipeline/remediation/confusion_fix.py` (`NGUON_VERDICT`),
`pipeline/tools/variant_table.py`, `pipeline/export_review_excel.py`, `run_pipeline.sh` (`FINAL_DIR`).

---

## 2. Bộ gộp chung `dataset/_ALL/`

Sinh bởi `pipeline/tools/merge_datasets.py` (bước **B7** của `./run_pipeline.sh --book all`, hoặc
`./run_pipeline.sh --merge`). **Chỉ ĐỌC** các thư mục bộ; ảnh **copy hẳn** để bàn giao độc lập.

| tệp | nội dung |
|---|---|
| `labels.csv` | **17 cột** = 12 cột giao nộp + `cell_uid`, `book_set`, `evaluation_only`, `split_hint`, `image_dup` |
| `labels_trace.csv` | sidecar chẩn đoán, **cùng số dòng, cùng thứ tự**, khoá `cell_uid` |
| `columns.csv` | một dòng mỗi cột trang, thêm `book_set` |
| `crops/<Bộ>/gold\|syllable/*.png` | ảnh crop copy hẳn |
| `labels.xlsx` | bản đọc bằng Excel (mọi cột là CHUỖI trừ vài cột đo) |
| `SOURCES.json` | mỗi bộ: số dòng, số ảnh, cờ `evaluation_only`, **sha256 `labels.csv` nguồn** + bảng bất biến |
| `CHECKSUMS.txt` | sha256 mọi tệp của bộ gộp |
| `README.md` · `DATASHEET.md` · `TAP_DANH_GIA.md` | tài liệu, **mọi con số đọc từ dữ liệu thật** |
| `TRUNG_ANH.csv` | các dòng dùng chung một tệp ảnh (§3) |

### Cột thêm

| cột | nghĩa |
|---|---|
| `cell_uid` | **KHOÁ CHÍNH** `<Bộ>/<book>/<page>/c<cột>/n<nom_idx>/s<syl_idx>` — đo được: duy nhất trên cả 140.733 dòng |
| `book_set` | tên thư mục bộ nguồn |
| `evaluation_only` | `1` khi bộ có `evaluation_only.json` (nguồn sự thật: `mark_eval_dataset.is_marked()`) |
| `split_hint` | `train` \| `eval` — **gợi ý** chia, không phải split chính thức (split chính thức: `pipeline/publish`) |
| `image_dup` | `1` khi đường dẫn ảnh này bị hai dòng dùng chung (§3) |

`image` đổi thành `crops/<Bộ>/<đường dẫn cũ>`; 11 cột giao nộp còn lại **giữ nguyên từng ký tự**.

### Số phải khớp

| đại lượng | giá trị |
|---|---|
| tổng dòng | **140.733** = 71.610 + 11.938 + 20.546 + 5.130 + 11.769 + 19.740 |
| dòng có ảnh | **140.252** (= 140.733 − 481 dòng `GOLD_text_only`) |
| tệp crop trên đĩa | **140.231** (= 140.252 − 21 đường dẫn dùng chung, §3) |
| dòng `evaluation_only = 1` | **31.509** (11.769 + 19.740) |
| ô (kể cả REVIEW, từ `labels_gated`/`labels_final`) | **164.738** — bảng `CHOT_CUOI` §4 |

---

## 3. Khuyết tật KẾ THỪA vừa đo được: 21 đường dẫn ảnh dùng chung trong bộ STT

`dataset/SachThanhTruyen/labels.csv` có **71.610 dòng nhưng chỉ 71.589 đường dẫn ảnh khác nhau**.
21 tệp crop bị **hai ô khác nhau** (khác `bbox`, khác nhãn) cùng trỏ tới; trong mọi cặp, một dòng
mang `rule = self_training_rescue` còn dòng kia `rule = s1_inter_s2_direct`, và `image_md5` của hai
dòng **trùng tiền tố** ⇒ cả hai thật sự chỉ về **một tệp**. Nghĩa là **một trong hai dòng có ảnh sai**
(nhãn có thể vẫn đúng — nhãn sinh từ phép ghép DP, không từ ảnh).

Cơ chế: tên crop là `{book}_{page}_c{cột}_{nom_idx:03d}.png`; bước giải cứu `self_training_rescue`
cấp lại chỉ số ô nên đụng vào ô đã có. `pipeline/tools/selftest.py` có sẵn phép kiểm
*"image là khoá chính (không trùng)"* và nó **đang FAIL** — FAIL này **có từ trước vòng này**
(tệp `labels.csv` byte-identical với bản đã công bố, sha256 đã đối chiếu).

**Vòng này KHÔNG sửa** (sửa sẽ đổi bộ đã công bố, phải chạy lại + đổi mọi chuỗi băm). Đã làm:
giữ nguyên cả hai dòng, gắn `image_dup = 1` (42 dòng), liệt kê ra `_ALL/TRUNG_ANH.csv`, khai trong
`_ALL/README.md`, và **cấp khoá chính riêng `cell_uid`** để phép gộp vẫn có khoá duy nhất 100 %.
Ai huấn luyện nên loại `image_dup == 1`. Việc còn mở: đặt lại chỉ số ô trong `self_training_rescue`.

---

## 4. Chặn rò rỉ tập đánh giá trong `pipeline/publish/`

Trước: `splits.assign_page_disjoint` chia **mọi** dòng `usable` theo trang — hai bộ IHR-NomDB
(có **nhãn người từng chữ**) sẽ nằm trong `train`, làm mọi con số precision công bố về sau vô nghĩa.

Nay (`pipeline/publish/splits.py`):

- `eval_only_books(dataset_root)` quét `dataset/<Bộ>/evaluation_only.json` qua
  **`mark_eval_dataset.is_marked()`** — cùng một nguồn sự thật với bước đóng dấu của `run_pipeline.sh`.
- `eval_only_mask(df)` nhận **cả hai lối**: cột `evaluation_only` (bộ gộp `_ALL`) **hoặc** cột
  `book`/`book_set` khớp thư mục đã đóng dấu.
- Dòng thuộc tập đánh giá nhận `EVAL_SPLIT = "eval_only"` — **không** `""` (dòng REVIEW đã mang `""`),
  để phân biệt *"không chia vì REVIEW"* với *"không chia vì CẤM huấn luyện"*.
- Chúng bị loại khỏi cả phép đếm lớp singleton; `lobo_split` không bao giờ đưa chúng vào `train`;
  `_verify` thêm vi phạm `"N eval-only rows leaked into train/val/test"`.
- `export.build_hf_dataset` chỉ xuất `train/val/test` ⇒ dòng `eval_only` **tự rơi khỏi parquet**.

**13 phép kiểm mới** trong `pipeline/publish/selftest.py` (`test_eval_only_gate`) — tổng
**69 PASS / 0 FAIL** (trước: 56).

---

## 5. Quy tắc dọn rác

### `--prune [--keep-old N]` — CHUYỂN bản cũ vào `archive/`

Đúng **3 mẫu**, không đoán thêm:

| mẫu | ví dụ |
|---|---|
| `dataset/<Bộ>_v<số>*` | `LucVanTien1883_v7`, `Chrestomathie1872_v5_lang1` |
| `dataset/<Bộ>*probe` | `TruyenKieu1872_v1probe`, `LucVanTien1916_v2probe` |
| `prepared/<Sách>/dataset_out_*` | `dataset_out_v5_lang1` (KHÔNG khớp `dataset_out` chốt) |

`--keep-old N` giữ **N bản mới nhất theo mtime**. In danh sách + dung lượng rồi **hỏi** (gõ `CHUYEN`);
`--yes` bỏ hỏi. **Chuyển, không xoá** — muốn xoá thì `--clean`.

`--keep-old N` (N > 0) còn đổi hành vi của **chính lần chạy**: trước khi bước export ghi đè
`dataset/<Bộ>/`, bản dựng TRƯỚC được dời sang `archive/dataset/<Bộ>.<thời điểm>/` thay vì bị xoá,
và chỉ **N bản cũ nhất định** của bộ đó được giữ. Mặc định `N = 0` ⇒ **hành vi y hệt trước đây**
(export xoá sạch rồi ghi lại). Đây là chỗ thay cho thói quen `mv dataset/$b dataset/${b}_v9` đã
làm rải 20 thư mục bản cũ trong `dataset/`.

**KHÔNG BAO GIỜ đụng**: `dataset/<Bộ>` chốt · `dataset/_ALL` · `dataset_out/` (có trong git) ·
`prepared/<Sách>/{pages,detected,kim_raw,transcriptions,…}` — `detected/*_ocr_cache.json` là
**PRIMARY DATA**, xoá là mất tiền gọi API và mất tính tái lập.

### `--clean` — XOÁ HẲN `archive/` + `measure_out/` + `logs/`

Cả ba đều dựng lại được và đều trong `.gitignore`. In danh sách + dung lượng rồi hỏi (gõ `XOA`);
`--yes` bỏ hỏi. `measure_out/` dựng lại: `scripts/measure/measure.py --all` (~10 phút CPU, 0 token).

### Nguồn sinh rác đã rà

| chỗ | trước | nay |
|---|---|---|
| `--suffix _v1probe` | đẻ `dataset/<Sách>_v1probe` + `prepared/<Sách>/dataset_out_v1probe`, không ai dọn | `--prune` nhận diện cả hai |
| dựng lại thủ công (`mv dataset/$b dataset/${b}_v9`) | 20 thư mục tồn đọng trong `dataset/` | `--prune` dọn vào `archive/` |
| `logs/run_*.log`, `measure_out/` | cộng dồn mãi (45 M + 293 M) | `--clean` |
| `tick()` của bước gộp | sẽ ghi thêm dòng vào `dataset_out/CHECKSUMS.txt` (**có trong git**) | ghi vào `dataset/_ALL/THOI_GIAN.txt` |
| `auto_precision{,_gated,_verify}/` | tên cố định, ghi đè mỗi lần | giữ nguyên (không cộng dồn) |
| `prepared/<Sách>/transcriptions/*_qn_tmp.png` | 448 tệp, **82 MB**, không ai xoá | **CỐ Ý giữ**: đó là *cache ảnh render trang* (`step1_extract` chỉ vẽ lại khi thiếu), xoá chỉ làm extract chậm hơn chứ không gọi API — **không** đưa vào `--clean` để khỏi đụng nhầm `prepared/` |
| tệp tạm khác (`auto_precision/_tmp_*.png`, `ocr_api` NamedTemporaryFile, `layout_lithograph` tesseract, `enrich_crop_quality` `.tmp`) | — | đã rà: **tất cả đều `unlink`/`replace` trong `finally`**; đo trên đĩa: 0 tệp sót |

**Đã dọn vòng này**: 25 thư mục / **3.722 MB** → `dataset/` **3,8 G → 1,1 G**, `prepared/` **3,1 G → 2,2 G**.

---

## 6. Nghiệm thu — đã CHẠY THẬT (23/09, `./run_pipeline.sh --book all --yes`, 1.517 s, 0 gọi API)

| Phép | Kết quả |
|---|---|
| Bảng 8 bộ | **TRÙNG `CHOT_CUOI` §4 từng ô**: tổng **164.738 ô** · **140.252 ảnh export** |
| Bộ gộp | **140.733 dòng** · 140.252 dòng có ảnh · **140.231 tệp crop** · **31.509** dòng `evaluation_only` · `image_dup` **42 dòng / 21 nhóm** · ảnh trùng byte giữa hai bộ khác nhau: **0** |
| `--verify` | **6/6 PASS, 0 FAIL cứng** (align_audit · ihr_endtoend · auto_precision cross ×2 · measure --all --report-only · **bộ gộp --check 16/16**) |
| precision trên NHÃN NGƯỜI | LVT1916 **98,04 %** (n 11.496) · TK1872 **98,61 %** (n 18.550) — trùng §4 `CHOT_CUOI` |
| hồi quy STT 3 trang | `labels.csv` md5 **`59e436d7641fa849bb6759868ac29259`**, 556 dòng — **trùng mốc** |
| `code_facts.py --check` | **18/18** |
| selftest | `merge_datasets` **23/23** · `publish` **69/0** · `mark_eval_dataset` **15/15** · `tools` 139/3 (3 FAIL có từ trước, §7.6) · `remediation` 175/4 (4 FAIL có từ trước) |
| `git status -- data dataset_out prepared` | **TRỐNG** |
| cổng rò rỉ chạy trên bộ gộp thật | `publish split --labels dataset/_ALL/labels.csv` → `train 95.533 / val 6.965 / test 6.571`, **31.509 dòng `eval_only` giữ ngoài**, `page-span 0 md5-span 0`, LOBO 8 quyển đều không kéo bộ đánh giá vào train |

⚠️ **Hồi quy STT dùng ĐÚNG lệnh trong tài liệu** (`--config config/pipeline.yaml --qd01-cells none --force
--pages pages3.csv`, tức `--reseg` MẶC ĐỊNH `midpoint`). Thêm `--reseg detector` cho md5 khác
(`6556ce3a…`) — không phải hồi quy, chỉ là khác tham số.

## 7. Lệnh

```bash
./run_pipeline.sh --book all --yes        # 8 bộ + B7 gộp + nghiệm thu (0 gọi API)
./run_pipeline.sh --merge                 # chỉ dựng lại dataset/_ALL/ từ các bộ trên đĩa
./run_pipeline.sh --merge --merge-mode link   # hardlink thay vì copy (tiết kiệm đĩa, cùng ổ)
./run_pipeline.sh --summary-only          # bảng 8 bộ + dòng bộ gộp
./run_pipeline.sh --verify                # + phép kiểm bộ gộp (khoá duy nhất · ảnh đủ · cờ eval)
./run_pipeline.sh --prune --keep-old 1    # dọn bản cũ, giữ 1 bản mới nhất -> archive/
./run_pipeline.sh --clean                 # xoá archive/ + measure_out/ + logs/
.venv/bin/python -m pipeline.tools.merge_datasets --selftest    # 23/23
.venv/bin/python -m pipeline.publish.selftest                   # 69/0
```

---

## 8. Giới hạn TRUNG THỰC

1. **Bộ gộp không phải nguồn sự thật.** Nguồn là `dataset/<Bộ>/`. `_ALL/` dựng lại bằng một lệnh;
   sửa tay vào đó sẽ bị lần gộp sau xoá sạch (bước gộp **xoá hết** thư mục đích trước khi ghi).
2. **`split_hint` không phải split chính thức.** Nó chỉ nói "được/không được huấn luyện". Split
   page-disjoint + LOBO chính thức vẫn do `pipeline/publish` dựng.
3. **Cờ `evaluation_only` chỉ mạnh bằng việc đóng dấu.** Bộ nào chưa chạy `mark_eval_dataset` thì
   cổng không biết. Hiện chỉ 2 bộ IHR được đóng dấu, và đúng chúng là 2 bộ có nhãn người.
4. **21 đường dẫn ảnh dùng chung chưa được sửa** (§3) — khuyết tật của bộ đã công bố, không phải
   của phép gộp.
5. **`--clean` xoá `measure_out/`** nên `--verify` chạy ngay sau đó sẽ phải dựng lại bộ đo (~10 phút).
6. `pipeline/tools/selftest.py` còn **3 FAIL có từ trước**: 2 là khuyết tật dữ liệu ở §3
   (`image` trùng, `image_md5` 32 hex do `self_training_rescue`), 1 là phép kiểm mẫu mã
   `|| n_old=0` của một khối preflight đã bị gỡ khỏi `run_pipeline.sh` từ trước vòng này.
   (`git show HEAD:run_pipeline.sh | grep -c '|| n_old=0'` = 0 ⇒ FAIL này có ở HEAD.)
7. **`dataset/SachThanhTruyen{2,4,11}/` xuất hiện lúc 19:29 KHÔNG do pipeline sinh.** Ba thư mục
   là bản **tách bộ STT theo từng quyển** (24.023 / 23.829 / 23.758 dòng, khớp từng dòng với phần
   `book == stt*` của `dataset/SachThanhTruyen/labels.csv`); 4 tệp tài liệu trong đó là **bản sao
   giữ mtime** của tài liệu bộ STT (19:22). **Không tìm thấy đường mã nào trong kho sinh ra chúng**
   và **cả 13 selftest đều không tái hiện** — cùng kiểu với việc `web/app.js` + `web/style.css` bị
   sửa ngoài phiên này. **Đã KHÔNG xoá** (không phải rác do vòng này tạo, cũng không khớp mẫu bản
   dựng cũ). Rủi ro duy nhất: gộp nhầm là **đếm hai lần** cả bộ STT — đã chặn bằng cảnh báo trong
   `merge_datasets` (thư mục trông-như-bộ mà không khai trong `--books` thì in cảnh báo và bỏ qua)
   và bằng việc `run_pipeline.sh` luôn khai **danh sách bộ tường minh**.
8. **`NGUON_VERDICT` của `confusion_fix.py` CỐ Ý không thêm đường dẫn mới.** Thêm vào là 0 chức năng
   (chưa thư mục nào có `verdicts.csv`) mà lại đổi nội dung `dataset_out/confusion_fix_report.json`
   — tệp **có trong git**. Khi nào đội chấm giao verdict thật thì thêm.
9. **`dataset_out/{summary.json, CHECKSUMS.txt}` đã `git checkout` về HEAD sau lần chạy.** Lý do:
   `CHECKSUMS.txt` là nhật ký cộng dồn, còn `summary.json` bản trong git có TRƯỚC khi mã thêm khoá
   `detector_params_by_book` nên mọi lần chạy lại ở HEAD đều làm nó bẩn. `labels*.csv` của
   `dataset_out/` **không đổi một byte**, nên đây là hoàn nguyên hiện vật sinh lại được, không phải
   che giấu khác biệt. Cổng `code_facts dataset_out_tracked_clean` nhờ đó **18/18**.
