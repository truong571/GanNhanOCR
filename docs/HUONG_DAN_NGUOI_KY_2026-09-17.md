# HƯỚNG DẪN NGƯỜI KÝ — hai việc còn lại của Khối A (≈30 phút)

**Ai làm:** người chấm (bạn). **Máy không được làm thay** — đây là hai điểm duy nhất trong pipeline có phán quyết
người, và mọi phán quyết phải có xuất xứ (tệp có thật) để `decisions.py`/`glyph_fix --mode kiem` chấp nhận.

Ba trang xem crop đã dựng sẵn (mở bằng trình duyệt, không cần mạng):

| Trang | Cho việc |
|---|---|
| `dataset_out_v3/ky/qd01a_review.html` | Việc 1 — 14 ô QĐ-01a (crop + ngữ cảnh trang, hộp đỏ) |
| `dataset_out_v3/ky/di_the_review.html` | Việc 2a — 14 cặp dị thể, 10 crop/mã/sách |
| `dataset_out_v3/ky/corpus_readings_review.html` | Việc 2b — 8 cặp cách đọc riêng của ngữ liệu, 12 crop/cặp |

---

## VIỆC 1 — Điền `quyet` cho 14 ô trong `config/qd01a_decisions.csv`

### 1.1 Chuẩn bị tệp (2 phút)

Tệp hiện có **12 dòng** (ô GOLD 'người' ngoài QĐ-01, cột `quyet` trống). **Thêm 2 dòng** cho 2 ô QĐ-01 trôi, lấy từ
`dataset_out_v3/qd01a_pending.csv` (2 dòng có `ly_do` = `am_khac:…`):

```
stt2,page_0052,4,4,"[960, 853, 1078, 963]",người,㝵,𠊚,"QĐ-01 trôi: v3 ghép âm 'mà'",,,,
stt4,page_0230,5,19,"[576, 1505, 642, 1584]",người,晁,𠊚,"QĐ-01 trôi: v3 ghép âm 'con'",,,,
```

(cột: `book,page,column,nom_idx,bbox_cu,syllable,ocr_char,label_hien_tai,ly_do,quyet,nguoi_ky,ngay,xuat_xu`).

### 1.2 Cách quyết (mở `qd01a_review.html`, đi từng dòng)

Câu hỏi cho MỖI ô: *"Crop này có phải chữ Nôm của âm **người** không?"* — nhìn crop, nhìn ngữ cảnh cột (hộp đỏ), đọc
cột "âm ghép mới / cũ".

| Thấy gì | Điền `quyet` | Hệ quả |
|---|---|---|
| Đúng là chữ "người" (dạng 𠊚/㝵 viết tay, như 2.012 ô đã khoá) | `giu_2029A` | nhãn 𠊚, tier GOLD, rule `quyet_dinh_nguoi:qd01a` |
| Không phải, hoặc không chắc, hoặc máy v3 đã ghép ô này với âm **khác** (chịu, một, lời, lên, đã, ẩn, mà, con — 8/14 ô) | `bo` | không ép nhãn; ô đi theo tier_v3 (thường REVIEW/SYL); `qd01_excluded=1` |
| Là một chữ khác đọc rõ được (ví dụ 昆 "côn") | `khac:昆` | nhãn chữ ấy, GOLD, rule `quyet_dinh_nguoi:qd01a` |

Gợi ý từ số liệu (bạn vẫn là người quyết):
- 6 ô mà v3 **vẫn ghép "người"** (stt11/page_0106, 0114, 0130; stt4/page_0034, 0090, 0184 — OCR 具/昆, nhãn cũ 昆): nhìn
  xem hình có phải "người" viết tay không; nếu có → `giu_2029A`; nếu là 昆 thật → `khac:昆`.
- 8 ô mà v3 **ghép âm khác**: bản cũ ghép sai thanh ghi nên nhãn "người" cũ vô nghĩa → gần như chắc `bo`.

### 1.3 Ba cột còn lại (bắt buộc, nếu thiếu máy từ chối)

- `nguoi_ky`: tên bạn (như trong `docs/NGUOI_CHAM_QUYET_DINH.md`).
- `ngay`: `2026-09-17` (ISO).
- `xuat_xu`: đường dẫn tệp **có thật** ghi lại phiên quyết. Cách gọn nhất: thêm **Phụ lục a** vào cuối
  `docs/NGUOI_CHAM_QUYET_DINH.md` (mẫu ở §3 dưới) rồi điền `docs/NGUOI_CHAM_QUYET_DINH.md` cho cả 14 dòng.

### 1.4 Kiểm (30 giây)

```
.venv/bin/python - <<'EOF'
import csv; r=list(csv.DictReader(open('config/qd01a_decisions.csv')))
bad=[x for x in r if x['quyet'].strip()=='' or not x['nguoi_ky'] or not x['xuat_xu']]
print('dòng', len(r), '· thiếu', len(bad), [ (x['book'],x['page'],x['column'],x['nom_idx']) for x in bad])
EOF
```
Kỳ vọng: `dòng 14 · thiếu 0`. Sau đó rebuild (§4) — `assert_qd01` sẽ đếm 2.014 = 2.012 lock + 2 (giu/khac) hoặc
2.012 + 0 nếu bạn `bo` cả hai ô trôi (khi đó tệp `qd01_cells.csv` giữ nguyên, hai ô ấy chuyển sang `qd01_excluded`).

---

## VIỆC 2 — Ký `config/decisions.yaml`

Mở tệp; mỗi mục có `trang_thai: cho_ky`, `xuat_xu: ''`. Máy **chỉ áp mục `da_ky` có `xuat_xu` trỏ tới tệp có thật**.
Không ký mục nào cũng chạy được — chỉ thiếu cột `label_canonical` (= `label`) và ~250 ô corpus_readings ở tầng SYL thay vì GOLD.

### 2a · 14 cặp dị thể (`di_the`) — mở `di_the_review.html`

Câu hỏi cho MỖI cặp: *"Hai mã này trên giấy có phải MỘT hình chữ không?"* (nhìn 10 crop mỗi mã, theo từng sách).

| Thấy gì | Làm gì |
|---|---|
| MỘT hình (chỉ khác cách OCR/khác chuẩn gõ: 徳/德, 亜/亞, 実/实, 毎/每, 沒/没, 廩/廪, 別/别, 亊/事, 垩/堊, 宐/宜, 庄/庒, 烝/蒸, 為/爲) | điền `chuan:` = mã bạn chọn làm chuẩn (một trong hai `quan_sat`); `quy_tac: unicode` nếu chọn theo dạng chính thống Unicode/Khang Hy (thường là dạng phồn thể: 德, 亞, 實?, 每, 沒, 廩, 別, 事, 堊, 宜, 爲), hoặc `quy_tac: da_so_ngu_lieu` nếu chọn mã xuất hiện nhiều hơn (số `so_o` đã ghi sẵn); `xuat_xu:` tệp phụ lục; `trang_thai: da_ky` |
| HAI hình rõ rệt ở một sách | để `cho_ky` (không ký), hoặc ghi `book: stt4` để chỉ gộp ở sách một hình (mục `dt_70ba_7232_stt4` đã thu hẹp sẵn vì E3 đo (stt2, vì) là hai hình) |
| Không chắc | để `cho_ky` — không mất gì |

**Riêng `dt_5171_5176` (共/其, "cùng")**: không phải dị thể Unicode; 其 nằm trong từ điển của âm "cùng" nhưng ảnh (E3) nói
303 ô 其 cùng hình với 共. Ký `chuan: 共` = tuyên bố "其 là OCR đọc nhầm 共" cho 303 ô — chỉ ký nếu 10 crop 其 trông
giống 10 crop 共; nếu do dự, để `cho_ky` (báo cáo 15/09 xếp việc này vào Khối B-4).

Nhớ: ký dị thể **không đổi `label`** (mã quan sát vẫn giữ), chỉ ghi thêm `label_canonical`. Sai thì sửa yaml và rebuild.

### 2b · 8 cặp `corpus_readings` — mở `corpus_readings_review.html`

Câu hỏi: *"Trong sách này, chữ X có được đọc là âm GỐC của bản in không?"* (ví dụ 無 đọc "vồ" trong "Vít-vồ";
僥 đọc "nhiều"). Mỗi mục ghi sẵn số ô mà 2-gram ủng hộ âm gốc / đổi / hoà.

| Thấy gì | Làm gì |
|---|---|
| Crop + ngữ cảnh khớp âm gốc, số "ủng hộ gốc" áp đảo (無/vồ 69-0, 僥/nhiều 62-0, 係/hế 9-0…) | `trang_thai: da_ky` + `xuat_xu` — ô thành GOLD rule `corpus_reading:<id>` (không vào từ điển, không làm neo) |
| Số ủng hộ gốc / đổi gần nhau (凜 lăm 28-31, 礼 lê 11-27) | để `cho_ky` — máy tiếp tục giữ âm bản in ở tầng SYL, không ép GOLD |

Không được đổi `ocr_char`/`syllable_raw` của mục — muốn thêm cặp mới thì thêm mục mới cùng schema.

### 2c · Kiểm (30 giây)

```
.venv/bin/python -m pipeline.decisions --config config/decisions.yaml
```
(hoặc `.venv/bin/python pipeline/decisions_selftest.py`). Kỳ vọng: in số mục `da_ky`/`cho_ky`, 0 lỗi schema, mọi
`xuat_xu` của mục `da_ky` là tệp tồn tại.

---

## 3. MẪU PHỤ LỤC a (dán vào cuối `docs/NGUOI_CHAM_QUYET_DINH.md`)

```
## Phụ lục a — Phán quyết QĐ-01a và ký bảng quy ước (2026-09-17)

Người quyết: <tên>. Phiên: 2026-09-17, <giờ bắt đầu–kết thúc>, xem trực tiếp
dataset_out_v3/ky/{qd01a_review,di_the_review,corpus_readings_review}.html.

a.1 QĐ-01a — 14 ô (config/qd01a_decisions.csv): <n> giu_2029A · <n> bo · <n> khac. Nguyên tắc: chỉ giu_2029A khi crop
    là chữ "người"; ô mà căn chỉnh v3 ghép sang âm khác thì bo.
a.2 Dị thể — ký <n>/14 cặp, quy tắc chọn mã chuẩn: <unicode | da_so_ngu_lieu>. Không ký: <liệt kê id> vì <lý do>.
a.3 corpus_readings — duyệt <n>/8 cặp: <liệt kê id>.
```

Đường dẫn `docs/NGUOI_CHAM_QUYET_DINH.md` chính là giá trị `xuat_xu` cho cả 3 việc.

---

## 4. SAU KHI KÝ — máy làm phần còn lại (bạn chỉ chạy lệnh)

```
# 1) build lại bản thử nghiệm, kiểm không còn pending, số áp decisions
NONINTERACTIVE=1 DS_OUT=dataset_out_v3 ./run_pipeline.sh
.venv/bin/python -m pipeline.tools.doi_soat_the_he --old dataset_out/labels_final.csv \
    --new dataset_out_v3/labels_final.csv --qd01 config/qd01_cells.csv --out docs/DOI_SOAT_v3_2026-09-16.md
# 2) commit mã + config (34 tệp sửa, 19 mới) — commit này định danh mã đã chạy
# 3) thăng cấp (ghi đè có chủ ý bộ 64.525 đã đóng băng) → evidence + bảng số liệu tự chạy
NONINTERACTIVE=1 FROZEN_OVERRIDE=GHIDE DS_OUT=dataset_out ./run_pipeline.sh
bash scripts/check_consistency.sh        # kỳ vọng 3/3 sau commit thứ hai
```
Chi tiết: `dataset_out_v3/HUONG_DAN_THANG_CAP.md`. Nếu bạn muốn, tôi chạy giúp bước 1 và 3 sau khi bạn báo đã ký.
