# Bộ dữ liệu gán nhãn chữ Nôm — Sách Thánh Truyện

**51,922 nhãn cấp KÝ TỰ** + **18,404 chú giải cấp ÂM TIẾT**
= 70,326 dòng · 1,331 lớp chữ · 447 trang · 4,010 cột

> ⚠️ **Đừng phát biểu là "70,326 nhãn".** 18,404 dòng tầng SYLLABLE có cột
> `label` **rỗng** — chúng chỉ ghi âm Quốc ngữ, KHÔNG gán chữ Nôm. Con số đem so với các bộ
> dữ liệu Hán Nôm khác là **51,922**.

## Cấu trúc

| | |
|---|---|
| `labels.csv` | một dòng mỗi ô, **12 cột cố định**, đường dẫn ảnh tương đối |
| `labels_trace.csv` | sidecar chẩn đoán, cùng số dòng/thứ tự, khoá `image` (không cần để dùng nhãn) |
| `columns.csv` | một dòng mỗi cột trang (`book,page,column`) |
| `gold/` | ảnh crop của các ô có nhãn cấp ký tự |
| `syllable/` | ảnh crop của các ô chỉ có chú giải âm |

## Cột của `labels.csv`

| cột | nghĩa |
|---|---|
| `image` | đường ảnh crop tương đối, **khoá chính** (duy nhất) |
| `book` | mã sách (`stt2`/`stt4`/`stt11`) |
| `page` | tên trang trên bản quét — **đơn vị chia tách** nếu người dùng cần |
| `column` | số cột trên trang (1–9, bố cục ván khắc luôn 9 cột) |
| `ocr_char` | chữ OCR Nôm (S1) — bằng chứng trực tiếp, không rỗng |
| `syllable` | âm Quốc ngữ (lower/NFC) lấy từ bản phiên âm in kèm — nhãn tầng SYLLABLE |
| `label` | chữ Nôm được gán. **Rỗng** ở tầng SYLLABLE |
| `unicode` | mã của `label` (`U+XXXX`), để Excel không hiện được Ext-B vẫn tra được |
| `tier` | `GOLD` = nhãn cấp ký tự · `SYLLABLE` = chỉ có âm (thay cột `label_level` cũ) |
| `rule` | luật quyết nhãn (`quyet_dinh_nguoi:*` = phán quyết NGƯỜI, QĐ-01) — xem DATASHEET |
| `bbox` | toạ độ `[x1, y1, x2, y2]` trên ảnh trang gốc — không tái lập được từ crop |
| `image_md5` | md5 (12 hex đầu) của tệp crop: kiểm toàn vẹn khi sao chép, bắt trùng crop |

## Sidecar `labels_trace.csv` — chẩn đoán, CÙNG số dòng và thứ tự, khoá `image`

Không cần cho việc dùng nhãn; giữ để truy vết vì sao một ô được gán như vậy.

| cột | nghĩa |
|---|---|
| `nom_idx` | thứ tự chữ trong cột OCR Nôm (khoá bền cùng `book,page,column`) |
| `syl_idx` | thứ tự âm trong cột Quốc ngữ |
| `syllable_ocr` | âm Quốc ngữ NGUYÊN VĂN VietOCR (lower) — "ghi đè bản in" đo trên cột này |
| `syllable_raw` | âm sau `normalize_column` (vá dấu), TRƯỚC mọi sửa L1 |
| `tier_v3` | tầng theo luật v3 (`CHAR_A`/`CHAR_B`/`SYL`/…) trước khi ánh xạ sang `tier` |
| `tier_goc` | tier trước khi bị hạ (rỗng nếu chưa từng hạ) |
| `rule_goc` | luật trước khi bị hạ (rỗng nếu chưa từng hạ) |
| `p_register` | xác suất hậu nghiệm P(chữ i ↔ âm j) — forward–backward trên lưới DP có băng |
| `dict_support` | số chữ từ điển cho âm này (|R(s)|); `corpus` khi nhãn do `corpus_readings` |
| `context_evidence` | cờ ngữ cảnh ĐÚNG, nối bằng `|`: `bigram|corpus4|tone|corpus2|direct|sim_unique` |
| `l1_support` | L1: n(âm mới) − n(âm gốc) theo 2-gram âm–âm (LOO trang); > 0 mới đổi âm |
| `l1_tie` | 1 = L1 hoà → giữ âm gốc |
| `flank_gold` | số ô kề (`syl_idx` ± 1, cùng cột) có `tier_v3 == CHAR_A` ∈ {0, 1, 2} |
| `box_source` | nguồn hộp `bbox`: `detector` / `split` / `midpoint` / `legacy_locked_col` / `qd01_locked` |
| `qd01_locked` | 1 = ô khoá theo phán quyết NGƯỜI QĐ-01 (giữ `bbox` cũ) |
| `qd01_excluded` | 1 = ô QĐ-01 bị loại (`bo`) hoặc còn chờ người |
| `label_canonical` | mã chuẩn dị thể theo bảng đã ký (mặc định = `label`; `label` không đổi) |
| `crop_quality_flag` | `ok` / `bleed` (dính mực chữ bên) / `truncated` / `blank` — **`blank`/`truncated` = ảnh hỏng, đừng chấm chiều ảnh** |
| `stray_ink` | tỷ lệ mực lạc (không thuộc chữ chính) |
| `border_ink` | mực sát biên crop |
| `ink_pct` | tỷ lệ điểm mực trên crop |
| `crop_w` | rộng crop (px) |
| `crop_h` | cao crop (px) |
| `seg_flag` | cờ tách chữ của bước crop |
| `s3_cosine` | cosine với nguyên mẫu thị giác S3 (rỗng khi không tính / S3 tắt) |

## `columns.csv` — một dòng mỗi cột trang, khoá `book,page,column`

Dựng trên TOÀN BỘ ô (kể cả ô không giao nộp) nên đủ để suy trang thiếu cột;
các cột `n_ocr` (số chữ OCR Nôm), `n_qn` (số âm Quốc ngữ), `n_det` (số hộp
detector), `count_source` (`equal_qn`/`equal_ocr`/`conflict`/`legacy_locked_col`)
chỉ có ở bộ dựng từ 16/09 trở đi.

## Không chia train/val/test

Bộ này **không** mang cột `split`/`split_group`/`label_in_train` (bỏ từ 16/09). Ba cột ấy
là hàm thuần của `book` + `page`, không mất thông tin khi bỏ; người dùng tự chia theo
**trang** — hai cột cạnh nhau trên cùng một trang dùng chung nét bút, chung mực, chung
lần quét, chia theo cột thì mô hình học được *diện mạo trang* rồi được chấm lại trên
chính trang đó.

Công thức đã dùng cho các bản tới 25/08 (tái lập được):

```python
h = int(hashlib.md5(f'{book}|{page}'.encode()).hexdigest(), 16) % 100
split = 'train' if h < 80 else 'val' if h < 90 else 'test'
```

Khi đánh giá **cấp ký tự** hãy tự tính "lớp chữ có mặt trong train" trên phần train
của chính phép chia mình dùng, và tính trong phạm vi `tier == "GOLD"` — lọc trên cả bộ
sẽ vứt luôn 18,404 dòng SYLLABLE có `label` rỗng.

## Ảnh hỏng và trang thiếu cột

- **464 ô** có ảnh trắng hoặc bị cắt mất nét (`crop_quality_flag` = `blank`/`truncated`
  trong `labels_trace.csv`). Nhãn có thể vẫn đúng; đừng chấm chiều ảnh ở các ô này.
- **214 ô trên 3 trang không đủ 9 cột** (stt11/page_0010_p0028, stt4/page_0110, stt4/page_0252):
  ghép cột Nôm↔Quốc ngữ trên trang đó có thể đã trượt — suy từ `columns.csv`.

## 🔴 Trạng thái kiểm định

**Chưa có phép đo precision nào còn hiệu lực.** Mọi con số precision trong các bản trước
đã bị **tước tư cách bằng chứng** vì nguồn phán quyết hoá ra là máy chấm chứ không phải
người.

Nghĩa là: bộ này dùng được để **huấn luyện** và **thăm dò**, nhưng **chưa được trích dẫn
như dữ liệu đã kiểm chứng**.

## Trích dẫn

Xem `NGUON_THU_TICH.md` cho lai lịch ba cuốn sách nguồn.

*Sinh tự động từ `labels.csv` ngày 2026-09-16 · commit `08e954752b`*
