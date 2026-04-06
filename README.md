# GanNhanOCR

**Gán nhãn tự động cho kho ngữ liệu Hán Nôm viết tay từ bản dịch Quốc ngữ**

Hệ thống xử lý sách Hán Nôm viết tay (dạng PDF) kết hợp với bản dịch Quốc ngữ, tự động gán nhãn từng ký tự Nôm: crop ảnh viết tay, chữ Nôm, mã Unicode, âm đọc Quốc ngữ, và độ tin cậy.

---

## Tổng quan quy trình

```
PDF sách Hán Nôm ──► Bước 1: prepare_data.py
                         │  Trích xuất ảnh trang Nôm + OCR text Quốc ngữ
                         ▼
                     Bước 2: detect_characters.py
                         │  Phát hiện cột + cắt từng ký tự Nôm
                         ▼
                     Bước 3: clean_crops.py
                         │  Khử nhiễu ảnh ký tự (Sauvola binarization)
                         ▼
                     Bước 4-6: label_characters.py
                         │  Alignment + Tra từ điển + Gán nhãn + Xuất dataset
                         ▼
                     Dataset (JSON + CSV + Excel + ảnh crop)
```

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- Tesseract OCR (cài sẵn, dùng cho bước OCR text Quốc ngữ)

### Cài đặt thư viện

```bash
pip install PyMuPDF opencv-python numpy scipy requests urllib3 pytesseract Pillow xlsxwriter
```

### Cấu trúc thư mục

```
GanNhanOCR/
├── prepare_data.py          # Bước 1: Trích xuất PDF
├── detect_characters.py     # Bước 2: Phát hiện ký tự
├── clean_crops.py           # Bước 3: Khử nhiễu ảnh
├── label_characters.py      # Bước 4-6: Gán nhãn + xuất dataset
├── data/                    # Dữ liệu PDF đầu vào
│   ├── CacThanhTruyen4.pdf
│   └── SachThanhTruyen4.pdf
├── Alignment/Code/dict/     # Từ điển
│   ├── QuocNgu_SinoNom_Merged.csv      # Từ điển QN→Nôm chính (104,164 cặp)
│   ├── QuocNgu_SinoNom_TongHop3.csv    # Từ điển gốc (99,859 cặp)
│   ├── SinoNom_Similar_Dic_v2.csv      # Từ điển ký tự tương tự
│   └── QuocNgu_SinoNom_Dic.xlsx        # Dic bổ sung (4,345 cặp mới)
├── FontDiffusion/
│   ├── fonts/NomNaTong-Regular.ttf     # Font Nôm để render ảnh đánh máy
│   └── scripts/clean_image_enhance.py  # CharacterImageCleaner
└── OCR/                     # Tham khảo code OCR API
```

---

## Hướng dẫn chạy

### Bước 1: Trích xuất dữ liệu từ PDF (`prepare_data.py`)

Tách trang ảnh Nôm (trang lẻ) và text Quốc ngữ (trang chẵn) từ PDF.

```bash
# Cơ bản
python prepare_data.py data/CacThanhTruyen4.pdf

# Tuỳ chỉnh thư mục output và DPI
python prepare_data.py data/SachThanhTruyen4.pdf --output-dir data/prepared/STT4 --dpi 400

# Re-OCR bằng Tesseract (cần cho SachThanhTruyen - text không có sẵn trong PDF)
python prepare_data.py data/SachThanhTruyen4.pdf --reocr --ocr-lang vie

# Xử lý nhiều file cùng lúc
python prepare_data.py data/CacThanhTruyen4.pdf data/SachThanhTruyen4.pdf
```

**Tham số:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `pdf_files` | (bắt buộc) | Đường dẫn file PDF |
| `--output-dir` | `data/prepared/<tên_pdf>/` | Thư mục output |
| `--dpi` | `300` | Độ phân giải ảnh trích xuất |
| `--reocr` | `False` | Re-OCR trang text bằng Tesseract |
| `--ocr-lang` | `vie` | Ngôn ngữ Tesseract |
| `--quiet` | `False` | Không hiển thị chi tiết |

**Output:**

```
data/prepared/CacThanhTruyen4/
├── images/page_0001.png, page_0002.png, ...      # Ảnh trang Nôm
├── transcriptions/page_0001.txt, ...              # Text QN tương ứng
├── text_pages/page_0001.png, ...                  # Ảnh trang QN gốc
└── manifest.json                                  # Metadata (số trang, DPI, ...)
```

---

### Bước 2: Phát hiện và cắt ký tự (`detect_characters.py`)

Phát hiện cột (vertical projection + find_peaks) và cắt từng ký tự Nôm (horizontal projection).

```bash
# Cơ bản
python detect_characters.py data/prepared/CacThanhTruyen4

# Lưu ảnh debug với bounding box
python detect_characters.py data/prepared/CacThanhTruyen4 --debug

# Chỉ xử lý 1 trang
python detect_characters.py data/prepared/CacThanhTruyen4 --page 2
```

**Tham số:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `prepared_dir` | (bắt buộc) | Thư mục từ bước 1 |
| `--output-dir` | `<prepared_dir>/detected/` | Thư mục output |
| `--debug` | `False` | Lưu ảnh debug với bbox |
| `--page` | Tất cả | Chỉ xử lý 1 trang |
| `--quiet` | `False` | Không hiển thị chi tiết |

**Output:**

```
data/prepared/CacThanhTruyen4/detected/
├── crops/page_0001/char_0001.png, ...             # Ảnh crop từng ký tự
├── page_0001_detection.json                       # Bbox + metadata
├── debug/page_0001_debug.png                      # (--debug) Ảnh với bbox
└── detection_summary.json                         # Thống kê
```

**Thuật toán phát hiện:**
1. Adaptive thresholding → ảnh nhị phân
2. Vertical projection profile → tìm ranh giới cột (find_peaks với constraint 9 cột)
3. Horizontal projection trong từng cột → tìm ranh giới ký tự
4. Merge box nhỏ (nét bút bị tách) + Split box lớn (2 ký tự dính)

---

### Bước 3: Khử nhiễu ảnh ký tự (`clean_crops.py`)

Áp dụng Sauvola binarization — phương pháp tối ưu cho tài liệu lịch sử viết tay có nền không đều.

```bash
# Cơ bản (Sauvola, 64x64)
python clean_crops.py data/prepared/CacThanhTruyen4/detected

# Tuỳ chỉnh kích thước và phương pháp
python clean_crops.py data/prepared/CacThanhTruyen4/detected --size 128 --method adaptive

# Xem ảnh so sánh before/after
python clean_crops.py data/prepared/CacThanhTruyen4/detected --verify --verify-samples 20

# Chỉ 1 trang
python clean_crops.py data/prepared/CacThanhTruyen4/detected --page 5
```

**Tham số:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `detected_dir` | (bắt buộc) | Thư mục detected/ từ bước 2 |
| `--output-name` | `crops_cleaned` | Tên thư mục output |
| `--size` | `64` | Kích thước output vuông (pixel) |
| `--method` | `sauvola` | Phương pháp binarization: `auto`, `otsu`, `adaptive`, `sauvola` |
| `--denoise` | `3` | Strength khử nhiễu |
| `--min-stroke` | `2` | Độ dày nét tối thiểu |
| `--padding` | `5` | Padding quanh ký tự |
| `--page` | Tất cả | Chỉ xử lý 1 trang |
| `--verify` | `False` | Tạo ảnh so sánh before/after |
| `--verify-samples` | `10` | Số trang tạo ảnh verify |

**Output:**

```
data/prepared/CacThanhTruyen4/detected/
├── crops_cleaned/page_0001/char_0001.png, ...     # Ảnh cleaned 64x64
├── verify/page_0001_verify.png                    # (--verify) So sánh before/after
└── crops_cleaned/clean_summary.json               # Thống kê
```

**Pipeline khử nhiễu:**
1. Đọc ảnh crop grayscale
2. Sauvola binarization (xử lý uneven illumination)
3. Morphological cleanup
4. Connected component noise removal
5. Stroke thickness normalization
6. Center + resize về kích thước chuẩn
7. Output: nét đen trên nền trắng

---

### Bước 4-6: Gán nhãn + Xuất dataset (`label_characters.py`)

Gán nhãn tự động bằng Levenshtein alignment + tra từ điển + OCR API (tuỳ chọn).

```bash
# Cơ bản
python label_characters.py data/prepared/CacThanhTruyen4

# Với OCR API để cải thiện accuracy (cần kết nối mạng)
python label_characters.py data/prepared/CacThanhTruyen4 --ocr

# Tạo ảnh review (so sánh viết tay | đánh máy)
python label_characters.py data/prepared/CacThanhTruyen4 --review

# Xuất file Excel
python label_characters.py data/prepared/CacThanhTruyen4 --excel

# Đầy đủ tất cả tuỳ chọn
python label_characters.py data/prepared/CacThanhTruyen4 --ocr --review --excel

# Chỉ 1 trang
python label_characters.py data/prepared/CacThanhTruyen4 --page 2 --review
```

**Tham số:**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `prepared_dir` | (bắt buộc) | Thư mục prepared data |
| `--font` | `FontDiffusion/fonts/NomNaTong-Regular.ttf` | Font Nôm TTF |
| `--page` | Tất cả | Chỉ xử lý 1 trang |
| `--review` | `False` | Tạo ảnh review (viết tay \| đánh máy) |
| `--excel` | `False` | Xuất file Excel với kết quả có màu |
| `--ocr` | `False` | Dùng API OCR tools.clc.hcmus.edu.vn |

**Output:**

```
data/prepared/CacThanhTruyen4/labeled/
├── dataset.json             # Toàn bộ nhãn
├── labels.csv               # Dataset chuẩn nghiên cứu
├── summary.json             # Thống kê accuracy
├── typed_nom/               # Ảnh Nôm đánh máy render từ font
│   └── page_0001/char_0001.png, ...
├── review/                  # (--review) Ảnh debug
│   └── page_0001_review.png
└── labels_CacThanhTruyen4.xlsx  # (--excel) File Excel
```

**Chi tiết từng bước con:**

**Bước 4 — Levenshtein Alignment:**
- Đối sánh N ký tự detected ↔ M âm tiết Quốc ngữ
- Chi phí xoá ký tự dựa trên kích thước ảnh (ảnh nhỏ → chi phí thấp → ưu tiên xoá nhiễu)
- Kết quả: mỗi ký tự được gán 1 âm đọc QN hoặc đánh dấu `[GAP]`

**Bước 5 — Tra từ điển + Gán Unicode (có xếp hạng ứng viên):**
- Từ điển chính: `QuocNgu_SinoNom_Merged.csv` (104,164 cặp QN→Nôm, 7,521 từ QN)
- Nếu 1 âm QN có 1 ứng viên → gán luôn (`high`)
- Nếu nhiều ứng viên → **xếp hạng bằng combined score** rồi chọn tốt nhất (`medium`):
  - **Visual similarity (60%)**: so ảnh crop viết tay vs ảnh render font (IoU + projection profile correlation)
  - **Corpus frequency (40%)**: đếm tần suất ký tự xuất hiện trong corpus transcription
  - Cải thiện accuracy +15.3% so với lấy mặc định ứng viên đầu tiên (20.9% → 36.2%)
- Nếu có OCR (`--ocr`): gọi API tools.clc.hcmus.edu.vn, so khớp bbox overlap, ưu tiên kết quả OCR → nâng lên `high`
- Font render bằng **pygame.freetype** (từ FontDiffusion) cho chất lượng cao hơn PIL

**Bước 5.5 — Render ảnh đánh máy:**
- Render chữ Nôm Unicode bằng font NomNaTong-Regular.ttf
- Dùng cho visual review (so sánh viết tay vs đánh máy)

**Bước 7 — Self-Consistency (khi dùng `--ocr`):**
- Sau khi xử lý tất cả trang, thu thập các cặp (QN → Nôm) đã được OCR xác nhận
- Nếu cùng từ QN được OCR xác nhận ≥2 lần cho cùng ký tự Nôm → dùng ký tự đó cho tất cả medium còn lại
- Hiệu quả: High tăng từ 14.7% → **64.8%** (+41,757 ký tự trên toàn bộ 6 bộ sách)

**Bước 8 — Xuất dataset:**
- `dataset.json`: đầy đủ tất cả trường (nom_char, nom_unicode, reading, confidence, ranking_score, bbox, candidates, ...)
- `labels.csv`: format chuẩn nghiên cứu — `image, nom_char, label, reading, confidence, bbox, page, source`
- Excel (tuỳ chọn): có màu theo confidence (xanh=high, vàng=medium, đỏ=low)

---

## Hệ thống confidence

| Mức | Ý nghĩa | Điều kiện |
|-----|---------|-----------|
| `high` | Tin cậy cao | Chỉ 1 ứng viên, hoặc OCR xác nhận, hoặc Self-Consistency |
| `medium` | Cần review | Nhiều ứng viên, chưa được OCR/Consistency xác nhận |
| `low` | Không tìm thấy | Từ QN không có trong từ điển |
| `gap` | Insertion/Deletion | Ký tự thừa hoặc thiếu từ alignment |

---

## OCR API (tuỳ chọn)

Khi dùng flag `--ocr`, hệ thống gọi API nhận dạng chữ Nôm tại `tools.clc.hcmus.edu.vn`:

1. Upload ảnh trang Nôm → nhận bounding boxes + ký tự nhận dạng
2. Tổ chức kết quả theo cột (sắp xếp x phải→trái)
3. So khớp bbox overlap với ký tự detected:
   - **Cột**: tìm cột OCR có x-center gần nhất (threshold 100px)
   - **Ký tự**: trong cột, tìm box OCR có y-center gần nhất (threshold max(det_h, 50)px)
4. Nếu khớp → dùng ký tự OCR, confidence = `high`

**Cache**: Kết quả OCR được cache theo hash ảnh tại `<detected_dir>/ocr_cache/`, tránh gọi lại API cho ảnh đã xử lý.

---

## Chạy toàn bộ pipeline

```bash
# === CacThanhTruyen4 ===
python prepare_data.py data/CacThanhTruyen4.pdf
python detect_characters.py data/prepared/CacThanhTruyen4
python clean_crops.py data/prepared/CacThanhTruyen4/detected
python label_characters.py data/prepared/CacThanhTruyen4 --ocr --review --excel

# === SachThanhTruyen4 (cần --reocr vì PDF không có text layer) ===
python prepare_data.py data/SachThanhTruyen4.pdf --reocr
python detect_characters.py data/prepared/SachThanhTruyen4
python clean_crops.py data/prepared/SachThanhTruyen4/detected
python label_characters.py data/prepared/SachThanhTruyen4 --ocr --review --excel
```

---

## Kết quả đạt được

Trên toàn bộ 6 bộ dữ liệu (với `--ocr`):

| Bộ dữ liệu | Trang | Detected | Matched | High | Medium | Low | Gap |
|-------------|-------|----------|---------|------|--------|-----|-----|
| CacThanhTruyen2 | 5 | 866 | 864 | 38.9% | 61.0% | 0.1% | 10.3% |
| CacThanhTruyen4 | 4 | 756 | 753 | 49.1% | 50.2% | 0.7% | 2.0% |
| CacThanhTruyen11 | 4 | 767 | 767 | 49.2% | 50.8% | 0.0% | 0.5% |
| SachThanhTruyen2 | 160 | 28,454 | 27,973 | 64.6% | 34.0% | 1.3% | 4.6% |
| SachThanhTruyen4 | 145 | 27,441 | 26,471 | 67.5% | 31.7% | 0.8% | 8.6% |
| SachThanhTruyen11 | 143 | 27,160 | 26,646 | 63.9% | 35.3% | 0.8% | 6.4% |
| **TỔNG** | **461** | **85,444** | **83,474** | **64.8%** | **34.3%** | **0.9%** | **6.5%** |

**Chuỗi cải tiến:**

| Giai đoạn | High confidence |
|-----------|-----------------|
| Ban đầu (lấy dict[0]) | ~0% |
| + Ranking (visual 60% + frequency 40%) | ~0% (thay đổi ứng viên, không đổi confidence) |
| + OCR API bbox matching | 14.7% (+12,289 ký tự) |
| + **Self-Consistency** | **64.8%** (+41,757 ký tự) |

**Lưu ý:**
- **Self-Consistency** là cải tiến lớn nhất: dùng kết quả OCR đã xác nhận để nâng medium→high cho cùng từ QN
- SachThanhTruyen hưởng lợi nhiều nhất (140-160 trang → nhiều lần lặp lại cùng từ)
- `gap` ở SachThanhTruyen cao hơn do Tesseract OCR text QN kém chính xác hơn text layer có sẵn

---

## Từ điển

### QuocNgu_SinoNom_Merged.csv
- **104,164 cặp** (QN → Nôm)
- **7,521 từ QN** khác nhau, **41,501 ký tự Nôm** khác nhau
- Được merge từ TongHop3.csv (99,859 cặp) + Dic.xlsx (4,345 cặp mới)
- Format: `quoc_ngu,sino_nom` (UTF-8, không header)

### SinoNom_Similar_Dic_v2.csv
- Từ điển ký tự Nôm tương tự (dùng cho so sánh hình dạng)

---

## Format dataset đầu ra

### labels.csv (format chuẩn nghiên cứu)

```csv
image,nom_char,label,reading,confidence,bbox,page,source
crops_cleaned/page_0001/char_0001.png,經,U+7D93,kinh,high,"[x1,y1,x2,y2]",1,CacThanhTruyen4
```

| Trường | Mô tả |
|--------|-------|
| `image` | Đường dẫn ảnh crop (tương đối từ thư mục labeled/) |
| `nom_char` | Chữ Nôm (ký tự Unicode gốc) |
| `label` | Mã Unicode dạng `U+XXXX` |
| `reading` | Âm đọc Quốc ngữ |
| `confidence` | Mức độ tin cậy: `high`, `medium`, `low`, `gap` |
| `bbox` | Bounding box `[x1, y1, x2, y2]` |
| `page` | Số trang |
| `source` | Tên bộ sách |

### dataset.json

Chứa đầy đủ thông tin hơn labels.csv, bao gồm: danh sách tất cả ứng viên (`candidates`), thông tin alignment, OCR result, v.v.
