# GanNhanOCR

**Gán nhãn tự động cho kho ngữ liệu Hán Nôm viết tay từ bản dịch Quốc ngữ**

Hệ thống xử lý sách Hán Nôm viết tay (PDF) kết hợp bản dịch Quốc ngữ, tự động gán nhãn Unicode cho từng ký tự Nôm với độ tin cậy phân tầng (high/medium/low).

---

## Tổng quan pipeline

```
PDF sách Hán Nôm
│
├─ Giai đoạn 1: prepare_data.py ─── Tách PDF + OCR text QN + Khử nhiễu ảnh
│     PyMuPDF · Tesseract OCR · Background Normalization
│
├─ Giai đoạn 2: detect_characters.py ─── Phát hiện cột + Cắt ký tự
│     Otsu · Morphological Line Detection · Vertical/Horizontal Projection
│
├─ Giai đoạn 3: clean_crops.py ─── Làm sạch ảnh ký tự
│     Sauvola Binarization · Connected Component · Stroke Normalization
│
├─ Giai đoạn 4: label_characters.py ─── Gán nhãn tự động
│     Levenshtein DP · Dictionary Lookup · OCR API · Self-Consistency
│
├─ Giai đoạn 5: (tích hợp trong label_characters.py) ─── Xuất dataset
│     JSON · CSV · XlsxWriter · Review Images
│
└─ Nâng cao (embedding/) ─── Deep Metric Learning
      ResNet50 + SupConLoss · ViT + ConMIM · FAISS · Iterative Refinement
```

---

## Cài đặt

### Yêu cầu

- Python 3.10+
- Tesseract OCR (`brew install tesseract` hoặc `apt install tesseract-ocr`)

### Thư viện Python

```bash
# Core pipeline
pip install PyMuPDF opencv-python numpy scipy pytesseract Pillow requests

# Xuất Excel (tuỳ chọn)
pip install xlsxwriter

# Deep embedding (tuỳ chọn)
pip install torch torchvision faiss-cpu
```

---

## Cấu trúc dự án

```
GanNhanOCR/
├── prepare_data.py            # GĐ1: Tách PDF → ảnh Nôm + text QN
├── detect_characters.py       # GĐ2: Detect cột + cắt ký tự
├── clean_crops.py             # GĐ3: Làm sạch ảnh (Sauvola)
├── label_characters.py        # GĐ4-5: Gán nhãn + xuất dataset
├── export_dataset.py          # Tổng hợp nhiều bộ sách → dataset chuẩn
│
├── embedding/                 # Deep Metric Learning (nâng cao)
│   ├── prepare_data.py        #   Chuẩn bị dữ liệu training
│   ├── train_embedding.py     #   ResNet50 + SupConLoss
│   ├── train_conmim.py        #   ViT + ConMIM pre-training
│   ├── embed_ranker.py        #   FAISS ranking interface
│   └── iterative_refine.py    #   Vòng lặp label → train → re-label
│
├── Alignment/Code/dict/       # Từ điển QN → Nôm
│   ├── QuocNgu_SinoNom_Merged.csv   # 104,164 cặp (chính)
│   └── SinoNom_Similar_Dic_v2.csv   # Ký tự tương tự
│
├── FontDiffusion/
│   └── fonts/NomNaTong-Regular.ttf   # Font render ảnh đánh máy
│
└── data/                      # PDF đầu vào
    ├── CacThanhTruyen4.pdf
    └── SachThanhTruyen4.pdf
```

---

## Công nghệ theo giai đoạn

| Giai đoạn | Công nghệ | Lĩnh vực |
|-----------|-----------|----------|
| Tách PDF | PyMuPDF + Tesseract OCR | Document Processing |
| Khử nhiễu ảnh | Background Normalization (Morph. Closing 51×51) | Image Processing |
| Binarize ảnh | Otsu thresholding + Close/Open | Image Processing |
| Detect vùng text | Morphological line detection | Classical CV |
| Detect cột | Vertical Projection + find_peaks | Classical CV |
| Detect ký tự | Horizontal Projection + merge/split | Classical CV |
| Làm sạch crop | Sauvola binarization (R=128) + morphology | Image Processing |
| Alignment | Levenshtein DP (variable deletion cost) | Sequence Alignment |
| Tra từ điển | Dictionary lookup + heuristic ranking | NLP / Lexicography |
| OCR bổ trợ | REST API (HCMUS server) | Cloud OCR |
| Self-Consistency | Statistical label propagation | Semi-supervised Learning |
| Deep ranking | ResNet50/ViT + SupConLoss + FAISS | Deep Metric Learning |

---

## Hướng dẫn chạy

### Giai đoạn 1: Tách dữ liệu từ PDF

```bash
# CacThanhTruyen (text nhúng sẵn trong PDF)
python prepare_data.py data/CacThanhTruyen4.pdf

# SachThanhTruyen (cần re-OCR bằng Tesseract)
python prepare_data.py data/SachThanhTruyen4.pdf --reocr

# Với khử nhiễu ảnh Nôm (lưu song song bản denoised)
python prepare_data.py data/SachThanhTruyen4.pdf --reocr --denoise

# Nhiều file + tuỳ chỉnh DPI
python prepare_data.py data/*.pdf --dpi 400
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `pdf_files` | (bắt buộc) | File PDF đầu vào |
| `--output-dir` | `data/prepared/<tên_pdf>/` | Thư mục output |
| `--dpi` | `300` | Độ phân giải ảnh |
| `--reocr` | `False` | Re-OCR trang text bằng Tesseract |
| `--denoise` | `False` | Lưu ảnh Nôm đã khử nhiễu vào `pages_denoised/` |
| `--ocr-lang` | `vie` | Ngôn ngữ Tesseract |

**Pipeline khử nhiễu (khi `--reocr` hoặc `--denoise`):**
```
Ảnh gốc → GaussianBlur(3,3) → Background Estimation (Morph. Closing 51×51)
→ Background Removal (pixel ÷ background × 255) → Contrast Stretching
→ (OCR: + Otsu → Close 2×2 → Open 3×3)
```

**Output:**
```
data/prepared/CacThanhTruyen4/
├── pages/page_0012.png              # Ảnh trang Nôm (gốc)
├── pages_denoised/page_0012.png     # (--denoise) Ảnh đã khử nhiễu
├── transcriptions/page_0012.txt     # Text QN (1 dòng = 1 cột)
├── transcriptions/page_0012.json    # Chi tiết âm tiết
└── manifest.json                    # Metadata
```

---

### Giai đoạn 2: Phát hiện và cắt ký tự

```bash
python detect_characters.py data/prepared/CacThanhTruyen4
python detect_characters.py data/prepared/CacThanhTruyen4 --debug    # Ảnh debug
python detect_characters.py data/prepared/CacThanhTruyen4 --page 12  # 1 trang
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `prepared_dir` | (bắt buộc) | Thư mục từ GĐ1 |
| `--debug` | `False` | Lưu ảnh debug với bounding box |
| `--page` | Tất cả | Chỉ xử lý 1 trang |

**Pipeline (Bước 2.1 → 2.4):**
```
Ảnh gốc (hoặc denoised nếu có)
│
├─ 2.1 Binarization:
│   GaussianBlur(3,3) → Morph. Closing(51×51) → Divide → Otsu
│   → Close(2×2) → Open(3×3) → ảnh nhị phân (1=mực, 0=nền)
│
├─ 2.2 Text Box Detection:
│   Morph. Open (w/3×1) → đường ngang
│   Morph. Open (1×h/3) → đường dọc
│   → 4 đường → hình chữ nhật vùng text
│
├─ 2.3 Column Detection:
│   Vertical Projection → Smoothing → find_peaks
│   → N cột (auto-detect từ transcription hoặc projection)
│   Thứ tự: phải → trái (cột 1 ở bên phải)
│
└─ 2.4 Character Segmentation:
    Horizontal Projection → find_peaks → ranh giới ký tự
    Merge: box < 40% expected height → gộp với box liền kề
    Split: box > 160% expected height → chia bằng projection valley
    Adaptive retry: nếu lệch > 15% expected count → thử lại
```

**Output:**
```
data/prepared/CacThanhTruyen4/detected/
├── crops/page_0012/col01_char000.png    # Ảnh crop từng ký tự
├── page_0012_detection.json             # Bbox + metadata
├── debug/page_0012_debug.png            # (--debug) Ảnh với bbox
└── summary.json                         # Thống kê
```

---

### Giai đoạn 3: Làm sạch ảnh ký tự

```bash
python clean_crops.py data/prepared/CacThanhTruyen4/detected
python clean_crops.py data/prepared/CacThanhTruyen4/detected --verify   # So sánh before/after
python clean_crops.py data/prepared/CacThanhTruyen4/detected --page 12
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `detected_dir` | (bắt buộc) | Thư mục detected/ từ GĐ2 |
| `--size` | `64` | Kích thước output vuông |
| `--sauvola-k` | `0.2` | Sauvola sensitivity |
| `--sauvola-window` | `25` | Sauvola window size |
| `--denoise` | `3` | Median blur kernel size |
| `--min-stroke` | `2` | Độ dày nét tối thiểu |
| `--verify` | `False` | Tạo ảnh so sánh before/after |

**Pipeline (self-contained, không phụ thuộc FontDiffusion):**
```
Ảnh crop thô
│
├─ Median Blur: khử nhiễu muối tiêu
│
├─ Sauvola Binarization:
│   T(x,y) = mean(x,y) × [1 + k × (std(x,y)/R - 1)]
│   k=0.2, R=128 (cố định theo paper gốc)
│   → Ngưỡng cục bộ, xử lý tốt nền sáng tối không đều
│
├─ Morphological Close(2×2) → Open(3×3)
│   Close: nối nét bị đứt  ·  Open: xóa chấm nhiễu
│
├─ Connected Component: xóa vùng < 0.5% diện tích
│
├─ Stroke Normalization: distance transform → dilate/erode
│
└─ Center + Resize → 64×64 (nét đen, nền trắng)
```

**Output:**
```
data/prepared/CacThanhTruyen4/detected/
├── crops_cleaned/page_0012/col01_char000.png   # Ảnh cleaned 64×64
├── crops_cleaned/clean_summary.json            # Thống kê
└── verify/page_0012_verify.png                 # (--verify) Before/after
```

---

### Giai đoạn 4-5: Gán nhãn + Xuất dataset

```bash
# Cơ bản (Self-Consistency luôn chạy)
python label_characters.py data/prepared/CacThanhTruyen4

# Với OCR API (cần mạng)
python label_characters.py data/prepared/CacThanhTruyen4 --ocr

# Với deep embedding ranking
python label_characters.py data/prepared/CacThanhTruyen4 \
    --embedding embedding/checkpoints/best.pt \
    --gallery embedding/data/gallery

# Đầy đủ
python label_characters.py data/prepared/CacThanhTruyen4 \
    --ocr --review --excel
```

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `prepared_dir` | (bắt buộc) | Thư mục prepared data |
| `--font` | `FontDiffusion/fonts/NomNaTong-Regular.ttf` | Font Nôm TTF |
| `--ocr` | `False` | Dùng API OCR HCMUS |
| `--embedding` | `None` | Checkpoint deep embedding |
| `--gallery` | `None` | Gallery directory |
| `--review` | `False` | Tạo ảnh review [viết tay \| đánh máy \| nhãn] |
| `--excel` | `False` | Xuất file Excel có màu |
| `--page` | Tất cả | Chỉ xử lý 1 trang |

**Bước 4.1 — Chuẩn hóa text QN:**
```
"1. trời đất sinh ra Đo-minh-gô1 ..."
  → Tách tên riêng: "Dominhgô" → "do minh cô" (bảng SAINT_NAMES)
  → Xóa chú thích, dấu câu, artifact
  → Tách âm tiết: ["trời", "đất", "sinh", "ra", "do", "minh", "cô"]
```

**Bước 4.2 — Levenshtein Alignment (DP):**
```
N ảnh ký tự ↔ M âm tiết QN (N có thể ≠ M)
Chi phí deletion theo kích thước:
  < 30% median height → cost 0.3 (nhiễu)
  < 50% median height → cost 0.6 (nhỏ)
  ≥ 50% median height → cost 1.2 (ký tự thật)
Insertion cost = 1.0
```

**Bước 4.3 — Tra từ điển + Xếp hạng:**
```
"trời" → tra từ điển (104,164 cặp) → [𡗶, 𠅜, 天, ...]

1 ứng viên  → gán luôn                → confidence = "high"
N ứng viên  → xếp hạng → chọn top-1  → confidence = "medium"
0 ứng viên  → không gán               → confidence = "low"

Xếp hạng 3 cấp (tùy tài nguyên có sẵn):
  Deep embedding:  0.5 × embed_sim + 0.3 × specificity + 0.2 × cjk_block
  Classical visual: 0.4 × IoU/proj   + 0.3 × specificity + 0.3 × cjk_block
  Fallback:                            0.6 × specificity + 0.4 × cjk_block
```

**Bước 4.4 — OCR API (tuỳ chọn, `--ocr`):**
```
Ảnh trang Nôm → Upload HCMUS → OCR → boxes → columns
So khớp bbox y-overlap: nếu OCR char ∈ candidates → nâng lên "high"
Cache: labeled/ocr_cache/ (tránh gọi lại API)
```

**Bước 4.5 — Self-Consistency (LUÔN chạy):**
```
Thu thập TẤT CẢ cặp (QN, Nôm) có confidence = "high"
  (bao gồm: 1-candidate dict lookup + OCR-confirmed)
Đếm: "trời" → {𡗶: 8 lần}
Nếu 1 ký tự xuất hiện ≥2 lần → propagate cho tất cả medium cùng từ
→ medium → high
```

**Output (Giai đoạn 5):**
```
data/prepared/CacThanhTruyen4/labeled/
├── dataset.json              # Đầy đủ metadata
├── labels.csv                # Format chuẩn nghiên cứu (UTF-8 BOM)
├── summary.json              # Thống kê confidence
├── typed_nom/000000.png      # Ảnh Nôm render từ font (pygame/PIL)
├── review/page_0012.png      # (--review) [viết tay | đánh máy | nhãn]
├── ocr_cache/                # (--ocr) Cache kết quả OCR
└── *.xlsx                    # (--excel) Excel có màu
```

---

### Tổng hợp nhiều bộ sách (`export_dataset.py`)

```bash
python export_dataset.py data/prepared/*/labeled \
    --output dataset/ --split 0.8 0.1 0.1

# Chỉ high confidence
python export_dataset.py data/prepared/*/labeled \
    --output dataset/ --min-confidence high
```

**Output:**
```
dataset/
├── labels.csv       # Tổng hợp tất cả
├── train.csv        # 80% (stratified by source)
├── val.csv          # 10%
├── test.csv         # 10%
├── class_map.json   # class_id → (char, unicode)
└── metadata.json    # Thống kê
```

---

## Deep Embedding (nâng cao)

### Chuẩn bị dữ liệu

```bash
python embedding/prepare_data.py \
    --kaggle-dataset path/to/New-SinoNom_Dataset \
    --font FontDiffusion/fonts/NomNaTong-Regular.ttf \
    --output embedding/data
```

### Training ResNet50 + SupConLoss

```bash
python embedding/train_embedding.py \
    --manifest embedding/data/manifest.csv \
    --output-dir embedding/checkpoints \
    --epochs 50 --device cuda
```

### ConMIM Pre-training (ViT-Small)

```bash
# Phase 1: Self-supervised pre-training
python embedding/train_conmim.py \
    --manifest embedding/data/manifest.csv \
    --output-dir embedding/checkpoints/conmim \
    --pretrain-epochs 100 --finetune-epochs 50
```

### Iterative Refinement

```bash
python embedding/iterative_refine.py \
    --prepared-dir data/prepared/SachThanhTruyen4 \
    --manifest embedding/data/manifest.csv \
    --checkpoint embedding/checkpoints/best.pt \
    --gallery embedding/data/gallery \
    --max-rounds 3
```

---

## Hệ thống confidence

| Mức | Ý nghĩa | Điều kiện |
|-----|---------|-----------|
| `high` | Tin cậy cao | 1 ứng viên duy nhất / OCR xác nhận / Self-Consistency (≥2 lần) |
| `medium` | Cần review | Nhiều ứng viên, chọn bằng ranking |
| `low` | Không tìm thấy | Từ QN không có trong từ điển |
| `gap` | Thừa/thiếu | Insertion hoặc Deletion từ alignment |

---

## Chạy toàn bộ pipeline

```bash
# === CacThanhTruyen4 (text nhúng PDF) ===
python prepare_data.py data/CacThanhTruyen4.pdf --denoise
python detect_characters.py data/prepared/CacThanhTruyen4
python clean_crops.py data/prepared/CacThanhTruyen4/detected
python label_characters.py data/prepared/CacThanhTruyen4 --ocr --review --excel

# === SachThanhTruyen4 (cần Tesseract re-OCR) ===
python prepare_data.py data/SachThanhTruyen4.pdf --reocr --denoise
python detect_characters.py data/prepared/SachThanhTruyen4
python clean_crops.py data/prepared/SachThanhTruyen4/detected
python label_characters.py data/prepared/SachThanhTruyen4 --ocr --review --excel

# === Tổng hợp dataset ===
python export_dataset.py data/prepared/*/labeled --output dataset/
```

---

## Format dataset

### labels.csv

```csv
image,nom_char,label,reading,confidence,bbox,page,source
crops_cleaned/page_0012/col01_char000.png,經,U+7D93,kinh,high,"100,200,150,260",12,CacThanhTruyen4
```

| Trường | Mô tả |
|--------|-------|
| `image` | Đường dẫn ảnh crop (tương đối) |
| `nom_char` | Ký tự Nôm (Unicode gốc) |
| `label` | Mã Unicode `U+XXXX` |
| `reading` | Âm đọc Quốc ngữ |
| `confidence` | `high` / `medium` / `low` |
| `bbox` | Bounding box `x1,y1,x2,y2` |
| `page` | Số trang sách |
| `source` | Tên bộ sách |

---

## Tài liệu tham khảo

- [New-SinoNom Dataset (Kaggle)](https://www.kaggle.com/datasets/5c09041f61f1bd528a0281281a55ed4ddb6b4aa1c83bdb0c0e21a1553339ad32)
- [SinoNom Similarity Retrieval (Kaggle)](https://www.kaggle.com/code/hongduyhng/sinonom-img-to-img-similarity-retrieval)
- [ConMIM - Masked Image Modeling (GitHub)](https://github.com/TencentARC/ConMIM)
- [FontDiffuser / Font Architect (HuggingFace)](https://huggingface.co/dzungpham/font-architect)
- [FontTransfer Dataset (HuggingFace)](https://huggingface.co/datasets/dzungpham/FontTransfer)
