# GanNhanOCR

**Tu dong gan nhan Unicode cho kho ngu lieu Han Nom viet tay tu ban dich Quoc ngu**

He thong xu ly sach Han Nom viet tay (PDF) ket hop ban dich Quoc ngu, tu dong gan nhan Unicode cho tung ky tu Nom thong qua 3 tang tra cuu: tu dien song huong, ky tu tuong tu, va so khop anh (DINOv2).

---

## Tong quan pipeline

```
PDF sach co (Han Nom + Quoc Ngu)
│
├── Buoc 0: Setup ─── Kiem tra moi truong, tao thu muc
│
├── Buoc 1: Extract ─── PDF → anh goc + anh khu nhieu + OCR + crop ky tu + text QN
│     • original_image  → luu vao pages/           (giu nguyen, dung cho dataset)
│     • processed_image → luu vao pages_denoised/  (chi dung noi bo cho OCR)
│     • processed_image → Kimhannom API → bbox + OCR so bo
│     • original_image  + bbox → crop ky tu goc (khong xu ly them)
│
├── Buoc 2: Align ─── Can chinh Levenshtein (N ky tu ↔ M am tiet QN)
│
├── Buoc 3: Label ─── Gan nhan 3 tang
│     • Tang 1: Tu dien song huong (QN↔Nom)
│     • Tang 2: Mo rong qua chu tuong tu
│     • Tang 3: So khop anh (DINOv2 cosine similarity)
│
└── Buoc 4: Export ─── Gop sach → Loc chat luong → dataset cuoi cung
```

**Nguyen tac cot loi:** `processed_image` chi dung noi bo cho OCR. Moi anh luu ra dataset deu la crop tu `original_image`.

---

## Cai dat

### Yeu cau

- Python 3.10+
- Font NomNaTong (co san tai `FontDiffusion/fonts/NomNaTong-Regular.ttf`)
- Token API Kimhannom (dat trong file `.env`)

### Tao moi truong ao va cai thu vien

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lan sau mo lai project chi can kich hoat lai:

```bash
source .venv/bin/activate
```

### Cau hinh API

Tao file `.env` tai thu muc goc:

```
SN_OCR_TOKEN=<token_cua_ban>
```

API mac dinh: `https://kimhannom.fit.hcmus.edu.vn`
Doi domain: them `SN_DOMAIN=domain.khac.vn` vao `.env`.

---

## Cach chay

### Chay toan bo pipeline

```bash
./run_pipeline.sh
```

### Chay tung buoc

```bash
# Buoc 0: Setup
python -m pipeline.step0_setup config/pipeline.yaml

# Buoc 1: Tach du lieu tu PDF
python -m pipeline.step1_extract config/pipeline.yaml CacThanhTruyen2

# Buoc 2: Can chinh Levenshtein
python -m pipeline.step2_align config/pipeline.yaml CacThanhTruyen2

# Buoc 3: Gan nhan 3 tang
python -m pipeline.step3_label config/pipeline.yaml CacThanhTruyen2

# Buoc 4: Xuat dataset
python -m pipeline.step4_export config/pipeline.yaml
```

### Tuy chon run_pipeline.sh

```bash
./run_pipeline.sh --step 1                     # Chi chay buoc 1
./run_pipeline.sh --book CacThanhTruyen2       # Chi 1 sach
./run_pipeline.sh --step 3 --book CacThanhTruyen4   # Buoc 3, 1 sach
./run_pipeline.sh --config config/custom.yaml  # Config khac
```

---

## Kiem tra tung buoc (tests/)

Moi test chay pipeline roi in bao cao de nguoi dung kiem tra bang mat.
Chay **tuan tu** — moi buoc can output cua buoc truoc.

```bash
# Buoc 0: Kiem tra config, thu muc, dictionaries
python tests/test_step0.py

# Buoc 1: Extract 1 book, kiem tra anh goc vs denoised, crops, OCR cache
python tests/test_step1.py CacThanhTruyen2

# Buoc 2: Alignment, kiem tra match/deletion/insertion, syllable count
python tests/test_step2.py CacThanhTruyen2

# Buoc 3: Labeling, kiem tra tier, verify crop_file tro toi anh goc
python tests/test_step3.py CacThanhTruyen2

# Buoc 4: Export, kiem tra split, verify dataset khong chua anh da xu ly
python tests/test_step4.py
```

Moi test in **PASS/FAIL** cho cac diem quan trong. Doc output va mo file thu cong de xac nhan.

---

## Push code len GitHub

```bash
./push.sh "noi dung commit"
```

---

## Cau truc du an

```
GanNhanOCR/
├── config/
│   └── pipeline.yaml              # Cau hinh trung tam
│
├── lib/                            # Thu vien modules
│   ├── text_utils.py               # Text cleaning, saint names, syllables
│   ├── dictionary.py               # Dict loading, reverse lookup, specificity
│   ├── pdf_parser.py               # PDF page classification, image/text extraction
│   ├── image_processing.py         # Denoise, binarize, text box detection
│   ├── column_detector.py          # Column detection (projection + ruling lines)
│   ├── char_segmenter.py           # Character segmentation (projection profile)
│   ├── crop_cleaner.py             # Sauvola binarization, cleanup, 64x64 resize
│   ├── ocr_api.py                  # Kimhannom OCR API client
│   ├── qn_ocr.py                   # PaddleOCR + VietOCR for QN text
│   ├── alignment.py                # Levenshtein DP alignment
│   ├── ranker.py                   # 3-tier ranking (dict → similar → DINOv2)
│   └── dinov2_ranker.py            # DINOv2 cosine similarity ranker
│
├── pipeline/                       # Pipeline steps
│   ├── step0_setup.py
│   ├── step1_extract.py
│   ├── step2_align.py
│   ├── step3_label.py
│   └── step4_export.py
│
├── tests/                          # Kiem tra tung buoc
│   ├── test_step0.py
│   ├── test_step1.py
│   ├── test_step2.py
│   ├── test_step3.py
│   └── test_step4.py
│
├── tools/                          # Visualization
│   ├── verify_labels.py
│   ├── visualize_labels.py
│   └── visualize_results.py
│
├── embedding/                      # Deep Metric Learning (nang cao)
│   ├── train_embedding.py
│   ├── embed_ranker.py
│   └── iterative_refine.py
│
├── FontDiffusion/                  # Font style transfer model
│   └── fonts/NomNaTong-Regular.ttf
│
├── Dict/                           # Tu dien
│   ├── QuocNgu_SinoNom_TongHop3.csv
│   └── SinoNom_Similar_Dic_v2.csv
│
├── Data/                           # PDF dau vao (chi chua file goc)
│
├── prepared/                       # Output trung gian cua pipeline (moi book 1 folder)
│
├── dataset/                        # Ket qua cuoi cung
│   ├── CacThanhTruyen2/            #   tung book rieng
│   ├── CacThanhTruyen4/
│   └── all/                        #   gop tat ca book
│
├── requirements.txt
├── run_pipeline.sh
├── push.sh
├── .env                            # Token API (KHONG commit)
└── README.md
```

---

## Cau hinh (config/pipeline.yaml)

```yaml
books:
  - name: CacThanhTruyen2
    pdf: Data/CacThanhTruyen2.pdf
    reocr: false          # true neu can re-OCR trang QN bang PaddleOCR+VietOCR

paths:
  data_dir: prepared
  output_dir: dataset
  qn_to_nom_dict: Dict/QuocNgu_SinoNom_TongHop3.csv
  similar_dict: Dict/SinoNom_Similar_Dic_v2.csv
  font_path: FontDiffusion/fonts/NomNaTong-Regular.ttf

step1:
  dpi: 300
  denoise: true
  crop_size: 64
  sauvola_k: 0.2
  use_ocr_api: true       # Dung Kimhannom API

step3:
  use_dinov2: true         # DINOv2 cho tang 3
  dinov2_threshold: 0.75

step4:
  min_samples_per_class: 3
```

---

## Chi tiet tung buoc

### Buoc 1 — Tach du lieu

1. **Phan loai trang**: `is_image_page()` phan biet trang Han Nom vs trang Quoc Ngu
2. **Trich xuat anh**: Render PDF → `pages/` (anh goc) + `pages_denoised/` (khu nhieu)
3. **OCR trang Nom**: Upload anh **khu nhieu** len Kimhannom API → bbox + transcription
4. **Phan tach ky tu**: Projection Profile cat tung ky tu tu cot
5. **Crop ky tu**: Crop tu anh **goc** (`crops/`) + Sauvola cleanup (`crops_cleaned/`)
6. **Trich xuat text QN**: Doc text tu PDF (hoac PaddleOCR + VietOCR khi `reocr=true`)
7. **Normalize syllables**: Tach ten thanh (dominhgo → do minh co) ngay tu buoc nay

### Buoc 2 — Can chinh Levenshtein

Can chinh N ky tu detected voi M am tiet QN bang Levenshtein DP:

| Chieu cao ky tu | Chi phi xoa | Ly do |
|-----------------|-------------|-------|
| < 30% median   | 0.3         | Nhieu/dau |
| 30-50% median  | 0.6         | Ky tu nho |
| >= 50% median  | 1.2         | Ky tu that |

### Buoc 3 — Gan nhan 3 tang

**Tang 1: Tu dien song huong**
- QN → Nom: tra tap ung vien S2
- Nom OCR → QN: tra nguoc S1
- S2 co 1 ung vien duy nhat → matched (DEN)
- OCR nam trong S2 VA QN nam trong S1 → matched (DEN)

**Tang 2: Chu tuong tu**
- Tra SinoNom_Similar_Dic cho chu OCR
- Tim chu tuong tu nam trong S2 → matched (DEN)

**Tang 3: So khop anh (DINOv2)**
- Tap ung vien = S2 ∪ danh sach tuong tu
- Render font NomNaTong → DINOv2 embedding → cosine similarity
- Score > 0.75 → matched (DEN), nguoc lai → unmatched (DO)

### Buoc 4 — Xuat dataset

- Gop labels.csv tu tat ca sach
- Loc crop loi (trang, qua den, kich thuoc bat thuong)
- Loai class hiem (< 3 mau)
- Xuat rieng tung book + gop tat ca vao `all/`

---

## He thong matched/unmatched

| Trang thai | Mau | Y nghia |
|------------|-----|---------|
| `matched = True` | **DEN** | Nhan dung (xac nhan qua tu dien hoac visual) |
| `matched = False` | **DO** | Nhan sai hoac khong xac nhan duoc |

Khong dung confidence score. Chi co 2 trang thai.

---

## Format dataset

### labels.csv

```csv
crop_file,nom_char,unicode,syllable,matched,tier,bbox,page,source
crops/page_0012/col01_char000.png,經,U+7D93,kinh,True,1,"[100,200,150,260]",page_0012,CacThanhTruyen2
```

| Truong | Mo ta |
|--------|-------|
| `crop_file` | Duong dan anh crop **goc** (tu `crops/`, khong phai `crops_cleaned/`) |
| `nom_char` | Ky tu Nom (Unicode) |
| `unicode` | Ma Unicode `U+XXXX` |
| `syllable` | Am doc Quoc ngu |
| `matched` | `True` (den) / `False` (do) |
| `tier` | Tang da gan: 1 (dict), 2 (similar), 3 (visual), 0 (none) |
| `bbox` | Bounding box `[x1,y1,x2,y2]` |
| `page` | Ten trang |
| `source` | Ten bo sach |

### Output cuoi cung

```
dataset/
├── CacThanhTruyen2/       # Du lieu rieng tung book
│   ├── labels.csv
│   ├── class_map.json
│   └── metadata.json
├── CacThanhTruyen4/
│   ├── labels.csv
│   ├── class_map.json
│   └── metadata.json
└── all/                   # Gop tat ca book
    ├── labels.csv
    ├── class_map.json
    └── metadata.json
```

---

## Visualization

```bash
# Ve bbox + nhan len anh goc (den=dung, do=sai)
python tools/visualize_labels.py Data/prepared/CacThanhTruyen2

# Chi 1 trang
python tools/visualize_labels.py Data/prepared/CacThanhTruyen2 --page 12

# Xuat PDF
python tools/visualize_labels.py Data/prepared/CacThanhTruyen2 --pdf output.pdf
```

---

## Cong nghe su dung

| Thanh phan | Cong nghe |
|------------|-----------|
| Tach PDF | PyMuPDF |
| Khu nhieu | Morph. Closing 51x51 + Contrast Stretching |
| Nhi phan hoa | Otsu + Sauvola (k=0.2, R=128) |
| Phat hien cot | Vertical Projection + Ruling Line Morphology |
| Phan tach ky tu | Horizontal Projection + Merge/Split |
| OCR Nom | Kimhannom API (kimhannom.fit.hcmus.edu.vn) |
| OCR Quoc Ngu | PaddleOCR + VietOCR |
| Can chinh | Levenshtein DP (variable deletion cost) |
| Tra tu dien | Song huong QN↔Nom + Fuzzy matching |
| Chu tuong tu | SinoNom_Similar_Dic |
| So khop anh | DINOv2 ViT-S/14 cosine similarity |
| Xuat dataset | Merge + Quality filter + Class map |

---

## Tai lieu tham khao

- [New-SinoNom Dataset (Kaggle)](https://www.kaggle.com/datasets/5c09041f61f1bd528a0281281a55ed4ddb6b4aa1c83bdb0c0e21a1553339ad32)
- [SinoNom Similarity Retrieval (Kaggle)](https://www.kaggle.com/code/hongduyhng/sinonom-img-to-img-similarity-retrieval)
- [FontDiffuser / Font Architect (HuggingFace)](https://huggingface.co/dzungpham/font-architect)
- [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2)
