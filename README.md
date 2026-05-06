# GanNhanOCR

**Tu dong gan nhan Unicode cho kho ngu lieu Han Nom viet tay tu ban dich Quoc ngu**

He thong xu ly sach Han Nom viet tay (PDF) ket hop ban dich Quoc ngu, tu dong
gan nhan Unicode cho tung ky tu Nom thong qua 3 tang tra cuu: tu dien song
huong, ky tu tuong tu, va so khop anh DINOv2 voi anh tham chieu sinh boi
FontDiffusion.

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
│     • Tang 3: So khop anh (DINOv2 cosine vs anh sinh boi FontDiffusion)
│
└── Buoc 4: Export ─── Gop sach → Loc chat luong → dataset cuoi cung
```

**Sinh anh tham chieu (FontDiffusion):** chay 1 lan tren Kaggle GPU bang
[`kaggle_diffusion/diffusion_run.ipynb`](kaggle_diffusion/diffusion_run.ipynb)
de tao **universal cache** ~21,837 ky tu chu Nom (toan bo CJK ranges trong font
NomNaTong). Cache nay duoc luu len HuggingFace Hub roi keo ve
`prepared/_universal_fd_cache/`. Tat ca 6 cuon sach dung chung 1 cache nay,
khong can sinh lai.

**Nguyen tac cot loi:** `processed_image` chi dung noi bo cho OCR. Moi anh luu
ra dataset deu la crop tu `original_image`.

---

## Cai dat

### Yeu cau

- Python 3.10+
- Font NomNaTong (co san tai `font_diffusion/fonts/NomNaTong-Regular.ttf`)
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

## Quy trinh chay (lan dau)

### Buoc A — Sinh universal fd_cache tren Kaggle (1 lan duy nhat)

Lam theo [`kaggle_diffusion/README.md`](kaggle_diffusion/README.md):

1. Chay `python kaggle_diffusion/build_char_universe.py` tren may local de tao
   `kaggle_diffusion/exports/char_universe.txt` (~21,837 ky tu).
2. Tao 1 dataset repo tren HuggingFace Hub, dat ten tuy y (mac dinh notebook
   dung `mdnt571/gannhanocr-universal-fd-cache`).
3. Mo `kaggle_diffusion/diffusion_run.ipynb` tren Kaggle (GPU T4 x2).
4. Notebook tu sinh, push moi 500 ky tu, resume duoc neu Kaggle reset
   (~10–12h tong).
5. Sau khi xong, keo cache ve may local:
   ```bash
   huggingface-cli download mdnt571/gannhanocr-universal-fd-cache \
     --repo-type=dataset \
     --local-dir prepared/_universal_fd_cache/
   ```

### Buoc B — Chay pipeline 6 cuon

```bash
./run_pipeline.sh                    # tat ca 6 cuon, du 5 buoc 0→4
```

Hoac chay tung phan:

```bash
./run_pipeline.sh --step 1                          # chi buoc 1, tat ca sach
./run_pipeline.sh --book CacThanhTruyen2            # 1 sach, du 5 buoc
./run_pipeline.sh --step 3 --book CacThanhTruyen4   # buoc 3, 1 sach
./run_pipeline.sh --config config/pipeline.yaml     # chi dinh config
```

Hoac goi tung step Python:

```bash
python -m pipeline.step0_setup config/pipeline.yaml
python -m pipeline.step1_extract config/pipeline.yaml CacThanhTruyen2
python -m pipeline.step2_align   config/pipeline.yaml CacThanhTruyen2
python -m pipeline.step3_label   config/pipeline.yaml CacThanhTruyen2
python -m pipeline.step4_export  config/pipeline.yaml
```

---

## Cau truc du an

```
GanNhanOCR/
├── config/
│   └── pipeline.yaml             # Cau hinh trung tam (6 sach)
│
├── core/                         # Thu vien shared (import boi pipeline/)
│   ├── image/                    # Crop, denoise, column/char segmentation
│   ├── pdf/                      # PDF parser
│   ├── ocr/                      # Kimhannom API client + QN OCR
│   ├── ranking/                  # Ranker 3-tang + DINOv2 + FontDiffusion
│   └── text/                     # Dictionary, syllable utils
│
├── pipeline/                     # 5 buoc thuc thi
│   ├── step0_setup.py
│   ├── step1_extract.py
│   ├── step2_align.py
│   ├── step3_label.py
│   └── step4_export.py
│
├── kaggle_diffusion/             # One-shot generator universal fd_cache
│   ├── README.md
│   ├── build_char_universe.py    # Trich xuat 21k ky tu tu font NomNaTong
│   ├── extract_book_chars.py     # Trich xuat ky tu rieng tung sach (optional)
│   ├── run_local_sanity.py       # Sanity check truoc khi day Kaggle
│   ├── diffusion_run.ipynb       # Notebook chinh (universal cache)
│   ├── diffusion_per_book.ipynb  # Bien the per-book (neu can style rieng)
│   └── exports/                  # char_universe.txt + .json
│
├── font_diffusion/               # Submodule FontDiffuser (model + ckpt)
│
├── deep_seek-OCR/                # Nhanh nghien cuu OCR DeepSeek (chua tich hop)
│
├── Data/                         # PDF goc 6 cuon (KHONG commit)
├── Dict/                         # Tu dien QN↔Nom + Similar dic
├── prepared/                     # Output trung gian (S0→S3) per-book
│   ├── _universal_fd_cache/      # Universal FontDiffusion cache (tu Kaggle)
│   ├── CacThanhTruyen2/
│   │   ├── pages/                # Anh trang goc
│   │   ├── pages_denoised/       # Anh khu nhieu (chi dung cho OCR)
│   │   ├── detected/             # Crops + bbox + OCR cache
│   │   ├── aligned/              # Levenshtein alignment JSON
│   │   ├── labeled/              # labels.csv tung sach
│   │   └── fd_cache/             # (optional) per-book FD cache override
│   └── ...
├── dataset/                      # Output cuoi cung (S4)
│   ├── CacThanhTruyen2/
│   ├── ...
│   └── all/                      # Gop tat ca sach
│
├── requirements.txt
├── run_pipeline.sh               # Orchestrator chinh
├── push.sh                       # Push len GitHub
├── .env                          # Token API (KHONG commit)
└── README.md
```

---

## Cau hinh (config/pipeline.yaml)

```yaml
books:
  - { name: CacThanhTruyen2,   pdf: Data/CacThanhTruyen2.pdf,   reocr: false }
  - { name: CacThanhTruyen4,   pdf: Data/CacThanhTruyen4.pdf,   reocr: false }
  - { name: CacThanhTruyen11,  pdf: Data/CacThanhTruyen11.pdf,  reocr: false }
  - { name: SachThanhTruyen2,  pdf: Data/SachThanhTruyen2.pdf,  reocr: false }
  - { name: SachThanhTruyen4,  pdf: Data/SachThanhTruyen4.pdf,  reocr: false }
  - { name: SachThanhTruyen11, pdf: Data/SachThanhTruyen11.pdf, reocr: false }

paths:
  data_dir: prepared
  output_dir: dataset
  qn_to_nom_dict: Dict/QuocNgu_SinoNom_TongHop3.csv
  similar_dict:   Dict/SinoNom_Similar_Dic_v2.csv
  font_path:      font_diffusion/fonts/NomNaTong-Regular.ttf
  fontdiffusion_ckpt:        font_diffusion/ckpt/PROD
  fontdiffusion_phase1_ckpt: font_diffusion/ckpt/PROD
  fd_cache_universal:        prepared/_universal_fd_cache

step1: { dpi: 300, denoise: true, crop_size: 64, sauvola_k: 0.2, use_ocr_api: true }
step2: { deletion_cost_small: 0.3, deletion_cost_medium: 0.6, deletion_cost_normal: 1.2 }
step3:
  use_dinov2: true
  dinov2_model: dinov2_vitb14_reg
  dinov2_threshold: 0.75
  use_fontdiffusion: true
  require_fontdiffusion: true     # tier 3 chi dung anh trong fd_cache
step4: { min_samples_per_class: 1 }
```

---

## Chi tiet tung buoc

### Buoc 1 — Tach du lieu

1. **Phan loai trang**: phan biet trang Han Nom vs trang Quoc Ngu
2. **Trich xuat anh**: Render PDF → `pages/` (anh goc) + `pages_denoised/` (khu nhieu)
3. **OCR trang Nom**: Upload anh khu nhieu len Kimhannom API → bbox + transcription
4. **Phan tach ky tu**: Projection Profile cat tung ky tu tu cot
5. **Crop ky tu**: Crop tu anh **goc** vao `crops/` + Sauvola cleanup vao `crops_cleaned/`
6. **Trich xuat text QN**: Doc text tu PDF (hoac PaddleOCR + VietOCR khi `reocr=true`)
7. **Normalize syllables**: Tach ten thanh ngay tu buoc nay

### Buoc 2 — Can chinh Levenshtein

| Chieu cao ky tu | Chi phi xoa | Ly do |
|-----------------|-------------|-------|
| < 30% median   | 0.3         | Nhieu/dau |
| 30-50% median  | 0.6         | Ky tu nho |
| >= 50% median  | 1.2         | Ky tu that |

### Buoc 3 — Gan nhan 3 tang

**Tang 1: Tu dien song huong (QN↔Nom)** — neu S2 co duy nhat 1 ung vien hoac
OCR ∈ S2 va QN ∈ S1 → matched.

**Tang 2: Chu tuong tu** — tra `SinoNom_Similar_Dic`, tim chu tuong tu nam
trong S2 → matched.

**Tang 3: So khop anh (DINOv2 + FontDiffusion)** — tap ung vien la S2; voi moi
ung vien, lay anh tu `fd_cache` (universal hoac per-book), tinh cosine
similarity DINOv2 voi crop. Score > 0.75 → matched, nguoc lai → unmatched.

Voi `require_fontdiffusion: true`, tang 3 **CHI** dung anh trong fd_cache;
khong fallback ve render tu font, dam bao do dong nhat ve style.

### Buoc 4 — Xuat dataset

- Gop labels.csv tu tat ca sach
- Loc crop loi (trang, qua den, kich thuoc bat thuong)
- Loai class hiem (`min_samples_per_class`)
- Xuat rieng tung book + gop tat ca vao `all/`

---

## He thong matched/unmatched

| Trang thai | Mau | Y nghia |
|------------|-----|---------|
| `matched = True`  | **DEN** | Nhan dung (xac nhan qua tu dien hoac visual) |
| `matched = False` | **DO**  | Nhan sai hoac khong xac nhan duoc |

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
| `nom_char`  | Ky tu Nom (Unicode) |
| `unicode`   | Ma Unicode `U+XXXX` |
| `syllable`  | Am doc Quoc ngu |
| `matched`   | `True` (den) / `False` (do) |
| `tier`      | Tang da gan: 1 (dict), 2 (similar), 3 (visual), 0 (none) |
| `bbox`      | Bounding box `[x1,y1,x2,y2]` |
| `page`      | Ten trang |
| `source`    | Ten bo sach |

---

## Push code len GitHub

```bash
./push.sh "noi dung commit"
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
| Sinh anh tham chieu | FontDiffuser (NomNaTong style transfer) tren Kaggle T4 |
| So khop anh | DINOv2 ViT-B/14 + registers, cosine similarity |
| Xuat dataset | Merge + Quality filter + Class map |

---

## Tai lieu tham khao

- [New-SinoNom Dataset (Kaggle)](https://www.kaggle.com/datasets/5c09041f61f1bd528a0281281a55ed4ddb6b4aa1c83bdb0c0e21a1553339ad32)
- [SinoNom Similarity Retrieval (Kaggle)](https://www.kaggle.com/code/hongduyhng/sinonom-img-to-img-similarity-retrieval)
- [FontDiffuser / Font Architect (HuggingFace)](https://huggingface.co/dzungpham/font-architect)
- [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2)
