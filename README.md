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
`prepared/_universal_fd_cache/`. Tat ca 3 cuon sach dung chung 1 cache nay,
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

Tao file `.env` tai thu muc goc. Co 2 cach cau hinh OCR token:

**Cach 1 — Auto-login (khuyen nghi)**

Code se tu POST email/password den `/account/login` cua HCMUS moi khi
cache idToken con <5 phut, lay token moi (60 min TTL). Pipeline chay mai
khong can thao tac tay.

```
SN_OCR_USERNAME=your_email@example.com   # tai khoan dang ky tools.clc.hcmus.edu.vn
SN_OCR_PASSWORD=your_password
```

Vi sao khong dung Firebase refresh_token? HCMUS dat Firebase API key o
server-side (khong lo cho client) nen khong the goi `securetoken.googleapis.com`
truc tiep. Auto-login don gian va hieu qua hon — co the goi /account/login
bat ky luc nao de lay token moi.

**Cach 2 — Manual token (rotate moi 1 gio)**

Neu khong muon luu password trong `.env`, paste idToken thu cong:

```
SN_OCR_TOKEN=eyJhbGciOiJSUzI1NiIs...  # Firebase ID token, het han sau 1h
```

Lay tu DevTools: F12 -> Storage -> Cookies -> `tools.clc.hcmus.edu.vn` ->
chon cookie `token` -> copy full value. Sau 1 gio phai lay lai.

**Check trang thai token:**

```sh
python3 scripts/check_ocr_token.py
```

In ra `idToken` con bao lau, auto-login co dang active khong.

**Doi domain:** them `SN_DOMAIN=domain.khac.vn` vao `.env`.
API mac dinh: `https://tools.clc.hcmus.edu.vn`.

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

### Buoc B — Chay pipeline 3 cuon

```bash
./run_pipeline.sh                    # tat ca 3 cuon, 3 buoc: 0 (setup) -> 1 (extract) -> 2 (build dataset)
```

Hoac chay tung phan:

```bash
./run_pipeline.sh --step 1                          # chi buoc 1 (extract OCR), tat ca sach
./run_pipeline.sh --step 2                          # chi buoc 2 (build dataset)
./run_pipeline.sh --book SachThanhTruyen2            # 1 sach, du 3 buoc
./run_pipeline.sh --config config/pipeline.yaml     # chi dinh config
```

Buoc 2 = goi truc tiep pipeline/align_engine (banded-DP align + consensus GOLD/SILVER/
SYLLABLE/REVIEW, S3 = encoder Nom da train, + xuat 3 chuan quoc te):

```bash
# Step 2 lives in the main package pipeline/align_engine/ (moved out of evaluation/).
.venv/bin/python -m pipeline.align_engine.build_dataset --config config/pipeline.yaml --use-s3 --strict
.venv/bin/python -m pipeline.align_engine.to_standard    # HF imagefolder + Frictionless + Croissant

# Phase 0 (ground truth) and Phase 1 (remediation) run on the built dataset:
.venv/bin/python -m pipeline.remediation apply           # fix proven errors -> labels_remediated.csv
.venv/bin/python -m pipeline.ground_truth rank           # rank crops by error suspicion for audit
```

> ⚠️ Cac lenh cu `python -m pipeline.step2_align / step3_label / step4_export`
> da **NGHI (RETIRED)** — gop het vao Buoc 2 o tren. DINOv2 da bi **tat**
> (thay bang encoder Nom da train).
>
> **S3 checkpoints (bat buoc cho SILVER):** `gannhanocr-fd/` (89,898-glyph font bank)
> va `nom-embed/` (ArcFace encoder) la gitlink **chua dang ky trong `.gitmodules`** —
> clone sach se rong. Lay ban sao 2 thu muc nay truoc khi build; neu thieu, dung
> `--strict` de **fail loud** thay vi SILVER am tham tut het thanh REVIEW.

---

## Cau truc du an

```
GanNhanOCR/
├── config/
│   └── pipeline.yaml             # Cau hinh trung tam (3 sach)
│
├── core/                         # Thu vien shared (import boi pipeline/)
│   ├── image/                    # Crop, denoise, column/char segmentation
│   ├── pdf/                      # PDF parser
│   ├── ocr/                      # Kimhannom API client + QN OCR
│   ├── ranking/                  # Ranker 3-tang + DINOv2 + FontDiffusion
│   └── text/                     # Dictionary, syllable utils
│
├── pipeline/                     # Buoc 0-1 active; step2/3/4 da NGHI
│   ├── step0_setup.py            #   (active) setup & validate
│   ├── step1_extract.py          #   (active) PDF -> crop khung -> OCR 9 cot
│   ├── step2_align.py            #   [RETIRED] thay boi pipeline/align_engine/build_dataset.py
│   ├── step3_label.py            #   [RETIRED] duong DINOv2, da tat
│   └── step4_export.py           #   [RETIRED] thay boi pipeline/align_engine/to_standard.py
│
├── pipeline/align_engine/           # *** Buoc 2 hien hanh (build dataset) ***
│   ├── build_dataset.py          #   align banded-DP + consensus + crops + labels.csv
│   ├── to_standard.py            #   xuat HF / Frictionless / Croissant
│   ├── visual_signal.py          #   S3 = encoder Nom da train (NomEncoder)
│   ├── nom_classifier/           #   train encoder Nom (Kaggle P100)
│   └── FLOW.md                   #   mo ta flow chi tiet
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
├── Data/                         # PDF goc 3 cuon (KHONG commit)
├── Dict/                         # Tu dien QN↔Nom + Similar dic
├── prepared/                     # Output trung gian (S0→S3) per-book
│   ├── _universal_fd_cache/      # Universal FontDiffusion cache (tu Kaggle)
│   ├── SachThanhTruyen2/
│   │   ├── pages/                # Anh trang goc
│   │   ├── pages_denoised/       # Anh khu nhieu (chi dung cho OCR)
│   │   ├── detected/             # Crops + bbox + OCR cache
│   │   ├── aligned/              # Levenshtein alignment JSON
│   │   ├── labeled/              # labels.csv tung sach
│   │   └── fd_cache/             # (optional) per-book FD cache override
│   └── ...
├── dataset/                      # Output cuoi cung (S4)
│   ├── SachThanhTruyen2/
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
  - { name: SachThanhTruyen2,  pdf: Data/SachThanhTruyen2.pdf,  reocr: true }
  - { name: SachThanhTruyen4,  pdf: Data/SachThanhTruyen4.pdf,  reocr: true }
  - { name: SachThanhTruyen11, pdf: Data/SachThanhTruyen11.pdf, reocr: true }

paths:
  data_dir: prepared
  output_dir: dataset
  qn_to_nom_dict: Dict/QuocNgu_SinoNom.csv
  similar_dict:   Dict/SinoNom_Similar.csv
  font_path:      font_diffusion/fonts/NomNaTong-Regular.ttf
  fontdiffusion_ckpt:        font_diffusion/ckpt/PROD
  fontdiffusion_phase1_ckpt: font_diffusion/ckpt/PROD
  fd_cache_universal:        prepared/_universal_fd_cache

step1: { dpi: 300, denoise: true, crop_size: 64, sauvola_k: 0.2, use_ocr_api: true }
step2: { deletion_cost_small: 0.3, deletion_cost_medium: 0.6, deletion_cost_normal: 1.2 }
# step3/step4 = duong DINOv2/export cu, da NGHI. use_dinov2: false (encoder Nom
# da train thay the). Giu lai cho doi chieu trong luan van, khong xoa.
step3:
  use_dinov2: false               # TAT — thay bang encoder Nom da train (S3)
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

### Buoc 2 — Build dataset (align_engine)  ·  THAY cho Buoc 2/3/4 cu

> Day la flow hien hanh. Mo ta day du: `pipeline/align_engine/README.md`.

`pipeline/align_engine/build_dataset.py --use-s3` lam tat ca trong 1 buoc:

1. **Can chinh banded-DP neo tu dien** (thay ghep theo index cu): chi phi xoa
   theo chieu cao ky tu (<30% median = 0.3 / 30-50% = 0.6 / >=50% = 1.2).
2. **3 tin hieu** moi cap Nom↔QN: S1 = ky tu OCR SinoNom · S2 = tu dien
   QN↔Nom + chu tuong tu (`SinoNom_Similar`) · S3 = **so khop anh bang
   encoder Nom da train** (ArcFace) voi glyph FontDiffusion cua ung vien.
   *(DINOv2 da bi tat — khong phan biet duoc chu Nom; xem
   `pipeline/align_engine/README.md`.)*
3. **Tang dong thuan**: GOLD (tu dien xac nhan, char) · SILVER (S3 sua thi giac,
   char) · SYLLABLE (vay muon nhat quan giua cac trang, am tiet) · REVIEW.
4. **Re-segment cot + sua bbox khung**, crop tu anh goc -> `dataset_out/`.
5. **Xuat 3 chuan quoc te** (`to_standard.py`): HuggingFace imagefolder,
   Frictionless Data Package, MLCommons Croissant.

---

## He thong matched/unmatched

| Trang thai | Mau | Y nghia |
|------------|-----|---------|
| `matched = True`  | **DEN** | Nhan dung (xac nhan qua tu dien hoac visual) |
| `matched = False` | **DO**  | Nhan sai hoac khong xac nhan duoc |

Khong dung confidence score. Chi co 2 trang thai.

---

## Format dataset

### labels.csv (align_engine — 20 cot)

Xuat tai `dataset_out/labels.csv`. Header thuc te:

```csv
image,book,page,column,ocr_char,syllable,label,unicode,label_level,tier,rule,s3_cosine,ink_pct,crop_w,crop_h,image_md5,seg_flag,split,split_group,bbox
```

| Truong | Mo ta |
|--------|-------|
| `image` | Duong dan anh crop (goc) |
| `ocr_char` | Ky tu OCR SinoNom doc duoc (S1) |
| `label` / `unicode` | Nhan cuoi (chu Nom) + ma `U+XXXX` |
| `syllable` | Am doc Quoc ngu |
| `label_level` | `char` (tung chu) hoac `syllable` (am tiet) |
| `tier` | **GOLD** (tu dien xac nhan) · **SILVER** (S3 sua thi giac) · **SYLLABLE** (vay muon nhat quan) · **REVIEW** (can soat) |
| `rule` | Luat sinh nhan (vd dict_confirm, similar_bridge, below_visual_threshold...) |
| `s3_cosine` | Cosine encoder Nom train (S3) — co khi dung SILVER |
| `split` / `split_group` | train/val/test (chia theo nhom book·page·column, chong ro ri) |
| `bbox` | Bounding box `[x1,y1,x2,y2]` |

Phan bo hien tai: GOLD 51.195 · SILVER 6.747 · SYLLABLE 5.486 · REVIEW 18.840
(tong 82.268). 3 ban chuan quoc te kem theo trong `dataset_out/` (xem `to_standard.py`).

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
| So khop anh (S3) | **Encoder Nom da train (ResNet + ArcFace)** — cosine vs glyph FD. *(DINOv2 da tat: khong phan biet duoc Nom)* |
| Xuat dataset | labels.csv + 3 chuan: HF imagefolder / Frictionless / Croissant |

---

## Tai lieu tham khao

- [New-SinoNom Dataset (Kaggle)](https://www.kaggle.com/datasets/5c09041f61f1bd528a0281281a55ed4ddb6b4aa1c83bdb0c0e21a1553339ad32)
- [SinoNom Similarity Retrieval (Kaggle)](https://www.kaggle.com/code/hongduyhng/sinonom-img-to-img-similarity-retrieval)
- [FontDiffuser / Font Architect (HuggingFace)](https://huggingface.co/dzungpham/font-architect)
- [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2)
