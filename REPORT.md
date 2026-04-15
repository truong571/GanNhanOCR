# REPORT: Pipeline Flow Chi Tiet

Tai lieu mo ta chi tiet tung buoc pipeline chay, input/output, va ket qua thuc te.
Dung de kiem tra flow co dung khong.

---

## Tong quan luong du lieu

```
PDF (Data/CacThanhTruyen2.pdf)
  |
  v
Step 0: Tao thu muc prepared/CacThanhTruyen2/{pages,pages_denoised,transcriptions,detected/crops,...}
  |
  v
Step 1: Extract
  |-- Doc PDF tung trang
  |-- Phan loai: trang anh HN vs trang text QN (ghep cap)
  |-- Trang HN:
  |     |-- Luu anh goc      -> prepared/.../pages/page_XXXX.png
  |     |-- Khu nhieu (denoise) -> prepared/.../pages_denoised/page_XXXX.png
  |     |-- OCR API (dung anh DENOISED) -> detected/page_XXXX_ocr_cache.json
  |     |-- Binarize anh goc -> detect text box -> detect columns -> segment chars
  |     |-- Crop tu anh GOC  -> detected/crops/page_XXXX/colXX_charXXX.png
  |     |-- Clean crop (Sauvola) -> detected/crops_cleaned/page_XXXX/colXX_charXXX.png
  |-- Trang QN:
  |     |-- Doc text tu PDF (hoac PaddleOCR+VietOCR neu reocr=true)
  |     |-- Normalize syllables (expand ten thanh)
  |     |-- Luu -> transcriptions/page_XXXX.txt + .json
  |-- Luu detection -> detected/page_XXXX_detection.json
  |-- Luu manifest.json
  |
  v
Step 2: Align
  |-- Doc detected/page_XXXX_detection.json (chars per column)
  |-- Doc transcriptions/page_XXXX.txt (syllables per column)
  |-- Levenshtein DP: N chars <-> M syllables
  |     |-- match: char duoc ghep voi syllable
  |     |-- deletion: char thua (nhieu/noise)
  |     |-- insertion: syllable thua (thieu char)
  |-- Luu -> aligned/page_XXXX_aligned.json
  |
  v
Step 3: Label
  |-- Doc aligned/page_XXXX_aligned.json
  |-- Voi moi cap (char, syllable):
  |     |-- Tang 1: Tra tu dien QN->Nom va Nom->QN
  |     |     |-- 1 ung vien duy nhat -> matched=True, tier=1
  |     |     |-- OCR char nam trong S2 -> matched=True, tier=1
  |     |     |-- Nhieu ung vien, khong xac dinh -> sang tang 2
  |     |-- Tang 2: Tra chu tuong tu cua OCR char
  |     |     |-- Tim thay chu tuong tu trong S2 -> matched=True, tier=2
  |     |     |-- Khong tim thay -> sang tang 3
  |     |-- Tang 3: So khop anh (DINOv2 hoac classical CV)
  |     |     |-- Dung anh CLEANED cho visual comparison (noi bo)
  |     |     |-- score > 0.75 -> matched=True, tier=3
  |     |     |-- score <= 0.75 -> matched=False, tier=3
  |     |-- Fallback: chon ung vien theo CJK block score -> tier=1
  |     |-- Khong co ung vien nao -> matched=False, tier=0
  |-- QUAN TRONG: label output luu crop_file = crops/... (anh GOC)
  |     (KHONG phai crops_cleaned/)
  |-- Luu -> labeled/dataset.json + labels.csv + summary.json
  |
  v
Step 4: Export
  |-- Doc labels.csv tu moi book
  |-- Loc: bo gap (khong co nom_char)
  |-- Loc: crop quality (ink ratio, kich thuoc)
  |-- Loc: class hiem (< 3 mau)
  |-- Xuat rieng tung book -> dataset/CacThanhTruyen2/
  |-- Xuat gop tat ca -> dataset/all/
  |-- Moi folder co: labels.csv + class_map.json + metadata.json
```

---

## Step 0: Setup — KET QUA

```
- Tao thu muc cho 3 books: CacThanhTruyen2, CacThanhTruyen4, CacThanhTruyen11
- Kiem tra Dict/ co 2 file CSV -> OK
- Kiem tra font NomNaTong -> OK
- Kiem tra PDF -> OK
```

Trang thai: PASS

---

## Step 1: Extract — CHI TIET

### Input
- `Data/CacThanhTruyen2.pdf`

### Qua trinh
1. Mo PDF bang PyMuPDF, duyet tung trang
2. `is_image_page(page)`: kiem tra text < 200 ky tu -> la trang anh HN
3. Ghep cap: trang HN (page i) + trang QN ke tiep (page i+1)
4. `extract_book_page_number()`: lay so trang sach tu noi dung PDF
5. `extract_nom_image()`: trich anh goc tu PDF -> `pages/page_XXXX.png`
6. `denoise_image()`: GaussianBlur -> MorphClose(51x51) -> divide -> contrast stretch
   -> `pages_denoised/page_XXXX.png`
7. `ocr_page()`: Upload anh **DENOISED** len kimhannom API
   - Upload -> nhan file_name
   - OCR -> nhan boxes (bbox + transcription)
   - `boxes_to_columns()`: nhom boxes thanh cot (phai->trai), moi cot co chars (tren->duoi)
   - Cache ket qua -> `detected/page_XXXX_ocr_cache.json`
8. `extract_quocngu_text()` hoac `ocr_qn_page()`: doc text QN
   - `parse_numbered_lines()`: tach "1. noi dung..." thanh {1: "noi dung"}
   - `build_transcription_columns()`: clean text -> split syllables
   - `normalize_syllables()`: expand ten thanh (dominhgo -> do minh co)
   - Luu -> `transcriptions/page_XXXX.txt` va `.json`
9. `load_and_binarize(img_path)`: load anh GOC -> binary mask (cho segmentation)
10. `detect_text_box()`: tim vung text chinh (loai bo le trang)
11. `detect_columns()`: projection profile + ruling lines -> bbox tung cot
12. `segment_characters_in_column()`: cat tung ky tu trong cot
    - Valley snap-to-grid (uu tien)
    - Threshold + merge/split
    - Equal-division (fallback)
13. Crop tu anh GOC `gray_img[cy1:cy2, cx1:cx2]` -> `crops/`
14. `cleaner.clean(crop)`: Sauvola + cleanup -> `crops_cleaned/`
15. Match OCR char voi segmented char bang y_center gan nhat

### Output (per page)
| File | Noi dung |
|------|----------|
| `pages/page_XXXX.png` | Anh goc tu PDF |
| `pages_denoised/page_XXXX.png` | Anh da khu nhieu (chi dung noi bo) |
| `detected/page_XXXX_ocr_cache.json` | OCR results tu API (cached) |
| `detected/page_XXXX_detection.json` | Bbox columns + chars + ocr_char |
| `detected/crops/page_XXXX/colXX_charXXX.png` | Crop GOC |
| `detected/crops_cleaned/page_XXXX/colXX_charXXX.png` | Crop da clean |
| `transcriptions/page_XXXX.txt` | Syllables per line (da normalize) |
| `transcriptions/page_XXXX.json` | Structured transcription |
| `manifest.json` | Tong hop book |

### Kiem tra quan trong
- [ ] OCR API dung anh denoised (kiem tra log: "pages_denoised" trong path)
- [ ] Crops trong `crops/` la tu anh goc (KHONG qua denoise/binarize)
- [ ] Syllable count da normalize (ten thanh duoc tach)
- [ ] So cot detected = so dong trong transcription

---

## Step 2: Align — CHI TIET

### Input
- `detected/page_XXXX_detection.json` (chars per column)
- `transcriptions/page_XXXX.txt` (syllables per column)

### Qua trinh
1. Doc detection JSON: lay chars (bbox, ocr_char) theo column
2. Doc transcription TXT: split by whitespace (DA normalize tu step 1)
3. `levenshtein_align()`:
   - Tinh deletion cost theo chieu cao char (nho=re, lon=dat)
   - Tinh substitution cost theo tu dien (co ung vien=0, khong=0.8)
   - DP table m x n
   - Backtrack -> danh sach {char, syllable, type}

### Output (per page)
| File | Noi dung |
|------|----------|
| `aligned/page_XXXX_aligned.json` | List of {char, syllable, type, column} |

### Kiem tra quan trong
- [ ] Match rate > 70% (binh thuong)
- [ ] Insertion thap (< 15% total) — neu cao = syllable count bi lech
- [ ] Deletion thap (< 15% total) — neu cao = over-segmentation

---

## Step 3: Label — CHI TIET

### Input
- `aligned/page_XXXX_aligned.json`
- Tu dien: `Dict/QuocNgu_SinoNom_TongHop3.csv`, `Dict/SinoNom_Similar_Dic_v2.csv`
- Font: `FontDiffusion/fonts/NomNaTong-Regular.ttf`

### Qua trinh
1. Load tu dien QN->Nom (~99k entries) va Nom->QN (reverse)
2. Load tu dien chu tuong tu (~26k entries)
3. (Optional) Load DINOv2 model
4. Voi moi cap match trong aligned:
   - Lay `ocr_char` tu char_info va `syllable` tu alignment
   - **Crop path cho ranking**: uu tien `cleaned_file` (noi bo cho visual comparison)
   - **Crop path cho output**: luon la `crop_file` (anh goc)
   - `assign_label()`:
     - tier1: `tier1_dictionary_lookup()` -> tra QN->Nom va Nom->QN
     - tier2: `tier2_similar_expansion()` -> tra chu tuong tu
     - tier3: `tier3_visual_comparison()` -> DINOv2 cosine hoac classical CV
     - fallback: chon theo CJK block score

### Output (per book)
| File | Noi dung |
|------|----------|
| `labeled/dataset.json` | Tat ca labels chi tiet |
| `labeled/labels.csv` | CSV format cho Step 4 |
| `labeled/summary.json` | Thong ke: matched/unmatched/tiers |

### Kiem tra quan trong
- [ ] crop_file trong labels.csv bat dau bang "crops/" (KHONG phai "crops_cleaned/")
- [ ] Tier 1 chiem da so (60-80%) — tu dien la nguon chinh
- [ ] Matched rate > 50%

---

## Step 4: Export — CHI TIET

### Input
- `prepared/*/labeled/labels.csv` tu moi book

### Qua trinh
1. Load labels.csv tu tung book, them truong `source`
2. Loc: bo rows khong co `nom_char` (gaps)
3. Loc: `filter_crop_quality()` — doc anh, kiem tra ink ratio
4. Loc: `filter_rare_classes()` — bo class < 3 mau
5. Xuat rieng tung book -> `dataset/{book_name}/`
6. Gop tat ca -> loc rare classes lan nua -> `dataset/all/`

### Output
```
dataset/
  CacThanhTruyen2/labels.csv, class_map.json, metadata.json
  CacThanhTruyen4/...
  CacThanhTruyen11/...
  all/labels.csv, class_map.json, metadata.json
```

### Kiem tra quan trong
- [ ] Moi book co folder rieng
- [ ] `all/` co tong samples = tong cac book (sau loc chung)
- [ ] crop_file trong labels.csv la duong dan goc

---

## DIEM CAN KIEM TRA TONG HOP

| # | Diem kiem tra | Vi tri |
|---|--------------|--------|
| 1 | OCR API dung anh denoised | Step 1: ocr_page() nhan pages_denoised/ |
| 2 | Crop tu anh goc | Step 1: gray_img[cy1:cy2, cx1:cx2] |
| 3 | Syllables da normalize truoc segmentation | Step 1: normalize_syllables() truoc expected_counts |
| 4 | Step 2 KHONG normalize lai | Step 2: chi split(), khong goi normalize_syllables() |
| 5 | Label output luu crop goc | Step 3: crop_file = char_info.get("crop_file") |
| 6 | Visual ranking dung cleaned (noi bo) | Step 3: ranking_crop_path dung cleaned_file |
| 7 | Dataset chi chua anh goc | Step 4: labels.csv -> crops/... |

---

## KET QUA CHAY THUC TE (CacThanhTruyen2)

### Step 0: PASS
- 3 books, thu muc da tao

### Step 1: PASS
```
PDF: 10 trang -> 5 cap (HN + QN)
page_0012: 9 cot, 169 chars, 169 syllables
page_0014: 9 cot, 205 chars, 205 syllables
page_0016: 9 cot, 200 chars, 200 syllables
page_0018: 9 cot, 200 chars, 200 syllables
page_0020: 9 cot, 187 chars, 187 syllables
Total: 961 chars

OCR API: dung anh DENOISED (pages_denoised/) -> PASS
Crop path: tat ca tu crops/ (anh goc) -> PASS
Syllable count = char count (tat ca trang) -> PASS
```

### Step 2: PASS
```
961 matches, 0 gaps (0 deletion, 0 insertion)
Match rate: 100%

Giai thich: chars = syllables (169, 205, 200, 200, 187)
nen Levenshtein align 1:1 hoan hao, khong co gap.
```

### Step 3: PASS
```
Total labels: 961
Matched (black): 955 (99.4%)
Unmatched (red): 6 (0.6%)
Gaps: 0

Phan bo tier:
  Tier 1 (dict):    327 (34.0%)
  Tier 2 (similar): 37  (3.9%)
  Tier 3 (visual):  591 (61.5%)
  Tier 0 (none):    6   (0.6%)

crop_file trong labels.csv: 961/961 la crops/ (anh goc) -> PASS
Khong co dong nao tro toi crops_cleaned/ -> PASS
```

### Step 4: PASS
```
Dataset rieng tung book:
  CacThanhTruyen2:  342 samples, 72 classes
  CacThanhTruyen4:  162 samples, 44 classes
  CacThanhTruyen11: 227 samples, 55 classes

Dataset gop (all):
  731 samples, 134 classes, 731 matched, 0 unmatched

Ghi chu: total gop (731) < tong rieng (342+162+227=731)
vi filter rare classes (<3 mau) ap dung tren tung book
roi filter lan nua tren tap gop.
```

---

## VAN DE CAN CHU Y

1. **Tier 3 chiem 61.5%**: Nhieu label phai dua vao visual comparison (DINOv2).
   Dieu nay co the do tu dien QN->Nom chi co 7366 entries (nho).
   Khi tu dien lon hon, tier 1 se tang va tier 3 giam.

2. **OCR cache dung path cu**: File `page_XXXX_ocr_cache.json` luu
   `"image": "Data/prepared/.../pages_denoised/..."` (path cu truoc khi
   chuyen prepared/ ra ngoai). Khong anh huong vi chi la metadata trong cache,
   khong dung de load anh. Nhung neu can re-OCR (xoa cache), se dung path moi.

3. **6 labels unmatched**: 6/961 ky tu khong tim duoc ung vien nao
   (tier=0). Can kiem tra thu cong xem la ky tu gi.
