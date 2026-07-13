# Giai đoạn 3 — Công bố đạt chuẩn quốc tế

Tầng **pipeline chính** biến dataset đã remediation (Giai đoạn 1) thành bản công bố đạt
chuẩn quốc tế: split group-aware, metadata Frictionless + Croissant hợp lệ, datasheet
di sản văn hóa, HF Parquet typed, và một **CI gate fail-loud**.

Chỉ `numpy + pandas + PIL` (thống kê/hash tự viết) + `datasets` cho Parquet.
`frictionless`/`mlcroissant`/`imagehash` KHÔNG cần — spec tự cài, validator nội bộ.

## Sáu module

| Module | Vai trò |
|---|---|
| `splits.py` | **page-disjoint** + **leave-one-book-out** + audit rò chéo-split (md5 + dHash) |
| `metadata.py` | Frictionless Data Package + Croissant JSON-LD — **sha256 thật, PK không null** |
| `datasheet.py` | datasheet Gebru + mở rộng di sản văn hóa JOHD (11 mục) |
| `export.py` | HF Parquet typed Features (`Image()` + `ClassLabel(1592)`) + imagefolder |
| `validate.py` | CI gate — bất biến mà các chuẩn đòi hỏi; fail loud |
| `hashing.py` | sha256 (integrity) + dHash (near-dup) tự viết |

## Lệnh

```bash
PY=.venv/bin/python
$PY -m pipeline.publish all            # split -> metadata -> datasheet -> validate
$PY -m pipeline.publish split          # gán split page-disjoint + báo LOBO
$PY -m pipeline.publish metadata       # datapackage.json + croissant.json + card
$PY -m pipeline.publish datasheet      # DATASHEET.md
$PY -m pipeline.publish export --sample 200   # smoke parquet; bỏ --sample = full 82k
$PY -m pipeline.publish validate       # CI gate (exit 1 nếu fail)
```

Đầu vào mặc định: `dataset_out/labels_remediated.csv` (Giai đoạn 1), fallback
`labels.csv`. Đầu ra: `dataset_out/release/`.

## Kết quả đo trên dataset hiện tại

```
split    : train 60,675 / val 1,777 / test 2,577 · page-span=0 md5-span=0 · singletons->train 424
lobo     : holdout yen2/yen4/yen11 = test 22,310 / 21,146 / 21,573
metadata : crops.csv 68,076 rows · 1,592 classes · sha256 THẬT · PK=['image']
datasheet: 11 mục (Gebru + JOHD)
validate : 11/11 checks passed
```

## Sửa đúng các phát hiện audit (A8)

- **PK null**: cũ `primaryKey=['image']` null trên 14,192 REVIEW → package invalid. Nay
  resource crop **chỉ chứa dòng có image** → `image` là khóa non-null unique thật.
- **sha256='n/a'**: nay là sha256 **thật** của `crops.csv` trong cả datapackage +
  croissant.
- **rò split**: page-disjoint đảm bảo không trang/md5 nào span >1 split (assert trong
  validate).
- **orphan/typed**: validate kiểm mọi crop char-labeled tồn tại trên đĩa + enum tier/split.

## Bất biến CI gate (validate)

class count = declared · không page/md5 span split · PK image unique+non-null · mọi
crop tồn tại · sha256 thật (không 'n/a') · tier/split trong enum · (tùy chọn)
`load_dataset` mở được parquet.

## Drivers — baseline ngoài (chạy riêng, KHÔNG ở đây)

Theo pattern KMNIST/NomNaOCR, báo baseline trên split chính thức: kNN trên embedding
ArcFace, ResNet-18, ViT nhỏ; GOLD-only vs GOLD+SILVER; page-disjoint vs LOBO. Train
offline rồi ghi số vào dataset card. Mint DOI (HF/Zenodo) cho v1.0 đóng băng.

## Test

```bash
.venv/bin/python -m pipeline.publish.selftest   # 56 assertions, exit 0 = pass
```

Kiểm: hashing (sha256 vs hashlib, dHash identical/khác, sentinel -1); splits (không
span, deterministic, LOBO, phát hiện leak exact + perceptual chéo-split); metadata (PK
image, sha256 thật, typed, enum, croissant recordSet); datasheet (11 mục + interpolate);
export (char_labeled, ClassLabel+Image feature, parquet round-trip THẬT); validate (pass
sạch, fail khi PK trùng / sha256 'n/a'); integration thật trên labels_remediated.csv.

## Vị trí phát hành

`dataset_out/release/` — thay thế `pipeline/align_engine/to_standard.py` cũ (PK-null +
sha256='n/a'). REVIEW giữ lại (đánh cờ, không xóa) theo khuyến nghị.
