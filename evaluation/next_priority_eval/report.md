# Đánh giá scale 4 đề xuất ưu tiên kế tiếp

Dataset: dataset/all/labels.csv (83652 records, 83564 matched-type) + dataset.json 3 sách.

## A — Q1 Diacritic restore (toàn bộ 84k record)

- Records xét: **83652**
- Syllable đã có trong dict: **83052** (99.3%)
- Syllable KHÔNG có trong dict: **600** (0.7%)
- Trong số đó, **có thể khôi phục** (cách dấu ≤1): **475** (79.2%)
- Số cặp khôi phục unique: **141**
- **Promotion T2/T3 → T1** nếu áp Q1: **0** record

### Top 20 restoration phổ biến

| syl từ | syl đến | n |
|---|---|---:|
| `Giu` | `giù` | 83 |
| `Ay` | `ầy` | 31 |
| `trấy` | `trẩy` | 31 |
| `giu` | `giù` | 30 |
| `ga` | `gá` | 25 |
| `truyen` | `truyện` | 16 |
| `but` | `bụt` | 12 |
| `ay` | `ầy` | 11 |
| `goi` | `gối` | 8 |
| `hế` | `hề` | 7 |
| `Ây` | `ầy` | 7 |
| `Ga` | `gá` | 6 |
| `dố` | `dỗ` | 6 |
| `ây` | `ầy` | 6 |
| `góm` | `gốm` | 5 |
| `CHIN` | `chín` | 5 |
| `Hế` | `hề` | 5 |
| `chin` | `chín` | 5 |
| `Trằng` | `trắng` | 4 |
| `muon` | `muợn` | 4 |

## B — F2' Bucket-aware T3 threshold simulation

Ngưỡng: HAN_BASIC ≥ 0.85 · Ext A ≥ 0.87 · Ext B+/PUA ≥ 0.90

| Book | n | baseline matched | demote | new matched | recall drop |
|---|---:|---:|---:|---:|---:|
| SachThanhTruyen2 | 28656 | 28130 (98.16%) | 4271 | 23859 (83.26%) | 14.9pp |
| SachThanhTruyen4 | 27938 | 26830 (96.03%) | 5641 | 21189 (75.84%) | 20.19pp |
| SachThanhTruyen11 | 27510 | 26838 (97.56%) | 3779 | 23059 (83.82%) | 13.74pp |

Phân bucket demote / sách:

- **SachThanhTruyen2**: {'HAN_BASIC': 2444, 'NOM_EXT_B_PLUS': 1581, 'CJK_EXT_A': 240, 'OTHER': 6}
- **SachThanhTruyen4**: {'HAN_BASIC': 3439, 'NOM_EXT_B_PLUS': 1869, 'CJK_EXT_A': 318, 'OTHER': 15}
- **SachThanhTruyen11**: {'HAN_BASIC': 2104, 'NOM_EXT_B_PLUS': 1409, 'CJK_EXT_A': 256, 'OTHER': 10}

## C — V2 sqpad candidate audit (extreme aspect)

Crop có aspect ratio (w/h) ngoài [0.5, 2.0] là candidate cho sqpad rerank.

| Book | n total | extreme_aspect | extreme T3 matched | extreme T3 total | AR range |
|---|---:|---:|---:|---:|---|
| SachThanhTruyen2 | 28656 | 194 | 175 | 180 | (0.352, 5.174) |
| SachThanhTruyen4 | 27938 | 441 | 287 | 425 | (0.04, 21.667) |
| SachThanhTruyen11 | 27510 | 412 | 322 | 387 | (0.115, 19.125) |

## D — Per-col ensemble cost/impact projection

**Quan sát baseline** (2 trang debug, 373 record): 12 promotion (3.22%)

| Scope | API calls | Runtime ước | Promotion dự kiến | Cost/promotion |
|---|---:|---:|---:|---:|
| **Full** (toàn 9 cột/447 trang) | **4,023** | 134.1 phút | **2,688** | 1.5 call/promotion |
| **Selective ~10%** (cột flag count_ok=False / null ocr) | 402 | 13.4 phút | ~268 | (gần như free) |

## Tổng kết khuyến nghị

| Đề xuất | Verdict | Lý do |
|---|---|---|
| **A — Q1 Diacritic** | ⚪ Tác động nhỏ | 0 promotion T1 trên 84k = 0.00% |
| **B — F2' bucket threshold** | ⚠️ Cần gold set để chốt | Demote 13,691 record (precision↑, recall↓) |
| **C — V2 sqpad** | 🟡 Đáng implement | 1,047 candidate trên 3 sách (cần viết code rerank) |
| **D — Per-col ensemble full** | ⚠️ Đắt vs benefit | 4,023 API call để vớt ~2,688 promotion |