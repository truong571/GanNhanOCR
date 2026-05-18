# Image-projection PRIMARY vs Kimhannom-filter FALLBACK

Coverage of `detect_columns(binary, text_box, n_expected=9) == 9` and `Kimhannom filter len≥4 → 9 cols`.

| Book | Pages | Proj=9 | Filter=9 | Both | Proj-only | Filter-only | Neither |
|---|---:|---:|---:|---:|---:|---:|---:|
| SachThanhTruyen2 | 159 | 159 | 156 | 156 | 3 | 0 | 0 |
| SachThanhTruyen4 | 145 | 145 | 136 | 136 | 9 | 0 | 0 |
| SachThanhTruyen11 | 141 | 141 | 132 | 132 | 9 | 0 | 0 |
| **TOTAL** | **445** | **445** (100.0%) | **424** (95.3%) | **424** | **21** | **0** | **0** |

Fallback chain coverage (Proj → Filter):
- Proj primary: 445/445 (100.0%)
- Plus filter fallback: +0 → **445/445 (100.0%)**
- Neither (must flag suspect): 0
