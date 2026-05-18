# Full evaluation — 3 books × 2 nôm-col methods

Methods:

- **cluster**: x-overlap clustering of Kimhannom cols (threshold=0.3).
- **filter**: drop Kimhannom cols with `len ≤ 3`, keep rest right-to-left.
- **hybrid**: filter as anchors + re-attach short cols by nearest x-distance.

Tiers (without visual rank):

- Tier 1: `ocr_char ∈ qn_to_nom[qn_syl]`.
- Tier 2: `similar_dict[ocr_char] ∩ qn_to_nom[qn_syl] ≠ ∅`.
- Tier 3: otherwise (would need FontDiffusion+DINOv2 to confirm).

3-tier dataset:

- **Gold**: `alignment_ok` AND tier ∈ {1, 2}.
- **Silver**: `alignment_ok` AND tier 3 (visual-rank downstream).
- **Review**: alignment suspect / missing OCR / no QN.


## Method: `cluster`

| Book | Pages | QN=9 | col_match | page_ok | Pairs | Gold | Silver | Review | T1 | T2 | T3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SachThanhTruyen2 | 159 | 144 | 139 | 114 | 27568 | 13811 | 12873 | 884 | 12614 | 1372 | 13582 |
| SachThanhTruyen4 | 145 | 117 | 107 | 95 | 25988 | 13243 | 11765 | 980 | 11992 | 1312 | 12684 |
| SachThanhTruyen11 | 141 | 101 | 81 | 69 | 24355 | 11401 | 11516 | 1438 | 10357 | 1153 | 12845 |
| **TOTAL** | **445** | **362** | **327** | **278** | **77911** | **38455** | **36154** | **3302** | **34963** | **3837** | **39111** |

Gold % of pairs: **49.36%**
Tier1+2 hit rate (across all pairs, NOT just gold): **49.80%**
Pages structurally OK: **278/445 (62.5%)**

## Method: `filter`

| Book | Pages | QN=9 | col_match | page_ok | Pairs | Gold | Silver | Review | T1 | T2 | T3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SachThanhTruyen2 | 159 | 144 | 141 | 56 | 27403 | 13334 | 10653 | 3416 | 13140 | 1423 | 12840 |
| SachThanhTruyen4 | 145 | 117 | 108 | 54 | 25779 | 12910 | 10473 | 2396 | 12230 | 1312 | 12237 |
| SachThanhTruyen11 | 141 | 101 | 95 | 20 | 24116 | 10828 | 8212 | 5076 | 11279 | 1258 | 11579 |
| **TOTAL** | **445** | **362** | **344** | **130** | **77298** | **37072** | **29338** | **10888** | **36649** | **3993** | **36656** |

Gold % of pairs: **47.96%**
Tier1+2 hit rate (across all pairs, NOT just gold): **52.58%**
Pages structurally OK: **130/445 (29.2%)**

## Method: `hybrid`

| Book | Pages | QN=9 | col_match | page_ok | Pairs | Gold | Silver | Review | T1 | T2 | T3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SachThanhTruyen2 | 159 | 144 | 141 | 117 | 27576 | 13724 | 12958 | 894 | 12530 | 1368 | 13678 |
| SachThanhTruyen4 | 145 | 117 | 108 | 95 | 25905 | 13052 | 11875 | 978 | 11824 | 1288 | 12793 |
| SachThanhTruyen11 | 141 | 101 | 95 | 79 | 24412 | 11442 | 11507 | 1463 | 10387 | 1164 | 12861 |
| **TOTAL** | **445** | **362** | **344** | **291** | **77893** | **38218** | **36340** | **3335** | **34741** | **3820** | **39332** |

Gold % of pairs: **49.06%**
Tier1+2 hit rate (across all pairs, NOT just gold): **49.51%**
Pages structurally OK: **291/445 (65.4%)**
