# Image-projection segmentation test (Final Plan Step 2B)

For each col, run `segment_characters_in_column(binary, bbox, expected_count=len(qn_line))` and check:

- count returned == expected (segmenter guarantees this when given expected_count).
- ≤25% of bboxes have ink_ratio < 0.03 (rules out empty/noise bboxes).

## Per book

| Book | Pages | Page seg ok | Cols total | Cols seg ok | Pages w/ low-ink |
|---|---:|---:|---:|---:|---:|
| SachThanhTruyen2 | 159 | 141 | 1379 | 1379 | 13 |
| SachThanhTruyen4 | 145 | 108 | 1209 | 1206 | 59 |
| SachThanhTruyen11 | 141 | 95 | 1132 | 1130 | 50 |
| **TOTAL** | **445** | **344** (77.3%) | **3720** | **3715** (99.9%) | **122** |
