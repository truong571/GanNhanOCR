# Column Align Strategy A vs B — SachThanhTruyen2

Metric: tier-1 probe rate `ocr_char ∈ qn_to_nom[qn_syllable]`.

Strategy A = existing pipeline `aligned/*_aligned.json` matches.

Strategy B = align_v2 (cluster OCR cols by x-overlap → strip top marker → 1-1 pair by index).

Strategy B_ok = subset of B pairs where the column is `col_align_ok=True` (no underflow).


## Headline

- Pages evaluated: **159**
- Strategy A probe rate: **6.39%** (1187/18562)
- Strategy B probe rate: **42.17%** (11759/27883)
- Strategy B_ok probe rate: **43.56%** (11627/26694)
- Pages with cluster count == QN line count: **117/159** (73.6%)
- Pages fully `alignment_ok` (all cols clean): **100/159** (62.9%)

## Per-page rate distribution

Strategy A:    <10%:128 | 10-20%:17 | 20-30%:8 | 30-40%:3 | 40-50%:2 | 50-60%:1 | 60-70%:0 | 70-80%:0 | 80%+:0
Strategy B:    <10%:9 | 10-20%:7 | 20-30%:12 | 30-40%:17 | 40-50%:63 | 50-60%:46 | 60-70%:5 | 70-80%:0 | 80%+:0
Strategy B_ok: <10%:9 | 10-20%:3 | 20-30%:16 | 30-40%:12 | 40-50%:67 | 50-60%:47 | 60-70%:5 | 70-80%:0 | 80%+:0

## Top 10 pages where B beats A (delta)

| page | A% | B% | delta | clusters/qn | align_ok |
|---|---:|---:|---:|---|:-:|
| page_0128 | 0.7 | 62.7 | +62.0 | 9/9 | Y |
| page_0324 | 0.0 | 61.2 | +61.2 | 9/9 | Y |
| page_0014 | 1.7 | 62.9 | +61.2 | 9/9 | Y |
| page_0254 | 1.0 | 58.9 | +57.9 | 9/9 | Y |
| page_0316 | 0.0 | 57.5 | +57.5 | 10/9 | N |
| page_0166 | 0.9 | 58.3 | +57.5 | 9/9 | Y |
| page_0240 | 0.0 | 57.4 | +57.4 | 10/9 | N |
| page_0326 | 0.0 | 57.4 | +57.4 | 9/9 | Y |
| page_0076 | 0.0 | 57.1 | +57.1 | 9/9 | Y |
| page_0230 | 0.7 | 56.7 | +56.0 | 10/9 | N |

## Top 10 pages where B regresses vs A

| page | A% | B% | delta | clusters/qn | align_ok |
|---|---:|---:|---:|---|:-:|
| page_0096 | 59.3 | 49.2 | -10.1 | 9/9 | N |
| page_0116 | 19.0 | 17.0 | -2.0 | 10/9 | N |
| page_0032 | 2.3 | 0.6 | -1.7 | 9/8 | N |
| page_0060 | 1.4 | 0.6 | -0.7 | 9/7 | N |
| page_0088 | 1.3 | 0.7 | -0.7 | 9/8 | N |
| page_0034 | 1.1 | 0.6 | -0.5 | 9/8 | N |
| page_0046 | 0.0 | 0.0 | +0.0 | 9/8 | N |
| page_0184 | 15.0 | 15.0 | +0.0 | 11/9 | N |
| page_0052 | 14.7 | 15.1 | +0.4 | 10/7 | N |
| page_0092 | 31.9 | 32.5 | +0.6 | 10/9 | N |

## Notes & next steps

- B_ok is the headline rate to compare. It restricts to columns where strategy B produced no underflow (i.e. clusters where Kimhannom had ≥ expected_count chars, so marker-strip is valid).
- Pages where `clusters > qn_lines` after x-overlap clustering indicate Kimhannom still over-split (marker x-shift > 50% overlap threshold). Tune `overlap_threshold` in nom_column_cluster.py.
- Pages where `clusters < qn_lines` indicate two real columns got merged (rare). Inspect manually.
- Underflow columns (cluster.actual < qn.expected) are NOT included in B_ok — these are candidates for projection-based re-segmentation on the original image (next iteration).
- Merge into pipeline only after B_ok ≥ A + 10pp AND no critical regression (>5pp drop) on ≥ 5% of pages.
