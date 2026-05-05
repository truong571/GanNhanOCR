# test_crop/ — A/B/C visual comparison

Compare 3 character-crop cleaning approaches on real samples from all 6 books.

## How to view

1. **Open `results/grid.png`** — 36 samples, each row is one character with 4 columns:
   - **original** (raw crop, varying size — straight from `prepared/<book>/detected/crops/...`)
   - **A: current full** — what `core/image/crop_cleaner.py:CharacterCleaner` produces today
   - **B: Sauvola only** — Sauvola binarize + trim + center 64×64 (Mức 2 đề xuất)
   - **C: NEW4 adaptive** — Adaptive Gaussian threshold + trim + center 64×64

2. **Read `results/summary.md`** — counts + per-sample table.

3. **Drill into one sample**: `results/samples/sample_NNN/` contains the 4 PNGs + `info.json` (book, page, syllable, nom_char, ocr_char, tier).

## What to look for

For each row, compare A / B / C to the original:

| Concern | Visual signal | Verdict |
|---|---|---|
| **Stroke fidelity** | Are thin nét / dấu chấm preserved? | If C/B preserve them but A misses them → A is over-cleaning |
| **Stroke distortion** | Does any version look thicker/thinner than original? | If A normalizes thickness aggressively → A distorts handwriting |
| **Component loss** | Are radical dots (心 ⺗, 氵) intact? | If A drops them → A's CC noise removal is too aggressive |
| **Background noise** | Is paper texture removed cleanly? | If C still has texture but A/B don't → C is too lenient |
| **Border lines** | Are mép sách (vertical/horizontal rules) gone? | If C keeps lines but A/B don't → A/B better |
| **Centering / scale** | Is character centered consistently? | All 3 should match here |

## Approaches in detail

| # | Name | Steps | Lines |
|---|---|---|---|
| **A** | current full ([crop_cleaner.py](../core/image/crop_cleaner.py)) | medianBlur → bg_normalize → Sauvola → border_line_removal → morph_open → CC_noise_removal → stroke_normalize → trim → center | ~200 |
| **B** | Sauvola only | Sauvola → trim → center | ~25 |
| **C** | NEW4 paper | Adaptive Gaussian → trim → center | ~15 |

## Expected outcomes

If **A** is over-cleaning (the suspicion): some samples will show **A missing strokes that B/C preserve**. That's the smoking gun — if you see thin strokes / dots clearly in B but absent in A, the current pipeline is destroying real character data.

If **C** is too simple: some samples will show **C with leftover paper noise / border lines** that B/A handle. That justifies keeping Sauvola.

If **B** is the sweet spot (the hypothesis): B preserves strokes like C but cleans paper like A. This is the recommended replacement.

## Re-run with different sample size

Edit `run_compare.py` config block:

```python
N_PER_BOOK = 6              # change this
BOOKS = [...]               # add/remove books
RNG_SEED = 42               # change for different random samples
```

Then:

```sh
PATH="$PWD/.venv/bin:$PATH" python test_crop/run_compare.py
```

## After deciding

If B wins (most likely outcome based on prior analysis):

```sh
# Replace CharacterCleaner.clean() with the simpler 4-step approach.
# Keep _sauvola_binarize() and _normalize_background() helpers.
# Drop _morphological_cleanup, _normalize_stroke, _remove_noise_components.
# Keep _remove_border_lines optionally.
```

I will write the patch once you confirm.
