# Deep fd_cache evaluation — 500-crop test

Sample: 500 random labeled crops from dataset/all/labels.csv
Device: mps

## Cosine distribution (vs REAL book crops)

| Metric | n | mean | median | p25 | p75 | min | max | ≥0.75 |
|--------|--:|-----:|-------:|----:|----:|----:|----:|------:|
| self | 487 | 0.744 | 0.761 | 0.667 | 0.837 | 0.369 | 0.942 | 261 (53.6%) |
| cache | 499 | 0.705 | 0.719 | 0.630 | 0.789 | 0.335 | 0.891 | 189 (37.9%) |
| dilated | 499 | 0.685 | 0.706 | 0.578 | 0.798 | 0.287 | 0.924 | 200 (40.1%) |
| font | 500 | 0.598 | 0.615 | 0.518 | 0.675 | 0.224 | 0.864 | 25 (5.0%) |

**self** = crop[X] vs another crop[same X] (real-vs-real, upper bound)
**cache** = real crop vs fd_cache[X] (FontDiffusion output)
**dilated** = real crop vs fd_cache_dilated[X]
**font** = real crop vs NomNaTong font-rendered[X]

## Han Unified vs Nôm Ext breakdown

| Type | cache mean (n) | dilated mean (n) | font mean (n) |
|------|---------------:|-----------------:|--------------:|
| Han Unified | 0.702 (366) | 0.686 (366) | 0.599 (367) |
| Nôm Ext B+ | 0.714 (133) | 0.684 (133) | 0.596 (133) |

## Pipeline-realistic top-1 accuracy

For each crop, pick top-1 fd_cache candidate among dict[syllable].
Top-1 correct: **468/497 = 94.2%**

This is what Tier-3 actually delivers (combining dict + fd_cache).

## Verdict

- Self-similarity ceiling (real-vs-real): 0.744
- fd_cache:         0.705  (37.9% pass threshold 0.75)
- fd_cache_dilated: 0.685  (40.1% pass)
- Font baseline:    0.598

**fd_cache > font** → FontDiffusion adds value. Cache acceptable as-is.

## Recommendation for next action

- Switch to fd_cache_dilated (slight improvement)

- Current cache is acceptable. Generation of missing 5,532 chars (Kaggle) would improve coverage but not necessarily per-char quality.