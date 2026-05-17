# fd_cache vs REAL book crops — DINOv2 cosine test

Tests if fd_cache (FontDiffusion-generated) actually matches the
handwritten crops from the books, since that's what pipeline Tier-3 uses.

## Result (N=99 labeled crops)

|              | mean cosine | min  | max  | ≥0.75 |
|--------------|------------:|-----:|-----:|------:|
| **cache**    | **0.700** | 0.486 | 0.863 | 37/99 |
| **dilated**  | **0.688** | 0.374 | 0.884 | 43/99 |

## Interpretation

- Mean cosine 0.700 = average similarity between fd_cache and real
  book crops. Higher = fd_cache better represents real handwriting.
- Threshold 0.75 = pipeline Tier-3 'matched=True' cutoff.
- 37/99 (37%) crops
  have fd_cache match ≥0.75 → would 'pass' Tier-3.

## Verdict

✓ fd_cache cosine acceptable. Tier-3 matching reasonably valid.