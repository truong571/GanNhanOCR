# fd_cache stroke loss — DINOv2 evaluation

Using **DINOv2 cosine similarity** (the actual Tier-3 metric, not pixel IoU).

## Test A: 30 random chars — similarity to font baseline

|              | mean | min | max | ≥0.75 (threshold) |
|--------------|-----:|----:|----:|------------------:|
| **cache**    | 0.670 | 0.410 | 0.854 | 10/30 |
| **dilated**  | 0.695 | 0.528 | 0.837 | 9/30 |

Pipeline threshold = 0.75 (set in config). Higher mean = more confident matching.

## Test B: confusable pairs (each differs by 1-2 strokes)

**1/11** pairs unambiguously distinguished correct identity.

| pair | cache(a)·font(a) | cache(a)·font(b) | a→a? | cache(b)·font(b) | cache(b)·font(a) | b→b? |
|------|-----------------:|-----------------:|-----|-----------------:|-----------------:|-----|
| 大/天 | 0.699 | 0.717 | **WRONG** | 0.544 | 0.491 | OK |
| 大/夫 | 0.699 | 0.642 | OK | 0.698 | 0.75 | **WRONG** |
| 千/干 | 0.602 | 0.45 | OK | 0.446 | 0.56 | **WRONG** |
| 土/士 | 0.48 | 0.5 | **WRONG** | 0.569 | 0.539 | OK |
| 人/入 | 0.662 | 0.619 | OK | 0.703 | 0.7 | OK |
| 日/目 | 0.611 | 0.592 | OK | 0.619 | 0.623 | **WRONG** |
| 石/右 | 0.697 | 0.669 | OK | 0.767 | 0.807 | **WRONG** |
| 未/末 | 0.633 | 0.609 | OK | 0.551 | 0.561 | **WRONG** |
| 木/本 | 0.588 | 0.563 | OK | 0.623 | 0.643 | **WRONG** |
| 田/由 | 0.524 | 0.562 | **WRONG** | 0.623 | 0.539 | OK |
| 白/百 | 0.507 | 0.485 | OK | 0.612 | 0.616 | **WRONG** |