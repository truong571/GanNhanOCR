# Coverage report

- Total samples: **52,645**
- Matched: **50,458** (95.85%)
- Unmatched: **2,187** (4.15%)
- Unique nom_char: **4,653**

## Per-book breakdown

| source            |   rows |   unique_chars |   matched |   match_rate |
|:------------------|-------:|---------------:|----------:|-------------:|
| SachThanhTruyen11 |  15548 |           2665 |     14962 |        96.23 |
| SachThanhTruyen2  |  18509 |           3050 |     17943 |        96.94 |
| SachThanhTruyen4  |  18588 |           3215 |     17553 |        94.43 |

## Tier distribution per book

Tier 1 = dict, Tier 2 = similar, Tier 3 = DINOv2+FD, 0 = none.

| source            |    1 |   2 |     3 |
|:------------------|-----:|----:|------:|
| SachThanhTruyen11 |  419 | 264 | 14865 |
| SachThanhTruyen2  | 1199 | 422 | 16888 |
| SachThanhTruyen4  | 1455 | 374 | 16759 |

## Long tail (class -> sample count)

- Total classes: 4,653
- Classes with 1 sample only: 1,524 (32.8%)
- Classes with <5 samples: 2,947
- Classes with >=10 samples (train-ready): 1,012
- Classes with >=50 samples: 217

## Health flags

- **SachThanhTruyen11**: tier-1 only 2.7% -> dictionary lookup is failing, tier-3 (14,865) is masking it.