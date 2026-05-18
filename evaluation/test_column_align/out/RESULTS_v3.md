# Strategy B++ — parser_v3 + overlap_threshold sweep

Book: **SachThanhTruyen2**, pages: **159**, parser_v3 used on: **158** pages, pages with QN=9: **145/159**


## Sweep over overlap_threshold

| t | rate (all pairs) | rate (col_ok) | pages col_count_match | pages all_ok |
|---:|---:|---:|---:|---:|
| 0.3 | 45.74% (12626/27602) | 46.78% (12478/26672) | 140/159 (88.1%) | 114/159 (71.7%) |
| 0.4 | 45.76% (12629/27600) | 46.83% (12480/26649) | 134/159 (84.3%) | 110/159 (69.2%) |
| 0.5 | 45.40% (12479/27489) | 46.64% (12329/26433) | 124/159 (78.0%) | 102/159 (64.2%) |
| 0.6 | 44.78% (12244/27345) | 46.11% (12094/26230) | 114/159 (71.7%) | 95/159 (59.7%) |

## Comparison vs baselines

| Variant | rate (col_ok) | pages all_ok | pages col_match |
|---|---:|---:|---:|
| A (current pipeline) | 6.39% | — | — |
| B (v1 parser, t=0.5) | 43.56% | 100/159 (62.9%) | 117/159 (73.6%) |
| B+ (v2 parser, t=0.3) | 41.96% | 128/159 (80.5%) | 151/159 (95.0%) |
| **B++ (v3 parser, t=0.3)** | **46.78%** | **114/159 (71.7%)** | **140/159 (88.1%)** |
