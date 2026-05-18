# Strategy B+ — parser_v2 fallback + overlap_threshold sweep

Book: **SachThanhTruyen2**, pages: **159**, parser_v2 adopted on: **21** pages


## Sweep over overlap_threshold

| threshold | rate (all pairs) | rate (col_ok) | pages col_count_match | pages all_ok |
|---:|---:|---:|---:|---:|
| 0.3 | 41.50% (11758/28330) | 41.96% (11645/27752) | 151/159 (95.0%) | 128/159 (80.5%) |
| 0.4 | 41.52% (11761/28328) | 42.01% (11648/27729) | 145/159 (91.2%) | 124/159 (78.0%) |
| 0.5 | 41.22% (11630/28215) | 41.86% (11518/27513) | 133/159 (83.6%) | 114/159 (71.7%) |
| 0.6 | 40.87% (11474/28071) | 41.64% (11362/27285) | 122/159 (76.7%) | 104/159 (65.4%) |

## Comparison vs baselines

| Variant | rate | pages all_ok |
|---|---:|---:|
| A (current pipeline) | 6.39% | — |
| B baseline (t=0.5, v1 only) | 42.17% | 100/159 |
| **B+ best (t=0.4, v2 fallback)** | **42.01%** | **124/159** |

## Conclusion

- parser_v2 fallback adopted on pages where v1 returned < 9 lines.
- overlap_threshold sweep verifies merge behaviour on cases where Kimhannom split marker stack into a separate column.
- Best threshold: **0.4**.
