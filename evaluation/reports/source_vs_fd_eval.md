# Source font vs fd_cache — stroke preservation check

Compares the **input** to FontDiffusion (NomNaTong source render) vs the
**output** (fd_cache PNG). Tells us if FontDiffusion is dropping strokes
between input and output.

Sample: 30 random chars present in BOTH source font + fd_cache.

## Aggregate

| Metric | source | fd_cache | change |
|--------|-------:|---------:|-------:|
| Ink ratio (mean %) | 14.1% | 7.9% | -42.7% |
| IoU(source, fd_cache pixels) | — | — | **0.142** |
| DINOv2 cosine(source, fd_cache) | — | — | **0.673** |
| Chars with ink LOSS > 20% | — | — | 27/30 (90%) |
| Chars with ink GAIN > 20% | — | — | 0/30 (0%) |

## Per-char details (sorted by ink change)

| char | cp | source ink% | fd_cache ink% | change | IoU | cosine |
|------|----|------------:|--------------:|-------:|----:|-------:|
| 𫾛 | U+2BF9B | 16.6% | 5.3% | **-68.4%** | 0.083 | 0.45 |
| 溺 | U+6EBA | 16.9% | 6.3% | **-62.5%** | 0.157 | 0.86 |
| 郜 | U+90DC | 13.9% | 5.7% | **-58.9%** | 0.067 | 0.63 |
| 簾 | U+7C3E | 15.5% | 6.7% | **-57.0%** | 0.172 | 0.416 |
| 踽 | U+8E3D | 16.4% | 7.1% | **-56.6%** | 0.071 | 0.772 |
| 𤾱 | U+24FB1 | 15.6% | 7.1% | **-54.1%** | 0.074 | 0.659 |
| 葖 | U+8456 | 12.0% | 5.5% | **-54.0%** | 0.123 | 0.805 |
| 𤆷 | U+241B7 | 14.1% | 6.5% | **-53.9%** | 0.137 | 0.766 |
| 𡅐 | U+21150 | 12.6% | 6.3% | **-49.8%** | 0.129 | 0.679 |
| 𧊥 | U+272A5 | 15.4% | 7.9% | **-48.6%** | 0.093 | 0.558 |
| 蠕 | U+8815 | 17.1% | 8.8% | **-48.4%** | 0.118 | 0.799 |
| 萻 | U+843B | 11.7% | 6.3% | **-46.2%** | 0.101 | 0.576 |
| 餛 | U+991B | 16.0% | 8.7% | **-45.5%** | 0.149 | 0.793 |
| 𡋁 | U+212C1 | 11.5% | 6.3% | **-45.2%** | 0.187 | 0.68 |
| 𡁚 | U+2105A | 14.6% | 8.1% | **-44.6%** | 0.145 | 0.679 |
| 蹴 | U+8E74 | 14.9% | 8.3% | **-44.0%** | 0.078 | 0.671 |
| 𫸽 | U+2BE3D | 17.0% | 9.5% | **-43.9%** | 0.139 | 0.801 |
| 䕹 | U+4579 | 15.0% | 8.4% | **-43.9%** | 0.212 | 0.537 |
| 梐 | U+6890 | 13.5% | 7.6% | **-43.7%** | 0.168 | 0.648 |
| 𫃸 | U+2B0F8 | 13.4% | 7.6% | **-42.9%** | 0.163 | 0.462 |
| 𬕼 | U+2C57C | 13.1% | 7.7% | **-41.3%** | 0.096 | 0.702 |
| 𫵛 | U+2BD5B | 10.8% | 6.7% | **-38.2%** | 0.264 | 0.744 |
| 𠿱 | U+20FF1 | 13.0% | 8.0% | **-38.1%** | 0.104 | 0.55 |
| 鮪 | U+9BAA | 15.5% | 10.4% | **-33.2%** | 0.138 | 0.65 |
| 𥚤 | U+256A4 | 15.0% | 10.1% | **-32.2%** | 0.11 | 0.599 |
| 󱭼 | U+F1B7C | 15.0% | 10.2% | **-31.8%** | 0.167 | 0.691 |
| 𤝍 | U+2474D | 12.4% | 9.1% | **-27.0%** | 0.19 | 0.771 |
| 䋥 | U+42E5 | 14.8% | 12.3% | **-17.0%** | 0.147 | 0.83 |
| 椈 | U+6908 | 15.8% | 13.1% | **-16.8%** | 0.144 | 0.656 |
| 𬼀 | U+2CF00 | 5.2% | 5.5% | **+7.2%** | 0.326 | 0.759 |

## Interpretation

⚠️  fd_cache has MUCH less ink than source — significant stroke loss.

- IoU < 0.30 = pixels barely overlap (style change drastic)

- DINOv2 cosine 0.6-0.8 = embedding similar (style change but identity preserved)

## Visual

`source_vs_fd_grid.png` — left=NomNaTong source, mid=fd_cache, right=dilated.
Inspect: do the strokes in fd_cache MATCH the source? Are any minor
strokes (radicals, dots) missing?