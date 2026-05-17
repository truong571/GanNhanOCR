# fd_cache stroke loss evaluation

Sample: 30 random chars + 8 confusable pairs.
Columns: font (NomNaTong baseline) | fd_cache | fd_cache_dilated.

## Aggregate (Test 1)

- Ink ratio:  cache **7.9%**, dilated **13.5%**, font **13.4%**
- IoU vs font:  cache **0.133**, dilated **0.189**
- IoU buckets (cache vs font):  bad <0.40: **30**  medium 0.40-0.55: **0**  good ≥0.55: **0**

**Interpretation:**
- IoU < 0.40 = severe stroke loss, char likely unrecognizable
- IoU 0.40-0.55 = noticeable degradation, may confuse DINOv2
- IoU ≥ 0.55 = acceptable for visual matching

## Per-char sample (Test 1)

| char | ink cache % | ink dilated % | ink font % | IoU cache | IoU dilated |
|------|------------:|--------------:|-----------:|----------:|------------:|
| 餛 (U+991B) | 8.7 | 15.2 | 14.9 | 0.164 | 0.232 |
| 𥚤 (U+256A4) | 10.1 | 16.5 | 13.6 | 0.094 | 0.129 |
| 𠿱 (U+20FF1) | 8.0 | 13.7 | 12.3 | 0.081 | 0.137 |
| 䋥 (U+42E5) | 12.3 | 19.4 | 14.3 | 0.147 | 0.17 |
| 𬼀 (U+2CF00) | 5.5 | 7.7 | 4.9 | 0.326 | 0.365 |
| 𫾛 (U+2BF9B) | 5.3 | 10.4 | 15.8 | 0.077 | 0.138 |
| 𧊥 (U+272A5) | 7.9 | 14.3 | 14.6 | 0.108 | 0.174 |
| 𤾱 (U+24FB1) | 7.1 | 12.6 | 14.9 | 0.057 | 0.099 |
| 󱭼 (U+F1B7C) | 10.2 | 17.0 | 13.7 | 0.155 | 0.202 |
| 葖 (U+8456) | 5.5 | 9.9 | 11.0 | 0.111 | 0.173 |
| 𤆷 (U+241B7) | 6.5 | 11.9 | 13.1 | 0.15 | 0.208 |
| 蹴 (U+8E74) | 8.3 | 15.1 | 14.4 | 0.077 | 0.134 |
| 椈 (U+6908) | 13.1 | 18.8 | 15.3 | 0.136 | 0.164 |
| 𡋁 (U+212C1) | 6.3 | 10.2 | 11.0 | 0.15 | 0.243 |
| 𡅐 (U+21150) | 6.3 | 11.4 | 12.0 | 0.093 | 0.156 |
| 𤝍 (U+2474D) | 9.1 | 14.1 | 11.9 | 0.217 | 0.309 |
| 𫵛 (U+2BD5B) | 6.7 | 11.6 | 10.5 | 0.197 | 0.28 |
| 𬕼 (U+2C57C) | 7.7 | 14.1 | 12.5 | 0.086 | 0.134 |
| 簾 (U+7C3E) | 6.7 | 11.7 | 14.8 | 0.168 | 0.24 |
| 郜 (U+90DC) | 5.7 | 10.0 | 13.4 | 0.061 | 0.095 |
| 𡁚 (U+2105A) | 8.1 | 13.7 | 14.1 | 0.14 | 0.205 |
| 蠕 (U+8815) | 8.8 | 15.7 | 16.6 | 0.103 | 0.162 |
| 𫃸 (U+2B0F8) | 7.6 | 14.0 | 12.9 | 0.141 | 0.189 |
| 鮪 (U+9BAA) | 10.4 | 17.8 | 15.1 | 0.119 | 0.174 |
| 萻 (U+843B) | 6.3 | 9.9 | 11.0 | 0.094 | 0.114 |
| 梐 (U+6890) | 7.6 | 12.9 | 13.1 | 0.156 | 0.223 |
| 𫸽 (U+2BE3D) | 9.5 | 16.5 | 15.9 | 0.142 | 0.183 |
| 溺 (U+6EBA) | 6.3 | 11.6 | 16.2 | 0.146 | 0.216 |
| 踽 (U+8E3D) | 7.1 | 12.6 | 15.2 | 0.064 | 0.133 |
| 䕹 (U+4579) | 8.4 | 14.5 | 13.4 | 0.229 | 0.274 |

## Confusable pairs (Test 2)

**1/8 pairs** unambiguously matched correct identity.

| pair | why | IoU cache(a)·font(a) | IoU cache(a)·font(b) | correct? |
|------|-----|---------------------:|---------------------:|---------|
| 大/天 | 天 has 1 extra horizontal stroke at top | 0.13 | 0.137 | ✗ WRONG |
| 大/夫 | 夫 has 1 extra horizontal across | 0.13 | 0.121 | ✓ |
| 千/干 | 千 has slanted top, 干 horizontal | 0.264 | 0.345 | ✗ WRONG |
| 土/士 | 士 top horizontal longer than bottom | 0.336 | 0.21 | ✓ |
| 人/入 | 入 has left stroke crossing | 0.097 | 0.066 | ✓ |
| 日/目 | 目 has 1 extra horizontal | 0.186 | 0.313 | ✗ WRONG |
| 石/右 | right side differs | 0.193 | 0.293 | ✗ WRONG |
| 未/末 | 末 top longer than middle | 0.185 | 0.164 | ✓ |

## Visual
- `fd_cache_grid.png`        — 30 chars × 3 columns (font / cache / dilated)
- `fd_cache_confusable.png` — 8 confusable pairs × 4 columns