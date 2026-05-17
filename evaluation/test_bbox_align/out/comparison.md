# Bbox-aware vs Levenshtein alignment comparison

Tier-1 hit rate = % of match pairs where ocr_char ∈ dict candidates
for the aligned syllable. Higher = alignment kept right pairs together.

## Per-book

| Book | Cols | OLD hits/total | OLD % | NEW hits/total | NEW % | Δ pp |
|------|-----:|---------------:|------:|---------------:|------:|-----:|
| SachThanhTruyen2 | 1,404 | 1,233/18,562 | 6.64% | 1,239/18,562 | 6.67% | +0.03 |
| SachThanhTruyen4 | 1,270 | 1,647/18,636 | 8.84% | 1,648/18,636 | 8.84% | +0.01 |
| SachThanhTruyen11 | 1,247 | 418/15,603 | 2.68% | 461/15,603 | 2.95% | +0.28 |

## Overall

- OLD: 3,298/52,801 = **6.25%**
- NEW: 3,348/52,801 = **6.34%**
- Δ: **+0.09pp**

## Sample wins (NEW gained ≥3 hits on a column)

- SachThanhTruyen2 page_0016 col4: 23 chars / 67 syllables  OLD 0/23 → NEW 6/23
- SachThanhTruyen2 page_0040 col0: 20 chars / 42 syllables  OLD 0/20 → NEW 6/20
- SachThanhTruyen2 page_0054 col8: 15 chars / 23 syllables  OLD 0/15 → NEW 3/15
- SachThanhTruyen2 page_0056 col0: 22 chars / 46 syllables  OLD 0/22 → NEW 10/22
- SachThanhTruyen2 page_0132 col3: 23 chars / 24 syllables  OLD 1/23 → NEW 7/23
- SachThanhTruyen4 page_0126 col0: 21 chars / 44 syllables  OLD 0/21 → NEW 11/21
- SachThanhTruyen11 page_0072 col0: 22 chars / 47 syllables  OLD 0/22 → NEW 10/22
- SachThanhTruyen11 page_0092 col0: 22 chars / 44 syllables  OLD 0/22 → NEW 15/22
- SachThanhTruyen11 page_0134 col8: 22 chars / 23 syllables  OLD 3/22 → NEW 11/22
- SachThanhTruyen11 page_0292 col8: 21 chars / 22 syllables  OLD 0/21 → NEW 4/21