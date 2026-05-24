# Test toàn bộ đề xuất — ST2 page_0012 + page_0014

**Tổng records**: 373

## So sánh các variant

| Variant | matched | rate | T1 | T2 | T3 | T0 | Han | Ext A | NomB+ | PUA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 361 | 96.78% | 216 | 28 | 128 | 1 | 322 | 14 | 36 | 0 |
| F1_strict | 328 | 87.94% | 200 | 25 | 114 | 34 | 294 | 14 | 31 | 0 |
| F1_relaxed | 360 | 96.51% | 216 | 27 | 128 | 2 | 321 | 14 | 36 | 0 |
| F2_bucket_only | 310 | 83.11% | 216 | 28 | 128 | 1 | 322 | 14 | 36 | 0 |
| F3_only | 361 | 96.78% | 216 | 28 | 128 | 1 | 322 | 14 | 36 | 0 |
| F1r+F2+F3 | 309 | 82.84% | 216 | 27 | 128 | 2 | 321 | 14 | 36 | 0 |
| +Q1 (full) | 309 | 82.84% | 216 | 27 | 128 | 2 | 321 | 14 | 36 | 0 |

## Đếm số lần sửa

| Fix | Số record bị tác động |
|---|---:|
| F1_strict_demoted | 33 |
| F1_relaxed_demoted | 1 |
| F1_relaxed_kept_due_to_dict | 32 |
| F2_bucket_demoted | 51 |
| F3_promoted | 0 |
| Q1_diacritic_fixed | 2 |

## F4 — audit cjk_block_score filter (>0.1) trong nom_candidates

- Tổng entry PUA trong các pool nom_candidates: **141** (nếu >0 → filter `>0.1` loại bỏ chúng trước Tier 3 ranking)

## F5 — audit pool size nom_candidates

- Records có pool: 372
- Pool size max: 10
- Pool size mean: 9.56
- Records đạt cap 20: 0 (nếu cao → cần tăng cap)
- Records pool < 5: 12

## F3 promotions (Hán-shared shortcut)

(không có)

## F1' decisions

- **Giữ lại** (dict có entry): 32
- **Demote** (dict không có): 1

### Kept

| page | col | syl | nom |
|---|---:|---|---|
| page_0012 | 3 | Giê | 支 |
| page_0012 | 3 | su | 秋 |
| page_0012 | 4 | Bà | 婆 |
| page_0012 | 4 | Ma | 嗎 |
| page_0012 | 4 | ri | 𪅨 |
| page_0012 | 4 | a | 亜 |
| page_0012 | 5 | Rô | 嚕 |
| page_0012 | 5 | ma | 嫫 |
| page_0014 | 5 | sa | 沙 |
| page_0014 | 5 | se | 𠈴 |
| page_0014 | 5 | do | 由 |
| page_0014 | 5 | tê | 痺 |
| page_0014 | 5 | mi | 眉 |
| page_0014 | 5 | sa | 娑 |
| page_0012 | 6 | An | 氨 |
| page_0012 | 6 | ti | 𤰞 |
| page_0012 | 6 | ô | 烏 |
| page_0012 | 6 | ki | 璣 |
| page_0014 | 6 | Giê | 支 |
| page_0014 | 6 | su | 秋 |

## Q1 diacritic restore

- Số syllable được khôi phục dấu: **2**

| page | col | từ | sang |
|---|---:|---|---|
| page_0012 | 2 | `truyen` | `truyến` |
| page_0012 | 3 | `Giu` | `giữ` |