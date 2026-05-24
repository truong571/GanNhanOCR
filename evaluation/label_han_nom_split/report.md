# Test A — Phân loại nhãn cuối theo Hán shared vs Nôm thuần

Bucket dựa trên Unicode block của `nom_char`:

- `HAN_BASIC` (U+4E00–U+9FFF) — CJK Unified, dùng chung với Trung. Trong corpus tiếng Việt cổ thường mang nghĩa Hán-Việt.
- `CJK_EXT_A` (U+3400–U+4DBF) — Ext A. Có cả CJK hiếm và một số Nôm.
- `NOM_EXT_B_PLUS` (U+20000+) — Ext B/C/D/E/F. Đa số là Nôm thuần.
- `NOM_PUA` — Private-Use Area, Nôm legacy.

## (1) Sample 2 trang debug (page_0012 + page_0014)

### debug_p12_p14

- Tổng record: **373**
- Có `nom_char`: **372**

| Bucket | Mô tả | n | % | matched True | matched False | T1 | T2 | T3 | T0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HAN_BASIC` | Hán-share (U+4E00–U+9FFF) | 322 | 86.56% | 314 | 8 | 203 | 27 | 92 | 0 |
| `CJK_EXT_A` | Ext A (U+3400–U+4DBF) — rare CJK + một phần Nôm | 14 | 3.76% | 14 | 0 | 11 | 0 | 3 | 0 |
| `NOM_EXT_B_PLUS` | Ext B+ (U+20000+) — gần như chắc là Nôm thuần | 36 | 9.68% | 33 | 3 | 2 | 1 | 33 | 0 |

## (2) Toàn bộ dataset/all/labels.csv (84k record)

### dataset_all

- Tổng record: **83564**
- Có `nom_char`: **83564**

| Bucket | Mô tả | n | % | matched True | matched False | T1 | T2 | T3 | T0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HAN_BASIC` | Hán-share (U+4E00–U+9FFF) | 68227 | 81.65% | 67196 | 1031 | 36368 | 3689 | 28170 | 0 |
| `CJK_EXT_A` | Ext A (U+3400–U+4DBF) — rare CJK + một phần Nôm | 2448 | 2.93% | 2403 | 45 | 1178 | 330 | 940 | 0 |
| `NOM_EXT_B_PLUS` | Ext B+ (U+20000+) — gần như chắc là Nôm thuần | 9988 | 11.95% | 9744 | 244 | 390 | 483 | 9115 | 0 |
| `NOM_PUA` | PUA — Nôm legacy | 2327 | 2.78% | 2220 | 107 | 0 | 0 | 2327 | 0 |
| `OTHER` | Khác / không CJK | 574 | 0.69% | 561 | 13 | 13 | 0 | 561 | 0 |

## Đọc số liệu

- **debug_p12_p14**: Hán shared **322** (86.6%),  Ext A 14 (3.8%),  **Nôm thuần** (Ext B+ & PUA) **36** (9.7%).
- **dataset_all**: Hán shared **68227** (81.6%),  Ext A 2448 (2.9%),  **Nôm thuần** (Ext B+ & PUA) **12315** (14.7%).

## Cấu trúc Tier theo bucket — dataset toàn phần

Mục tiêu: xem các bucket Nôm thuần (Ext B+ / PUA) thường được gán bằng tầng nào — nếu chủ yếu T3 → cảnh báo độ tin cậy.

| Bucket | n | T1% | T2% | T3% | matched% |
|---|---:|---:|---:|---:|---:|
| `HAN_BASIC` | 68227 | 53.3% | 5.4% | 41.3% | 98.5% |
| `CJK_EXT_A` | 2448 | 48.1% | 13.5% | 38.4% | 98.2% |
| `NOM_EXT_B_PLUS` | 9988 | 3.9% | 4.8% | 91.3% | 97.6% |
| `NOM_PUA` | 2327 | 0.0% | 0.0% | 100.0% | 95.4% |
| `OTHER` | 574 | 2.3% | 0.0% | 97.7% | 97.7% |