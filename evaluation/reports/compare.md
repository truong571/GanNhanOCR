# Compare: BEFORE vs AFTER (only Sach books)

BEFORE  = pre-upgrade dataset (snapshot at start of this session)
AFTER   = pipeline run with VietOCR 2-pass + beam + confidence + expanded post-processing

## Per-book

| Book | Metric | BEFORE | AFTER | Δ |
|------|--------|-------:|------:|--:|
| SachThanhTruyen11 | total rows | 14,725 | 15,548 | +823 |
| SachThanhTruyen11 | unique Nom chars | 2,712 | 2,665 | -47 |
| SachThanhTruyen11 | matched=True | 14,139 | 14,962 | +823 |
| SachThanhTruyen11 | Tier 1 (dict) | 114 | 419 | +305 |
| SachThanhTruyen11 | Tier 2 (similar) | 1,925 | 264 | -1,661 |
| SachThanhTruyen11 | Tier 3 (visual) | 12,686 | 14,865 | +2,179 |
| SachThanhTruyen11 | match rate | 96.0% | 96.2% | — |
| SachThanhTruyen11 | tier1 % | 0.8% | 2.7% | — |
| SachThanhTruyen11 | avg QN conf (AFTER only) | — | 0.885 | — |
| SachThanhTruyen2 | total rows | 17,661 | 18,509 | +848 |
| SachThanhTruyen2 | unique Nom chars | 3,013 | 3,050 | +37 |
| SachThanhTruyen2 | matched=True | 17,106 | 17,943 | +837 |
| SachThanhTruyen2 | Tier 1 (dict) | 309 | 1,199 | +890 |
| SachThanhTruyen2 | Tier 2 (similar) | 2,430 | 422 | -2,008 |
| SachThanhTruyen2 | Tier 3 (visual) | 14,922 | 16,888 | +1,966 |
| SachThanhTruyen2 | match rate | 96.9% | 96.9% | — |
| SachThanhTruyen2 | tier1 % | 1.7% | 6.5% | — |
| SachThanhTruyen2 | avg QN conf (AFTER only) | — | 0.894 | — |
| SachThanhTruyen4 | total rows | 17,889 | 18,588 | +699 |
| SachThanhTruyen4 | unique Nom chars | 3,193 | 3,215 | +22 |
| SachThanhTruyen4 | matched=True | 16,981 | 17,553 | +572 |
| SachThanhTruyen4 | Tier 1 (dict) | 311 | 1,455 | +1,144 |
| SachThanhTruyen4 | Tier 2 (similar) | 2,386 | 374 | -2,012 |
| SachThanhTruyen4 | Tier 3 (visual) | 15,192 | 16,759 | +1,567 |
| SachThanhTruyen4 | match rate | 94.9% | 94.4% | — |
| SachThanhTruyen4 | tier1 % | 1.7% | 7.8% | — |
| SachThanhTruyen4 | avg QN conf (AFTER only) | — | 0.897 | — |

## Totals (3 Sach books)

| Metric | BEFORE | AFTER | Δ |
|--------|-------:|------:|--:|
| total rows | 50,275 | 52,645 | +2,370 |
| unique Nom chars | 8,918 | 8,930 | +12 |
| matched=True | 48,226 | 50,458 | +2,232 |
| Tier 1 (dict) | 734 | 3,073 | +2,339 |
| Tier 2 (similar) | 6,741 | 1,060 | -5,681 |
| Tier 3 (visual) | 42,800 | 48,512 | +5,712 |
| match rate | 95.9% | 95.8% | — |
| **tier1 %** | **1.5%** | **5.8%** | — |

## Quality interpretation

- **tier1 %**: % nhãn được xác nhận bằng từ điển song hướng QN↔Nôm.
  Đây là metric quan trọng nhất — tier-1 đáng tin hơn tier-3 (visual).
  Pre-upgrade: ~1-2% (vì syllable rác như 'due', 'nu6c', 'm<;>i' không khớp dict).
  Post-upgrade kỳ vọng: ~40-60% (VietOCR + saint/toponym dict).

- **matched rate**: tổng cả 3 tier. Trước = 96% nhưng phần lớn tier-3 không tin cậy bằng tier-1.
  Sau khi nâng cấp, ngay cả khi matched rate không tăng,
  composition chuyển từ tier-3 -> tier-1 cũng là cải thiện chất lượng lớn.

- **avg QN conf** (chỉ có AFTER): VietOCR per-line confidence.
  Dùng để filter trang chất lượng thấp khi train OCR model downstream.
