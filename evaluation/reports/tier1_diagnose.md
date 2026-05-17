# Tier-1 diagnostic

Cot quan trong:
- `tier1_pct`     = % mau duoc gan bang Tier-1 (tu dien).
- `syl_in_dict_pct` = % syllable XUAT HIEN trong tu dien QN.
- `syl_empty_pct` = % syllable rong / NaN.

Neu `syl_in_dict_pct` cao ma `tier1_pct` thap -> alignment hong (syllable
khop voi sai ky tu). Neu ca hai deu thap -> syllable bi noise tu Tesseract.

| source            |   rows |   tier1_pct |   syl_in_dict_pct |   syl_empty_pct |
|:------------------|-------:|------------:|------------------:|----------------:|
| SachThanhTruyen11 |  15548 |        2.69 |             99.62 |            0.01 |
| SachThanhTruyen2  |  18509 |        6.48 |             99.27 |            0    |
| SachThanhTruyen4  |  18588 |        7.83 |             99.71 |            0.01 |

## Top 15 syllable KHONG co trong tu dien (per source)

### SachThanhTruyen11

| syl_norm   |   count |
|:-----------|--------:|
| ay         |       4 |
| giu        |       3 |
| trấy       |       3 |
| truyen     |       3 |
| hế         |       2 |
| ngays      |       2 |
| rac        |       2 |
| truoc      |       2 |
| com        |       2 |
| ây         |       1 |
| imình      |       1 |
| ldm        |       1 |
| tran       |       1 |
| góm        |       1 |
| nàv        |       1 |

### SachThanhTruyen2

| syl_norm   |   count |
|:-----------|--------:|
| giu        |      28 |
| ay         |       9 |
| trấy       |       8 |
| truyen     |       6 |
| ga         |       4 |
| muon       |       4 |
| but        |       3 |
| th         |       2 |
| vao        |       2 |
| goi        |       2 |
| dố         |       2 |
| xac        |       2 |
| ina        |       1 |
| nh         |       1 |
| nóing      |       1 |

### SachThanhTruyen4

| syl_norm   |   count |
|:-----------|--------:|
| giu        |      14 |
| ga         |       5 |
| ây         |       4 |
| trấy       |       3 |
| ay         |       2 |
| hế         |       2 |
| sàu        |       2 |
| truyen     |       2 |
| but        |       2 |
| dau        |       1 |
| góm        |       1 |
| thấv       |       1 |
| ơc         |       1 |
| trằng      |       1 |
| hồm        |       1 |

## Doc ket qua

- Neu top-list co nhieu chuoi 1-2 ky tu (`fa`, `ay`, `ia`, `phai`, `ro`,...)
  -> QN OCR bi Tesseract chia nho sai. Can: re-run buoc QN OCR voi config tot hon,
  hoac dung text PDF embedded thay vi OCR.
- Neu top-list co nhieu ten rieng / dia danh -> mo rong tu dien (add aliases).
- Neu sach Sach co nhieu syllable in_dict NHUNG tier1 thap -> alignment lech,
  xem lai chi phi xoa/them trong Buoc 2.
