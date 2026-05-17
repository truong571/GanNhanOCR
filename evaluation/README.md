# evaluation/

Folder rieng cho cac script danh gia ket qua pipeline. Khong nam trong runtime
(`pipeline/`, `core/`) — chi de phan tich offline sau khi pipeline da chay xong.

## Cau truc

```
evaluation/
├── README.md
├── reports/                          # output (json / md / xlsx)
├── coverage_report.py                # tong ket dataset cuoi (tier, match-rate, long-tail)
├── font_coverage.py                  # check fd_cache cover bao nhieu % dataset + de xuat font bo sung
├── tier1_diagnose.py                 # vi sao tier 1 sup do tren cac sach Sach
├── alignment_quality.py              # AMR + CER (muc 4.4 — can ground truth)
└── export_unmatched_review.py        # xuat 2k mau matched=False kem thumbnail de duyet tay
```

## Cach chay

```bash
# 1. Tong ket dataset
python evaluation/coverage_report.py

# 2. Phan tich font (chu nao trong dataset khong co trong fd_cache, nen them font gi)
python evaluation/font_coverage.py

# 3. Soi tier 1 — vi sao 3 sach Sach chi gan duoc ~1-2% qua tu dien
python evaluation/tier1_diagnose.py

# 4. AMR / CER — can chuan bi truoc ground truth o evaluation/gt/<book>/<page>.json
python evaluation/alignment_quality.py

# 5. Xuat unmatched de review tay
python evaluation/export_unmatched_review.py
```

Tat ca script doc tu `dataset/all/labels.csv` va `prepared/<book>/...`, ghi
output vao `evaluation/reports/`. Khong sua state pipeline.
