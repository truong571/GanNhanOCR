"""Xuất bộ dataset CUỐI CÙNG (usable: GOLD+SILVER+SYLLABLE) từ dataset_out/ ra
một thư mục TỰ CHỨA (labels.csv + ảnh crop copy hẳn, không phụ thuộc dataset_out/).

Gọi bởi run_pipeline.sh sau bước remediate. XOÁ SẠCH --out trước khi ghi, nên
thư mục đó LUÔN LÀ bản mới nhất của lần chạy gần nhất — không cộng dồn qua các
lần chạy trước.

Usage:
    python3 pipeline/export_final_dataset.py \
        --labels dataset_out/labels_remediated.csv --src-root dataset_out --out dataset
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

USABLE_TIERS = {"GOLD", "SILVER", "SYLLABLE"}


def export_dataset(labels_path: Path, src_root: Path, out_root: Path) -> int:
    if not labels_path.exists():
        print(f"[export] không thấy {labels_path}", file=sys.stderr)
        return 1

    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r.get("tier") in USABLE_TIERS]

    if not rows:
        print("[export] 0 dòng usable (GOLD/SILVER/SYLLABLE) trong "
              f"{labels_path} — không ghi gì vào {out_root}.", file=sys.stderr)
        return 1

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    n_copied = 0
    n_missing = 0
    for r in rows:
        rel = r["image"]
        src = src_root / rel
        if not src.exists():
            n_missing += 1
            continue
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    with open(out_root / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    tiers = Counter(r["tier"] for r in rows)
    print(f"[export] {out_root}/labels.csv — {len(rows)} dòng usable "
          f"(GOLD {tiers.get('GOLD', 0)}, SILVER {tiers.get('SILVER', 0)}, "
          f"SYLLABLE {tiers.get('SYLLABLE', 0)})")
    print(f"[export] ảnh: {n_copied} đã copy, {n_missing} thiếu trên đĩa")
    if n_missing:
        print(f"[export] CẢNH BÁO: {n_missing} ảnh có trong {labels_path.name} "
              f"nhưng KHÔNG có file thật trong {src_root}/ — kiểm tra lại bước build.",
              file=sys.stderr)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="dataset_out/labels_remediated.csv")
    ap.add_argument("--src-root", default="dataset_out")
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()
    sys.exit(export_dataset(Path(args.labels), Path(args.src_root), Path(args.out)))


if __name__ == "__main__":
    main()
