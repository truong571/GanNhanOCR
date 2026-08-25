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
        all_rows = list(reader)
        rows = [r for r in all_rows if r.get("tier") in USABLE_TIERS]

    # LOẠI TRỪ PHẢI ỒN ÀO. Từ 2026-08-19 tier SILVER_uncalibrated (10.890 ô do S3
    # quyết, 0 verdict người) nằm ngoài USABLE_TIERS nên tự rơi khỏi bộ giao nộp —
    # nếu im lặng thì một hôm nào đó 10.890 ô biến mất mà không ai biết vì sao.
    excluded = Counter(r.get("tier") for r in all_rows if r.get("tier") not in USABLE_TIERS)
    for tier, n in sorted(excluded.items(), key=lambda kv: -kv[1]):
        print(f"[export] LOẠI khỏi bộ giao nộp: {tier or '(trống)':22} {n:>7,} dòng")

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
        if "usable_image" not in fieldnames:
            fieldnames = list(fieldnames) + ["usable_image"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    tiers = Counter(r["tier"] for r in rows)
    # TÁCH BẠCH HAI LOẠI NHÃN (2026-08-25, sau phản biện hội đồng). Gộp GOLD+SYLLABLE
    # thành một con số là THỔI SỐ: tầng SYLLABLE có cột `label` RỖNG — nó chỉ ghi ÂM
    # Quốc ngữ, KHÔNG gán chữ Nôm nào. Sản phẩm gán nhãn CẤP KÝ TỰ chỉ là phần GOLD.
    # GẮN CỜ ẢNH DÙNG ĐƯỢC (2026-08-25). Đo được 424 ô có ảnh hỏng (34+330 GOLD, 6+54
    # SYLLABLE) vẫn nằm trong bộ giao nộp. KHÔNG loại chúng: nhãn có thể vẫn ĐÚNG dù
    # ảnh hỏng, và loại đi sẽ đổi số + phá chuỗi băm. Thay vào đó gắn cờ để (a) người
    # chấm biết bỏ qua chiều ẢNH thay vì chấm nhầm thành "nhãn sai", (b) người huấn
    # luyện mô hình lọc được.
    for r in rows:
        r["usable_image"] = "0" if r.get("crop_quality_flag") in ("blank", "truncated") else "1"
    _nbad = sum(1 for r in rows if r["usable_image"] == "0")
    _nchar = sum(1 for r in rows if (r.get("label") or "").strip())
    _nsyl = len(rows) - _nchar
    print(f"[export] {out_root}/labels.csv — {_nchar:,} nhãn CẤP KÝ TỰ "
          f"+ {_nsyl:,} chú giải CẤP ÂM TIẾT = {len(rows):,} dòng "
          f"(GOLD {tiers.get('GOLD', 0)}, SILVER {tiers.get('SILVER', 0)}, "
          f"SYLLABLE {tiers.get('SYLLABLE', 0)})")
    print(f"[export] ⚠️ con số đem so với bộ dữ liệu Hán Nôm khác là {_nchar:,}, "
          f"KHÔNG phải {len(rows):,}")
    print(f"[export] usable_image=0 (ảnh trắng/cụt, ĐỪNG chấm chiều ảnh): {_nbad:,} ô")
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
