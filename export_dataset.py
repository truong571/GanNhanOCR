#!/usr/bin/env python3
"""
export_dataset.py - Xuất dataset chuẩn cuối cùng

Tổng hợp kết quả từ tất cả các bộ sách đã label, xuất thành dataset
chuẩn với train/val/test splits.

Input:
  Nhiều thư mục labeled/:
    data/prepared/SachThanhTruyen2/labeled/labels.csv
    data/prepared/SachThanhTruyen4/labeled/labels.csv
    data/prepared/CacThanhTruyen2/labeled/labels.csv
    ...

Output:
  dataset/
  ├── images/              # Symlink hoặc copy ảnh crop
  ├── typed_images/        # Symlink hoặc copy ảnh font render
  ├── labels.csv           # Nhãn tổng hợp
  ├── train.csv            # Training split
  ├── val.csv              # Validation split
  ├── test.csv             # Test split
  ├── metadata.json        # Thống kê dataset
  └── class_map.json       # Mapping: class_id → (char, unicode, reading)

Usage:
  python export_dataset.py data/prepared/*/labeled \\
      --output dataset/ \\
      --split 0.8 0.1 0.1

  # Chỉ HIGH confidence:
  python export_dataset.py data/prepared/*/labeled \\
      --output dataset/ --min-confidence high
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


def load_labels(labels_csv: Path, source_name: str) -> list[dict]:
    """Load labels.csv, thêm source_name."""
    rows = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["source"] = source_name
            rows.append(row)
    return rows


def filter_by_confidence(rows: list[dict], min_confidence: str) -> list[dict]:
    """Lọc theo mức confidence tối thiểu."""
    levels = {"high": 3, "medium": 2, "low": 1, "gap": 0}
    min_level = levels.get(min_confidence, 0)

    return [r for r in rows if levels.get(r.get("confidence", ""), 0) >= min_level]


def build_class_map(rows: list[dict]) -> dict:
    """Tạo mapping class_id cho mỗi ký tự Nôm unique."""
    chars = set()
    for row in rows:
        nom_char = row.get("nom_char", "")
        if nom_char:
            chars.add(nom_char)

    class_map = {}
    for idx, char in enumerate(sorted(chars)):
        code = f"U+{ord(char):04X}"
        class_map[char] = {
            "class_id": idx,
            "unicode": code,
            "hex": f"{ord(char):04X}",
        }

    return class_map


def split_dataset(
    rows: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Chia dataset thành train/val/test.

    Chiến lược: stratified split theo source (mỗi bộ sách có đại diện trong cả 3 set).
    """
    random.seed(seed)

    # Nhóm theo source
    by_source = {}
    for row in rows:
        src = row.get("source", "unknown")
        by_source.setdefault(src, []).append(row)

    train, val, test = [], [], []

    for src, src_rows in by_source.items():
        random.shuffle(src_rows)
        n = len(src_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(src_rows[:n_train])
        val.extend(src_rows[n_train:n_train + n_val])
        test.extend(src_rows[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def copy_images(rows: list[dict], output_dir: Path, prefix: str = "images"):
    """Copy ảnh crop vào thư mục output, cập nhật đường dẫn trong rows."""
    img_dir = output_dir / prefix
    img_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        src_path = row.get("image", "")
        if not src_path or not Path(src_path).exists():
            continue

        # Tạo tên file unique: source_page_char.png
        src = row.get("source", "unk")
        basename = Path(src_path).name
        new_name = f"{src}_{basename}"
        dst_path = img_dir / new_name

        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

        row["image"] = str(dst_path.relative_to(output_dir))


def save_split(rows: list[dict], output_path: Path, fieldnames: list[str]):
    """Lưu 1 split thành CSV."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Xuất dataset chuẩn")
    parser.add_argument("labeled_dirs", nargs="+", type=str,
                        help="Thư mục labeled/ (chứa labels.csv)")
    parser.add_argument("--output", type=str, default="dataset",
                        help="Thư mục output")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Tỷ lệ train/val/test (mặc định: 0.8 0.1 0.1)")
    parser.add_argument("--min-confidence", choices=["high", "medium", "low", "gap"],
                        default="low",
                        help="Confidence tối thiểu để đưa vào dataset")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy ảnh vào thư mục output (thay vì giữ path gốc)")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load tất cả labels ---
    all_rows = []
    for labeled_dir in args.labeled_dirs:
        labeled_path = Path(labeled_dir)
        labels_csv = labeled_path / "labels.csv"
        if not labels_csv.exists():
            # Thử tìm trong subdirectory
            labels_csv = labeled_path / "labeled" / "labels.csv"
        if not labels_csv.exists():
            print(f"[SKIP] Không tìm thấy labels.csv trong {labeled_dir}")
            continue

        source_name = labeled_path.parent.name if labeled_path.name == "labeled" else labeled_path.name
        rows = load_labels(labels_csv, source_name)
        print(f"  {source_name}: {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] Không có dữ liệu nào")
        return

    print(f"\nTổng: {len(all_rows)} rows")

    # --- Lọc theo confidence ---
    filtered = filter_by_confidence(all_rows, args.min_confidence)
    print(f"Sau lọc (>= {args.min_confidence}): {len(filtered)} rows")

    # --- Thống kê ---
    conf_counts = Counter(r.get("confidence", "") for r in filtered)
    char_counts = Counter(r.get("nom_char", "") for r in filtered if r.get("nom_char"))

    print(f"\nPhân bố confidence:")
    for conf, count in sorted(conf_counts.items()):
        pct = count / len(filtered) * 100
        print(f"  {conf}: {count} ({pct:.1f}%)")

    print(f"\nSố ký tự unique: {len(char_counts)}")
    print(f"Ký tự phổ biến nhất: {char_counts.most_common(5)}")

    # --- Copy images (nếu cần) ---
    if args.copy_images:
        print("\nCopy ảnh...")
        copy_images(filtered, output_dir)

    # --- Build class map ---
    class_map = build_class_map(filtered)
    class_map_path = output_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)
    print(f"\nClass map: {len(class_map)} classes → {class_map_path}")

    # --- Split ---
    train_ratio, val_ratio, test_ratio = args.split
    train, val, test = split_dataset(filtered, train_ratio, val_ratio, test_ratio)

    print(f"\nSplit: train={len(train)}, val={len(val)}, test={len(test)}")

    # --- Save ---
    fieldnames = [
        "image", "nom_char", "unicode", "reading", "confidence",
        "ranking_score", "bbox", "page", "source",
    ]

    # Thêm trường unicode nếu chưa có
    for row in filtered:
        if "unicode" not in row and row.get("nom_char"):
            row["unicode"] = f"U+{ord(row['nom_char']):04X}"

    save_split(filtered, output_dir / "labels.csv", fieldnames)
    save_split(train, output_dir / "train.csv", fieldnames)
    save_split(val, output_dir / "val.csv", fieldnames)
    save_split(test, output_dir / "test.csv", fieldnames)

    # --- Metadata ---
    metadata = {
        "total_samples": len(filtered),
        "num_classes": len(class_map),
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "confidence_distribution": dict(conf_counts),
        "min_confidence": args.min_confidence,
        "sources": list(set(r.get("source", "") for r in filtered)),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"DATASET XUẤT THÀNH CÔNG")
    print(f"{'='*60}")
    print(f"  Output: {output_dir}/")
    print(f"  labels.csv:  {len(filtered)} rows")
    print(f"  train.csv:   {len(train)} rows")
    print(f"  val.csv:     {len(val)} rows")
    print(f"  test.csv:    {len(test)} rows")
    print(f"  Classes:     {len(class_map)}")
    print(f"  metadata:    {meta_path}")


if __name__ == "__main__":
    main()
