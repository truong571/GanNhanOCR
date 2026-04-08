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

import cv2
import numpy as np


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


def filter_crop_quality(rows: list[dict], base_dirs: dict[str, Path],
                        min_size: int = 10, max_size: int = 256,
                        min_ink_ratio: float = 0.02,
                        max_ink_ratio: float = 0.85) -> tuple[list[dict], dict]:
    """Issue #6: Lọc crop quality — loại bỏ ảnh blank/noise/corrupt.

    Kiểm tra:
      1. File tồn tại và đọc được
      2. Kích thước hợp lý (không quá nhỏ = noise, không quá lớn = 2 ký tự)
      3. Tỷ lệ ink pixel hợp lý (không blank, không toàn đen)

    Returns: (filtered_rows, quality_stats)
    """
    quality_stats = {
        "total": len(rows),
        "missing_file": 0,
        "too_small": 0,
        "too_large": 0,
        "blank_image": 0,
        "too_dark": 0,
        "passed": 0,
    }
    filtered = []
    for row in rows:
        img_path_str = row.get("image", "")
        if not img_path_str:
            quality_stats["missing_file"] += 1
            continue

        # Resolve path relative to labeled dir
        source = row.get("source", "")
        base = base_dirs.get(source)
        if base:
            full_path = base / "detected" / img_path_str
        else:
            full_path = Path(img_path_str)

        if not full_path.exists():
            quality_stats["missing_file"] += 1
            continue

        # Read image
        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            quality_stats["missing_file"] += 1
            continue

        h, w = img.shape[:2]

        # Size check
        if h < min_size or w < min_size:
            quality_stats["too_small"] += 1
            continue
        if h > max_size or w > max_size:
            quality_stats["too_large"] += 1
            continue

        # Ink ratio check (Otsu threshold)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_ratio = binary.sum() / 255 / (h * w)

        if ink_ratio < min_ink_ratio:
            quality_stats["blank_image"] += 1
            continue
        if ink_ratio > max_ink_ratio:
            quality_stats["too_dark"] += 1
            continue

        quality_stats["passed"] += 1
        filtered.append(row)

    return filtered, quality_stats


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


def filter_rare_classes(rows: list[dict], min_samples: int = 3) -> tuple[list[dict], int]:
    """Issue #7: Loại bỏ class có quá ít samples (không thể train/test).

    Returns: (filtered_rows, num_removed_classes)
    """
    char_counts = Counter(r.get("nom_char", "") for r in rows if r.get("nom_char"))
    rare_chars = {c for c, count in char_counts.items() if count < min_samples}
    if not rare_chars:
        return rows, 0
    filtered = [r for r in rows if r.get("nom_char", "") not in rare_chars]
    return filtered, len(rare_chars)


def split_dataset(
    rows: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Chia dataset thành train/val/test.

    Issue #7: Stratified split theo CHARACTER CLASS (nom_char) thay vì chỉ source.
    Đảm bảo mỗi ký tự có đại diện trong cả train/val/test.
    Ký tự có ≤2 samples → chỉ vào train (không đủ để split).
    """
    random.seed(seed)

    # Nhóm theo character class
    by_char: dict[str, list[dict]] = {}
    for row in rows:
        char = row.get("nom_char", "_unknown_")
        by_char.setdefault(char, []).append(row)

    train, val, test = [], [], []

    for char, char_rows in by_char.items():
        random.shuffle(char_rows)
        n = len(char_rows)

        if n <= 2:
            # Quá ít → chỉ vào train
            train.extend(char_rows)
            continue

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Đảm bảo test có ít nhất 1 sample nếu đủ data
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_train -= 1
            n_test = 1

        train.extend(char_rows[:n_train])
        val.extend(char_rows[n_train:n_train + n_val])
        test.extend(char_rows[n_train + n_val:])

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
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Bỏ qua crop quality filter (mặc định: bật)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Loại bỏ class có ít hơn N samples (mặc định: 3)")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load tất cả labels ---
    all_rows = []
    base_dirs = {}  # source_name → prepared_dir Path (for quality filter)
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
        # Resolve prepared dir (parent of labeled/)
        prepared_dir = labeled_path.parent if labeled_path.name == "labeled" else labeled_path
        base_dirs[source_name] = prepared_dir

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

    # --- Issue #6: Crop quality filter ---
    if not args.no_quality_filter:
        print("\nKiểm tra chất lượng crop...")
        filtered, qstats = filter_crop_quality(filtered, base_dirs)
        print(f"  Tổng kiểm tra  : {qstats['total']}")
        if qstats["missing_file"]:
            print(f"  File thiếu/lỗi : {qstats['missing_file']}")
        if qstats["too_small"]:
            print(f"  Quá nhỏ (noise): {qstats['too_small']}")
        if qstats["too_large"]:
            print(f"  Quá lớn        : {qstats['too_large']}")
        if qstats["blank_image"]:
            print(f"  Blank/trắng    : {qstats['blank_image']}")
        if qstats["too_dark"]:
            print(f"  Quá đen        : {qstats['too_dark']}")
        print(f"  Đạt chất lượng : {qstats['passed']}")

    # --- Issue #7: Loại class hiếm ---
    if args.min_samples > 1:
        filtered, n_removed = filter_rare_classes(filtered, args.min_samples)
        if n_removed > 0:
            print(f"\nLoại {n_removed} class có <{args.min_samples} samples "
                  f"→ còn {len(filtered)} rows")

    # --- Thống kê ---
    conf_counts = Counter(r.get("confidence", "") for r in filtered)
    char_counts = Counter(r.get("nom_char", "") for r in filtered if r.get("nom_char"))

    print(f"\nPhân bố confidence:")
    for conf, count in sorted(conf_counts.items()):
        pct = count / len(filtered) * 100 if filtered else 0
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
