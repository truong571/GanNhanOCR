"""Step 4: Export dataset — per-book + all-in-one.

Output structure:
  dataset/
    CacThanhTruyen2/    <- per-book
      labels.csv
      class_map.json
      metadata.json
    CacThanhTruyen4/
      ...
    all/                <- merged from all books
      labels.csv
      class_map.json
      metadata.json
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

from pipeline.step0_setup import load_config


def load_labels(labels_csv: Path, source_name: str) -> list[dict]:
    """Load labels.csv with source tracking."""
    rows = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["source"] = source_name
            rows.append(row)
    return rows


def filter_crop_quality(
    rows: list[dict],
    base_dirs: dict[str, Path],
    min_ink: float = 0.02,
    max_ink: float = 0.85,
) -> tuple[list[dict], dict]:
    """Filter out blank, corrupt, or over-dark crop images."""
    stats = {"total": len(rows), "passed": 0, "filtered": 0}
    filtered = []

    for row in rows:
        crop_file = row.get("crop_file", "")
        if not crop_file:
            stats["filtered"] += 1
            continue

        source = row.get("source", "")
        base = base_dirs.get(source)
        if base:
            full_path = base / "detected" / crop_file
        else:
            full_path = Path(crop_file)

        if not full_path.exists():
            stats["filtered"] += 1
            continue

        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["filtered"] += 1
            continue

        h, w = img.shape[:2]
        if h < 10 or w < 10 or h > 256 or w > 256:
            stats["filtered"] += 1
            continue

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_ratio = binary.sum() / 255 / (h * w)
        if ink_ratio < min_ink or ink_ratio > max_ink:
            stats["filtered"] += 1
            continue

        stats["passed"] += 1
        filtered.append(row)

    return filtered, stats


def filter_rare_classes(rows: list[dict], min_samples: int = 3) -> tuple[list[dict], int]:
    char_counts = Counter(r.get("nom_char", "") for r in rows if r.get("nom_char"))
    rare = {c for c, n in char_counts.items() if n < min_samples}
    if not rare:
        return rows, 0
    return [r for r in rows if r.get("nom_char", "") not in rare], len(rare)


def build_class_map(rows: list[dict]) -> dict:
    chars = sorted(set(r.get("nom_char", "") for r in rows if r.get("nom_char")))
    return {
        char: {
            "class_id": idx,
            "unicode": f"U+{ord(char):04X}",
            "hex": f"{ord(char):04X}",
        }
        for idx, char in enumerate(chars)
    }


FIELDNAMES = [
    "crop_file", "nom_char", "unicode", "syllable", "matched",
    "tier", "bbox", "page", "source",
]


def save_csv(rows: list[dict], path: Path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_crops(rows: list[dict], out_dir: Path, base_dirs: dict[str, Path]) -> int:
    """Copy crop images into dataset output folder."""
    copied = 0
    for row in rows:
        crop_file = row.get("crop_file", "")
        source = row.get("source", "")
        if not crop_file or not source:
            continue
        base = base_dirs.get(source)
        if not base:
            continue
        src = base / "detected" / crop_file
        dst = out_dir / crop_file
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            copied += 1
    return copied


def save_subset(
    rows: list[dict],
    out_dir: Path,
    name: str,
    base_dirs: dict[str, Path],
    verbose: bool = True,
):
    """Filter, build class_map, copy crops, save one subset."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add unicode field
    for row in rows:
        if "unicode" not in row and row.get("nom_char"):
            row["unicode"] = f"U+{ord(row['nom_char']):04X}"

    # Build class map
    class_map = build_class_map(rows)
    with open(out_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    # Save labels
    save_csv(rows, out_dir / "labels.csv")

    # Copy crop images into dataset folder
    copied = copy_crops(rows, out_dir, base_dirs)

    # Metadata
    matched_count = sum(1 for r in rows if r.get("matched") in (True, "True"))
    unmatched_count = len(rows) - matched_count
    metadata = {
        "total_samples": len(rows),
        "num_classes": len(class_map),
        "matched": matched_count,
        "unmatched": unmatched_count,
        "sources": sorted(set(r.get("source", "") for r in rows)),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"    {name}: {len(rows)} samples, "
              f"{len(class_map)} classes, "
              f"{matched_count} matched, {unmatched_count} unmatched, "
              f"{copied} crops copied")


def export_dataset(config: dict, verbose: bool = True):
    """Run Step 4: export per-book + all-in-one dataset."""
    paths = config["paths"]
    step4_cfg = config.get("step4", {})
    data_dir = Path(paths["data_dir"])
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    min_samples = step4_cfg.get("min_samples_per_class", 3)

    if verbose:
        print(f"\n{'='*60}")
        print("Step 4: Export Dataset")
        print(f"{'='*60}")

    # Load and export per-book
    all_filtered: list[dict] = []
    base_dirs: dict[str, Path] = {}

    for book in config["books"]:
        name = book["name"]
        labels_csv = data_dir / name / "labeled" / "labels.csv"
        if not labels_csv.exists():
            if verbose:
                print(f"  [SKIP] {name}: no labels.csv")
            continue

        rows = load_labels(labels_csv, name)
        base_dirs[name] = data_dir / name

        # Filter: remove gaps
        rows = [r for r in rows if r.get("nom_char")]

        # Filter: crop quality
        rows, _ = filter_crop_quality(rows, base_dirs)

        # Filter: rare classes (per-book)
        if min_samples > 1:
            rows, _ = filter_rare_classes(rows, min_samples)

        # Save per-book
        book_dir = output_dir / name
        save_subset(rows, book_dir, name, base_dirs, verbose=verbose)

        all_filtered.extend(rows)

    if not all_filtered:
        print("[ERROR] No data.", file=sys.stderr)
        return

    # Filter rare classes again on merged set
    if min_samples > 1:
        all_filtered, _ = filter_rare_classes(all_filtered, min_samples)

    # Save all-in-one
    all_dir = output_dir / "all"
    save_subset(all_filtered, all_dir, "all", base_dirs, verbose=verbose)

    if verbose:
        print(f"\n  {'='*50}")
        print(f"  DATASET EXPORTED -> {output_dir}/")
        print(f"  {'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Step 4: Export Dataset")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    export_dataset(config)


if __name__ == "__main__":
    main()
