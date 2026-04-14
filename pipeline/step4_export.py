"""Step 4: Merge labels from all books, filter, stratified split -> final dataset.

No confidence filtering. Labels are either matched (black) or unmatched (red).
"""

import argparse
import csv
import json
import random
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


def split_dataset(
    rows: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by character class."""
    random.seed(seed)
    train_r, val_r, _ = ratios

    by_char: dict[str, list[dict]] = {}
    for row in rows:
        char = row.get("nom_char", "_unknown_")
        by_char.setdefault(char, []).append(row)

    train, val, test = [], [], []
    for char, char_rows in by_char.items():
        random.shuffle(char_rows)
        n = len(char_rows)
        if n <= 2:
            train.extend(char_rows)
            continue
        n_train = max(1, int(n * train_r))
        n_val = max(1, int(n * val_r))
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


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_dataset(config: dict, verbose: bool = True):
    """Run Step 4: export final dataset."""
    paths = config["paths"]
    step4_cfg = config.get("step4", {})
    data_dir = Path(paths["data_dir"])
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print("Step 4: Export Dataset")
        print(f"{'='*60}")

    # Load all labels
    all_rows = []
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
        if verbose:
            print(f"  {name}: {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("[ERROR] No data.", file=sys.stderr)
        return

    # Only keep rows with nom_char (skip gaps)
    filtered = [r for r in all_rows if r.get("nom_char")]
    if verbose:
        print(f"\n  After removing gaps: {len(filtered)}")

    # Filter quality
    filtered, qstats = filter_crop_quality(filtered, base_dirs)
    if verbose:
        print(f"  After quality filter: {qstats['passed']} passed, {qstats['filtered']} filtered")

    # Filter rare classes
    min_samples = step4_cfg.get("min_samples_per_class", 3)
    if min_samples > 1:
        filtered, n_removed = filter_rare_classes(filtered, min_samples)
        if verbose and n_removed:
            print(f"  Removed {n_removed} classes with <{min_samples} samples")

    # Add unicode field
    for row in filtered:
        if "unicode" not in row and row.get("nom_char"):
            row["unicode"] = f"U+{ord(row['nom_char']):04X}"

    # Build class map
    class_map = build_class_map(filtered)
    with open(output_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    # Split
    ratios = tuple(step4_cfg.get("split_ratios", [0.8, 0.1, 0.1]))
    seed = step4_cfg.get("seed", 42)
    train, val, test = split_dataset(filtered, ratios, seed)

    # Save
    fieldnames = [
        "crop_file", "nom_char", "unicode", "syllable", "matched",
        "tier", "bbox", "page", "source",
    ]
    save_csv(filtered, output_dir / "labels.csv", fieldnames)
    save_csv(train, output_dir / "train.csv", fieldnames)
    save_csv(val, output_dir / "val.csv", fieldnames)
    save_csv(test, output_dir / "test.csv", fieldnames)

    # Metadata
    matched_count = sum(1 for r in filtered if r.get("matched") in (True, "True"))
    unmatched_count = len(filtered) - matched_count
    metadata = {
        "total_samples": len(filtered),
        "num_classes": len(class_map),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "matched": matched_count,
        "unmatched": unmatched_count,
        "sources": sorted(set(r.get("source", "") for r in filtered)),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n  {'='*50}")
        print(f"  DATASET EXPORTED")
        print(f"  {'='*50}")
        print(f"  Output:    {output_dir}/")
        print(f"  Total:     {len(filtered)} samples")
        print(f"  Classes:   {len(class_map)}")
        print(f"  Matched:   {matched_count} (black)")
        print(f"  Unmatched: {unmatched_count} (red)")
        print(f"  Train:     {len(train)}")
        print(f"  Val:       {len(val)}")
        print(f"  Test:      {len(test)}")


def main():
    parser = argparse.ArgumentParser(description="Step 4: Export Dataset")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    export_dataset(config)


if __name__ == "__main__":
    main()
