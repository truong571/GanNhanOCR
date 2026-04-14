"""Test Step 4: Export — kiem tra dataset cuoi cung.

Chay: python tests/test_step4.py
  Yeu cau: Step 3 da chay xong cho it nhat 1 book.

Kiem tra bang mat:
  1. Moi book co folder rieng trong dataset/?
  2. Folder all/ gop du lieu tu tat ca book?
  3. crop_file tro toi anh GOC?
  4. Anh crop co doc duoc?
"""

import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step0_setup import load_config
from pipeline.step4_export import export_dataset

CONFIG_PATH = "config/pipeline.yaml"


def check_subset(subset_dir: Path, data_dir: Path, name: str):
    """Kiem tra 1 subset (per-book hoac all)."""
    print(f"\n  [{name}]")

    meta_path = subset_dir / "metadata.json"
    if not meta_path.exists():
        print(f"    NOT FOUND: {subset_dir}/")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"    Samples: {meta['total_samples']}, "
          f"Classes: {meta['num_classes']}, "
          f"Matched: {meta['matched']}, "
          f"Unmatched: {meta['unmatched']}")

    # Check crop paths
    labels_path = subset_dir / "labels.csv"
    if not labels_path.exists():
        print(f"    labels.csv NOT FOUND")
        return

    with open(labels_path) as f:
        rows = list(csv.DictReader(f))

    cleaned_count = 0
    original_count = 0
    for r in rows:
        cf = r.get("crop_file", "")
        if "cleaned" in cf:
            cleaned_count += 1
        elif "crops/" in cf:
            original_count += 1

    print(f"    Crop paths: {original_count} original, {cleaned_count} cleaned")
    if cleaned_count > 0:
        print(f"    FAIL: Contains processed image paths!")
    else:
        print(f"    PASS: All original")

    # Sample image check
    readable = 0
    not_found = 0
    for r in rows[:20]:
        cf = r.get("crop_file", "")
        source = r.get("source", "")
        if cf and source:
            full_path = data_dir / source / "detected" / cf
            if full_path.exists():
                img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    readable += 1
            else:
                not_found += 1
    print(f"    Image check (20 samples): {readable} readable, {not_found} not found")


def check_outputs(config: dict):
    """Kiem tra output Step 4."""
    output_dir = Path(config["paths"]["output_dir"])
    data_dir = Path(config["paths"]["data_dir"])

    print("\n" + "=" * 60)
    print("STEP 4 OUTPUT CHECK")
    print("=" * 60)

    # Per-book subsets
    for book in config["books"]:
        name = book["name"]
        check_subset(output_dir / name, data_dir, name)

    # All-in-one
    check_subset(output_dir / "all", data_dir, "all")

    print("\n" + "=" * 60)
    print(f"Output: {output_dir}/")
    print("=" * 60)


def main():
    config = load_config(CONFIG_PATH)

    print("=" * 60)
    print("TEST STEP 4: Export Dataset")
    print("=" * 60)

    data_dir = Path(config["paths"]["data_dir"])
    has_labels = False
    for book in config["books"]:
        if (data_dir / book["name"] / "labeled" / "labels.csv").exists():
            has_labels = True
            break
    if not has_labels:
        print("ERROR: No labels.csv found. Run test_step3.py first.")
        sys.exit(1)

    export_dataset(config, verbose=True)
    check_outputs(config)


if __name__ == "__main__":
    main()
