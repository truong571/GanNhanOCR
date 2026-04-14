"""Test Step 4: Export — kiểm tra dataset cuối cùng.

Chạy: python tests/test_step4.py
  Yêu cầu: Step 3 đã chạy xong cho ít nhất 1 book.

Kiểm tra bằng mắt:
  1. train/val/test split hợp lý?
  2. crop_file trong CSV trỏ tới ảnh GỐC?
  3. class_map.json đúng format?
  4. Ảnh crop có đọc được không?
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


def check_outputs(config: dict):
    """Kiểm tra output Step 4."""
    output_dir = Path(config["paths"]["output_dir"])
    data_dir = Path(config["paths"]["data_dir"])

    print("\n" + "=" * 60)
    print("STEP 4 OUTPUT CHECK")
    print("=" * 60)

    # 1. Metadata
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n[1] Metadata:")
        print(f"    Total samples: {meta['total_samples']}")
        print(f"    Classes:       {meta['num_classes']}")
        print(f"    Train:         {meta['splits']['train']}")
        print(f"    Val:           {meta['splits']['val']}")
        print(f"    Test:          {meta['splits']['test']}")
        print(f"    Matched:       {meta['matched']}")
        print(f"    Unmatched:     {meta['unmatched']}")
        print(f"    Sources:       {meta['sources']}")

    # 2. Class map
    cm_path = output_dir / "class_map.json"
    if cm_path.exists():
        with open(cm_path) as f:
            cm = json.load(f)
        print(f"\n[2] Class map: {len(cm)} classes")
        sample = list(cm.items())[:5]
        for char, info in sample:
            print(f"    '{char}' -> id={info['class_id']} unicode={info['unicode']}")

    # 3. CSV splits
    print(f"\n[3] CSV files:")
    for name in ["labels.csv", "train.csv", "val.csv", "test.csv"]:
        p = output_dir / name
        if p.exists():
            with open(p) as f:
                rows = list(csv.DictReader(f))
            print(f"    {name}: {len(rows)} rows")
        else:
            print(f"    {name}: NOT FOUND")

    # 4. CRITICAL: Verify crop paths are original
    print(f"\n[4] Crop path check (CRITICAL):")
    labels_path = output_dir / "labels.csv"
    if labels_path.exists():
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

        print(f"    Original (crops/):        {original_count}")
        print(f"    Cleaned (crops_cleaned/): {cleaned_count}")

        if cleaned_count > 0:
            print("    FAIL: Dataset contains processed image paths!")
        else:
            print("    PASS: All paths point to original crops")

    # 5. Verify sample images are readable
    print(f"\n[5] Image readability check (sample 20):")
    if labels_path.exists():
        with open(labels_path) as f:
            rows = list(csv.DictReader(f))

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
                        print(f"    CANNOT READ: {full_path}")
                else:
                    not_found += 1
                    if not_found <= 3:
                        print(f"    NOT FOUND: {full_path}")

        print(f"    Readable: {readable}, Not found: {not_found}")

    print("\n" + "=" * 60)
    print(f"Output directory: {output_dir}/")
    print("Open labels.csv and verify crop images manually.")
    print("=" * 60)


def main():
    config = load_config(CONFIG_PATH)

    print("=" * 60)
    print("TEST STEP 4: Export Dataset")
    print("=" * 60)

    # Check prerequisites
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
