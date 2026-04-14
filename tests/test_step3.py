"""Test Step 3: Label — kiểm tra gán nhãn 3 tầng.

Chạy: python tests/test_step3.py [book_name]
  Mặc định: CacThanhTruyen2
  Yêu cầu: Step 2 đã chạy xong.

Kiểm tra bằng mắt:
  1. Tỷ lệ matched vs unmatched
  2. Phân bố tier 1/2/3/0
  3. crop_file trong label trỏ tới crops/ (GỐC), KHÔNG phải crops_cleaned/
  4. Mở dataset.json xem các nhãn có hợp lý không
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step0_setup import load_config
from pipeline.step3_label import label_book

CONFIG_PATH = "config/pipeline.yaml"
BOOK = sys.argv[1] if len(sys.argv) > 1 else "CacThanhTruyen2"


def check_outputs(data_dir: Path):
    """Kiểm tra output Step 3."""
    print("\n" + "=" * 60)
    print("STEP 3 OUTPUT CHECK")
    print("=" * 60)

    labeled_dir = data_dir / "labeled"

    # 1. Summary
    summary_path = labeled_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n[1] Summary:")
        print(f"    Total:     {summary['total_labels']}")
        print(f"    Matched:   {summary['matched']}")
        print(f"    Unmatched: {summary['unmatched']}")
        print(f"    Gaps:      {summary['gaps']}")
        tiers = summary.get("tiers", {})
        print(f"    Tier 1 (dict):    {tiers.get('1', 0)}")
        print(f"    Tier 2 (similar): {tiers.get('2', 0)}")
        print(f"    Tier 3 (visual):  {tiers.get('3', 0)}")
        print(f"    Tier 0 (none):    {tiers.get('0', 0)}")

    # 2. Dataset labels
    dataset_path = labeled_dir / "dataset.json"
    if not dataset_path.exists():
        print("\n  ERROR: dataset.json not found")
        return

    with open(dataset_path) as f:
        labels = json.load(f)

    match_labels = [l for l in labels if l["type"] == "match"]
    print(f"\n[2] Sample labels (first 10 matches):")
    for l in match_labels[:10]:
        print(f"    '{l.get('syllable', '')}' -> {l.get('nom_char', '?')} "
              f"({l.get('unicode', '?')}) "
              f"tier={l.get('tier')} matched={l.get('matched')} "
              f"crop={l.get('crop_file', '')}")

    # 3. CRITICAL: Verify crop_file points to original crops, NOT cleaned
    print(f"\n[3] Crop path check (CRITICAL):")
    cleaned_count = 0
    original_count = 0
    missing_count = 0
    for l in match_labels:
        cf = l.get("crop_file", "")
        if not cf:
            missing_count += 1
        elif "cleaned" in cf:
            cleaned_count += 1
        elif "crops/" in cf:
            original_count += 1

    print(f"    Original (crops/):        {original_count}")
    print(f"    Cleaned (crops_cleaned/): {cleaned_count}")
    print(f"    Missing:                  {missing_count}")

    if cleaned_count > 0:
        print("    FAIL: Dataset references processed images!")
        print("    crop_file should ALWAYS point to crops/ (original), not crops_cleaned/")
    else:
        print("    PASS: All crop paths point to original images")

    # 4. Verify crop files exist
    print(f"\n[4] Crop file existence check:")
    exists = 0
    not_found = 0
    for l in match_labels[:50]:
        cf = l.get("crop_file", "")
        if cf:
            p = data_dir / "detected" / cf
            if p.exists():
                exists += 1
            else:
                not_found += 1
                if not_found <= 3:
                    print(f"    NOT FOUND: {p}")

    print(f"    Exists: {exists}, Not found: {not_found} (checked first 50)")

    print("\n" + "=" * 60)
    print("Open labeled/dataset.json to manually verify labels.")
    print("Check: syllable <-> nom_char makes sense for your domain knowledge.")
    print("=" * 60)


def main():
    config = load_config(CONFIG_PATH)

    print("=" * 60)
    print(f"TEST STEP 3: Label — {BOOK}")
    print("=" * 60)

    data_dir = Path(config["paths"]["data_dir"]) / BOOK

    # Check prerequisites
    aligned_files = list((data_dir / "aligned").glob("*_aligned.json"))
    if not aligned_files:
        print(f"ERROR: No aligned files. Run test_step2.py first.")
        sys.exit(1)

    label_book(config, BOOK, verbose=True)
    check_outputs(data_dir)


if __name__ == "__main__":
    main()
