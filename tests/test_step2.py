"""Test Step 2: Alignment — kiểm tra kết quả căn chỉnh.

Chạy: python tests/test_step2.py [book_name]
  Mặc định: CacThanhTruyen2
  Yêu cầu: Step 1 đã chạy xong cho book này.

Kiểm tra bằng mắt:
  1. Số match vs deletion vs insertion có hợp lý?
  2. Syllable count (normalized) khớp với char count?
  3. Mở aligned JSON xem các cặp (char, syllable) có đúng không?
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step0_setup import load_config
from pipeline.step2_align import align_book

CONFIG_PATH = "config/pipeline.yaml"
BOOK = sys.argv[1] if len(sys.argv) > 1 else "CacThanhTruyen2"


def check_outputs(data_dir: Path):
    """Kiểm tra output Step 2."""
    print("\n" + "=" * 60)
    print("STEP 2 OUTPUT CHECK")
    print("=" * 60)

    aligned_dir = data_dir / "aligned"
    aligned_files = sorted(aligned_dir.glob("*_aligned.json"))
    print(f"\nAligned files: {len(aligned_files)}")

    total_match = 0
    total_del = 0
    total_ins = 0

    for af in aligned_files:
        with open(af) as f:
            alignment = json.load(f)

        matches = [a for a in alignment if a["type"] == "match"]
        deletions = [a for a in alignment if a["type"] == "deletion"]
        insertions = [a for a in alignment if a["type"] == "insertion"]

        total_match += len(matches)
        total_del += len(deletions)
        total_ins += len(insertions)

        page_name = af.stem.replace("_aligned", "")
        print(f"\n  {page_name}:")
        print(f"    Match: {len(matches)}, Deletion: {len(deletions)}, Insertion: {len(insertions)}")

        # Show first few matches for manual inspection
        print("    Sample matches:")
        for m in matches[:5]:
            char_info = m.get("char", {})
            ocr = char_info.get("ocr_char", "?") if char_info else "?"
            syl = m.get("syllable", "")
            bbox = char_info.get("bbox", []) if char_info else []
            print(f"      OCR='{ocr}' <-> QN='{syl}'  bbox={bbox}")

        # Check: syllable count vs char count
        det_path = data_dir / "detected" / f"{page_name}_detection.json"
        trans_path = data_dir / "transcriptions" / f"{page_name}.txt"
        if det_path.exists() and trans_path.exists():
            with open(det_path) as f:
                det = json.load(f)
            lines = trans_path.read_text(encoding="utf-8").strip().split("\n")
            n_chars = det["total_chars"]
            n_syls = sum(len(line.split()) for line in lines)
            diff = abs(n_chars - n_syls)
            status = "OK" if diff <= 3 else f"MISMATCH (diff={diff})"
            print(f"    Chars={n_chars}, Syllables={n_syls} — {status}")

    print(f"\n  TOTAL: {total_match} matches, {total_del} deletions, {total_ins} insertions")
    match_rate = total_match / max(1, total_match + total_del + total_ins) * 100
    print(f"  Match rate: {match_rate:.1f}%")

    if total_ins > total_match * 0.3:
        print("  WARNING: High insertion count — possible syllable count mismatch")
    if total_del > total_match * 0.3:
        print("  WARNING: High deletion count — possible over-segmentation")

    print("\n" + "=" * 60)
    print("Open aligned/*.json to manually verify (char, syllable) pairs.")
    print("=" * 60)


def main():
    config = load_config(CONFIG_PATH)

    print("=" * 60)
    print(f"TEST STEP 2: Alignment — {BOOK}")
    print("=" * 60)

    data_dir = Path(config["paths"]["data_dir"]) / BOOK

    # Check prerequisites
    det_files = list((data_dir / "detected").glob("*_detection.json"))
    if not det_files:
        print(f"ERROR: No detection files. Run test_step1.py first.")
        sys.exit(1)

    align_book(config, BOOK, verbose=True)
    check_outputs(data_dir)


if __name__ == "__main__":
    main()
