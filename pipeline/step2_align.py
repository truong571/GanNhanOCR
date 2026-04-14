"""Step 2: Levenshtein alignment — match N detected chars with M QN syllables."""

import argparse
import json
import sys
from pathlib import Path

from lib.alignment import levenshtein_align
from lib.dictionary import load_qn_to_nom

from pipeline.step0_setup import load_config


def align_page(
    detection_path: Path,
    transcription_path: Path,
    qn_to_nom: dict,
) -> list[dict]:
    """Align one page: detected chars <-> QN syllables per column."""
    with open(detection_path, "r", encoding="utf-8") as f:
        detection = json.load(f)

    # Load transcription
    with open(transcription_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    page_alignment = []

    for col_data in detection["columns"]:
        col_idx = col_data["column"] - 1
        chars = col_data["chars"]

        # Get syllables for this column (already normalized in Step 1)
        if col_idx < len(lines):
            syllables = lines[col_idx].split()
        else:
            syllables = []

        # Run Levenshtein alignment
        aligned = levenshtein_align(
            chars, syllables,
            qn_to_nom=qn_to_nom,
        )

        for pair in aligned:
            pair["column"] = col_data["column"]

        page_alignment.extend(aligned)

    return page_alignment


def align_book(config: dict, book_name: str, verbose: bool = True):
    """Run alignment for all pages of a book."""
    paths = config["paths"]
    data_dir = Path(paths["data_dir"]) / book_name

    # Load dictionary
    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])

    detected_dir = data_dir / "detected"
    trans_dir = data_dir / "transcriptions"
    aligned_dir = data_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    # Find all detection files
    det_files = sorted(detected_dir.glob("page_*_detection.json"))
    if not det_files:
        print(f"[ERROR] No detection files in {detected_dir}", file=sys.stderr)
        return

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 2: Alignment — {book_name}")
        print(f"  Pages: {len(det_files)}")
        print(f"{'='*60}")

    total_matches = 0
    total_gaps = 0

    for det_path in det_files:
        page_name = det_path.stem.replace("_detection", "")
        trans_path = trans_dir / f"{page_name}.txt"

        if not trans_path.exists():
            if verbose:
                print(f"  [SKIP] {page_name}: no transcription")
            continue

        alignment = align_page(det_path, trans_path, qn_to_nom)

        # Save alignment
        out_path = aligned_dir / f"{page_name}_aligned.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(alignment, f, ensure_ascii=False, indent=2)

        matches = sum(1 for a in alignment if a["type"] == "match")
        gaps = sum(1 for a in alignment if a["type"] in ("deletion", "insertion"))
        total_matches += matches
        total_gaps += gaps

        if verbose:
            print(f"  {page_name}: {matches} matches, {gaps} gaps")

    if verbose:
        print(f"\n  Total: {total_matches} matches, {total_gaps} gaps")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Levenshtein Alignment")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    parser.add_argument("book", type=str, help="Book name")
    args = parser.parse_args()

    config = load_config(args.config)
    align_book(config, args.book)


if __name__ == "__main__":
    main()
