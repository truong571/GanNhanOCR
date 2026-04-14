"""Step 3: 3-tier label assignment — dictionary -> similar -> FontDiffusion+DINOv2.

Labels are either:
  matched=True  -> correct (BLACK in visualization)
  matched=False -> incorrect/unconfirmed (RED in visualization)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from lib.dictionary import (
    load_qn_to_nom, build_nom_to_qn, load_similarity_dict,
)
from lib.ranker import assign_label, get_dinov2_ranker

from pipeline.step0_setup import load_config


def label_book(config: dict, book_name: str, verbose: bool = True):
    """Run 3-tier labeling for all pages of a book."""
    paths = config["paths"]
    step3_cfg = config.get("step3", {})
    data_dir = Path(paths["data_dir"]) / book_name

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 3: Label — {book_name}")
        print(f"{'='*60}")
        print("  Loading dictionaries...")

    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])
    nom_to_qn = build_nom_to_qn(qn_to_nom)
    similar_dict = load_similarity_dict(paths["similar_dict"])

    if verbose:
        print(f"    QN->Nom: {len(qn_to_nom)} entries")
        print(f"    Similar: {len(similar_dict)} entries")

    # Optional: DINOv2 ranker
    dinov2 = None
    if step3_cfg.get("use_dinov2", False):
        font_path = paths.get("font_path")
        dinov2 = get_dinov2_ranker(font_path)
        if dinov2 and verbose:
            print("    DINOv2 ranker loaded.")

    font_path = paths.get("font_path")
    if font_path and not Path(font_path).exists():
        font_path = None

    aligned_dir = data_dir / "aligned"
    labeled_dir = data_dir / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    aligned_files = sorted(aligned_dir.glob("page_*_aligned.json"))
    if not aligned_files:
        print(f"[ERROR] No aligned files in {aligned_dir}", file=sys.stderr)
        return

    all_labels = []
    tier_counts = {1: 0, 2: 0, 3: 0, 0: 0}
    matched_count = 0
    unmatched_count = 0
    gap_count = 0

    for aligned_path in aligned_files:
        page_name = aligned_path.stem.replace("_aligned", "")

        with open(aligned_path, "r", encoding="utf-8") as f:
            alignment = json.load(f)

        page_labels = []
        for pair in alignment:
            if pair["type"] in ("deletion", "insertion"):
                label = {
                    "page": page_name,
                    "column": pair.get("column"),
                    "type": pair["type"],
                    "syllable": pair.get("syllable"),
                    "nom_char": None,
                    "matched": False,
                    "tier": 0,
                }
                gap_count += 1
            else:
                # type == "match"
                char_info = pair.get("char", {})
                syllable = pair.get("syllable", "")
                ocr_char = char_info.get("ocr_char") if char_info else None

                # Resolve crop path for visual ranking (prefer cleaned for better comparison)
                ranking_crop_path = None
                if char_info:
                    cleaned_file = char_info.get("cleaned_file", "")
                    if cleaned_file:
                        p = data_dir / "detected" / cleaned_file
                        if p.exists():
                            ranking_crop_path = str(p)
                    if not ranking_crop_path:
                        crop_file = char_info.get("crop_file", "")
                        if crop_file:
                            p = data_dir / "detected" / crop_file
                            if p.exists():
                                ranking_crop_path = str(p)

                # 3-tier assignment
                result = assign_label(
                    ocr_char=ocr_char,
                    qn_syllable=syllable,
                    crop_path=ranking_crop_path,
                    qn_to_nom=qn_to_nom,
                    nom_to_qn=nom_to_qn,
                    similar_dict=similar_dict,
                    font_path=font_path,
                    dinov2_ranker=dinov2,
                )

                is_matched = bool(result["matched"])  # convert numpy bool_ to Python bool

                label = {
                    "page": page_name,
                    "column": pair.get("column"),
                    "type": "match",
                    "syllable": syllable,
                    "nom_char": result["nom_char"],
                    "unicode": f"U+{ord(result['nom_char']):04X}" if result["nom_char"] else None,
                    "matched": is_matched,
                    "tier": result["tier"],
                    "nom_candidates": result.get("nom_candidates", []),
                    "ocr_char": ocr_char,
                    "bbox": char_info.get("bbox") if char_info else None,
                    # Always save original crop path — processed images never go into dataset
                    "crop_file": char_info.get("crop_file") if char_info else None,
                }
                tier_counts[result["tier"]] += 1
                if is_matched:
                    matched_count += 1
                else:
                    unmatched_count += 1

            page_labels.append(label)
            all_labels.append(label)

        if verbose:
            page_matched = sum(1 for l in page_labels if l.get("matched"))
            page_total = sum(1 for l in page_labels if l["type"] == "match")
            print(f"  {page_name}: {page_total} labeled, "
                  f"{page_matched} matched (black), "
                  f"{page_total - page_matched} unmatched (red)")

    # Save dataset.json
    with open(labeled_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=2)

    # Save labels.csv
    fieldnames = [
        "page", "column", "syllable", "nom_char", "unicode",
        "matched", "tier", "ocr_char", "bbox", "crop_file",
    ]
    with open(labeled_dir / "labels.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for label in all_labels:
            row = {k: label.get(k) for k in fieldnames}
            if row.get("bbox"):
                row["bbox"] = str(row["bbox"])
            writer.writerow(row)

    # Save summary
    total = len(all_labels)
    summary = {
        "book": book_name,
        "total_labels": total,
        "matched": matched_count,
        "unmatched": unmatched_count,
        "gaps": gap_count,
        "tiers": {str(k): v for k, v in tier_counts.items()},
    }
    with open(labeled_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n  Summary:")
        print(f"    Total labels:  {total}")
        print(f"    Matched (black):  {matched_count}")
        print(f"    Unmatched (red):  {unmatched_count}")
        print(f"    Gaps (skipped):   {gap_count}")
        if matched_count + unmatched_count > 0:
            rate = matched_count / (matched_count + unmatched_count) * 100
            print(f"    Match rate:       {rate:.1f}%")
        print(f"    Tier 1 (dict):    {tier_counts[1]}")
        print(f"    Tier 2 (similar): {tier_counts[2]}")
        print(f"    Tier 3 (visual):  {tier_counts[3]}")
        print(f"    Tier 0 (none):    {tier_counts[0]}")
        print(f"  Output: {labeled_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Step 3: 3-Tier Label Assignment")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    parser.add_argument("book", type=str, help="Book name")
    args = parser.parse_args()

    config = load_config(args.config)
    label_book(config, args.book)


if __name__ == "__main__":
    main()
