"""Step 2: Align + Crop using OCR API bboxes.

Flow:
  1. Load OCR API results (bbox per character) as primary positions
  2. Levenshtein alignment: OCR chars <-> QN syllables → exact match count
  3. Crop matched characters from ORIGINAL image using OCR bboxes
  4. Fallback: projection profile when OCR bbox not available
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lib.alignment import levenshtein_align
from lib.char_segmenter import segment_characters_in_column
from lib.column_detector import detect_columns, auto_detect_n_columns
from lib.crop_cleaner import CharacterCleaner
from lib.dictionary import load_qn_to_nom
from lib.image_processing import load_and_binarize, detect_text_box

from pipeline.step0_setup import load_config


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _build_chars_from_ocr(ocr_col: list[dict]) -> list[dict]:
    """Build char info list from OCR API column data."""
    chars = []
    for i, oc in enumerate(ocr_col):
        bbox = oc["bbox"]  # [x_left, y_top, x_right, y_bottom]
        chars.append({
            "char_idx": i,
            "bbox": [int(b) for b in bbox],
            "width": int(bbox[2] - bbox[0]),
            "height": int(bbox[3] - bbox[1]),
            "ocr_char": oc.get("char"),
        })
    return chars


def _build_chars_from_projection(
    binary: np.ndarray,
    col_bbox: tuple,
    expected_count: int,
) -> list[dict]:
    """Fallback: build char info from projection-based segmentation."""
    char_bboxes = segment_characters_in_column(
        binary, col_bbox, expected_count=expected_count,
    )
    chars = []
    for i, (cx1, cy1, cx2, cy2) in enumerate(char_bboxes):
        chars.append({
            "char_idx": i,
            "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
            "width": int(cx2 - cx1),
            "height": int(cy2 - cy1),
            "ocr_char": None,
        })
    return chars


def process_page(
    page_name: str,
    data_dir: Path,
    qn_to_nom: dict,
    step1_cfg: dict,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Full pipeline for one page: align using OCR bbox → crop from original.

    Returns (alignment, stats).
    """
    pages_dir = data_dir / "pages"
    denoised_dir = data_dir / "pages_denoised"
    trans_dir = data_dir / "transcriptions"
    crops_dir = data_dir / "detected" / "crops"
    cleaned_dir = data_dir / "detected" / "crops_cleaned"

    img_path = pages_dir / f"{page_name}.png"
    trans_path = trans_dir / f"{page_name}.txt"

    if not img_path.exists() or not trans_path.exists():
        return [], {}

    # Load original grayscale (for cropping)
    gray_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        return [], {}

    # Read transcription (syllables per column)
    with open(trans_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    n_columns = len(lines)
    if n_columns == 0:
        return [], {}
    syllables_per_col = [line.split() for line in lines]

    # Load OCR API cache (primary source for character positions)
    ocr_columns = None
    ocr_cache_path = data_dir / "detected" / f"{page_name}_ocr_cache.json"
    if ocr_cache_path.exists():
        with open(ocr_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        ocr_columns = cached.get("columns")

    # Load denoised binary (fallback for projection-based segmentation)
    denoised_path = denoised_dir / f"{page_name}.png"
    binary = None  # lazy load only if needed

    # ── Phase 1: Build char list per column ──
    chars_per_col: list[list[dict]] = []
    use_ocr_bbox = ocr_columns is not None

    if use_ocr_bbox:
        # Use OCR API bboxes as character positions
        for col_idx in range(n_columns):
            if col_idx < len(ocr_columns) and ocr_columns[col_idx]:
                chars = _build_chars_from_ocr(ocr_columns[col_idx])
            else:
                # Fallback for columns missing from OCR
                if binary is None:
                    bin_path = str(denoised_path) if denoised_path.exists() else str(img_path)
                    _, binary = load_and_binarize(bin_path)
                text_box = detect_text_box(binary)
                col_bboxes = detect_columns(binary, text_box, n_expected=n_columns)
                exp = len(syllables_per_col[col_idx]) if col_idx < len(syllables_per_col) else 10
                chars = _build_chars_from_projection(binary, col_bboxes[col_idx], exp)
            chars_per_col.append(chars)
    else:
        # No OCR: full projection-based segmentation
        bin_path = str(denoised_path) if denoised_path.exists() else str(img_path)
        _, binary = load_and_binarize(bin_path)
        text_box = detect_text_box(binary)
        col_bboxes = detect_columns(binary, text_box, n_expected=n_columns)
        for col_idx in range(n_columns):
            if col_idx < len(col_bboxes):
                exp = len(syllables_per_col[col_idx]) if col_idx < len(syllables_per_col) else None
                chars = _build_chars_from_projection(binary, col_bboxes[col_idx], exp)
            else:
                chars = []
            chars_per_col.append(chars)

    # ── Phase 2: Levenshtein alignment ──
    page_alignment = []
    match_count_per_col: dict[int, int] = {}

    for col_idx in range(n_columns):
        chars = chars_per_col[col_idx]
        syllables = syllables_per_col[col_idx] if col_idx < len(syllables_per_col) else []
        col_num = col_idx + 1

        aligned = levenshtein_align(chars, syllables, qn_to_nom=qn_to_nom)
        matches = 0
        for pair in aligned:
            pair["column"] = col_num
            if pair["type"] == "match":
                matches += 1
        match_count_per_col[col_num] = matches
        page_alignment.extend(aligned)

    # ── Phase 3: Crop matched characters from ORIGINAL image ──
    crop_size = step1_cfg.get("crop_size", 64)
    cleaner = CharacterCleaner(
        target_size=crop_size,
        sauvola_k=step1_cfg.get("sauvola_k", 0.2),
        sauvola_window=step1_cfg.get("sauvola_window", 25),
        min_stroke=step1_cfg.get("min_stroke", 2),
    )

    detection_columns = []
    all_chars_count = 0

    # Rebuild: only keep matched chars, re-index, crop
    for col_idx in range(n_columns):
        col_num = col_idx + 1

        # Collect matched chars for this column (in order)
        matched_chars = [
            pair["char"] for pair in page_alignment
            if pair.get("column") == col_num
            and pair["type"] == "match"
            and pair.get("char")
        ]

        if not matched_chars:
            continue

        page_crops_dir = crops_dir / page_name
        page_cleaned_dir = cleaned_dir / page_name
        page_crops_dir.mkdir(parents=True, exist_ok=True)
        page_cleaned_dir.mkdir(parents=True, exist_ok=True)

        col_data = {
            "column": col_num,
            "num_chars": len(matched_chars),
            "chars": [],
        }

        for char_idx, char_info in enumerate(matched_chars):
            bbox = char_info["bbox"]
            cx1, cy1, cx2, cy2 = bbox

            crop_file = f"crops/{page_name}/col{col_num:02d}_char{char_idx:03d}.png"
            cleaned_file = f"crops_cleaned/{page_name}/col{col_num:02d}_char{char_idx:03d}.png"

            # Crop from ORIGINAL image
            crop = gray_img[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                cv2.imwrite(str(data_dir / "detected" / crop_file), crop)
                cleaned, _ = cleaner.clean(crop)
                if cleaned is not None:
                    cv2.imwrite(str(data_dir / "detected" / cleaned_file), cleaned)

            new_char_info = {
                "char_idx": char_idx,
                "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "width": int(cx2 - cx1),
                "height": int(cy2 - cy1),
                "crop_file": crop_file,
                "cleaned_file": cleaned_file,
                "ocr_char": char_info.get("ocr_char"),
            }
            col_data["chars"].append(new_char_info)

        detection_columns.append(col_data)
        all_chars_count += len(matched_chars)

    # Save detection JSON
    detection_data = {
        "book_page": page_name,
        "image_size": list(gray_img.shape[::-1]),
        "num_columns": len(detection_columns),
        "total_chars": all_chars_count,
        "columns": detection_columns,
    }
    det_path = data_dir / "detected" / f"{page_name}_detection.json"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(detection_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # ── Phase 4: Rebuild alignment with final char info ──
    final_alignment = []
    col_char_iter: dict[int, int] = {}

    for pair in page_alignment:
        col = pair.get("column", 0)
        if pair["type"] == "match":
            det_col = None
            for cd in detection_columns:
                if cd["column"] == col:
                    det_col = cd
                    break
            if det_col:
                idx = col_char_iter.get(col, 0)
                if idx < len(det_col["chars"]):
                    pair["char"] = det_col["chars"][idx]
                    col_char_iter[col] = idx + 1
        final_alignment.append(pair)

    matches = sum(1 for a in final_alignment if a["type"] == "match")
    gaps = sum(1 for a in final_alignment if a["type"] in ("deletion", "insertion"))

    return final_alignment, {"matches": matches, "gaps": gaps, "chars": all_chars_count}


def align_book(config: dict, book_name: str, verbose: bool = True):
    """Run align + crop for all pages of a book."""
    paths = config["paths"]
    step1_cfg = config.get("step1", {})
    data_dir = Path(paths["data_dir"]) / book_name

    qn_to_nom = load_qn_to_nom(paths["qn_to_nom_dict"])

    trans_dir = data_dir / "transcriptions"
    aligned_dir = data_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    trans_files = sorted(trans_dir.glob("page_*.txt"))
    if not trans_files:
        print(f"[ERROR] No transcription files in {trans_dir}", file=sys.stderr)
        return

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 2: Align + Crop — {book_name}")
        print(f"  Pages: {len(trans_files)}")
        print(f"{'='*60}")

    total_matches = 0
    total_gaps = 0
    total_chars = 0

    for trans_path in trans_files:
        page_name = trans_path.stem

        alignment, stats = process_page(
            page_name, data_dir, qn_to_nom, step1_cfg, verbose=verbose,
        )

        if not alignment:
            if verbose:
                print(f"  [SKIP] {page_name}")
            continue

        out_path = aligned_dir / f"{page_name}_aligned.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(alignment, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        total_matches += stats["matches"]
        total_gaps += stats["gaps"]
        total_chars += stats["chars"]

        if verbose:
            print(f"  {page_name}: {stats['matches']} matches, "
                  f"{stats['gaps']} gaps, {stats['chars']} chars cropped")

    if verbose:
        print(f"\n  Total: {total_matches} matches, {total_gaps} gaps, "
              f"{total_chars} chars cropped")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Align + Crop")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    parser.add_argument("book", type=str, help="Book name")
    args = parser.parse_args()

    config = load_config(args.config)
    align_book(config, args.book)


if __name__ == "__main__":
    main()
