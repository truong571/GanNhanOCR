"""Step 1: Extract data from PDF -> images + text + character crops."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lib.pdf_parser import (
    is_image_page, extract_book_page_number, extract_nom_image,
    extract_quocngu_text, build_transcription_columns,
)
from lib.image_processing import denoise_image, load_and_binarize, detect_text_box
from lib.column_detector import detect_columns, auto_detect_n_columns
from lib.char_segmenter import segment_characters_in_column
from lib.crop_cleaner import CharacterCleaner
from lib.ocr_api import ocr_page
from lib.qn_ocr import ocr_qn_page
from lib.text_utils import has_vietnamese_diacritics, normalize_syllables

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


def process_book(config: dict, book_name: str, verbose: bool = True):
    """Process one book: PDF -> images + text + crops."""
    book_cfg = None
    for b in config["books"]:
        if b["name"] == book_name:
            book_cfg = b
            break
    if book_cfg is None:
        print(f"[ERROR] Book '{book_name}' not in config.", file=sys.stderr)
        return

    paths = config["paths"]
    data_dir = Path(paths["data_dir"]) / book_name
    pdf_path = Path(book_cfg["pdf"])
    step1_cfg = config.get("step1", {})

    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}", file=sys.stderr)
        return

    import fitz
    doc = fitz.open(str(pdf_path))
    dpi = step1_cfg.get("dpi", 300)
    reocr = book_cfg.get("reocr", False)
    crop_size = step1_cfg.get("crop_size", 64)
    use_ocr_api = step1_cfg.get("use_ocr_api", False)

    pages_dir = data_dir / "pages"
    denoised_dir = data_dir / "pages_denoised"
    trans_dir = data_dir / "transcriptions"
    crops_dir = data_dir / "detected" / "crops"
    cleaned_dir = data_dir / "detected" / "crops_cleaned"

    cleaner = CharacterCleaner(
        target_size=crop_size,
        sauvola_k=step1_cfg.get("sauvola_k", 0.2),
        sauvola_window=step1_cfg.get("sauvola_window", 25),
        min_stroke=step1_cfg.get("min_stroke", 2),
    )

    results = []
    page_idx = 0
    total_pages = doc.page_count

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step 1: Extract — {book_name}")
        print(f"  PDF: {pdf_path} ({total_pages} pages)")
        print(f"  Output: {data_dir}/")
        print(f"{'='*60}")

    while page_idx < total_pages:
        page = doc[page_idx]

        if not is_image_page(page):
            page_idx += 1
            continue

        # Current page = Nom image
        text_page_idx = page_idx + 1
        if text_page_idx >= total_pages:
            page_idx += 1
            continue

        text_page = doc[text_page_idx]
        if is_image_page(text_page):
            page_idx += 1
            continue

        # Determine book page number
        book_page_img = extract_book_page_number(page)
        book_page_txt = extract_book_page_number(text_page)
        if book_page_img:
            book_page = book_page_img
        elif book_page_txt:
            book_page = book_page_txt - 1
        else:
            book_page = page_idx + 10

        page_name = f"page_{book_page:04d}"

        # 1.2: Extract & denoise Nom image
        img_path = pages_dir / f"{page_name}.png"
        img_info = extract_nom_image(page, img_path, dpi)

        if step1_cfg.get("denoise", True):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                denoised = denoise_image(gray)
                cv2.imwrite(str(denoised_dir / f"{page_name}.png"), denoised)

        # 1.3: OCR Nom page via API (get bbox + transcription)
        # Use denoised image for OCR (better quality), fallback to original
        ocr_columns = None
        if use_ocr_api:
            denoised_path = denoised_dir / f"{page_name}.png"
            ocr_image = str(denoised_path) if denoised_path.exists() else str(img_path)
            cache_path = str(data_dir / "detected" / f"{page_name}_ocr_cache.json")
            ocr_columns = ocr_page(ocr_image, cache_path=cache_path, verbose=verbose)

        # 1.6: Extract QN text
        if reocr:
            # Re-OCR with PaddleOCR + VietOCR
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = text_page.get_pixmap(matrix=mat)
            tmp_path = str(data_dir / "transcriptions" / f"{page_name}_qn_tmp.png")
            pix.save(tmp_path)
            ocr_text = ocr_qn_page(tmp_path, verbose=verbose)
            from lib.pdf_parser import parse_numbered_lines
            import re
            ocr_text = re.sub(r"\n\d+\s*$", "", ocr_text.strip())
            raw_lines = parse_numbered_lines(ocr_text)
            book_page_from_text = extract_book_page_number(text_page)
        else:
            book_page_from_text, raw_lines = extract_quocngu_text(text_page)

        columns = build_transcription_columns(raw_lines)

        # Normalize syllables (expand saint names) so segmentation count
        # matches the aligned syllable count in Step 2
        for col in columns:
            col["syllables"] = normalize_syllables(col["syllables"])
            col["num_syllables"] = len(col["syllables"])

        # Save transcription
        trans_path = trans_dir / f"{page_name}.txt"
        with open(trans_path, "w", encoding="utf-8") as f:
            for col in columns:
                f.write(" ".join(col["syllables"]) + "\n")

        trans_json_path = trans_dir / f"{page_name}.json"
        with open(trans_json_path, "w", encoding="utf-8") as f:
            json.dump({"book_page": book_page, "columns": columns}, f,
                      ensure_ascii=False, indent=2)

        # 1.4: Character segmentation (projection profile)
        gray_img, binary = load_and_binarize(str(img_path))
        text_box = detect_text_box(binary)

        n_columns = len(columns)
        if n_columns == 0:
            n_columns = auto_detect_n_columns(binary, text_box)

        col_bboxes = detect_columns(binary, text_box, n_expected=n_columns)

        expected_counts = [len(col["syllables"]) for col in columns]
        all_chars = []
        detection_data = {
            "book_page": book_page,
            "image_size": list(gray_img.shape[::-1]),
            "text_box": list(text_box),
            "num_columns": len(col_bboxes),
            "columns": [],
        }

        page_crops_dir = crops_dir / page_name
        page_cleaned_dir = cleaned_dir / page_name
        page_crops_dir.mkdir(parents=True, exist_ok=True)
        page_cleaned_dir.mkdir(parents=True, exist_ok=True)

        for col_idx, col_bbox in enumerate(col_bboxes):
            exp_count = expected_counts[col_idx] if col_idx < len(expected_counts) else None

            chars = segment_characters_in_column(
                binary, col_bbox,
                expected_count=exp_count,
            )

            col_data = {
                "column": col_idx + 1,
                "bbox": list(col_bbox),
                "num_chars": len(chars),
                "chars": [],
            }

            for char_idx, char_bbox in enumerate(chars):
                cx1, cy1, cx2, cy2 = char_bbox
                crop_file = f"crops/{page_name}/col{col_idx+1:02d}_char{char_idx:03d}.png"
                cleaned_file = f"crops_cleaned/{page_name}/col{col_idx+1:02d}_char{char_idx:03d}.png"

                # Save raw crop
                crop = gray_img[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    cv2.imwrite(str(data_dir / "detected" / crop_file), crop)

                    # 1.5: Clean crop
                    cleaned, _ = cleaner.clean(crop)
                    if cleaned is not None:
                        cv2.imwrite(str(data_dir / "detected" / cleaned_file), cleaned)

                # OCR char from API (if available)
                ocr_char = None
                if ocr_columns and col_idx < len(ocr_columns):
                    ocr_col = ocr_columns[col_idx]
                    cy_center = (cy1 + cy2) / 2
                    best_dist = float("inf")
                    for oc in ocr_col:
                        dist = abs(oc["y_center"] - cy_center)
                        if dist < best_dist:
                            best_dist = dist
                            ocr_char = oc["char"]

                char_info = {
                    "char_idx": char_idx,
                    "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                    "width": int(cx2 - cx1),
                    "height": int(cy2 - cy1),
                    "crop_file": crop_file,
                    "cleaned_file": cleaned_file,
                    "ocr_char": ocr_char,
                }
                col_data["chars"].append(char_info)
                all_chars.append(char_info)

            detection_data["columns"].append(col_data)

        detection_data["total_chars"] = len(all_chars)

        # Save detection JSON
        det_path = data_dir / "detected" / f"{page_name}_detection.json"
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(detection_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        results.append({
            "book_page": book_page,
            "num_columns": len(col_bboxes),
            "total_chars": len(all_chars),
            "total_syllables": sum(len(c["syllables"]) for c in columns),
        })

        if verbose:
            total_syls = sum(len(c["syllables"]) for c in columns)
            print(f"  {page_name}: {len(col_bboxes)} cols, "
                  f"{len(all_chars)} chars, {total_syls} syllables")

        page_idx = text_page_idx + 1

    doc.close()

    # Save manifest
    manifest = {
        "book": book_name,
        "pdf": str(pdf_path),
        "pages": results,
        "total_pages": len(results),
        "total_chars": sum(r["total_chars"] for r in results),
    }
    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n  Total: {len(results)} pages, "
              f"{sum(r['total_chars'] for r in results)} chars")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Extract data from PDF")
    parser.add_argument("config", type=str, help="Path to pipeline.yaml")
    parser.add_argument("book", type=str, help="Book name")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    config = load_config(args.config)
    process_book(config, args.book, verbose=args.verbose)


if __name__ == "__main__":
    main()
