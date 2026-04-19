"""Step 1: Extract data from PDF -> pages + denoised + OCR + QN text.

NO character segmentation or cropping here.
That happens in Step 2 after Levenshtein alignment determines exact char count.
"""

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
from lib.image_processing import denoise_image
from lib.ocr_api import ocr_page
from lib.qn_ocr import ocr_qn_page
from lib.text_utils import normalize_syllables

from pipeline.step0_setup import load_config


def process_book(config: dict, book_name: str, verbose: bool = True):
    """Process one book: PDF -> pages + denoised + OCR cache + transcriptions."""
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
    use_ocr_api = step1_cfg.get("use_ocr_api", False)

    pages_dir = data_dir / "pages"
    denoised_dir = data_dir / "pages_denoised"
    trans_dir = data_dir / "transcriptions"

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

        # 1a: Extract Nom image (original)
        img_path = pages_dir / f"{page_name}.png"
        extract_nom_image(page, img_path, dpi)

        # 1b: Denoise
        if step1_cfg.get("denoise", True):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                denoised = denoise_image(gray)
                cv2.imwrite(str(denoised_dir / f"{page_name}.png"), denoised)

        # 1c: OCR Nom page via API (raw image)
        # Benchmarked (tests/bench_results/): raw > denoised by ~0.7pp coverage
        # over 3 books × 13 pages. API expects noisy/framed input as context.
        ocr_columns = None
        if use_ocr_api:
            cache_path = str(data_dir / "detected" / f"{page_name}_ocr_cache.json")
            ocr_columns = ocr_page(str(img_path), cache_path=cache_path, verbose=verbose)

        # 1d: Extract QN text
        if reocr:
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
        else:
            _, raw_lines = extract_quocngu_text(text_page)

        columns = build_transcription_columns(raw_lines)

        # Normalize syllables (expand saint names)
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

        total_syls = sum(len(c["syllables"]) for c in columns)
        n_ocr_chars = sum(len(c) for c in ocr_columns) if ocr_columns else 0

        results.append({
            "book_page": book_page,
            "num_columns": len(columns),
            "total_syllables": total_syls,
            "ocr_chars": n_ocr_chars,
        })

        if verbose:
            print(f"  {page_name}: {len(columns)} cols, "
                  f"{total_syls} syllables, {n_ocr_chars} OCR chars")

        page_idx = text_page_idx + 1

    doc.close()

    # Save manifest
    manifest = {
        "book": book_name,
        "pdf": str(pdf_path),
        "pages": results,
        "total_pages": len(results),
        "total_syllables": sum(r["total_syllables"] for r in results),
    }
    with open(data_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n  Total: {len(results)} pages, "
              f"{sum(r['total_syllables'] for r in results)} syllables")


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
