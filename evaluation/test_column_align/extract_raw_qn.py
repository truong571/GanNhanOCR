"""Walk the source PDF and emit raw QN text per Nôm page name.

Re-implements the page-pairing logic of pipeline/step1_extract.py just enough
to recover raw text for the 22 short-QN pages where parser_v1 lost lines.
Output: prepared/<book>/transcriptions_raw/<page>.txt
"""

import argparse
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.pdf.pdf_parser import (
    is_image_page, extract_book_page_number,
)


def extract_raw(book_name: str, repo: Path):
    pdf_path = repo / "Data" / f"{book_name}.pdf"
    out_dir = repo / "prepared" / book_name / "transcriptions_raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    page_idx = 0
    written = 0

    while page_idx < n:
        page = doc[page_idx]
        if not is_image_page(page):
            page_idx += 1
            continue
        text_idx = page_idx + 1
        if text_idx >= n:
            break
        text_page = doc[text_idx]
        if is_image_page(text_page):
            page_idx += 1
            continue

        book_page_img = extract_book_page_number(page)
        book_page_txt = extract_book_page_number(text_page)
        if book_page_img:
            book_page = book_page_img
        elif book_page_txt:
            book_page = book_page_txt - 1
        else:
            book_page = page_idx + 10

        page_name = f"page_{book_page:04d}"
        raw = text_page.get_text()
        # Trim leading bare page number line (like extract_quocngu_text does)
        raw = re.sub(r"^\d+\s*\n", "", raw.strip(), count=1)
        (out_dir / f"{page_name}.txt").write_text(raw, encoding="utf-8")
        written += 1
        page_idx += 2

    print(f"[extract_raw] wrote {written} raw text files to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()
    extract_raw(args.book, Path(args.repo))


if __name__ == "__main__":
    main()
