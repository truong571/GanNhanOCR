"""PDF page classification, image/text extraction."""

import re
from pathlib import Path

import fitz  # PyMuPDF

from core.text.text_utils import (
    clean_line_text, has_vietnamese_diacritics, split_to_syllables,
)


def is_image_page(page: fitz.Page) -> bool:
    """Classify page: True if Han Nom image, False if QN text.

    Nom image pages have short text (column numbers + page number).
    QN text pages have numbered lines and long text.
    """
    text = page.get_text().strip()
    has_numbered = bool(re.search(r"^\d+[.,]\s", text, re.MULTILINE))
    if has_numbered and len(text) > 200:
        return False
    return len(text) < 200


def extract_book_page_number(page: fitz.Page) -> int | None:
    """Extract book page number from PDF page content.

    Tries first line, last line, and last few lines.
    Numbers > 9 are considered page numbers (vs column numbers 1-9).
    """
    text = page.get_text().strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None

    threshold = 9

    # First line
    match = re.match(r"^(\d+)$", lines[0])
    if match and int(match.group(1)) > threshold:
        return int(match.group(1))

    # Last line
    match = re.match(r"^(\d+)$", lines[-1])
    if match and int(match.group(1)) > threshold:
        return int(match.group(1))

    # Last few lines
    for line in reversed(lines[-5:]):
        match = re.match(r"^(\d+)$", line)
        if match and int(match.group(1)) > threshold:
            return int(match.group(1))

    return None


def extract_nom_image(page: fitz.Page, output_path: Path, dpi: int = 300) -> dict:
    """Extract Nom image from PDF page, save as PNG.

    Prefers embedded original image (higher quality).
    Fallback: render full page.
    """
    images = page.get_images()

    if images:
        from PIL import Image
        import io

        xref = images[0][0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        img = Image.open(io.BytesIO(image_bytes))
        img.save(str(output_path), "PNG")
        return {
            "width": base_image["width"],
            "height": base_image["height"],
            "source": "embedded",
        }
    else:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pix.save(str(output_path))
        return {
            "width": pix.width,
            "height": pix.height,
            "dpi": dpi,
            "source": "rendered",
        }


def parse_numbered_lines(text: str) -> dict[int, str]:
    """Parse QN text into {line_number: content}.

    Robust with Tesseract noise: accepts 'N.' and 'N,' patterns.
    """
    lines: dict[int, str] = {}
    current_num = None
    current_content: list[str] = []
    pattern = re.compile(r"^[^a-zA-ZÀ-ỹ]*?(\d+)[.,]\s+(.*)")

    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue

        match = pattern.match(stripped)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 50:
                if current_num is not None:
                    lines[current_num] = " ".join(current_content)
                current_num = num
                current_content = [match.group(2)]
                continue

        if current_num is not None:
            if sum(1 for c in stripped if c.isalpha()) >= 2:
                current_content.append(stripped)

    if current_num is not None:
        lines[current_num] = " ".join(current_content)

    return lines


def extract_quocngu_text(page: fitz.Page) -> tuple[int | None, dict[int, str]]:
    """Extract QN text from a PDF text page.

    Returns: (book_page_number, {line_num: raw_text})
    """
    text = page.get_text()
    book_page = extract_book_page_number(page)
    text_clean = re.sub(r"^\d+\s*\n", "", text.strip(), count=1)
    lines = parse_numbered_lines(text_clean)
    return book_page, lines


def build_transcription_columns(raw_lines: dict[int, str]) -> list[dict]:
    """Clean text lines and split into syllable columns."""
    columns = []
    for line_num in sorted(raw_lines.keys()):
        raw_text = raw_lines[line_num]
        cleaned = clean_line_text(raw_text)
        syllables = split_to_syllables(cleaned)
        columns.append({
            "column": line_num,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "syllables": syllables,
            "num_syllables": len(syllables),
        })
    return columns
