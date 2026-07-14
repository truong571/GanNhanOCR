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


_LEAD_NUM = re.compile(r"^[\s\-–—.,:;)]*(\d+)([.,]?)")          # leading number (+opt dot)
_LEAD_STRIP = re.compile(r"^[\s\-–—.,:;)]*\d[\d.,:;)\s\-–—]*")   # number + trailing junk
_MID_RE = re.compile(r"^(\S{1,5})\s+(\d+)[.,]\s+(.+)$")          # 'word N. body' (mid-line)


def _detect_marker(line: str, expected: int, total: int):
    """Return (column_number, body) if `line` starts a column, else (None, None).

    Detects the tiny 1-9 markers through every corruption VietOCR produces (measured):
      - clean       'N.' / 'N,'                     (1<=N<=9)
      - period lost 'N '                            (N == expected)
      - stuck junk  'N.1 -' / '5.1 -' (no space)    (leading number, junk stripped)
      - extra digit 'NN' e.g. 28->8, 2017->7        (N>9 and N%10 == expected)
      - mid-line    'word N.'                        (N == expected)
    A bare dash '- ' (number fully gone) is resolved later inside the merged block,
    where the exact number of splits is known from the sequence.
    """
    m = _LEAD_NUM.match(line)
    if m:
        n, dot = int(m.group(1)), m.group(2) in (".", ",")
        col = None
        if 1 <= n <= total and (dot or n == expected):
            col = n
        elif n > total and expected <= total and n % 10 == expected:
            col = expected                     # extra-digit variant: last digit == column
        if col is not None:
            body = _LEAD_STRIP.sub("", line).strip()     # drop the number + stuck junk
            if body:                           # only a marker if real content follows
                return col, body
    m2 = _MID_RE.match(line)
    if m2 and expected <= total and int(m2.group(2)) == expected:
        return expected, m2.group(3)
    return None, None


def parse_numbered_lines(text: str, total: int = 9) -> dict[int, list[str]]:
    """Parse QN text into {column_number: [physical lines]} (marker stripped from the
    first line of each column). Content before the first marker is kept (not dropped)
    and assigned to column 1 if column 1's marker was lost.

    The QN translation is a fixed sequence of `total` numbered columns mirroring the
    woodblock columns. Marker corruption (see _detect_marker) otherwise merges columns;
    lines are kept so the merged block can be re-split at its true boundaries later.
    """
    cols: dict[int, list[str]] = {}
    leading: list[str] = []
    cur = None
    expected = 1

    for raw_line in text.split("\n"):
        s = raw_line.strip()
        if not s:
            continue
        num, body = _detect_marker(s, expected, total)
        if num is not None and (cur is None or num >= cur):      # markers only go forward
            cur = num
            cols[cur] = [body]
            expected = num + 1
            continue
        if cur is None:
            leading.append(s)                                    # keep — do not drop col 1
        elif sum(1 for c in s if c.isalpha()) >= 2:
            cols[cur].append(s)

    if leading and 1 not in cols:                                # dropped-leading -> col 1
        cols[1] = leading
    return cols


def extract_quocngu_text(page: fitz.Page) -> tuple[int | None, dict[int, list[str]]]:
    """Extract QN text from a PDF text page.

    Returns: (book_page_number, {column_number: [physical lines]})
    """
    text = page.get_text()
    book_page = extract_book_page_number(page)
    text_clean = re.sub(r"^\d+\s*\n", "", text.strip(), count=1)
    lines = parse_numbered_lines(text_clean)
    return book_page, lines


_DASH_RE = re.compile(r"^[-–—]\s*\S")
_DASH_MARK = re.compile(r"^[-–—]\s*(\w{1,2}\s+)?")     # dash + optional 1-2 char garbled digit


def build_transcription_columns(cols_lines: dict[int, list[str]],
                                total_columns: int = 9) -> list[dict]:
    """Turn {column_number: [lines]} into exactly `total_columns` syllable columns.

    Merged columns (a lost marker) are split at their TRUE internal boundaries — the
    dash-prefixed lines that are the corrupted markers — so no content is invented or
    lost. Only if a block lacks enough dash boundaries is the remainder split by word
    count (near-uniform columns); such approximate boundaries just send a few edge
    syllables to REVIEW, never a wrong GOLD label.
    """
    # accept both {n: [lines]} (current) and {n: "text"} (legacy) inputs
    cols_lines = {n: (v if isinstance(v, list) else [v]) for n, v in cols_lines.items()}
    filled = enforce_column_sequence(cols_lines, total_columns)   # {n: [lines]}, all n present
    columns = []
    for n in range(1, total_columns + 1):
        lines = filled.get(n, [])
        raw_text = " ".join(lines)
        cleaned = clean_line_text(raw_text)
        syllables = split_to_syllables(cleaned)
        columns.append({
            "column": n,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "syllables": syllables,
            "num_syllables": len(syllables),
        })
    return columns


def _split_lines(lines: list[str], k: int) -> list[list[str]]:
    """Split `lines` into k contiguous groups. Prefer dash-prefixed lines (corrupted
    markers) as boundaries; if too few, add cut points by cumulative word count."""
    if k <= 1:
        return [lines]
    n = len(lines)
    words = [len(x.split()) for x in lines]
    dash = [i for i in range(1, n) if _DASH_RE.match(lines[i])]
    if len(dash) >= k - 1:
        cuts = sorted(dash)[:k - 1]
    else:
        cuts = set(dash)
        total_w = sum(words) or 1
        cum, j = 0, 1
        for i in range(1, n):
            cum += words[i - 1]
            while j < k and cum >= total_w * j / k:
                cuts.add(i)
                j += 1
        cuts = sorted(cuts)[:k - 1]
    cutset = sorted(cuts)
    groups, prev = [], 0
    for c in cutset:
        groups.append(lines[prev:c])
        prev = c
    groups.append(lines[prev:])
    while len(groups) < k:
        groups.append([])
    groups = groups[:k]
    # strip the corrupted dash-marker ("- n ..." -> "...") from each split boundary
    for g in groups[1:]:
        if g and _DASH_RE.match(g[0]):
            g[0] = _DASH_MARK.sub("", g[0], count=1).strip()
    return groups


def enforce_column_sequence(cols_lines: dict[int, list[str]], total: int = 9
                            ) -> dict[int, list[str]]:
    """Return {1..total: [lines]} — split merged blocks to fill forward gaps."""
    by = {n: ls for n, ls in cols_lines.items() if 1 <= n <= total and ls}
    if not by:
        return {n: [] for n in range(1, total + 1)}
    present = sorted(by)
    out: dict[int, list[str]] = {}
    for idx, cur in enumerate(present):
        nxt = present[idx + 1] if idx + 1 < len(present) else total + 1
        span = min(nxt, total + 1) - cur
        if span <= 1:
            out[cur] = by[cur]
            continue
        for j, part in enumerate(_split_lines(by[cur], span)):     # split merged block
            out[cur + j] = part
    return {n: out.get(n, []) for n in range(1, total + 1)}
