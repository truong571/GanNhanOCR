"""Parser v2: QN numbered-line parser with 4 patterns + expected-marker guard.

Patterns handled:
  P1: '1. text...'                    standard numbered
  P2: bare '2' on its own line        marker stripped to a line
  P3: '2 chịu, vì dễ ...'             marker inline with content, no period
  P4: '2017. su chẳng ...'            digits glued to content; only accept when
                                      expected_next_marker matches first digit

Returns dict {1..N: [syllables]} plus diagnostics.
"""

import re
import unicodedata


VIETNAMESE_LETTER_RE = re.compile(r"[a-zA-ZÀ-ỹ]")


def _is_vietnamese_word(token: str) -> bool:
    """Heuristic: contains at least 2 letters and no CJK."""
    if sum(1 for c in token if VIETNAMESE_LETTER_RE.match(c)) < 1:
        return False
    for c in token:
        if "一" <= c <= "鿿" or "㐀" <= c <= "䶿":
            return False
    return True


def _split_syllables(text: str) -> list[str]:
    """Whitespace split + drop pure-punct tokens."""
    out = []
    for tok in text.split():
        # Strip surrounding punctuation but keep diacritics
        stripped = re.sub(r"^[^\wÀ-ỹ]+|[^\wÀ-ỹ]+$", "", tok)
        if stripped and VIETNAMESE_LETTER_RE.search(stripped):
            out.append(stripped)
    return out


def parse_v2(text: str, max_lines: int = 9) -> tuple[dict[int, list[str]], dict]:
    """Parse QN numbered text into {line_no: [syllables]}.

    `max_lines`: cap expected marker range. Markers beyond this are rejected.
    """
    text = unicodedata.normalize("NFC", text)
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]

    lines: dict[int, list[str]] = {}
    current_num: int | None = None
    current_buf: list[str] = []
    warnings: list[str] = []

    def flush():
        nonlocal current_num, current_buf
        if current_num is not None:
            lines[current_num] = _split_syllables(" ".join(current_buf))
        current_buf = []

    p1 = re.compile(r"^[^a-zA-ZÀ-ỹ\d]*?(\d{1,2})[.,]\s+(.+)$")
    p2 = re.compile(r"^\s*(\d{1,2})\s*$")
    # P3: leading 1-2 digit number followed by space + Vietnamese word (no period/comma).
    p3 = re.compile(r"^(\d{1,2})\s+([a-zA-ZÀ-ỹ].*)$")
    # P4: digit-run >= 3 glued; first 1-2 digits MIGHT be the marker, rest is junk.
    p4 = re.compile(r"^(\d)\d+[.,]?\s*(.+)$")

    for raw in raw_lines:
        expected_next = (current_num + 1) if current_num is not None else 1

        # P1
        m = p1.match(raw)
        if m:
            num = int(m.group(1))
            if 1 <= num <= max_lines:
                flush()
                current_num = num
                current_buf = [m.group(2)]
                continue

        # P2: bare number alone
        m = p2.match(raw)
        if m:
            num = int(m.group(1))
            if num == expected_next and 1 <= num <= max_lines:
                flush()
                current_num = num
                current_buf = []
                continue

        # P3: 'N word...' with N == expected_next
        m = p3.match(raw)
        if m:
            num = int(m.group(1))
            if num == expected_next and 1 <= num <= max_lines:
                rest = m.group(2)
                first_tok = rest.split()[0] if rest.split() else ""
                if _is_vietnamese_word(first_tok):
                    flush()
                    current_num = num
                    current_buf = [rest]
                    warnings.append(f"p3:{num}")
                    continue

        # P4: digit-glued line; accept first digit ONLY if equals expected_next
        m = p4.match(raw)
        if m:
            num = int(m.group(1))
            if num == expected_next and 1 <= num <= max_lines:
                flush()
                current_num = num
                current_buf = [m.group(2)]
                warnings.append(f"p4:{num}")
                continue

        # Otherwise: continuation of current line
        if current_num is not None:
            # require at least 2 letters to avoid noise lines
            if sum(1 for c in raw if VIETNAMESE_LETTER_RE.match(c)) >= 2:
                current_buf.append(raw)

    flush()

    diag = {
        "n_lines": len(lines),
        "expected": max_lines,
        "missing": [i for i in range(1, max_lines + 1) if i not in lines],
        "warnings": warnings,
    }
    return lines, diag


def load_v1_transcription(txt_path: str) -> dict[int, list[str]]:
    """Convert the existing transcriptions/*.txt (parser_v1 output) into the
    same dict format. Each non-empty line = one numbered line in order.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.read().split("\n") if l.strip()]
    return {i + 1: _split_syllables(l) for i, l in enumerate(lines)}
