"""Parser v3: very permissive marker detection + missing-line inference.

Marker patterns accepted (in priority order):
  M1  '1. text'                       digit + period/comma/colon/semicolon + space + content
  M2  '1 text'                        digit + space + content (no separator)
  M3  '1 - text'                      digit + space + dash + space + content
  M4  '4,' / '7:' / '4'               standalone marker (alone or with sep only)
  M5  '5.1 text' / '5..text'          digit + non-digit-junk + content (first digit is marker)
  M6  'Thế 7. text' / '- 1 text'      marker after leading 1-2 word prefix
  M7  Standalone bare digit on its    e.g. line "4" followed by next line "đạo thật..."
       own line (caught by M4)

After greedy parsing, if line N is missing but N-1 and N+1 exist:
  Scan accumulated content of line N-1 for an embedded marker == N → split.

Acceptance is gated by `expected_next` — only emit a new section if the
candidate number equals current_num + 1. This prevents year/decimal noise
('2017', '5.1') from being misread as markers.
"""

import re
import sys
import unicodedata
from pathlib import Path

# Reuse the canonical text-cleaning helpers from core/text/text_utils.py so the
# syllable count matches what the current pipeline (parser_v1) produces.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.text.text_utils import (  # noqa: E402
    clean_line_text, split_to_syllables, normalize_syllables,
)

LETTER_RE = re.compile(r"[a-zA-ZÀ-ỹ]")


def _split_syllables(text: str, qn_dict: set[str] | None = None) -> list[str]:
    """Match the v1 pipeline: clean_line_text → split_to_syllables → normalize."""
    cleaned = clean_line_text(text)
    raw = split_to_syllables(cleaned)
    return normalize_syllables(raw, qn_dict=qn_dict)


def _has_letters(s: str, min_count: int = 2) -> bool:
    return sum(1 for c in s if LETTER_RE.match(c)) >= min_count


def _strip_prefix_noise(line: str) -> str:
    """Strip up to ~12 chars of leading non-marker prefix.
    Handles 'Thế 7. text' / '- 1 text' / '* 3 text' cases.
    Returns the stripped suffix, or original if no marker-like pattern follows.
    """
    # Try to match: <up to 12 non-digit chars> + <digit> + <rest>
    m = re.match(r"^([^\d\n]{1,12})(\d.*)$", line)
    if m:
        prefix = m.group(1).strip()
        rest = m.group(2)
        # Only strip if prefix looks like noise (short, no marker chars)
        if len(prefix) <= 10 and not any(c.isdigit() for c in prefix):
            return rest
    return line


def _detect_marker(line: str, expected: int, max_lines: int = 9
                   ) -> tuple[int, str] | None:
    """If `line` starts with a valid marker == expected, return (marker, content_after).
    Otherwise None.
    """
    if expected < 1 or expected > max_lines:
        return None

    # Try the line as-is, then with prefix stripped, then with OCR-confusion
    # rewrite of the leading char (when it could be a misread of `expected`).
    candidates = [line]
    stripped = _strip_prefix_noise(line)
    if stripped != line:
        candidates.append(stripped)

    # OCR digit confusions: only apply when the line clearly looks like a
    # marker line — confused-char followed by a HARD separator [.,:;] (not
    # just a space, since 'Song' starts with 'S' followed by 'ong' and would
    # else be misread as marker 5).
    _OCR_DIGIT_CONFUSIONS = {
        "8": {"s", "S", "B"},
        "1": {"l", "I", "i"},
        "0": {"o", "O", "D"},
        "5": {"S", "s"},
        "2": {"Z", "z"},
        "9": {"g", "q"},
        "7": {"T"},
    }
    confused_chars = _OCR_DIGIT_CONFUSIONS.get(str(expected), set())
    for variant_src in list(candidates):
        v = variant_src.lstrip()
        # Require: first char ∈ confused set AND immediately followed by hard sep.
        if len(v) >= 2 and v[0] in confused_chars and v[1] in ".,:;":
            candidates.append(str(expected) + v[1:])

    for cand in candidates:
        cand = cand.lstrip()
        if not cand:
            continue

        # Must start with 1-2 digit number followed by non-digit or EOL.
        m = re.match(r"^(\d{1,2})(?!\d)(.*)$", cand)
        if not m:
            continue
        num = int(m.group(1))
        if num != expected:
            continue
        tail = m.group(2)

        # Case A: standalone marker (tail empty or only punctuation/space).
        if re.fullmatch(r"[.,:;\-\s]*", tail):
            return expected, ""

        # Case B: tail is separator junk + content.
        # Separator can be: any mix of [.,:;\-–—] and spaces, repeated.
        # Content must contain ≥ 2 Vietnamese letters somewhere.
        m2 = re.match(r"^([.,:;\-–—\s]*)(.+)$", tail)
        if m2:
            sep = m2.group(1)
            content = m2.group(2).strip()
            # If sep is empty but tail starts with a digit-like junk (e.g., '017')
            # → reject (it's likely '2017', not a marker).
            if not sep and re.match(r"^\d", content):
                continue
            # Content must look like a real line (≥ 2 letters).
            if _has_letters(content, 2):
                return expected, content

    return None


def parse_v3(text: str, max_lines: int = 9,
             qn_dict: set[str] | None = None) -> tuple[dict[int, list[str]], dict]:
    text = unicodedata.normalize("NFC", text)
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]

    lines: dict[int, str] = {}
    current = None
    buf: list[str] = []
    warnings: list[str] = []

    def flush():
        nonlocal current, buf
        if current is not None:
            lines[current] = " ".join(buf)
        buf = []

    for raw in raw_lines:
        expected = (current + 1) if current is not None else 1
        det = _detect_marker(raw, expected, max_lines)
        if det is not None:
            num, content = det
            flush()
            current = num
            if content:
                buf = [content]
            else:
                buf = []
            continue
        # Continuation
        if current is not None and _has_letters(raw, 2):
            buf.append(raw)
    flush()

    # ── Inference: recover missing middle markers ──
    found = sorted(lines.keys())
    missing = [i for i in range(1, max_lines + 1) if i not in lines]
    inferred = []
    for m in missing:
        if (m - 1) in lines and (m + 1) in lines:
            host = lines[m - 1]
            # Look for `m` followed by a sep or space inside host.
            # The marker must be preceded by a non-letter (to avoid matching
            # digits inside words like 'm6i') and followed by content.
            pat = re.compile(rf"(?<![\w]){m}[.,:;\-\s]+([a-zA-ZÀ-ỹ].*)$")
            mt = pat.search(host)
            if mt:
                # Split host at marker position
                lines[m - 1] = host[:mt.start()].strip()
                lines[m] = mt.group(1).strip()
                inferred.append(m)
                warnings.append(f"inferred:{m}")

    # Convert to syllables (matches v1 pipeline behavior).
    out = {k: _split_syllables(v, qn_dict=qn_dict) for k, v in lines.items()}
    diag = {
        "n_lines": len(out),
        "expected": max_lines,
        "missing": [i for i in range(1, max_lines + 1) if i not in out],
        "inferred": inferred,
        "warnings": warnings,
    }
    return out, diag
