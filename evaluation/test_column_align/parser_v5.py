"""Parser v5: fix v4 bugs found in 20 residual miss pages.

v5 vs v4:
  Bug 1: v4 standalone-marker case accepts stripped prefix as standalone.
         `TRE3` → strip_prefix gives `3` → fullmatch passes → treats as
         standalone marker 3. Fix: standalone only valid on ORIGINAL line
         after at most whitespace stripping, NOT after prefix noise strip.

  New P7: `19.` pattern — OCR adds rogue leading `1` before digit. When
         expected=N and line starts with `1N[.,:;]` (N ∈ 0..9), accept as N.

  New P8: First-line `- ` prefix. When expected=1 and line starts with
         `-\s+[A-Za-zÀ-ỹ]` (dash + space + capital letter), accept as
         marker 1 implicit.

  New P9: Last-line tail-fragment rescue. If after parsing we have lines
         1..8 but missing 9, AND the LAST raw line starts with junk + letters
         that doesn't match any pattern, treat that as line 9 content.
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.text.text_utils import (  # noqa: E402
    clean_line_text, split_to_syllables, normalize_syllables,
)

LETTER_RE = re.compile(r"[a-zA-ZÀ-ỹ]")


def _split_syllables(text: str, qn_dict=None):
    cleaned = clean_line_text(text)
    raw = split_to_syllables(cleaned)
    return normalize_syllables(raw, qn_dict=qn_dict)


def _has_letters(s, n=2):
    return sum(1 for c in s if LETTER_RE.match(c)) >= n


def _strip_prefix_noise(line):
    # Two-path strip to balance recall vs false-positive on mid-sentence digits:
    #
    # Path 1 — PUNCT-ONLY PREFIX (e.g., '- 1 ra', '* 3 text'): max 4 chars of
    # only symbols/whitespace before the digit. Loose sep after digit accepted
    # (so '- 1  ra' with space-only sep still works for missing-marker case).
    m = re.match(r"^([^\w\d\n]{1,4})(\d.*)$", line)
    if m:
        return m.group(2)
    # Path 2 — SHORT WORD PREFIX (e.g., 'Thế 7. text'): 1-12 chars including
    # letters. Strict — digit MUST be followed by hard sep [.,:;] to avoid
    # treating 'trên ấy 4 năm' as marker 4.
    m = re.match(r"^([^\d\n]{1,12})(\d{1,2}[.,:;].*)$", line)
    if m and len(m.group(1).strip()) <= 10 and not any(c.isdigit() for c in m.group(1)):
        return m.group(2)
    return line


_OCR_CONF = {
    "8": {"s", "S", "B"}, "1": {"l", "I", "i"},
    "0": {"o", "O", "D"}, "5": {"S", "s"},
    "2": {"Z", "z"}, "9": {"g", "q"}, "7": {"T"},
    "3": {"E"}, "4": {"A"}, "6": {"G", "b"},
}


def _try_marker(line: str, target: int, max_lines: int = 9):
    """Try to match marker == target. Returns content_after or None."""
    if target < 1 or target > max_lines:
        return None

    original = line.lstrip()
    candidates = [(original, "orig")]
    stripped = _strip_prefix_noise(line)
    if stripped != line:
        candidates.append((stripped.lstrip(), "stripped"))
    confused = _OCR_CONF.get(str(target), set())
    v = original
    if len(v) >= 2 and v[0] in confused and v[1] in ".,:;":
        candidates.append((str(target) + v[1:], "ocr_conf"))

    # NEW P7: `1N` rogue-1 pattern. E.g. `19.` when expected=9.
    # Restrict: require hard separator [.,:;] right after the rogue digit,
    # otherwise '19 m mà vè' (continuation text) is falsely treated as 9.
    if target <= 9 and len(original) >= 3 and original[0] == "1":
        next_c = original[1]
        sep_c = original[2]
        if next_c.isdigit() and int(next_c) == target and sep_c in ".,:;":
            candidates.append((original[1:], "rogue_1"))

    # NEW P8: dash + Capital letter at line start, only when expected=1.
    if target == 1 and re.match(r"^-\s+[A-ZÀ-Ỹ]", original):
        m = re.match(r"^-\s+(.*)$", original)
        if m and _has_letters(m.group(1), 2):
            candidates.append(("1. " + m.group(1), "dash_first"))

    for cand, src in candidates:
        cand = cand.lstrip()
        if not cand:
            continue
        m = re.match(r"^(\d{1,2})(?!\d)(.*)$", cand)
        if not m or int(m.group(1)) != target:
            continue
        raw_digits = m.group(1)
        tail = m.group(2)

        # Reject leading-zero forms like '03' — these are page numbers, not
        # markers. Standalone '3' is fine; '03' is not.
        if len(raw_digits) > 1 and raw_digits[0] == "0":
            continue

        # Case A: standalone (entire tail is just punct/whitespace).
        if re.fullmatch(r"[.,:;\-\s]*", tail):
            # FIX BUG 1: standalone only valid for ORIGINAL line (not after
            # prefix-noise strip). 'TRE3' would otherwise be standalone-3.
            if src == "orig":
                return ""
            else:
                continue

        # Case B: separator + content.
        m2 = re.match(r"^([.,:;\-–—\s]*)(.+)$", tail)
        if m2:
            sep = m2.group(1)
            content = m2.group(2).strip()
            if not sep and re.match(r"^\d", content):
                continue
            if _has_letters(content, 2):
                return content
    return None


def parse_v5(text: str, max_lines: int = 9, qn_dict=None):
    text = unicodedata.normalize("NFC", text)
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]

    lines: dict[int, str] = {}
    current = None
    buf: list[str] = []
    skipped = []
    warnings = []

    def flush():
        nonlocal current, buf
        if current is not None:
            lines[current] = " ".join(buf)
        buf = []

    for ridx, raw in enumerate(raw_lines):
        next_expected = (current + 1) if current is not None else 1

        matched_num = None
        matched_content = None
        for target in range(next_expected, max_lines + 1):
            content = _try_marker(raw, target, max_lines)
            if content is not None:
                matched_num = target
                matched_content = content
                break

        if matched_num is not None:
            flush()
            if matched_num > next_expected:
                for missing in range(next_expected, matched_num):
                    skipped.append(missing)
                    warnings.append(f"skipped_marker:{missing}")
                    lines[missing] = ""
            current = matched_num
            buf = [matched_content] if matched_content else []
            continue

        if current is not None and _has_letters(raw, 1):
            buf.append(raw)
    flush()

    # P9: last-line tail rescue. If line max_lines missing but max_lines-1
    # exists and has a lot of content with a marker-like break in it.
    if max_lines not in lines and (max_lines - 1) in lines:
        host = lines[max_lines - 1]
        # Look for embedded `9` marker inside host
        pat = re.compile(rf"(?<![\w]){max_lines}[.,:;\-\s]+([a-zA-ZÀ-ỹ].*)$")
        mt = pat.search(host)
        if mt:
            lines[max_lines - 1] = host[:mt.start()].strip()
            lines[max_lines] = mt.group(1).strip()
            warnings.append(f"p9_inferred:{max_lines}")

    # Inference for skipped middle markers
    for m in list(skipped):
        if lines.get(m, "").strip():
            continue
        if (m - 1) in lines and (m + 1) in lines:
            host = lines[m - 1]
            pat = re.compile(rf"(?<![\w]){m}[.,:;\-\s]+([a-zA-ZÀ-ỹ].*)$")
            mt = pat.search(host)
            if mt:
                lines[m - 1] = host[:mt.start()].strip()
                lines[m] = mt.group(1).strip()
                warnings.append(f"inferred:{m}")
                continue
            tokens = host.split()
            if len(tokens) >= 8:
                mid = len(tokens) // 2
                lines[m - 1] = " ".join(tokens[:mid])
                lines[m] = " ".join(tokens[mid:])
                warnings.append(f"split_half:{m}")

    out = {k: _split_syllables(v, qn_dict=qn_dict)
           for k, v in lines.items() if v.strip()}
    diag = {
        "n_lines": len(out),
        "missing": [i for i in range(1, max_lines + 1) if i not in out],
        "skipped_markers": skipped,
        "warnings": warnings,
    }
    return out, diag
