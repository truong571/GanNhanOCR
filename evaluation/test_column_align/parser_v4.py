"""Parser v4: resync-on-skip + greedy monotonic marker search.

Key insight from inspecting 10 failed pages:
  - VietOCR sometimes DROPS the marker entirely for a line.
  - When parser_v3 expects N but the next marker found is M > N, parser_v3
    rejects M (because it must equal expected_next) and the page collapses.

v4 strategy:
  1. Scan raw lines. For each line, try to detect a marker for ANY of
     {expected_next, expected_next+1, ..., max_lines}. Accept the smallest
     marker ≥ expected_next that produces a valid match.
  2. If marker M > expected_next is accepted, lines for {expected_next .. M-1}
     are emitted as EMPTY (or with whatever content lies between the previous
     marker and this one — split heuristically by line count if possible).
  3. After parsing, run the same content-inference pass as v3 to recover
     embedded markers inside accumulated content.

Plus all v3 patterns retained.
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
    m = re.match(r"^([^\d\n]{1,12})(\d.*)$", line)
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
    """Try to detect a marker == `target` at start of `line`.
    Returns content_after or None.
    """
    if target < 1 or target > max_lines:
        return None
    candidates = [line, _strip_prefix_noise(line)]
    confused = _OCR_CONF.get(str(target), set())
    for src in list(candidates):
        v = src.lstrip()
        if len(v) >= 2 and v[0] in confused and v[1] in ".,:;":
            candidates.append(str(target) + v[1:])

    for cand in candidates:
        cand = cand.lstrip()
        if not cand:
            continue
        m = re.match(r"^(\d{1,2})(?!\d)(.*)$", cand)
        if not m or int(m.group(1)) != target:
            continue
        tail = m.group(2)
        if re.fullmatch(r"[.,:;\-\s]*", tail):
            return ""
        m2 = re.match(r"^([.,:;\-–—\s]*)(.+)$", tail)
        if m2:
            sep = m2.group(1)
            content = m2.group(2).strip()
            if not sep and re.match(r"^\d", content):
                continue
            if _has_letters(content, 2):
                return content
    return None


def parse_v4(text: str, max_lines: int = 9,
             qn_dict=None) -> tuple[dict, dict]:
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

    for raw in raw_lines:
        next_expected = (current + 1) if current is not None else 1

        # RESYNC: try expected_next first; if fail, try expected_next+1, +2... up to max.
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
                # Skip markers — emit empty placeholders. We'll try to
                # back-fill content during inference.
                for missing in range(next_expected, matched_num):
                    skipped.append(missing)
                    warnings.append(f"skipped_marker:{missing}")
                    lines[missing] = ""  # empty placeholder
            current = matched_num
            buf = [matched_content] if matched_content else []
            continue

        # Continuation
        if current is not None and _has_letters(raw, 1):
            buf.append(raw)
    flush()

    # ── Inference for skipped markers ──
    for m in list(skipped):
        if lines.get(m, "").strip():
            continue
        if (m - 1) in lines and (m + 1) in lines:
            host = lines[m - 1]
            # Search for embedded marker == m inside host
            pat = re.compile(rf"(?<![\w]){m}[.,:;\-\s]+([a-zA-ZÀ-ỹ].*)$")
            mt = pat.search(host)
            if mt:
                lines[m - 1] = host[:mt.start()].strip()
                lines[m] = mt.group(1).strip()
                warnings.append(f"inferred:{m}")
                continue
            # If host has multiple sentences and we know one line is missing,
            # split host into 2 roughly equal halves by token count.
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
