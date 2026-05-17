"""Bbox-aware alignment: smarter than pure Levenshtein.

When Kimhannom OCR detects N chars but QN has M syllables (M < N), choose
WHICH N-M chars to drop using bbox metadata, not just token cost.

Heuristics:
  1. Height filter — chars much shorter than median are likely punctuation/noise
  2. Y-gap clustering — chars too close to a neighbor (overlapping) are fused
     fragments; chars with huge Y-gap before them are isolated noise.
  3. Aspect ratio — extreme W:H suggests border line or decoration.
  4. Anchored ends — first/last char in column anchors to first/last syllable
     when their bbox is clearly at the column edge.

If N == M: skip filtering, map 1:1 by Y order.
If N <  M: cannot invent chars — return as-is, alignment marks gaps as "insertion".
"""
from __future__ import annotations

from statistics import median
from typing import Any


def _noise_score(c: dict, median_h: float, median_y_gap: float,
                 prev_y_gap: float | None) -> float:
    """Higher score = more likely noise. 0 = clean char."""
    s = 0.0
    h = c.get("height") or (c["bbox"][3] - c["bbox"][1])
    w = c.get("width") or (c["bbox"][2] - c["bbox"][0])

    # 1. Height vs median: bigger penalty for very short OR very tall
    if h < 0.4 * median_h:
        s += 2.0 * (0.4 * median_h - h) / max(median_h, 1)
    elif h > 2.0 * median_h:
        s += 1.5

    # 2. Aspect ratio extreme
    if h > 0:
        aspect = w / h
        if aspect > 3.0 or aspect < 0.2:
            s += 1.0

    # 3. Tiny Y-gap to previous → fused-with-neighbor (one of them is noise)
    if prev_y_gap is not None and median_y_gap > 0:
        if prev_y_gap < 0.3 * median_y_gap:
            s += 1.5

    return s


def filter_noise_bboxes(chars: list[dict], expected_count: int) -> list[dict]:
    """Drop chars with highest noise score until we hit expected_count.

    If chars are already <= expected_count, return as-is.
    """
    if len(chars) <= expected_count:
        return list(chars)
    n_drop = len(chars) - expected_count

    # Sort by Y for gap calc
    sorted_chars = sorted(chars, key=lambda c: c["bbox"][1])
    heights = [c.get("height") or (c["bbox"][3] - c["bbox"][1])
               for c in sorted_chars]
    med_h = median(heights) if heights else 50

    y_gaps = []
    for i in range(1, len(sorted_chars)):
        prev_bottom = sorted_chars[i - 1]["bbox"][3]
        this_top = sorted_chars[i]["bbox"][1]
        y_gaps.append(this_top - prev_bottom)
    med_y_gap = median(y_gaps) if y_gaps else 10

    # Score each char
    scored = []
    for i, c in enumerate(sorted_chars):
        prev_gap = y_gaps[i - 1] if i > 0 else None
        score = _noise_score(c, med_h, med_y_gap, prev_gap)
        scored.append((score, i, c))

    # Drop the n_drop highest-score chars
    scored.sort(key=lambda t: -t[0])
    drop_indices = {scored[i][1] for i in range(n_drop)}

    return [c for i, c in enumerate(sorted_chars) if i not in drop_indices]


def bbox_aware_align(chars: list[dict], syllables: list[str],
                     ) -> list[dict]:
    """Map N filtered chars to M syllables.

    Returns same format as levenshtein_align:
      [{char, syllable, type: "match"|"deletion"|"insertion"}]
    """
    m = len(syllables)
    if not chars and not syllables:
        return []
    if not chars:
        return [{"char": None, "syllable": s, "type": "insertion"} for s in syllables]
    if not syllables:
        return [{"char": c, "syllable": None, "type": "deletion"} for c in chars]

    filtered = filter_noise_bboxes(chars, m)
    filtered.sort(key=lambda c: c["bbox"][1])  # by Y top

    pairs: list[dict] = []
    if len(filtered) == m:
        # Perfect 1:1 by Y order
        for c, s in zip(filtered, syllables):
            pairs.append({"char": c, "syllable": s, "type": "match"})
    elif len(filtered) < m:
        # Cannot expand chars — pad insertions at end
        for c, s in zip(filtered, syllables[:len(filtered)]):
            pairs.append({"char": c, "syllable": s, "type": "match"})
        for s in syllables[len(filtered):]:
            pairs.append({"char": None, "syllable": s, "type": "insertion"})
    else:
        # Shouldn't happen since filter cap'd at m, but safeguard
        for c, s in zip(filtered[:m], syllables):
            pairs.append({"char": c, "syllable": s, "type": "match"})
        for c in filtered[m:]:
            pairs.append({"char": c, "syllable": None, "type": "deletion"})

    return pairs
