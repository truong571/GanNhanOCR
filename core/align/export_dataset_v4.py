"""Projection re-segmentation of an under-counted Nôm column.

Trimmed to `resegment_col` — the only function still on the live path (used by
pipeline/step2_align.py and pipeline/align_engine/align_production.py). The old
parser_v4 batch export (export_book/main + parse_v4/probe imports) was removed
as dead code (2026-06).
"""
from __future__ import annotations

from core.image.char_segmenter import segment_characters_in_column


def resegment_col(binary, cluster, expected: int):
    """Project-segment a column body and assign Kimhannom chars by y-overlap.

    Returns: list of dicts {bbox, char (or None)} with len == expected, or None
    if the segmenter cannot hit `expected` boxes.
    """
    chars = cluster["chars"]
    x_lo = min(c["bbox"][0] for c in chars)
    x_hi = max(c["bbox"][2] for c in chars)
    y_lo = min(c["bbox"][1] for c in chars)
    y_hi = max(c["bbox"][3] for c in chars)
    H = binary.shape[0]
    pad = 12
    bbox = (x_lo, max(0, y_lo - pad), x_hi, min(H, y_hi + pad))
    new_bboxes = segment_characters_in_column(binary, bbox, expected_count=expected)
    if len(new_bboxes) != expected:
        return None

    # Map old kimhannom chars to new bboxes by max y-overlap.
    out = []
    used = set()
    for (nx1, ny1, nx2, ny2) in new_bboxes:
        best = None
        best_ov = 0
        for i, c in enumerate(chars):
            if i in used:
                continue
            cy1, cy2 = c["bbox"][1], c["bbox"][3]
            ov = max(0, min(ny2, cy2) - max(ny1, cy1))
            if ov > best_ov:
                best_ov = ov
                best = i
        ocr_char = None
        if best is not None and best_ov > 0:
            ocr_char = chars[best].get("char")
            used.add(best)
        out.append({"bbox": [int(nx1), int(ny1), int(nx2), int(ny2)], "char": ocr_char})
    return out
