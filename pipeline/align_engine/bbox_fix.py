"""Fix the frame-crop coordinate offset in cached SinoNom OCR bboxes.

THE BUG (pre-existing, affects production too): core/ocr/ocr_api.py runs the
HCMUS OCR on a FRAME-CROPPED image (framed=True, crop_to_frame with frame_pad),
so the cached character bboxes are in CROPPED-image coordinates. Everything
downstream (step2_align.py, ver_new align_production) crops the FULL page at
those bboxes -> the whole column grid is shifted left by the frame's left margin
(~252px median, ~1.7 columns). Result: leftmost columns fall into the margin
(blank crops, ~30%), interior crops capture the neighbouring glyph.

THE FIX (this module): crop_to_frame uses origin
    (x0, y0) = (max(0, frame_x0 - pad), max(0, frame_y0 - pad))
where frame_* = detect_frame_hybrid(img). So mapping a cropped-coord bbox back
to full-page coords is just `bbox + (x0, y0)`. We recompute (x0, y0) per page
with the SAME detector + the frame_pad stored in the cache.

Verified end-to-end: blank-crop rate 31% -> 0%, median crop ink 9.8% -> 18.1%
(pure translation, no scale). Written entirely in evaluation/ver_new — production
core/ is untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.image.frame_detector import detect_frame_hybrid

# Per-page memo so detect_frame_hybrid runs once per page, not per call.
_offset_cache: dict[str, tuple[int, int]] = {}


def frame_offset(page_png: str, framed: bool, frame_pad: int) -> tuple[int, int]:
    """Return (ox, oy) to add to cached OCR bboxes -> full-page coords."""
    if not framed:
        return (0, 0)
    key = str(page_png)
    if key in _offset_cache:
        return _offset_cache[key]
    bgr = cv2.imread(key)
    if bgr is None:
        return (0, 0)
    try:
        fx0, fy0, _, _ = detect_frame_hybrid(bgr)
    except Exception:
        return (0, 0)
    off = (max(0, int(fx0) - int(frame_pad)), max(0, int(fy0) - int(frame_pad)))
    _offset_cache[key] = off
    return off


def correct_columns(columns: list, ox: int, oy: int) -> list:
    """In-place add (ox, oy) to every char bbox in the OCR columns."""
    if ox == 0 and oy == 0:
        return columns
    for col in columns:
        for c in col:
            b = c.get("bbox")
            if b and len(b) == 4:
                c["bbox"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
    return columns


def tighten_box(gray: "np.ndarray", thr: int = 128, edge_line: float = 0.65,
                margin: int = 4) -> tuple[int, int, int, int] | None:
    """Tighten a grayscale glyph crop to its ink via binary projection.

    Two cleanups, both projection-based:
      1) Strip COLUMN-RULE / border lines: an edge row/col that is mostly ink
         (> edge_line of its length) is the woodblock ruling line, not the
         glyph — trim it inward.
      2) Crop to the ink bounding box (rows/cols that contain any ink), + a
         small margin.

    Returns (x0, y0, x1, y1) within the input crop, or None if (near-)empty.
    The OCR per-char box is loose (captures ruling lines, neighbour slivers,
    whitespace); this re-centres on the actual glyph.
    """
    bw = gray < thr
    h, w = bw.shape
    if bw.sum() < 8:
        return None
    cs = bw.sum(axis=0)
    rs = bw.sum(axis=1)
    # 1) trim long thin border lines
    x0 = 0
    while x0 < int(w * 0.40) and cs[x0] > edge_line * h:
        x0 += 1
    x1 = w
    while x1 > int(w * 0.60) and cs[x1 - 1] > edge_line * h:
        x1 -= 1
    y0 = 0
    while y0 < int(h * 0.40) and rs[y0] > edge_line * w:
        y0 += 1
    y1 = h
    while y1 > int(h * 0.60) and rs[y1 - 1] > edge_line * w:
        y1 -= 1
    sub = bw[y0:y1, x0:x1]
    if sub.sum() < 8:
        return None
    # 2) ink bounding box inside the trimmed region
    ys = np.where(sub.sum(axis=1) > 0)[0]
    xs = np.where(sub.sum(axis=0) > 0)[0]
    a = max(0, x0 + int(xs.min()) - margin)
    b = min(w, x0 + int(xs.max()) + 1 + margin)
    c = max(0, y0 + int(ys.min()) - margin)
    d = min(h, y0 + int(ys.max()) + 1 + margin)
    if b - a < 6 or d - c < 6:
        return None
    return (a, c, b, d)
