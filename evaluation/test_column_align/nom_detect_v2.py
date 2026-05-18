"""Final Plan-compliant Nôm column detector.

Order per spec:
  PRIMARY  : image projection — detect_columns(binary, text_box, n_expected=9).
  FALLBACK : Kimhannom filter len ≥ 4, if exactly 9.
  ELSE     : flag nom_column_suspect.

After columns are identified, Kimhannom chars are assigned to projection
bboxes by x-center membership (RIGHT-TO-LEFT order is preserved).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from core.image.column_detector import detect_columns  # noqa: E402
from core.image.image_processing import detect_text_box  # noqa: E402


def _assign_chars_to_projection(proj_bboxes, kim_columns):
    """For each projection bbox (x1, y1, x2, y2), gather Kimhannom chars
    by MAX x-overlap (winner-takes-all). Each Kimhannom char is assigned to
    exactly one projection col — the one with maximum x-overlap, provided
    overlap > 0. This handles marker chars whose x-range edge lies on a
    column boundary.

    Returns list of dicts {x_center, x_range, bbox, chars} per projection
    col, ordered right→left.
    """
    out = []
    for (x1, y1, x2, y2) in proj_bboxes:
        out.append({
            "x_range": (x1, x2),
            "y_range": (y1, y2),
            "x_center": (x1 + x2) / 2,
            "bbox": (x1, y1, x2, y2),
            "chars": [],
        })

    for col in kim_columns:
        if not col:
            continue
        for ch in col:
            cx1, cx2 = ch["bbox"][0], ch["bbox"][2]
            cxc = (cx1 + cx2) / 2
            best = None
            best_ov = 0
            best_d = float("inf")
            for o in out:
                px1, px2 = o["x_range"]
                ov = max(0, min(cx2, px2) - max(cx1, px1))
                if ov > best_ov:
                    best_ov = ov
                    best = o
                # Track nearest by x-center distance as fallback
                d = abs(cxc - o["x_center"])
                if d < best_d:
                    best_d = d
                    best_near = o
            if best is not None and best_ov > 0:
                best["chars"].append(ch)
            elif best_d < 80:
                # No overlap but very close — assign to nearest projection col.
                # Handles marker chars whose x-range is offset from body cols.
                best_near["chars"].append(ch)

    for o in out:
        o["chars"].sort(key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    out.sort(key=lambda m: -m["x_center"])
    return out


def _kim_filter_columns(kim_columns, min_len=4):
    """Filter Kimhannom cols by len, return same shape as projection-assigned."""
    body = []
    shorts = []
    for col in kim_columns:
        if not col:
            continue
        xs = [c["bbox"][0] for c in col] + [c["bbox"][2] for c in col]
        ys = [c["bbox"][1] for c in col] + [c["bbox"][3] for c in col]
        rec = {
            "x_range": (min(xs), max(xs)),
            "y_range": (min(ys), max(ys)),
            "x_center": (min(xs) + max(xs)) / 2,
            "bbox": (min(xs), min(ys), max(xs), max(ys)),
            "chars": sorted(col,
                            key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2),
        }
        if len(col) >= min_len:
            body.append(rec)
        else:
            shorts.append(rec)
    # Reattach shorts to nearest body
    for sh in shorts:
        if not body:
            break
        best = min(body, key=lambda b: abs(b["x_center"] - sh["x_center"]))
        if abs(best["x_center"] - sh["x_center"]) < 80:
            best["chars"].extend(sh["chars"])
            best["chars"].sort(
                key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    body.sort(key=lambda m: -m["x_center"])
    return body


def detect_nom_columns(binary: np.ndarray, kim_columns: list,
                       n_expected: int = 9) -> tuple[list, str]:
    """Detect Nôm columns per Final Plan order.

    Returns (cols, method) where method ∈ {"projection", "kim_filter", "suspect"}.
    """
    # Primary: image projection
    try:
        tb = detect_text_box(binary)
        proj_bboxes = detect_columns(binary, tb, n_expected=n_expected)
    except Exception:
        proj_bboxes = []

    if len(proj_bboxes) == n_expected:
        cols = _assign_chars_to_projection(proj_bboxes, kim_columns)
        return cols, "projection"

    # Fallback: Kimhannom filter
    body = _kim_filter_columns(kim_columns, min_len=4)
    if len(body) == n_expected:
        return body, "kim_filter"

    # Last resort: return whatever we have, marked suspect
    return body if body else _assign_chars_to_projection(proj_bboxes, kim_columns), "suspect"
