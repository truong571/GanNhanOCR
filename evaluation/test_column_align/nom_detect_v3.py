"""Nom column detector v3: hybrid PRIMARY + close-pair merge +
projection FALLBACK.

Strategy:
  1. hybrid (filter len≥4 + x-distance reattach short cols).
  2. If count > 9: merge any pair of adjacent body cols whose x-center gap
     is < 0.5 × median spacing (handles 10-col over-split).
  3. If still ≠ 9: fall back to image projection (covers 100% of pages per
     test_image_projection.py).
  4. After getting 9 col bboxes, assign Kimhannom chars by max x-overlap
     (with nearest-neighbour fallback for offset markers).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from core.image.column_detector import detect_columns  # noqa: E402
from core.image.image_processing import detect_text_box  # noqa: E402

from run_full import nom_cols_hybrid  # noqa: E402


def _close_pair_merge(cols, n_expected=9):
    """If we have n_expected+k cols, merge the k closest adjacent pairs."""
    while len(cols) > n_expected:
        # Compute gaps between adjacent (already right→left sorted) cols.
        gaps = []
        for i in range(len(cols) - 1):
            gaps.append((i, abs(cols[i]["x_center"] - cols[i + 1]["x_center"])))
        gaps.sort(key=lambda g: g[1])
        # Median gap
        med = sorted(g[1] for g in gaps)[len(gaps) // 2]
        # Only merge if smallest gap is < 0.55 * median
        if gaps[0][1] < 0.55 * med:
            i = gaps[0][0]
            merged = cols[i]
            merged["chars"].extend(cols[i + 1]["chars"])
            merged["chars"].sort(
                key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
            xs = ([c["bbox"][0] for c in merged["chars"]]
                  + [c["bbox"][2] for c in merged["chars"]])
            merged["x_range"] = (min(xs), max(xs))
            merged["x_center"] = (min(xs) + max(xs)) / 2
            cols.pop(i + 1)
        else:
            break
    return cols


def _assign_by_overlap(proj_bboxes, kim_columns):
    out = []
    for (x1, y1, x2, y2) in proj_bboxes:
        out.append({
            "x_range": (x1, x2), "y_range": (y1, y2),
            "x_center": (x1 + x2) / 2,
            "bbox": (x1, y1, x2, y2), "chars": [],
        })
    for col in kim_columns:
        if not col:
            continue
        for ch in col:
            cx1, cx2 = ch["bbox"][0], ch["bbox"][2]
            cxc = (cx1 + cx2) / 2
            best = None
            best_ov = 0
            best_near = None
            best_d = float("inf")
            for o in out:
                px1, px2 = o["x_range"]
                ov = max(0, min(cx2, px2) - max(cx1, px1))
                if ov > best_ov:
                    best_ov = ov
                    best = o
                d = abs(cxc - o["x_center"])
                if d < best_d:
                    best_d = d
                    best_near = o
            if best is not None and best_ov > 0:
                best["chars"].append(ch)
            elif best_near is not None and best_d < 80:
                best_near["chars"].append(ch)
    for o in out:
        o["chars"].sort(key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    out.sort(key=lambda m: -m["x_center"])
    return out


def detect_nom_columns_v3(binary, kim_columns, n_expected=9):
    """Returns (cols, method) where method ∈
    {hybrid_9, hybrid_merged, projection_fallback, suspect}.
    """
    cols = nom_cols_hybrid(kim_columns, min_len=4)
    if len(cols) == n_expected:
        return cols, "hybrid_9"

    # Try close-pair merge if too many cols
    if len(cols) > n_expected:
        cols = _close_pair_merge(cols, n_expected)
        if len(cols) == n_expected:
            return cols, "hybrid_merged"

    # Fall back to image projection
    try:
        tb = detect_text_box(binary)
        proj_bboxes = detect_columns(binary, tb, n_expected=n_expected)
        if len(proj_bboxes) == n_expected:
            return _assign_by_overlap(proj_bboxes, kim_columns), "projection_fallback"
    except Exception:
        pass

    return cols, "suspect"
