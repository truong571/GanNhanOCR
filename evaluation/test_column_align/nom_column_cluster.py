"""Cluster Kimhannom OCR columns by x-overlap to recover real Nôm columns.

Why: Kimhannom sometimes splits a column header marker (2-char stack at top of
column) into its own "column", producing 11 OCR cols for a page with 9 real cols.
Clustering by x-overlap merges marker stacks back with their host body column.

Output: list of clusters, sorted RIGHT-TO-LEFT (reading order; cluster 0 is the
rightmost = QN line 1 in this corpus).
"""


def _x_range(col_chars: list[dict]) -> tuple[int, int]:
    xs = [c["bbox"][0] for c in col_chars] + [c["bbox"][2] for c in col_chars]
    return min(xs), max(xs)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    inter = hi - lo
    smaller = min(a[1] - a[0], b[1] - b[0])
    return inter / max(smaller, 1)


def cluster_columns(ocr_columns: list[list[dict]],
                    overlap_threshold: float = 0.5) -> list[dict]:
    """Cluster OCR columns by x-overlap.

    Returns: list of {x_range, x_center, kim_col_ids: [...], chars: [...]}
    sorted right-to-left (reading order: cluster 0 = rightmost).

    `chars` are concatenated in TOP-TO-BOTTOM order across merged cols.
    """
    if not ocr_columns:
        return []

    items = []
    for idx, col in enumerate(ocr_columns):
        if not col:
            continue
        x_lo, x_hi = _x_range(col)
        items.append({
            "x_range": (x_lo, x_hi),
            "x_center": (x_lo + x_hi) / 2,
            "kim_col_ids": [idx],
            "chars": list(col),
        })

    # Greedy merge by x-overlap. Sort by x_center descending (right-to-left).
    items.sort(key=lambda it: -it["x_center"])
    merged: list[dict] = []
    for it in items:
        placed = False
        for m in merged:
            if _overlap(it["x_range"], m["x_range"]) >= overlap_threshold:
                # Merge
                m["chars"].extend(it["chars"])
                m["kim_col_ids"].extend(it["kim_col_ids"])
                lo = min(m["x_range"][0], it["x_range"][0])
                hi = max(m["x_range"][1], it["x_range"][1])
                m["x_range"] = (lo, hi)
                m["x_center"] = (lo + hi) / 2
                placed = True
                break
        if not placed:
            merged.append(it)

    # Sort chars within each cluster top-to-bottom (y_center asc).
    for m in merged:
        m["chars"].sort(key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)

    # Sort clusters right-to-left.
    merged.sort(key=lambda m: -m["x_center"])
    return merged
