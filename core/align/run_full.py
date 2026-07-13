"""Live column helpers used by pipeline/step2_align.py and pipeline/align_engine.

Trimmed to the two functions still on the live path (`nom_cols_hybrid`,
`load_similar`). The old v3-tier batch driver and its parse_v3 /
cluster_columns / probe imports were removed as dead code (2026-06).
"""
from __future__ import annotations

import ast
import csv


def load_similar(path: str) -> dict[str, set]:
    d: dict[str, set] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    sims = ast.literal_eval(row[1])
                    if isinstance(sims, list):
                        d[row[0].strip()] = set(sims)
                except (ValueError, SyntaxError):
                    pass
    return d


def nom_cols_hybrid(ocr_columns, min_len=4):
    """Filter cols with len ≥ min_len as 'body' cols, then re-attach short cols
    to the nearest body col by x-distance. Preserves marker chars that filter
    alone would drop, while keeping filter's better col identification.
    """
    body = []
    shorts = []
    for col in ocr_columns:
        if not col:
            continue
        xs = [c["bbox"][0] for c in col] + [c["bbox"][2] for c in col]
        cx = (min(xs) + max(xs)) / 2
        rec = {"x_center": cx, "x_range": (min(xs), max(xs)),
               "chars": list(col)}
        if len(col) >= min_len:
            body.append(rec)
        else:
            shorts.append(rec)

    # Attach each short col to nearest body by x-distance (only if x-overlap > 0
    # OR distance < 80px — to avoid stretching across the page).
    for sh in shorts:
        if not body:
            break
        best = None
        best_d = float("inf")
        for b in body:
            d = abs(sh["x_center"] - b["x_center"])
            overlap = max(0, min(sh["x_range"][1], b["x_range"][1])
                          - max(sh["x_range"][0], b["x_range"][0]))
            if overlap > 0 or d < 80:
                if d < best_d:
                    best_d = d
                    best = b
        if best is not None:
            best["chars"].extend(sh["chars"])

    for b in body:
        b["chars"].sort(key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    body.sort(key=lambda m: -m["x_center"])
    return body
