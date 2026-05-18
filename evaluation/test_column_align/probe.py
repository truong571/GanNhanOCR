"""Tier-1 probe helpers.

probe = (ocr_char in qn_to_nom[qn_syllable])

This is the proxy used to compare alignment strategies. NOT a final correctness
metric — Tier 2/3 valid matches will probe-miss. Use only for *relative*
comparison between alignment strategies on the same dataset.
"""

import csv
from pathlib import Path


def load_qn_to_nom(dict_path: str) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    with open(dict_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                w = row[0].strip().lower()
                c = row[1].strip()
                if w and c:
                    d.setdefault(w, []).append(c)
    return d


def probe_pair(ocr_char: str | None, qn_syllable: str | None,
               qn_to_nom: dict[str, list[str]]) -> bool:
    if not ocr_char or not qn_syllable:
        return False
    candidates = qn_to_nom.get(qn_syllable.strip().lower(), [])
    return ocr_char in candidates


def probe_pairs(pairs: list[tuple[str | None, str | None]],
                qn_to_nom: dict[str, list[str]]) -> tuple[int, int]:
    """Returns (hits, total). total counts only pairs with both sides non-empty."""
    hits = 0
    total = 0
    for oc, qn in pairs:
        if oc and qn:
            total += 1
            if probe_pair(oc, qn, qn_to_nom):
                hits += 1
    return hits, total
