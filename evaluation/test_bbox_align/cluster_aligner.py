"""Cluster-based Y-gap alignment — the FIXED heuristic.

Replaces the earlier naive bbox-aware aligner (which penalized small Y-gaps,
wrong for vertical text where all real chars touch). This one finds LARGE
Y-gaps as cluster boundaries, drops the small clusters (header/footer noise),
keeps the largest cluster.

Followed by Levenshtein WITHIN the main cluster if counts still don't match.
"""
from __future__ import annotations

from statistics import median


def split_clusters_by_y(chars: list[dict]) -> list[list[dict]]:
    """Sort chars top-to-bottom, split where Y-gap > 2×median (or 80px abs).

    Returns list of clusters (each = list[dict]).
    """
    if len(chars) <= 1:
        return [list(chars)]
    s = sorted(chars, key=lambda c: c["bbox"][1])
    gaps = [s[i]["bbox"][1] - s[i - 1]["bbox"][3] for i in range(1, len(s))]
    med_gap = max(median(gaps), 0)
    threshold = max(med_gap * 2 + 30, 80)
    clusters: list[list[dict]] = [[s[0]]]
    for i, g in enumerate(gaps, start=1):
        if g > threshold:
            clusters.append([])
        clusters[-1].append(s[i])
    return clusters


def keep_main_cluster(chars: list[dict]) -> list[dict]:
    """Return the cluster with the most chars (= main text column)."""
    cs = split_clusters_by_y(chars)
    return max(cs, key=len)


def cluster_align(chars: list[dict], syllables: list[str]) -> list[dict]:
    """Cluster-based alignment.

    1. Split chars by Y-gap clustering.
    2. Keep main (largest) cluster.
    3. If |main| == |syllables|: map 1:1 by Y-order.
    4. Else fall back to (cheap) deletion alignment:
       - If |main| > |syllables|: drop shortest chars iteratively from main
         until count matches, then map 1:1.
       - If |main| < |syllables|: pair first N chars with first N syllables,
         remaining syllables get "insertion" entries.

    Returns the same format as levenshtein_align — list of dicts with
    {char, syllable, type}.
    """
    if not chars and not syllables:
        return []
    if not chars:
        return [{"char": None, "syllable": s, "type": "insertion"}
                for s in syllables]
    if not syllables:
        return [{"char": c, "syllable": None, "type": "deletion"}
                for c in chars]

    main = keep_main_cluster(chars)
    main = sorted(main, key=lambda c: c["bbox"][1])
    m = len(syllables)

    pairs: list[dict] = []
    if len(main) == m:
        for c, s in zip(main, syllables):
            pairs.append({"char": c, "syllable": s, "type": "match"})
    elif len(main) > m:
        # Drop shortest chars from main until count matches
        # (keeps the visually tallest chars — likely the real ones).
        sorted_by_h = sorted(main,
                             key=lambda c: c.get("height") or
                                           (c["bbox"][3] - c["bbox"][1]),
                             reverse=True)
        keep = sorted_by_h[:m]
        keep_sorted = sorted(keep, key=lambda c: c["bbox"][1])
        for c, s in zip(keep_sorted, syllables):
            pairs.append({"char": c, "syllable": s, "type": "match"})
    else:
        # main < syllables: missing detections — pair what we have, insert rest
        for c, s in zip(main, syllables[:len(main)]):
            pairs.append({"char": c, "syllable": s, "type": "match"})
        for s in syllables[len(main):]:
            pairs.append({"char": None, "syllable": s, "type": "insertion"})
    return pairs
