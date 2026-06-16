"""Roadmap #5 (core) — turn detector boxes into EXACTLY N character boxes.

The leverage of this whole project over generic CJK segmentation: after the
Nôm↔QN alignment we KNOW the true character count `N` of a column (= the QN
syllable count) even when the OCR miscounts. A character detector (HRCenterNet /
CenterNet, trained on confirmed crops — see README) proposes boxes per column;
this module reconciles the proposal with the known `N`:

    len == N : accept.
    len  > N : detector over-segmented (a glyph split into radicals) -> repeatedly
               MERGE the adjacent pair (in reading order) with the smallest vertical
               gap, until N remain. Closest-together fragments are most likely one
               character.
    len  < N : detector merged/missed (touching glyphs) -> repeatedly SPLIT the
               tallest box at its mid-line (or, with `valley_split`, at the deepest
               horizontal-projection valley of that sub-image), until N exist.

This is what makes a "diverged" column re-segmentable into N clean crops instead of
being dumped to REVIEW. It is deliberately image-FREE (pure geometry) so it is
unit-testable and deterministic; pass a `valley_split` callback to make the split
ink-aware. Boxes are (x1, y1, x2, y2[, score]); reading order is top->bottom by
y-centre (vertical Nôm columns).

Self-test:
  .venv/bin/python evaluation/ver_new/char_detector/count_constrained.py
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence


def _yc(b) -> float:
    return (b[1] + b[3]) / 2.0


def constrain_to_count(
    boxes: Sequence[Sequence[float]],
    n: int,
    valley_split: Optional[Callable[[tuple], int]] = None,
) -> list[list[int]]:
    """Return EXACTLY `n` boxes (x1,y1,x2,y2), top->bottom, reconciling a detector
    proposal with the known character count.

    `valley_split(box)` -> a y split-coordinate inside the box (absolute), used when
    splitting a too-tall box; if None, split at the vertical midpoint.
    """
    bx = sorted(([int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes), key=_yc)
    if n <= 0:
        return []
    if not bx:
        return []
    if len(bx) == n:
        return bx

    # too many -> merge closest-adjacent until n
    while len(bx) > n:
        gaps = [(bx[i + 1][1] - bx[i][3], i) for i in range(len(bx) - 1)]  # top_next - bottom_cur
        _, i = min(gaps, key=lambda t: t[0])
        a, b = bx[i], bx[i + 1]
        bx[i:i + 2] = [[min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]]

    # too few -> split the tallest until n
    while len(bx) < n:
        i = max(range(len(bx)), key=lambda j: bx[j][3] - bx[j][1])
        a = bx[i]
        if valley_split is not None:
            cut = int(valley_split(tuple(a)))
            cut = min(max(cut, a[1] + 2), a[3] - 2)
        else:
            cut = (a[1] + a[3]) // 2
        bx[i:i + 1] = [[a[0], a[1], a[2], cut], [a[0], cut, a[2], a[3]]]
        bx.sort(key=_yc)
    return bx


# --------------------------------------------------------------------------- #
def _selftest():
    ok = 0
    # exact
    r = constrain_to_count([[0, 0, 10, 20], [0, 22, 10, 40]], 2)
    assert len(r) == 2, r; ok += 1
    # over-segmented: 3 detections, 2 chars; the two close fragments (gap 1) merge
    r = constrain_to_count([[0, 0, 10, 18], [0, 19, 10, 36], [0, 60, 10, 90]], 2)
    assert len(r) == 2, r
    assert r[0] == [0, 0, 10, 36], r            # first two merged (smallest gap)
    assert r[1] == [0, 60, 10, 90], r; ok += 1
    # under-segmented: 1 tall box, 3 chars -> split into 3 by midline twice
    r = constrain_to_count([[0, 0, 10, 90]], 3)
    assert len(r) == 3, r
    ys = [b[1] for b in r] + [r[-1][3]]
    assert ys[0] == 0 and ys[-1] == 90 and all(ys[i] < ys[i + 1] for i in range(len(ys) - 1)), r
    ok += 1
    # valley-aware split callback is honoured
    r = constrain_to_count([[0, 0, 10, 100]], 2, valley_split=lambda b: 30)
    assert r[0][3] == 30 and r[1][1] == 30, r; ok += 1
    # n==1 collapses everything into the union
    r = constrain_to_count([[2, 0, 8, 10], [0, 12, 10, 30]], 1)
    assert r == [[0, 0, 10, 30]], r; ok += 1
    # empty input
    assert constrain_to_count([], 3) == []; ok += 1
    print(f"count_constrained self-test: {ok}/6 passed ✅")


if __name__ == "__main__":
    _selftest()
