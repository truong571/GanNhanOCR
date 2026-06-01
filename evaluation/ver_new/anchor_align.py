"""Banded, dictionary-anchored monotonic alignment of one column.

This REPLACES the fragile positional index-pairing used in
pipeline/step2_align.py (which force-equalizes counts by truncating /
re-segmenting, so a single count divergence shifts the whole column tail and
mislabels every crop after it).

Idea (proven by measurement, see FLOW.md):
  - 1 chữ Nôm = 1 âm tiết Quốc-Ngữ, so per column the SinoNom-OCR character
    sequence and the QN-syllable sequence are a near-perfect 1-1 bitext.
  - The QN↔Nôm dictionary gives a STRONG anchor: a pair (ocr_char_i, syl_j) is
    "dict-confirmed" iff ocr_char_i is a listed Nôm reading of syl_j.
  - We align the two sequences with a Needleman-Wunsch DP whose substitution
    cost is driven by dict-compatibility (0.0 confirmed, cheap), so every
    dict-confirmable character is PINNED to its correct syllable and only the
    genuinely ambiguous run around a real insertion/deletion floats.
  - A BAND (|i-j| <= |m-n| + slack) bounds how far the alignment may deviate
    from the diagonal. This both (a) prevents the catastrophic whole-tail
    shift, and (b) stops the aligner from making large rearrangements when
    anchors are sparse — which is what collapsed the legacy global Levenshtein
    aligner down to 4,133 pairs. Costs are SOFT (never -inf), so one wrong
    anchor cannot drag a whole segment off-register.

Pure-Python, no third-party deps. Reusable: the production version belongs in
core/align/anchor_align.py and is called from process_page_structural.
"""
from __future__ import annotations

# ---- substitution / indel costs -------------------------------------------
# Tuned so that a 1-1 diagonal always wins on a clean matched column, but a
# single missing/extra glyph is cheaper to absorb as ONE local gap than to
# mis-pair the rest of the column.
COST_CONFIRM   = 0.0   # ocr_char is a dict reading of the syllable (S1 ∩ S2)
COST_SIMILAR   = 0.3   # a visually-similar char of ocr_char is a dict reading
COST_DICTMISS  = 1.0   # syllable IS in dict but ocr_char is not among readings
COST_NODICT    = 0.9   # syllable not in dict at all -> cannot judge, allow match
COST_DEL       = 0.7   # skip a Nôm char (spurious / over-segmented box)
COST_INS       = 0.7   # skip a QN syllable (Nôm OCR dropped a real glyph)
BAND_SLACK     = 2     # how far past |m-n| the alignment may bow off-diagonal


def _char_of(item) -> str | None:
    """Accept either a dict ({'char'|'ocr_char': ...}) or a bare string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("char") or item.get("ocr_char")
    return None


def substitution_cost(ocr_char: str | None, syllable: str,
                      qn_to_nom: dict[str, list[str]],
                      similar_dict: dict[str, list[str]] | None = None) -> float:
    """Cost of pairing a Nôm OCR char with a QN syllable (lower = better)."""
    cands = qn_to_nom.get((syllable or "").lower(), [])
    if ocr_char and ocr_char in cands:
        return COST_CONFIRM
    if ocr_char and similar_dict:
        sims = similar_dict.get(ocr_char, [])
        if any(s in cands for s in sims):
            return COST_SIMILAR
    if cands:                       # syllable known, but char doesn't match it
        return COST_DICTMISS
    return COST_NODICT              # syllable not in dict -> neutral-ish match


def is_confirmed(ocr_char: str | None, syllable: str,
                 qn_to_nom: dict[str, list[str]],
                 similar_dict: dict[str, list[str]] | None = None) -> bool:
    """True iff (ocr_char, syllable) is dict-confirmed (directly or via similar)."""
    cands = qn_to_nom.get((syllable or "").lower(), [])
    if ocr_char and ocr_char in cands:
        return True
    if ocr_char and similar_dict:
        sims = similar_dict.get(ocr_char, [])
        if any(s in cands for s in sims):
            return True
    return False


def realign_column(nom_chars: list, syllables: list[str],
                   qn_to_nom: dict[str, list[str]],
                   similar_dict: dict[str, list[str]] | None = None,
                   band_slack: int = BAND_SLACK) -> list[dict]:
    """Align one column's Nôm chars to its QN syllables.

    Args:
        nom_chars: list of Nôm items (dicts with 'char'/'ocr_char', or strings),
                   in reading order (top->bottom).
        syllables: list of QN syllable strings, in reading order.
        qn_to_nom: {qn_lower: [nom_char, ...]}.
        similar_dict: {nom_char: [similar, ...]} (optional, bridges OCR confusions).

    Returns:
        Ordered list of ops, each a dict:
          {'op': 'match',  'nom_idx': i, 'syl_idx': j, 'ocr_char': c,
           'syllable': s, 'confirmed': bool}
          {'op': 'del',    'nom_idx': i, 'ocr_char': c}      # extra Nôm box
          {'op': 'ins',    'syl_idx': j, 'syllable': s}      # missing Nôm glyph
    """
    m, n = len(nom_chars), len(syllables)
    chars = [_char_of(x) for x in nom_chars]

    # Edge cases
    if m == 0 and n == 0:
        return []
    if m == 0:
        return [{"op": "ins", "syl_idx": j, "syllable": syllables[j]} for j in range(n)]
    if n == 0:
        return [{"op": "del", "nom_idx": i, "ocr_char": chars[i]} for i in range(m)]

    band = abs(m - n) + max(1, band_slack)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    bt = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(m + 1):
        for j in range(n + 1):
            if abs(i - j) > band:
                continue
            if i == 0 and j == 0:
                continue
            best, op = INF, None
            # diagonal: match nom[i-1] with syl[j-1]
            if i > 0 and j > 0 and dp[i - 1][j - 1] < INF:
                c = dp[i - 1][j - 1] + substitution_cost(
                    chars[i - 1], syllables[j - 1], qn_to_nom, similar_dict)
                if c < best:
                    best, op = c, "M"
            # up: delete nom[i-1] (extra Nôm box, no syllable)
            if i > 0 and dp[i - 1][j] < INF:
                c = dp[i - 1][j] + COST_DEL
                if c < best:
                    best, op = c, "D"
            # left: insert syl[j-1] (Nôm OCR missed a glyph)
            if j > 0 and dp[i][j - 1] < INF:
                c = dp[i][j - 1] + COST_INS
                if c < best:
                    best, op = c, "I"
            dp[i][j], bt[i][j] = best, op

    # Backtrack
    ops: list[dict] = []
    i, j = m, n
    while i > 0 or j > 0:
        step = bt[i][j] if (i <= m and j <= n) else None
        if step is None:
            # Fell outside band at a corner — drain remaining greedily.
            if i > 0:
                step = "D"
            else:
                step = "I"
        if step == "M":
            i, j = i - 1, j - 1
            conf = is_confirmed(chars[i], syllables[j], qn_to_nom, similar_dict)
            ops.append({"op": "match", "nom_idx": i, "syl_idx": j,
                        "ocr_char": chars[i], "syllable": syllables[j],
                        "confirmed": conf})
        elif step == "D":
            i -= 1
            ops.append({"op": "del", "nom_idx": i, "ocr_char": chars[i]})
        else:  # "I"
            j -= 1
            ops.append({"op": "ins", "syl_idx": j, "syllable": syllables[j]})
    ops.reverse()
    return ops


def matched_pairs(ops: list[dict]) -> list[dict]:
    """Extract only the 'match' ops (the emitted Nôm-crop ↔ syllable labels)."""
    return [o for o in ops if o["op"] == "match"]
