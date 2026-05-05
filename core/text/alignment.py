"""Levenshtein alignment between detected characters and QN syllables."""

from typing import Callable


def levenshtein_align(
    chars: list[dict],
    syllables: list[str],
    deletion_cost_fn: Callable | None = None,
    qn_to_nom: dict | None = None,
) -> list[dict]:
    """Align N detected characters with M QN syllables using Levenshtein DP.

    Variable deletion costs based on character size:
      < 30% median height -> 0.3 (noise, cheap to skip)
      30-50% median      -> 0.6 (small char)
      >= 50% median      -> 1.2 (real char, expensive to skip)

    Dictionary-aware substitution cost:
      Syllable has candidates in dict -> 0.0 (likely match)
      Syllable not in dict            -> 0.8 (likely mismatch)

    Returns: list of aligned pairs:
      {char, syllable, type: "match"|"deletion"|"insertion"}
    """
    m = len(chars)
    n = len(syllables)

    if m == 0 and n == 0:
        return []
    if m == 0:
        return [{"char": None, "syllable": s, "type": "insertion"} for s in syllables]
    if n == 0:
        return [{"char": c, "syllable": None, "type": "deletion"} for c in chars]

    # Default deletion cost function
    if deletion_cost_fn is None:
        heights = [c.get("height", 50) for c in chars]
        median_h = sorted(heights)[len(heights) // 2] if heights else 50

        def deletion_cost_fn(c):
            ratio = c.get("height", 50) / median_h if median_h > 0 else 1
            if ratio < 0.3:
                return 0.3
            elif ratio < 0.5:
                return 0.6
            else:
                return 1.2

    # Pre-compute substitution costs
    MATCH_COST = 0.0
    MISMATCH_COST = 0.8
    syl_has_candidates = [False] * n
    if qn_to_nom:
        for j, syl in enumerate(syllables):
            syl_has_candidates[j] = bool(qn_to_nom.get(syl.lower(), []))

    # DP
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    bt = [[""] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = 0
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + deletion_cost_fn(chars[i - 1])
        bt[i][0] = "U"
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + 1
        bt[0][j] = "L"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if qn_to_nom:
                subst_cost = MATCH_COST if syl_has_candidates[j - 1] else MISMATCH_COST
            else:
                subst_cost = 0

            diag = dp[i - 1][j - 1] + subst_cost
            delete = dp[i - 1][j] + deletion_cost_fn(chars[i - 1])
            insert = dp[i][j - 1] + 1

            best = min(diag, delete, insert)
            dp[i][j] = best
            if best == diag:
                bt[i][j] = "D"
            elif best == delete:
                bt[i][j] = "U"
            else:
                bt[i][j] = "L"

    # Backtrack
    aligned = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] == "D":
            aligned.append({
                "char": chars[i - 1],
                "syllable": syllables[j - 1],
                "type": "match",
            })
            i -= 1
            j -= 1
        elif i > 0 and bt[i][j] == "U":
            aligned.append({
                "char": chars[i - 1],
                "syllable": None,
                "type": "deletion",
            })
            i -= 1
        elif j > 0:
            aligned.append({
                "char": None,
                "syllable": syllables[j - 1],
                "type": "insertion",
            })
            j -= 1
        else:
            break

    aligned.reverse()
    return aligned
