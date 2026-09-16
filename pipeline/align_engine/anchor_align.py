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

import math

# ---- substitution / indel costs -------------------------------------------
# Ma trận CALIB (2026-09-13): mỗi chi phí = −log tỉ số khả năng (nat) đo trên
# ngữ liệu thật — docs/RA_SOAT_GAN_NHAN_2026-09-13.md §2 N1 và
# lab/gan_nhan_2026-09-13/thuc_nghiem.py (cmd_calib, CALIB). Thay bộ cũ
# 0/0,3/1,0/0,9/0,7 (khe giả 346 cột) → khe giả 54 cột trên cột m=n.
# Một khe (8,6) đắt hơn một cặp lệch từ điển (6,7): chỉ mở khe khi ≥2 cặp
# liên tiếp không hợp từ điển — không còn "khe giả" vì một chữ OCR sai.
COST_CONFIRM   = 0.0   # ocr_char is a dict reading of the syllable (S1 ∩ S2)
COST_SIMILAR   = 2.5   # a visually-similar char of ocr_char is a dict reading
COST_DICTMISS  = 6.7   # syllable IS in dict but ocr_char is not among readings
COST_NODICT    = 5.1   # syllable not in dict at all -> cannot judge, allow match
COST_DEL       = 8.6   # skip a Nôm char (spurious / over-segmented box)
COST_INS       = 8.6   # skip a QN syllable (Nôm OCR dropped a real glyph)
BAND_SLACK     = 2     # how far past |m-n| the alignment may bow off-diagonal
# Trần chi phí cho cặp được NEO ngữ liệu (cặp (chữ, âm) thấy ở ≥2 trang khác,
# flow N4b): cost_fn = min(base, ANCHOR_CAP). 2,0 nat thay vì 0 để một cặp neo
# sai không kéo cả đoạn (khe đúng 86%, ghép sai 1,36% trên benchmark).
ANCHOR_CAP     = 2.0


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
                   band_slack: int = BAND_SLACK,
                   cost_fn=None, cost_ij=None,
                   cost_del: float | None = None,
                   cost_ins: float | None = None) -> list[dict]:
    """Align one column's Nôm chars to its QN syllables.

    Args:
        nom_chars: list of Nôm items (dicts with 'char'/'ocr_char', or strings),
                   in reading order (top->bottom).
        syllables: list of QN syllable strings, in reading order.
        qn_to_nom: {qn_lower: [nom_char, ...]}.
        similar_dict: {nom_char: [similar, ...]} (optional, bridges OCR confusions).
        cost_fn: callable (ocr_char, syllable) -> float ghi đè chi phí thay thế
                 (vd. neo ngữ liệu, phát xạ ảnh); None = substitution_cost.
                 Chỉ đổi ĐƯỜNG GHÉP; cờ 'confirmed' vẫn do is_confirmed quyết.
        cost_ij: B-2 (DANH_MUC 1B, visual_emission): callable (i, j, ocr_char,
                 syllable) -> float — chi phí thay thế THEO CHỈ SỐ (phát xạ ảnh
                 của hộp i cho âm j cộng vào chi phí văn bản). Ưu tiên hơn cost_fn
                 khi cả hai được truyền. None = dùng cost_fn/substitution_cost.
        cost_del, cost_ins: ghi đè COST_DEL/COST_INS (B-2: khe + λ·gap_vis, như
                 lab/tham_dinh_2026-09-16/dp_vis5.py:63). None = hằng mô-đun.

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
    _cost = cost_fn or (lambda c, s: substitution_cost(c, s, qn_to_nom, similar_dict))
    if cost_ij is not None:                     # B-2: chi phí theo chỉ số (i, j)
        _cost_ij = cost_ij
    else:
        _cost_ij = lambda i, j, c, s: _cost(c, s)      # noqa: E731
    c_del = COST_DEL if cost_del is None else float(cost_del)
    c_ins = COST_INS if cost_ins is None else float(cost_ins)

    for i in range(m + 1):
        for j in range(n + 1):
            if abs(i - j) > band:
                continue
            if i == 0 and j == 0:
                continue
            best, op = INF, None
            # diagonal: match nom[i-1] with syl[j-1]
            if i > 0 and j > 0 and dp[i - 1][j - 1] < INF:
                c = dp[i - 1][j - 1] + _cost_ij(i - 1, j - 1, chars[i - 1], syllables[j - 1])
                if c < best:
                    best, op = c, "M"
            # up: delete nom[i-1] (extra Nôm box, no syllable)
            if i > 0 and dp[i - 1][j] < INF:
                c = dp[i - 1][j] + c_del
                if c < best:
                    best, op = c, "D"
            # left: insert syl[j-1] (Nôm OCR missed a glyph)
            if j > 0 and dp[i][j - 1] < INF:
                c = dp[i][j - 1] + c_ins
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


def band_touched(ops: list[dict], m: int, n: int,
                 band_slack: int = BAND_SLACK) -> bool:
    """True nếu đường ghép chạm biên băng |i−j| == |m−n| + slack (cùng công thức
    băng với realign_column). Cột chạm biên = DP có thể đã bị băng CHẶN đường
    tốt hơn → cờ để rà soát (CALIB: 3/4.029 cột)."""
    band = abs(m - n) + max(1, band_slack)
    i = j = 0
    for o in ops:
        if o["op"] == "match":
            i += 1
            j += 1
        elif o["op"] == "del":
            i += 1
        else:  # "ins"
            j += 1
        if abs(i - j) >= band:
            return True
    return False


def _lse(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) ổn định số; −inf là phần tử trung hoà."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    mx = max(a, b)
    return mx + math.log(math.exp(a - mx) + math.exp(b - mx))


def posterior_matches(nom_chars: list, syllables: list[str],
                      qn_to_nom: dict[str, list[str]],
                      similar_dict: dict[str, list[str]] | None = None,
                      T: float = 1.0, cost_fn=None,
                      band_slack: int = BAND_SLACK, cost_ij=None,
                      cost_del: float | None = None,
                      cost_ins: float | None = None) -> dict[tuple[int, int], float]:
    """{(i, j): P(ghép nom i ↔ âm j)} bằng forward–backward trên đúng lưới NW có
    băng của realign_column, trọng số exp(−cost/T). Port từ
    lab/gan_nhan_2026-09-13/thuc_nghiem.py:118-173 (RA_SOAT_GAN_NHAN §4); T=1,0
    vì chi phí CALIB đã là nat. PHẢI truyền cùng cost_fn với realign_column để
    p đo đúng đường Viterbi (argmax theo hàng == cặp Viterbi; Σ_j p(i,j) ≤ 1,
    phần còn lại là P(del i)). cost_ij/cost_del/cost_ins (B-2): cùng nghĩa và
    PHẢI cùng giá trị với realign_column."""
    m, n = len(nom_chars), len(syllables)
    if m == 0 or n == 0:
        return {}
    chars = [_char_of(x) for x in nom_chars]
    band = abs(m - n) + max(1, band_slack)
    NEG = -math.inf
    _cost = cost_fn or (lambda c, s: substitution_cost(c, s, qn_to_nom, similar_dict))
    if cost_ij is not None:
        _cost_ij = cost_ij
    else:
        _cost_ij = lambda i, j, c, s: _cost(c, s)      # noqa: E731
    cache: dict[tuple[int, int], float] = {}

    def sc(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = -_cost_ij(i, j, chars[i], syllables[j]) / T
        return cache[(i, j)]
    c_del = COST_DEL if cost_del is None else float(cost_del)
    c_ins = COST_INS if cost_ins is None else float(cost_ins)
    d, ins = -c_del / T, -c_ins / T
    F = [[NEG] * (n + 1) for _ in range(m + 1)]
    F[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            if abs(i - j) > band or (i == 0 and j == 0):
                continue
            v = NEG
            if i > 0 and j > 0:
                v = _lse(v, F[i - 1][j - 1] + sc(i - 1, j - 1))
            if i > 0:
                v = _lse(v, F[i - 1][j] + d)
            if j > 0:
                v = _lse(v, F[i][j - 1] + ins)
            F[i][j] = v
    B = [[NEG] * (n + 1) for _ in range(m + 1)]
    B[m][n] = 0.0
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if abs(i - j) > band or (i == m and j == n):
                continue
            v = NEG
            if i < m and j < n:
                v = _lse(v, B[i + 1][j + 1] + sc(i, j))
            if i < m:
                v = _lse(v, B[i + 1][j] + d)
            if j < n:
                v = _lse(v, B[i][j + 1] + ins)
            B[i][j] = v
    Z = F[m][n]
    return {(i, j): math.exp(F[i][j] + sc(i, j) + B[i + 1][j + 1] - Z)
            for i in range(m) for j in range(n)
            if abs(i - j) <= band and F[i][j] > NEG and B[i + 1][j + 1] > NEG}
