"""tier_v3 — phân hạng ô theo NGỮ CẢNH loại-một-trang (A-7, flow N5a–N5d).

Chép NGUYÊN VĂN từ lab/gan_nhan_2026-09-13/thuc_nghiem.py: `Corpus` (:307-362, chỉ
giữ các cờ tier_v3 dùng) và `tier_v3` (:541-551). Mô phỏng đã đo ở đó (crosstab
47.482 / 3.338 / 19.288 / 12.983; ghép sai theo tier trên benchmark 600 cột) chỉ
còn giá trị khi mã sản xuất tính ĐÚNG những cờ ấy — selftest `tier_v3_selftest.py`
so `feats` với `thuc_nghiem.Corpus.feats` trên cols.pkl, phải giống 100%.

Nguyên tắc: MỌI thống kê ngữ liệu (cặp, 2-gram) loại trang đang xét — LOO theo
(book, page). Không LOO thì 81.755 ô "có ngữ cảnh" là tự khẳng định (N5a).
Posterior `p` chỉ gác cặp hoà; ngữ cảnh mới là cổng (N4c).
"""
from __future__ import annotations

from collections import defaultdict

from core.text.text_utils import strip_all, strip_tone

# tên tier_v3 → (tier, label_level) — cột `tier_v3` ghi riêng, `tier` giữ tên cũ để
# 7 chỗ đọc rule đúng-bằng (build_dataset/remediate/apply_verdicts/...) không gãy (N5d)
TIER_OF = {"CHAR_A": "GOLD", "CHAR_B": "GOLD", "SYL": "SYLLABLE", "REVIEW": "REVIEW"}
P_HI, P_MID = 0.8, 0.5
CTX_ORDER = ("bigram", "corpus4", "tone", "corpus2", "direct", "sim_unique")
SYL_CTX_ORDER = ("bigram", "corpus4", "tone")     # cờ đủ cho SYL (tier_v3 dòng 4)


class CorpusStats:
    """Thống kê ngữ liệu theo TRANG để loại-một-trang khi đánh giá (không tự khẳng định).

    `cols` = [{book, page, chars: [str], syl: [str], ops: [op]}] — cùng hình với
    cols.pkl của lab (xem `cols_from_page_states` để dựng từ col_states của build).
    pair_pages   {(chữ, âm-thường): {(book, page)}} từ MỌI op match của ops CUỐI;
    bigram_pages {(c1, c2, s1, s2): {(book, page)}} từ cặp match KỀ NHAU
                 (nom_idx+1 ∧ syl_idx+1 trong cùng cột).
    """

    def __init__(self, cols, qn_to_nom, similar):
        self.qn_to_nom = qn_to_nom
        self.similar = similar or {}
        self.Rset = {s: set(v) for s, v in qn_to_nom.items()}
        self.pair_pages: dict[tuple, set] = defaultdict(set)
        self.bigram_pages: dict[tuple, set] = defaultdict(set)
        for col in cols:
            pg = (col["book"], col["page"])
            mp = [o for o in col["ops"] if o["op"] == "match"]
            for o in mp:
                if o["ocr_char"]:
                    self.pair_pages[(o["ocr_char"], str(o["syllable"]).lower())].add(pg)
            for a, b in zip(mp, mp[1:]):
                if a["nom_idx"] + 1 == b["nom_idx"] and a["syl_idx"] + 1 == b["syl_idx"]:
                    self.bigram_pages[(a["ocr_char"], b["ocr_char"],
                                       str(a["syllable"]).lower(),
                                       str(b["syllable"]).lower())].add(pg)
        self.by_tone, self.by_all = defaultdict(set), defaultdict(set)
        for s in qn_to_nom:
            self.by_tone[strip_tone(s)].add(s)
            self.by_all[strip_all(s)].add(s)

    def bigrams(self, chars, syls, i, j):
        c, s = chars[i], syls[j].lower()
        out = []
        if i > 0 and j > 0:
            out.append((chars[i - 1], c, syls[j - 1].lower(), s))
        if i + 1 < len(chars) and j + 1 < len(syls):
            out.append((c, chars[i + 1], s, syls[j + 1].lower()))
        return out

    def feats(self, chars, syls, i, j, pg):
        """Cờ của cặp (chars[i], syls[j]) ở trang pg — mọi cờ ngữ liệu trừ chính pg.
        `dict_support` = |R(s)| (thêm so với lab, không đổi tier)."""
        c, s = chars[i], syls[j].lower()
        R = self.Rset.get(s, set())
        bg = self.bigrams(chars, syls, i, j)
        other = self.pair_pages.get((c, s), set()) - {pg}
        simc = set(self.similar.get(c, []))
        return dict(
            direct=c in R,
            sim_unique=len(simc & R) == 1,
            tone=any(c in self.Rset.get(v, ()) for v in self.by_tone.get(strip_tone(s), set()) - {s}),
            corpus2=len(other) >= 2, corpus4=len(other) >= 4,
            bigram=any(self.bigram_pages.get(k, set()) - {pg} for k in bg),
            dict_support=len(R),
        )

    def bridge_char(self, c, s) -> str:
        """Chữ cầu DUY NHẤT sim(c) ∩ R(s) (nhãn CHAR_B-cầu); '' nếu không đúng 1."""
        R = self.Rset.get((s or "").lower(), set())
        br = set(self.similar.get(c, [])) & R
        return next(iter(br)) if len(br) == 1 else ""


def tier_v3(f, p, P_HI=P_HI, P_MID=P_MID):
    ctx = f["bigram"] or f["corpus2"] or f["tone"]
    if f["direct"] and p >= P_HI:
        return "CHAR_A"
    if f["direct"]:
        return "CHAR_B"
    if f["sim_unique"] and ctx and p >= P_HI:
        return "CHAR_B"
    if p >= P_MID and (f["bigram"] or f["corpus4"] or f["tone"]):
        return "SYL"
    return "REVIEW"


def context_evidence(f) -> str:
    """Chuỗi 'bigram|corpus4|tone|corpus2|direct|sim_unique' gồm các cờ ĐÚNG."""
    return "|".join(k for k in CTX_ORDER if f.get(k))


def rule_of(t: str, f, p) -> str:
    """Rule id theo ánh xạ N5d. CHAR_A giữ tên `s1_inter_s2_direct` (7 chỗ đọc đúng-bằng),
    CHAR_B-cầu giữ tên `s1_inter_s2_similar`, CHAR_B-direct-p-thấp `_lowp` (không làm neo)."""
    if t == "CHAR_A":
        return "s1_inter_s2_direct"
    if t == "CHAR_B":
        return "s1_inter_s2_direct_lowp" if f["direct"] else "s1_inter_s2_similar"
    if t == "SYL":
        return "syl_ctx:" + next(k for k in SYL_CTX_ORDER if f[k])
    # REVIEW: không có ngữ cảnh nào -> no_context; có ngữ cảnh nhưng p thấp -> low_posterior
    if not any(f[k] for k in SYL_CTX_ORDER):
        return "no_context"
    return "low_posterior"


# --------------------------------------------------------------------------- #
# Dựng cols từ col_states của build (PASS 1b) + 2-gram ÂM–ÂM cho L1 (N5e)
# --------------------------------------------------------------------------- #
def cols_from_page_states(page_states):
    """[(book, page, page_png, rec)] -> cols cùng hình cols.pkl; ops = ops CUỐI
    (`ops2` của PASS 1b, rơi về `ops1` nếu không có)."""
    cols = []
    for book, page, _png, rec in page_states:
        for cs in rec.get("col_states") or ():
            cols.append({
                "book": book, "page": page, "column": cs["line_id"],
                "chars": [c.get("char") or c.get("ocr_char") or "" for c in cs["cluster"]["chars"]],
                "syl": list(cs["syllables"]),
                "ops": cs.get("ops2") or cs["ops1"],
            })
    return cols


def build_bigram_syl_pages(cols):
    """{(âm trái, âm phải): {(book, page)}} — 2-gram ÂM–ÂM kề nhau trong dòng QN
    (`syllables` sau normalize_column), ĐỘC LẬP với DP: dựng từ records thì 2-gram
    qua khe `ins` bị đứt (N5e)."""
    bsp: dict[tuple, set] = defaultdict(set)
    for col in cols:
        pg = (col["book"], col["page"])
        syl = [str(s).lower() for s in col["syl"]]
        for a, b in zip(syl, syl[1:]):
            bsp[(a, b)].add(pg)
    return bsp


def syl_bigram_count(bsp, syls, j, s, pg) -> int:
    """n(âm s đặt ở vị trí j) = |trang có (syls[j-1], s)| + |trang có (s, syls[j+1])|,
    loại chính trang pg, hai phía cộng."""
    s = str(s).lower()
    n = 0
    if j > 0:
        n += len(bsp.get((str(syls[j - 1]).lower(), s), set()) - {pg})
    if j + 1 < len(syls):
        n += len(bsp.get((s, str(syls[j + 1]).lower()), set()) - {pg})
    return n
