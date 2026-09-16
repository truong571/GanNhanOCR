"""Độ nhạy của 70.108 với: chi phí neo (0 vs 2.0), nguồn thống kê (PROD ops vs CALIB PASS-1), posterior có/không neo."""
import sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(REPO / "lab/gan_nhan_2026-09-13"))
import thuc_nghiem as T
from pipeline.align_engine import anchor_align as aa

cols = [c for c in T.load_cols() if c["tiers"] is not None]

def build_stats(pairs_of):
    pair_pages, bigram_pages = defaultdict(set), defaultdict(set)
    for col in cols:
        pg = (col["book"], col["page"]); ch, sy = col["chars"], col["syl"]
        pairs = sorted(pairs_of[id(col)].items())
        for i, j in pairs:
            if ch[i]:
                pair_pages[(ch[i], sy[j].lower())].add(pg)
        for (a, ja), (b, jb) in zip(pairs, pairs[1:]):
            if a + 1 == b and ja + 1 == jb:
                bigram_pages[(ch[a], ch[b], sy[ja].lower(), sy[jb].lower())].add(pg)
    return pair_pages, bigram_pages

# PASS 1 với CALIB không neo
T.set_matrix(T.CALIB)
p1 = {}
for col in cols:
    ch, sy = col["chars"], col["syl"]
    pairs, _ = T._dp_with(lambda i, j: T.substitution_cost(ch[i], sy[j], T.qn, T.sim), ch, sy)
    p1[id(col)] = pairs
C = T.Corpus(cols)   # thống kê từ ops PROD (như thuc_nghiem)
pp_calib, bg_calib = build_stats(p1)

def run(neo_cost, pair_pages, bigram_pages, post_anchored=True, label=""):
    T.set_matrix(T.CALIB)
    C.pair_pages, C.bigram_pages = pair_pages, bigram_pages
    cnt = Counter()
    for col in cols:
        pg = (col["book"], col["page"]); ch, sy = col["chars"], col["syl"]
        def sub(i, j):
            base = T.substitution_cost(ch[i], sy[j], T.qn, T.sim)
            return min(base, neo_cost) if len(pair_pages.get((ch[i], sy[j].lower()), set()) - {pg}) >= 2 else base
        pairs, _ = T._dp_with(sub, ch, sy)
        orig = T.substitution_cost
        if post_anchored:
            T.substitution_cost = lambda c, s, q, sm, _sub=orig: (min(_sub(c, s, q, sm), neo_cost)
                if len(pair_pages.get((c, s.lower()), set()) - {pg}) >= 2 else _sub(c, s, q, sm))
        try:
            post = T.posterior_matches(ch, sy, T=1.0)
        finally:
            T.substitution_cost = orig
        for i, j in pairs.items():
            cnt[T.tier_v3(C.feats(ch, sy, i, j, pg), post.get((i, j), 0.0))] += 1
    us = cnt["CHAR_A"] + cnt["CHAR_B"] + cnt["SYL"]
    print(f"{label:55s} {dict(cnt)} usable={us}")

pp_prod, bg_prod = C.pair_pages, C.bigram_pages
run(0.0, pp_prod, bg_prod, True, "neo=0.0, stats PROD-ops, posterior có neo (=thuc_nghiem)")
run(2.0, pp_prod, bg_prod, True, "neo=2.0 (kế hoạch A2), stats PROD-ops, posterior có neo")
run(0.0, pp_calib, bg_calib, True, "neo=0.0, stats từ PASS-1 CALIB, posterior có neo")
run(2.0, pp_calib, bg_calib, True, "neo=2.0, stats từ PASS-1 CALIB, posterior có neo")
run(2.0, pp_calib, bg_calib, False, "neo=2.0, stats PASS-1 CALIB, posterior KHÔNG neo")
