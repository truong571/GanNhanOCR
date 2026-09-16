"""Đo A4: tier_v3 nguyên văn (CALIB + neo LOO>=2 trang khác, T=1.0) trên cols.pkl, join labels_final."""
import sys, math, pickle
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
sys.path.insert(0, str(REPO / "lab/gan_nhan_2026-09-13"))
import thuc_nghiem as T
import pandas as pd
from pipeline.align_engine import anchor_align as aa

cols = T.load_cols()
C = T.Corpus(cols)
df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
rows_rule = df["rule"].values; rows_tier = df["tier"].values; rows_syl = df["syllable"].values
rows_label = df["label"].values; rows_img = df["image"].values

def tier_plan(f, p):   # chép NGUYÊN VĂN kế hoạch v3.1 §A4
    if f["direct"] and p >= 0.8:
        return "CHAR_A"
    if (f["direct"] and p < 0.8) or (f["sim_unique"] and (f["bigram"] or f["corpus2"] or f["tone"]) and p >= 0.8):
        return "CHAR_B"
    if p >= 0.5 and (f["bigram"] or f["corpus4"] or f["tone"]):
        return "SYL"
    return "REVIEW"

# non-LOO corpus counts
pair_pages_all = {k: len(v) for k, v in C.pair_pages.items()}
bigram_pages_all = {k: len(v) for k, v in C.bigram_pages.items()}

recs = []
n_diff = 0
T.set_matrix(T.CALIB)
for col in cols:
    if col["tiers"] is None:
        continue
    pg = (col["book"], col["page"])
    chars, syls = col["chars"], col["syl"]
    mp = [o for o in col["ops"] if o["op"] == "match"]
    prod_row = {o["nom_idx"]: r for o, r in zip(mp, col["rows"])}
    prod_syl = {o["nom_idx"]: o["syl_idx"] for o in mp}
    def sub(i, j):
        base = T.substitution_cost(chars[i], syls[j], T.qn, T.sim)
        if len(C.pair_pages.get((chars[i], syls[j].lower()), set()) - {pg}) >= 2:
            return min(base, aa.COST_CONFIRM)
        return base
    pairs, _ = T._dp_with(sub, chars, syls)
    orig = T.substitution_cost
    T.substitution_cost = lambda c, s, q, sm, _sub=orig: (min(_sub(c, s, q, sm), aa.COST_CONFIRM)
        if len(C.pair_pages.get((c, s.lower()), set()) - {pg}) >= 2 else _sub(c, s, q, sm))
    try:
        post = T.posterior_matches(chars, syls, T=1.0)
    finally:
        T.substitution_cost = orig
    for i, j in pairs.items():
        f = C.feats(chars, syls, i, j, pg)
        p = post.get((i, j), 0.0)
        tv3 = T.tier_v3(f, p); tpl = tier_plan(f, p)
        n_diff += tv3 != tpl
        c, s = chars[i], syls[j].lower()
        bg = C.bigrams(chars, syls, i, j)
        row = prod_row.get(i)
        recs.append(dict(book=col["book"], page=col["page"], column=col["column"], i=i, j=j, c=c, s=s,
            p=p, v3=tv3, direct=f["direct"], sim_unique=f["sim_unique"], bigram=f["bigram"],
            corpus2=f["corpus2"], corpus4=f["corpus4"], tone=f["tone"],
            corpus2_noLOO=pair_pages_all.get((c, s), 0) >= 2, corpus4_noLOO=pair_pages_all.get((c, s), 0) >= 4,
            bigram_noLOO=any(bigram_pages_all.get(k, 0) >= 1 for k in bg),
            row=row, syl_changed=(row is not None and prod_syl.get(i) != j),
            prod_tier=(rows_tier[row] if row is not None else "(khe)"),
            prod_rule=(rows_rule[row] if row is not None else ""),
            prod_syl=(rows_syl[row] if row is not None else ""),
            prod_label=(rows_label[row] if row is not None else ""),
            has_img=(bool(rows_img[row]) if row is not None else False)))
T.set_matrix(T.PROD)
R = pd.DataFrame(recs)
R.to_pickle("/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad/a4_recs.pkl")
print("cặp:", len(R), "· tier_v3 != tier_plan:", n_diff)
print("\n== crosstab prod_tier × v3 ==")
print(pd.crosstab(R.prod_tier, R.v3, margins=True).to_string())
usable = R.v3.isin(["CHAR_A", "CHAR_B", "SYL"]).sum()
print("usable v3:", usable)
