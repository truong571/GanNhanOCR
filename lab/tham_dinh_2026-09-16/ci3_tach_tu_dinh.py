"""Cải tiến 3 — tách từ dính VietOCR: đo lại đầy đủ trên cols.pkl.
(1) lớp OCR>QN, cột có token OOV>=5, BUG-1 vs từ dính thật
(2) tách theo từ điển (mảnh plausible ∈ qs), tổng mảnh thêm == gap → duy nhất / mơ hồ / không bù đủ
(3) 17 cột duy nhất: tier hiện tại (labels_final), sau tách → khớp số? → tier_v3 (p=1 và posterior) trước/sau
(4) detector |G| (det_count.csv) so với n_qn mới
"""
import sys, itertools, collections, json
from pathlib import Path
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR")
SCR = Path("/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/fb7207dc-964e-4d78-93c8-473f625e88a0/scratchpad")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"lab/gan_nhan_2026-09-13"))
import pandas as pd
import thuc_nghiem as tn
from thuc_nghiem import realign_column, qn, sim, set_matrix, PROD, CALIB, posterior_matches, Corpus, tier_v3
from pipeline.align_engine.syllable_normalize import build_readings
from core.text.text_utils import is_plausible_qn_syllable as plaus
cols = tn.load_cols(); qs = tn.qs; readings = build_readings(qn)
lex = {s for s in qs if plaus(s)}
df = pd.read_csv(REPO/"dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
USABLE = {"GOLD","SILVER","SYLLABLE"}
det = pd.read_csv(SCR/"det_count.csv", dtype={"column":int})
detG = {(r.book,r.page,int(r.column)): int(r.g) for r in det.itertuples()}

def splits(tok, k):
    tok = tok.lower(); out = []
    def rec(i, acc):
        if len(acc) == k:
            if i == len(tok): out.append(list(acc))
            return
        for j in range(i+1, len(tok)+1):
            p = tok[i:j]
            if p in lex: rec(j, acc+[p])
    rec(0, []); return out

def solve(c):
    m, n = len(c["chars"]), len(c["syl"]); gap = m-n
    oov = [(i, s) for i, s in enumerate(c["syl"]) if len(s) >= 5 and s.lower() not in qs]
    per = []
    for _, s in oov:
        opts = {}
        for k in range(1, min(len(s), gap+1)+1):
            sp = splits(s, k)
            if sp: opts[k] = sp
        per.append(opts)
    sols = []
    for ks in itertools.product(*[list(o.keys()) for o in per]):
        if sum(k-1 for k in ks) == gap:
            for combo in itertools.product(*[per[i][k] for i, k in enumerate(ks)]):
                sols.append(combo)
    return oov, sols

n_gt = 0; cand = []
for c in cols:
    m, n = len(c["chars"]), len(c["syl"])
    if m <= n: continue
    n_gt += 1
    oov, sols = solve(c)
    if oov: cand.append((c, oov, sols))
bug1 = [x for x in cand if x[0]["book"] == "stt4" and x[0]["page"] == "page_0146"]
real = [x for x in cand if not (x[0]["book"] == "stt4" and x[0]["page"] == "page_0146")]
print(f"[1] cột OCR>QN = {n_gt}; có token OOV>=5 = {len(cand)}; BUG-1 (stt4/page_0146) = {len(bug1)}; từ dính thật = {len(real)}")

uniq = [x for x in real if len(x[2]) == 1]; amb = [x for x in real if len(x[2]) > 1]; nofill = [x for x in real if not x[2]]
print(f"[2] duy nhất {len(uniq)} · mơ hồ {len(amb)} · không bù đủ {len(nofill)}")
print("\n== 36 CỘT TỪ DÍNH THẬT (book page cột | m n gap | token OOV | nghiệm | tier hiện tại labels_final | |G| det)")
for name, L in (("UNIQ", uniq), ("AMB", amb), ("NOFILL", nofill)):
    for c, oov, sols in L:
        key = (c["book"], c["page"], c["column"])
        g = df[(df.book == c["book"]) & (df.page == c["page"]) & (df.column == str(c["column"]))]
        tc = g.tier.value_counts().to_dict()
        s = "|".join("+".join(p) for p in sols[0]) if len(sols) == 1 else (f"{len(sols)} nghiệm: " + " ; ".join("|".join("+".join(p) for p in so) for so in sols[:3]))
        print(f"{name:6s} {key[0]:5s} {key[1]} c{key[2]} | {len(c['chars'])} {len(c['syl'])} {len(c['chars'])-len(c['syl'])} | {[s for _,s in oov]} | {s} | ô={len(g)} {tc} | G={detG.get(key,'?')}")

# ---------- [3] 17 cột duy nhất: trước/sau
set_matrix(PROD)
C = Corpus(cols)
def tiers_for(chars, syls, pg, ops, post):
    out = []
    for o in ops:
        if o["op"] != "match": continue
        i, j = o["nom_idx"], o["syl_idx"]
        f = C.feats(chars, syls, i, j, pg)
        out.append((i, j, tier_v3(f, 1.0), tier_v3(f, post.get((i, j), 0.0)), f["direct"], o.get("confirmed", False)))
    return out

tot = collections.Counter(); rows = []
cells_cur = collections.Counter(); n_cells_lf = 0; n_chars = 0; n_became_eq = 0; det_eq_q = 0; det_eq_c = 0
per_col = []
for c, oov, sols in uniq:
    m, n = len(c["chars"]), len(c["syl"]); pg = (c["book"], c["page"]); key = (c["book"], c["page"], c["column"])
    sol = sols[0]; repl = {i: sol[k] for k, (i, _) in enumerate(oov)}
    new = []
    for i, s in enumerate(c["syl"]):
        if i in repl: new.extend(repl[i])
        else: new.append(s)
    assert len(new) == m
    chars_d = [{"char": ch, "ocr_char": ch} for ch in c["chars"]]
    ops0 = realign_column(chars_d, c["syl"], qn, sim); ops1 = realign_column(chars_d, new, qn, sim)
    post0 = posterior_matches(c["chars"], c["syl"]); post1 = posterior_matches(c["chars"], new)
    t0 = tiers_for(c["chars"], c["syl"], pg, ops0, post0); t1 = tiers_for(c["chars"], new, pg, ops1, post1)
    gaps1 = sum(1 for o in ops1 if o["op"] != "match")
    n_became_eq += (gaps1 == 0)
    g = df[(df.book == c["book"]) & (df.page == c["page"]) & (df.column == str(c["column"]))]
    cells_cur.update(g.tier); n_cells_lf += len(g); n_chars += m
    G = detG.get(key)
    if G is not None:
        det_eq_q += (G == len(new)); det_eq_c += (G == m)
    c0p1 = collections.Counter(t[2] for t in t0); c1p1 = collections.Counter(t[2] for t in t1)
    c0pp = collections.Counter(t[3] for t in t0); c1pp = collections.Counter(t[3] for t in t1)
    # ô tại chính các mảnh mới
    new_idx = set()
    off = 0
    for i, s in enumerate(c["syl"]):
        if i in repl: new_idx |= set(range(off, off+len(repl[i]))); off += len(repl[i])
        else: off += 1
    piece = [(t[0], new[t[1]], c["chars"][t[0]], t[2], t[4]) for t in t1 if t[1] in new_idx]
    per_col.append(dict(key=key, m=m, n=n, sol=sol, gaps_after=gaps1, G=G,
                        p1_before=dict(c0p1), p1_after=dict(c1p1), post_before=dict(c0pp), post_after=dict(c1pp),
                        cur=g.tier.value_counts().to_dict(), piece=piece))
    for k in ("CHAR_A", "CHAR_B", "SYL", "REVIEW"):
        tot[("p1_before", k)] += c0p1[k]; tot[("p1_after", k)] += c1p1[k]
        tot[("post_before", k)] += c0pp[k]; tot[("post_after", k)] += c1pp[k]
    tot["match_before"] += len(t0); tot["match_after"] += len(t1)

print(f"\n== [3] 17 cột duy nhất: chữ OCR (m) tổng = {n_chars}; ô trong labels_final = {n_cells_lf} {dict(cells_cur)}")
print(f"    sau tách → cột KHÔNG còn khe (khớp số, DP toàn match) = {n_became_eq}/{len(uniq)}")
print(f"    match trước {tot['match_before']} → sau {tot['match_after']}")
for pre in ("p1_before", "p1_after", "post_before", "post_after"):
    print(f"    tier_v3[{pre:11s}]: " + " ".join(f"{k}={tot[(pre,k)]}" for k in ("CHAR_A", "CHAR_B", "SYL", "REVIEW")))
print(f"    detector (det_count.csv): |G| == n_qn MỚI: {det_eq_q}/{len(uniq)} · |G| == n_ocr: {det_eq_c}/{len(uniq)}")
print("\n== chi tiết từng cột (mảnh: nom_idx âm chữ tier_v3(p=1) direct)")
for r in per_col:
    print(f"{r['key'][0]} {r['key'][1]} c{r['key'][2]} m={r['m']} n={r['n']} {['+'.join(p) for p in r['sol']]} khe_sau={r['gaps_after']} G={r['G']} | hiện {r['cur']} | p1 {r['p1_before']} → {r['p1_after']} | post {r['post_before']} → {r['post_after']}")
    print("    mảnh:", [(i, s, ch, t, "D" if d else "-") for i, s, ch, t, d in r["piece"]])

# ---------- [4] toàn lớp OCR>QN và 36 cột: ô labels_final
gt_keys = {(c["book"], c["page"], str(c["column"])) for c in cols if len(c["chars"]) > len(c["syl"])}
sub = df[[(b, p, cc) in gt_keys for b, p, cc in zip(df.book, df.page, df.column)]]
print(f"\n== [4] lớp OCR>QN: ô labels_final {len(sub)} {sub.tier.value_counts().to_dict()}")
k36 = {(c["book"], c["page"], str(c["column"])) for c, _, _ in real}
sub36 = df[[(b, p, cc) in k36 for b, p, cc in zip(df.book, df.page, df.column)]]
print(f"    36 cột từ dính: ô labels_final {len(sub36)} {sub36.tier.value_counts().to_dict()}; chữ OCR tổng {sum(len(c['chars']) for c,_,_ in real)}")
