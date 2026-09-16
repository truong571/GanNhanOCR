import sys, random, math
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"); LAB = REPO / "lab/gan_nhan_2026-09-13"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(LAB))
from thuc_nghiem import load_cols, perturb, qn, sim, PROD, CALIB, set_matrix, aa
from pipeline.align_engine.anchor_align import substitution_cost
df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
z = np.load(LAB / "emb.npz", allow_pickle=True)
LP = z["LP"].astype(np.float32); classes = list(z["classes"]); cid = {s: i for i, s in enumerate(classes)}
cols = load_cols()
C0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5   # phát xạ trung tính cho âm ngoài 706 lớp

def pick(split):
    pages = set(map(tuple, df[df.split == split][["book", "page"]].drop_duplicates().itertuples(index=False)))
    return [c for c in cols if len(c["chars"]) == len(c["syl"]) >= 10 and c["rows"] is not None
            and len(c["rows"]) == len(c["syl"]) and all(o["op"] == "match" for o in c["ops"]) and (c["book"], c["page"]) in pages]

def dp_align(chars, syls, crop_rows, lam, gap_vis, M, CAP=12.0):
    m, n = len(chars), len(syls)
    band = abs(m - n) + max(1, aa.BAND_SLACK)
    V = np.full((m, n), C0)
    for j, s in enumerate(syls):
        if s.lower() in cid:
            for i in range(m):
                if crop_rows[i] is not None:
                    V[i, j] = min(-LP[crop_rows[i], cid[s.lower()]], CAP)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]; bt = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0
    cdel = M["COST_DEL"] + lam * gap_vis; cins = M["COST_INS"] + lam * gap_vis
    for i in range(m + 1):
        for j in range(n + 1):
            if abs(i - j) > band or (i == 0 and j == 0): continue
            best, op = INF, None
            if i > 0 and j > 0 and dp[i-1][j-1] < INF:
                c = dp[i-1][j-1] + substitution_cost(chars[i-1], syls[j-1], qn, sim) + lam * V[i-1, j-1]
                if c < best: best, op = c, "M"
            if i > 0 and dp[i-1][j] < INF:
                c = dp[i-1][j] + cdel
                if c < best: best, op = c, "D"
            if j > 0 and dp[i][j-1] < INF:
                c = dp[i][j-1] + cins
                if c < best: best, op = c, "I"
            dp[i][j], bt[i][j] = best, op
    pairs, dels, inss = [], [], []
    i, j = m, n
    while i > 0 or j > 0:
        s = bt[i][j]
        if s is None: s = "D" if i > 0 else "I"
        if s == "M": i -= 1; j -= 1; pairs.append((i, j))
        elif s == "D": i -= 1; dels.append(i)
        else: j -= 1; inss.append(j)
    return pairs[::-1], dels, inss

def crop_rows_for(kind, k, m2, rows):
    if kind == "drop_char":  return [rows[i if i < k else i + 1] for i in range(m2)]
    if kind == "drop2_char": return [rows[i if i < k else i + 2] for i in range(m2)]
    if kind == "split_char": return [rows[i] if i <= k else (None if i == k + 1 else rows[i - 1]) for i in range(m2)]
    return [rows[i] for i in range(m2)]

all_chars = [c for col in cols for c in col["chars"] if c]
def run(eq, kind, M, lam, gv, seed=2026):
    set_matrix(M); rng = random.Random(seed)
    n = wrong = gap_ok = ncol = 0
    for col in eq:
        chars, syls, rows = col["chars"], col["syl"], col["rows"]; m = len(chars)
        st = rng.getstate(); k = rng.randrange(2, m - 2); rng.setstate(st)
        c2, s2, truth = perturb(chars, syls, kind, rng, all_chars)
        pairs, dels, inss = dp_align(c2, s2, crop_rows_for(kind, k, len(c2), rows), lam, gv, M)
        ncol += 1
        for (i, j) in pairs: n += 1; wrong += truth.get(i) != j
        if kind == "drop_char": gap_ok += (inss == [k]) and not dels
        elif kind == "drop2_char": gap_ok += (sorted(inss) == [k, k+1]) and not dels
        elif kind == "drop_syl": gap_ok += (dels == [k]) and not inss
        elif kind == "split_char": gap_ok += (dels == [k + 1]) and not inss
        elif kind == "subst_char": gap_ok += (not dels and not inss)
        elif kind == "none": gap_ok += (not dels and not inss)
    return n, wrong, gap_ok, ncol

print(f"C0 (phát xạ trung tính âm ngoài lớp) = {C0}")
for split in ("test", "val"):
    eq = pick(split)
    print(f"\n##### {split}: {len(eq)} cột m=n≥10 · ô {sum(len(c['syl']) for c in eq)}")
    for kind in ("drop_char", "drop2_char", "drop_syl", "split_char", "subst_char", "none"):
        out = []
        for name, M, lam, gv in (("PROD thuần", PROD, 0, 0), ("CALIB thuần", CALIB, 0, 0),
                                 ("CALIB+λ0.25", CALIB, 0.25, 8.0), ("CALIB+λ0.5", CALIB, 0.5, 8.0), ("CALIB+λ1", CALIB, 1.0, 8.0), ("CALIB+λ2", CALIB, 2.0, 8.0)):
            n, w, g, nc = run(eq, kind, M, lam, gv)
            out.append(f"{name} sai {w/n:5.2%} khe {g/nc:5.1%}")
        print(f"  {kind:11s} | " + " | ".join(out))
