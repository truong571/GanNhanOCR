"""Phát xạ ảnh cắt từ HỘP OCR THÔ (có sẵn lúc DP) thay vì hộp cuối labels_final: benchmark có giữ không?"""
import sys, random, math, json, glob, time
from pathlib import Path
import numpy as np, pandas as pd, cv2
REPO = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR"); LAB = REPO / "lab/gan_nhan_2026-09-13"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(LAB))
import torch
from thuc_nghiem import load_cols, perturb, qn, sim, CALIB, set_matrix, aa, Corpus, BOOKS, qs
import thi_giac_am_tiet as T
from core.align.run_full import nom_cols_hybrid
from pipeline.step2_align import _get_qn_lines
from pipeline.align_engine.anchor_align import substitution_cost
df = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
z = np.load(LAB / "emb.npz", allow_pickle=True)
LP_final = z["LP"].astype(np.float32); classes = list(z["classes"]); cid = {s: i for i, s in enumerate(classes)}
cols = load_cols(); C = Corpus(cols)
net, classes2, dev = T.load_model(); assert classes2 == classes
C0 = 0.5; CAP = 12.0
def pick(split):
    pages = set(map(tuple, df[df.split == split][["book", "page"]].drop_duplicates().itertuples(index=False)))
    return [c for c in cols if len(c["chars"]) == len(c["syl"]) >= 10 and c["rows"] is not None
            and len(c["rows"]) == len(c["syl"]) and all(o["op"] == "match" for o in c["ops"]) and (c["book"], c["page"]) in pages]
eq = pick("test") + pick("val")
# hộp OCR thô cho từng cột
t0 = time.time(); need = {(c["book"], c["page"]) for c in eq}
ocr_box = {}
for book, code in BOOKS.items():
    dd = REPO / "prepared" / book
    for (b, page) in sorted(need):
        if b != code: continue
        od = json.load(open(dd / "detected" / f"{page}_ocr_cache.json", encoding="utf-8"))
        lines, _ = _get_qn_lines(dd, page, qs); keys = sorted(lines)
        cs = nom_cols_hybrid(od["columns"], min_len=4)
        if len(cs) != 9: cs = nom_cols_hybrid(od["columns"], min_len=1)
        for i in range(min(len(cs), len(keys))):
            ocr_box[(code, page, keys[i])] = [ch["bbox"] for ch in cs[i]["chars"]]
# cắt crop từ hộp OCR thô, chạy CNN
LP_ocr = {}   # (book,page,col) -> (m × 706)
imgs = {}
miss = 0
for c in eq:
    key = (c["book"], c["page"], c["column"])
    bbs = ocr_box.get(key)
    if bbs is None or len(bbs) != len(c["chars"]): miss += 1; continue
    ik = (c["book"], c["page"])
    if ik not in imgs:
        imgs[ik] = cv2.imread(str(REPO / "prepared" / T.BOOKDIR[c["book"]] / "pages" / f"{c['page']}.png"), cv2.IMREAD_GRAYSCALE)
    X = np.zeros((len(bbs), T.SZ, T.SZ), np.uint8)
    for i, bb in enumerate(bbs):
        g = T.cut(imgs[ik], bb)
        if g is not None: X[i] = g
    with torch.no_grad():
        lg, _ = net(T._prep_batch(X, np.arange(len(bbs)), dev))
        LP_ocr[key] = torch.log_softmax(lg, 1).cpu().numpy()
print(f"cột {len(eq)} · có hộp OCR khớp số chữ: {len(LP_ocr)} · lệch: {miss} · {time.time()-t0:.0f}s")
eq = [c for c in eq if (c["book"], c["page"], c["column"]) in LP_ocr]
# so sánh top-1 âm giữa hai nguồn crop
t1f = t1o = n = 0
for c in eq:
    key = (c["book"], c["page"], c["column"]); L = LP_ocr[key]
    for i, s in enumerate(c["syl"]):
        if s.lower() in cid:
            n += 1; y = cid[s.lower()]
            t1f += LP_final[c["rows"][i]].argmax() == y; t1o += L[i].argmax() == y
print(f"top-1 âm trên {n} ô: crop hộp cuối {t1f/n:.3f} · crop hộp OCR thô {t1o/n:.3f}")

def dp_align(chars, syls, emis, lam, gap_vis, M, pg, corpus):
    # emis: hàm (i, j) -> chi phí ảnh
    m, n = len(chars), len(syls); band = abs(m - n) + max(1, aa.BAND_SLACK)
    def tcost(i, j):
        base = substitution_cost(chars[i], syls[j], qn, sim)
        if corpus and len(C.pair_pages.get((chars[i], syls[j].lower()), set()) - {pg}) >= 2: return min(base, 0.0)
        return base
    INF = float("inf"); dp = [[INF] * (n + 1) for _ in range(m + 1)]; bt = [[None] * (n + 1) for _ in range(m + 1)]; dp[0][0] = 0.0
    cdel = M["COST_DEL"] + lam * gap_vis; cins = M["COST_INS"] + lam * gap_vis
    for i in range(m + 1):
        for j in range(n + 1):
            if abs(i - j) > band or (i == 0 and j == 0): continue
            best, op = INF, None
            if i > 0 and j > 0 and dp[i-1][j-1] < INF:
                c = dp[i-1][j-1] + tcost(i-1, j-1) + lam * emis(i-1, j-1)
                if c < best: best, op = c, "M"
            if i > 0 and dp[i-1][j] < INF and dp[i-1][j] + cdel < best: best, op = dp[i-1][j] + cdel, "D"
            if j > 0 and dp[i][j-1] < INF and dp[i][j-1] + cins < best: best, op = dp[i][j-1] + cins, "I"
            dp[i][j], bt[i][j] = best, op
    pairs, dels, inss = [], [], []; i, j = m, n
    while i > 0 or j > 0:
        s = bt[i][j] or ("D" if i > 0 else "I")
        if s == "M": i -= 1; j -= 1; pairs.append((i, j))
        elif s == "D": i -= 1; dels.append(i)
        else: j -= 1; inss.append(j)
    return pairs[::-1], dels, inss

all_chars = [c for col in cols for c in col["chars"] if c]
def run(kind, lam, corpus, source, seed=2026):
    set_matrix(CALIB); rng = random.Random(seed); n = wrong = gap_ok = ncol = 0
    for col in eq:
        chars, syls, rows = col["chars"], col["syl"], col["rows"]; m = len(chars)
        key = (col["book"], col["page"], col["column"]); Lo = LP_ocr[key]
        st = rng.getstate(); k = rng.randrange(2, m - 2); rng.setstate(st)
        c2, s2, truth = perturb(chars, syls, kind, rng, all_chars)
        orig = [(i if i < k else i + 1) for i in range(len(c2))] if kind == "drop_char" else list(range(len(c2)))
        def emis(i, j):
            s = s2[j].lower()
            if s not in cid: return C0
            lp = LP_final[rows[orig[i]], cid[s]] if source == "final" else Lo[orig[i], cid[s]]
            return float(min(-float(lp), CAP))
        pairs, dels, inss = dp_align(c2, s2, emis, lam, 8.0, CALIB, (col["book"], col["page"]), corpus)
        ncol += 1
        for (i, j) in pairs: n += 1; wrong += truth.get(i) != j
        if kind == "drop_char": gap_ok += (inss == [k]) and not dels
        elif kind == "drop_syl": gap_ok += (dels == [k]) and not inss
        elif kind == "none": gap_ok += (not dels and not inss)
    return n, wrong, gap_ok, ncol
for kind in ("drop_char", "drop_syl", "none"):
    out = []
    for name, lam, corpus, src in (("CALIB thuần", 0, False, "final"), ("+corpus", 0, True, "final"),
                                   ("+corpus+ảnh(hộp cuối)", 0.25, True, "final"), ("+corpus+ảnh(hộp OCR thô)", 0.25, True, "ocr"),
                                   ("+ảnh(hộp OCR thô) không corpus", 0.25, False, "ocr")):
        n, w, g, nc = run(kind, lam, corpus, src)
        out.append(f"{name}: sai {w}/{n} ({w/n:.2%}) khe {g/nc:.1%}")
    print(f"  {kind:10s} | " + " | ".join(out))
