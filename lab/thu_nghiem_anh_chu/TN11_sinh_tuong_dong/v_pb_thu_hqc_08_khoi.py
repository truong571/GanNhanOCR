"""v_pb_thu_hqc_08_khoi.py — PHẢN BIỆN độc lập (04/10): Δ(C1-C0, f_font) trên 5 KHỐI TRANG liền (T.page_blocks, như LOBO của TN8) và 200 lần chia ngẫu nhiên
các trang thành hai nửa (mẫu con khác nhau, không dùng lại nhúng mới): có khối/nửa nào đổi dấu không. Dùng nhúng C1 đã lưu + cột C0. Chỉ đọc."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import v_pb_thu_hqc_00_tai_tinh as A
T = A.T
out = {}
for b in (sys.argv[1:] or ["B34", "L16", "TK", "B18"]):
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_font"]]
    hy = F.groupby("i").y.max(); cells = hy[hy == 1].index.values
    F = F[F.i.isin(cells)].reset_index(drop=True)
    E = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    z = np.load(A.SRC / "emb" / f"primary_p0_{b}_font.npz", allow_pickle=False); emb = {str(k): v for k, v in zip(z["keys"], z["E"])}
    cats = pd.Categorical(F.c); G = np.zeros((len(cats.categories), 512), np.float32); has = np.zeros(len(cats.categories), bool)
    for j, ch in enumerate(cats.categories):
        v = emb.get(str(ch))
        if v is not None and not np.isnan(v[0]): G[j] = v; has[j] = True
    codes = cats.codes.astype(np.int64); ii = F.i.values.astype(np.int64)
    s1 = np.full(len(F), np.nan, np.float32); k = np.nonzero(has[codes])[0]
    for a in range(0, len(k), 300000):
        kk = k[a:a + 300000]; s1[kk] = np.einsum("ij,ij->i", E[ii[kk]], G[codes[kk]])
    y = F.y.values.astype(np.int8)
    o0, c0 = A.per_cell(F.i.values, F.f_font.values, y, "first"); o1, _ = A.per_cell(F.i.values, s1, y, "first")
    d = (o1 - o0) * 100
    pages = D.page.values[c0]
    blk = T.page_blocks(D.page.values, 5)[c0]
    r = dict(khoi={}, nua_ngau_nhien={})
    for g in range(5):
        m = blk == g
        mm, lo, hi, npg = A.cluster_ci(d[m], pages[m], seed=31 + g)
        r["khoi"][g] = dict(n=int(m.sum()), n_trang=npg, delta=round(mm, 2), ci=[round(lo, 2), round(hi, 2)])
    up = np.unique(pages); rng = np.random.default_rng(77)
    halves = []
    for _ in range(200):
        pick = set(rng.choice(up, size=len(up) // 2, replace=False))
        m = np.array([p in pick for p in pages])
        halves.append((float(d[m].mean()), float(d[~m].mean())))
    h = np.array(halves)
    r["nua_ngau_nhien"] = dict(min=round(float(h.min()), 2), max=round(float(h.max()), 2), p05=round(float(np.quantile(h, .05)), 2), n_nua_am=int((h <= 0).sum()), tong=int(h.size))
    out[b] = r
    print(b, json.dumps(r, ensure_ascii=False))
(A.OUT / "khoi_trang.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
