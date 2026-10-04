"""v_pb_thu_hqc_09_fd.py — PHẢN BIỆN độc lập (04/10): tính LẠI Δ(C1-C0) cho f_fd (K=3: trung bình chuẩn hoá của 3 mảng giấy primary_p0..2_<bộ>_fd.npz) trên mọi ô có chữ đúng,
CI cụm trang + cụm chữ-đúng. Đối chiếu số của người kiểm chứng (+0,32/-0,61/+2,03/+1,12). Chỉ đọc."""
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
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_fd"]]
    hy = F.groupby("i").y.max(); cells = hy[hy == 1].index.values
    F = F[F.i.isin(cells)].reset_index(drop=True)
    E = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    parts = []
    for k in range(3):
        z = np.load(A.SRC / "emb" / f"primary_p{k}_{b}_fd.npz", allow_pickle=False); parts.append({str(c): v for c, v in zip(z["keys"], z["E"])})
    keys = set(parts[0]).intersection(*[set(p) for p in parts[1:]])
    emb = {}
    for c in keys:
        v = np.mean([p[c] for p in parts], axis=0); n = np.linalg.norm(v); emb[c] = v / n if n > 0 else v
    cats = pd.Categorical(F.c); G = np.zeros((len(cats.categories), 512), np.float32); has = np.zeros(len(cats.categories), bool)
    for j, ch in enumerate(cats.categories):
        v = emb.get(str(ch))
        if v is not None and not np.isnan(v[0]): G[j] = v; has[j] = True
    codes = cats.codes.astype(np.int64); ii = F.i.values.astype(np.int64)
    s1 = np.full(len(F), np.nan, np.float32); k = np.nonzero(has[codes])[0]
    for a in range(0, len(k), 300000):
        kk = k[a:a + 300000]; s1[kk] = np.einsum("ij,ij->i", E[ii[kk]], G[codes[kk]])
    s0 = F.f_fd.values
    miss = int((~np.isnan(s0) & np.isnan(s1)).sum())
    y = F.y.values.astype(np.int8)
    o0, c0 = A.per_cell(F.i.values, s0, y, "first"); o1, _ = A.per_cell(F.i.values, s1, y, "first")
    d = (o1 - o0) * 100
    pg = D.page.values[c0]
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    tch = tru.c.reindex(c0).astype(str).values
    m, lo, hi, _ = A.cluster_ci(d, pg, seed=5); m2, lo2, hi2, _ = A.cluster_ci(d, tch, seed=6)
    out[b] = dict(n=int(len(d)), miss=miss, top1_C0=round(float(o0.mean() * 100), 2), top1_C1=round(float(o1.mean() * 100), 2), delta=round(m, 2),
                  ci_trang=[round(lo, 2), round(hi, 2)], ci_chu_dung=[round(lo2, 2), round(hi2, 2)])
    print(b, json.dumps(out[b], ensure_ascii=False))
(A.OUT / "fd_tai_tinh.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
