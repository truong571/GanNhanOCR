"""v_pb_thu_hqc_07_trang.py — PHẢN BIỆN độc lập (04/10): kiểm KHÔNG THAM SỐ cho Δ(C1-C0, f_font, mọi ô): (i) kiểm dấu theo TRANG (tỉ lệ trang có Δ>0),
(ii) kiểm hoán dấu cụm (sign-flip) mức trang VÀ mức chữ-đúng (1e5 lần, hạt giống của tôi), (iii) Δ theo ba nhóm cỡ crop (chiều dài cạnh dài).
Dùng cột sản xuất f_font (C0) và nhúng C1 đã lưu của họ. Chỉ đọc."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import v_pb_thu_hqc_00_tai_tinh as A
T = A.T
def signflip_p(d, cl, B=100000, seed=11):
    up, inv = np.unique(cl, return_inverse=True)
    k = np.bincount(inv, weights=d, minlength=len(up))
    obs = abs(k.sum())
    rng = np.random.default_rng(seed)
    cnt = 0; done = 0
    while done < B:
        n = min(5000, B - done)
        sgn = rng.choice([-1.0, 1.0], size=(n, len(up)))
        cnt += int((np.abs(sgn @ k) >= obs).sum()); done += n
    return (cnt + 1) / (B + 1)
out = {}
for b in (sys.argv[1:] or ["B34", "L16", "TK", "B18"]):
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font"]]
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
    d = o1 - o0
    page = D.page.values[c0]
    tru = F[F.y == 1].drop_duplicates("i").set_index("i")
    tch = tru.c.reindex(c0).astype(str).values
    up, inv = np.unique(page, return_inverse=True)
    kp = np.bincount(inv, weights=d, minlength=len(up)); npg = np.bincount(inv, minlength=len(up))
    pos = int((kp > 0).sum()); neg = int((kp < 0).sum()); zero = int((kp == 0).sum())
    # nhóm cỡ crop (cạnh dài) theo tam phân vị
    meta = np.load(T.OUT / "emb" / f"{b}_crop_meta.npy")
    L = np.maximum(meta[c0, 0], meta[c0, 1]); q = np.quantile(L, [1 / 3, 2 / 3])
    grp = np.digitize(L, q)
    r = dict(n=int(len(d)), delta=round(float(d.mean() * 100), 3), trang_duong=pos, trang_am=neg, trang_bang=zero, n_trang=int(len(up)),
             p_dau_trang=None, p_signflip_trang=round(signflip_p(d, page), 5), p_signflip_chu_dung=round(signflip_p(d, tch, seed=12), 5),
             theo_co_crop={f"tertile{g}": dict(n=int((grp == g).sum()), L_med=float(np.median(L[grp == g])), delta=round(float(d[grp == g].mean() * 100), 2)) for g in range(3)})
    from math import comb
    n_eff = pos + neg
    r["p_dau_trang"] = round(sum(comb(n_eff, j) for j in range(max(pos, neg), n_eff + 1)) * 2 / 2 ** n_eff, 6) if n_eff else None
    out[b] = r
    print(b, json.dumps(r, ensure_ascii=False))
(A.OUT / "trang_phi_tham_so.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
