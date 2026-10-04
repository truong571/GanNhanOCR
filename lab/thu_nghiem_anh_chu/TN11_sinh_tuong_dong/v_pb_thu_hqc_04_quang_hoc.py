"""v_pb_thu_hqc_04_quang_hoc.py — PHẢN BIỆN độc lập (04/10): tính LẠI (mã của tôi) tác dụng "quang học" vs "giấy" từ các nhúng đã lưu của người kiểm chứng
(primary_p0 = C1 giấy+quang học; ctrlwhite = cùng quang học nền TRẮNG; paperonly = giấy-chỉ norm=True; literal = K09 nguyên văn norm=False) trên mẫu
s3000 / s500 (mau_<bộ>.npz). Cho mỗi biến thể: Top-1, Δ so với C0 và Δ so với C1, CI cụm TRANG và cụm CHỮ-ĐÚNG (hạt giống của tôi).
Chỉ ĐỌC dữ liệu của họ; ghi measure_out/_tn11/verify/pb_thu_hqc/quang_hoc_<bộ>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import v_pb_thu_hqc_00_tai_tinh as A  # noqa: E402  (per_cell, cluster_ci của tôi)

T = A.T
SRC = A.SRC
OUT = A.OUT


def load(f):
    f = Path(f)
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return {str(k): v for k, v in zip(z["keys"], z["E"])}


def scores(F, E, emb):
    cats = pd.Categorical(F.c)
    G = np.zeros((len(cats.categories), 512), np.float32); has = np.zeros(len(cats.categories), bool)
    for j, ch in enumerate(cats.categories):
        v = emb.get(str(ch))
        if v is not None and not np.isnan(v[0]):
            G[j] = v; has[j] = True
    codes = cats.codes.astype(np.int64); ii = F.i.values.astype(np.int64)
    s = np.full(len(F), np.nan, np.float32)
    k = np.nonzero(has[codes])[0]
    for a in range(0, len(k), 300000):
        kk = k[a:a + 300000]
        s[kk] = np.einsum("ij,ij->i", E[ii[kk]], G[codes[kk]])
    return s


def run(b):
    D = T.load_base(b)
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font"]]
    hy = F.groupby("i").y.max(); truth = hy[hy == 1].index.values
    F = F[F.i.isin(truth)].reset_index(drop=True)
    E = np.load(T.OUT / "emb" / f"{b}_enc.npy").astype(np.float32)
    S = np.load(SRC / f"mau_{b}.npz")
    page = D.page.values
    res = dict(book=b, sets={})
    emb = {"C1": load(SRC / "emb" / f"primary_p0_{b}_font.npz"), "CTRLWHITE": load(SRC / "emb" / f"ctrlwhite_{b}_font.npz"),
           "PAPERONLY": load(SRC / "emb" / f"paperonly_{b}_font.npz"), "LIT": load(SRC / "emb" / f"literal_{b}_font.npz")}
    tru_all = F[F.y == 1].drop_duplicates("i").set_index("i")
    for sname, cells in (("s3000", S["s3000"]), ("s500", S["s1000"][:500])):
        m = F.i.isin(set(int(x) for x in cells)).values
        Fs = F[m].reset_index(drop=True)
        y = Fs.y.values.astype(np.int8)
        sc = {"C0": Fs.f_font.values}
        for nm, e in emb.items():
            if e is not None:
                s = scores(Fs, E, e)
                # chỉ giữ nếu phủ đủ (mọi hàng có glyph sản xuất đều có điểm mới)
                miss = int((~np.isnan(Fs.f_font.values) & np.isnan(s)).sum())
                sc[nm] = s
                res.setdefault("thieu_glyph", {}).setdefault(sname, {})[nm] = miss
        ok, cov = {}, {}
        for nm, s_ in sc.items():
            ok[nm], cells_ = A.per_cell(Fs.i.values, s_, y, "first")
            bad = ~np.isnan(sc["C0"]) & np.isnan(s_)
            badc = pd.Series(bad).groupby(Fs.i.values).any()
            cov[nm] = ~badc.reindex(cells_).values.astype(bool)
        pg = page[cells_]
        tch = tru_all.c.reindex(cells_).astype(str).values
        r = dict(n_cells_mau=int(len(cells_)), n_phu={nm: int(c.sum()) for nm, c in cov.items()}, top1={}, delta={})
        for nm, o in ok.items():
            r["top1"][nm] = round(float(o[cov[nm]].mean() * 100), 2)
        pairs = [(n1, "C0") for n1 in ok if n1 != "C0"] + [("C1", n2) for n2 in ok if n2 not in ("C0", "C1")]
        for a_, b_ in pairs:
            mk = cov[a_] & cov[b_]
            if mk.sum() < 50:
                continue
            d = (ok[a_][mk] - ok[b_][mk]) * 100
            mm, lo, hi, _ = A.cluster_ci(d, pg[mk], seed=7001)
            m2, lo2, hi2, _ = A.cluster_ci(d, tch[mk], seed=7002)
            r["delta"][f"{a_}-{b_}"] = dict(n=int(mk.sum()), delta=round(mm, 2), ci_trang=[round(lo, 2), round(hi, 2)], ci_chu_dung=[round(lo2, 2), round(hi2, 2)])
        res["sets"][sname] = r
    print(json.dumps(res, ensure_ascii=False))
    (OUT / f"quang_hoc_{b}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    for b in (sys.argv[1:] or ["B34", "L16", "TK", "B18"]):
        run(b)
