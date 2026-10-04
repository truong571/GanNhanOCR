"""v_thu_hqc_06_tn8.py — BƯỚC 2 (CÓ ĐIỀU KIỆN: chỉ khi C1 đạt "có ích"): thêm f_hqc (cos crop · glyph HQC2 của chữ ứng viên) vào BỘ CHỌN CHỮ TN8,
đo LOBO đúng như p05_bo_chon_them_f_map.py (cùng t06_eval.run_pair, cùng 4 cặp B34->B18, B18->B34, TK->L16, L16->TK), lượt "goc" (đặc trưng TN8 như cũ) vs
lượt "hqc" (thêm f_hqc ở tầng 1). Ghi vào measure_out/_tn11/verify/thu_hqc/tn8_hqc/ — KHÔNG ghi đè measure_out/_tn8.
--sample s1000|s3000|truth|all : chỉ giữ hàng ứng viên của các ô thuộc mẫu (cả bộ HỌC lẫn bộ THỬ) — lượt goc/hqc dùng CÙNG ô; 'truth' = mọi ô có chữ đúng (cần nhúng đủ chữ).
  .venv/bin/python .../v_thu_hqc_06_tn8.py --sample s1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T = H.T
SRC = T.OUT
W = H.OUT / "tn8_hqc"
PAIRS = [("B34", "B18"), ("B18", "B34"), ("TK", "L16"), ("L16", "TK")]


def load_c1(b, ks):
    parts = []
    for k in ks:
        z = np.load(H.OUT / "emb" / f"primary_p{k}_{b}_font.npz", allow_pickle=False)
        parts.append({str(c): v for c, v in zip(z["keys"], z["E"])})
    out = {}
    for c in set(parts[0]).intersection(*[set(p) for p in parts[1:]]):
        v = np.mean([p[c] for p in parts], 0); n = np.linalg.norm(v)
        out[c] = v / n if n > 0 else v
    return out


def prep(sample, ks, pairs=None):
    pairs = pairs or PAIRS
    (W / "cand").mkdir(parents=True, exist_ok=True)
    for d in ("base", "emb"):
        if not (W / d).exists():
            os.symlink(SRC / d, W / d)
    cov = {}
    for b in {x for p in pairs for x in p}:
        F = pd.read_pickle(SRC / "cand" / f"{b}.pkl")
        if sample == "truth":                                   # mọi ô có chữ đúng trong ứng viên (bỏ ô không có sự thật: chưa nhúng chữ của chúng)
            hy = F.groupby("i").y.max()
            F = F[F.i.isin(set(hy[hy == 1].index.values))].reset_index(drop=True)
        elif sample != "all":
            S = np.load(H.OUT / f"mau_{b}.npz")[sample]
            F = F[F.i.isin(set(int(x) for x in S))].reset_index(drop=True)
        E = np.load(SRC / "emb" / f"{b}_enc.npy").astype(np.float32)
        emb = load_c1(b, ks)
        cats = pd.Categorical(F.c)
        G = np.zeros((len(cats.categories), 512), np.float32); has = np.zeros(len(cats.categories), bool)
        for j, ch in enumerate(cats.categories):
            v = emb.get(ch)
            if v is not None and not np.isnan(v[0]):
                G[j] = v; has[j] = True
        codes = cats.codes.astype(np.int64)
        s = H.rowdot(E, F.i.values.astype(np.int64), G, np.where(has[codes], codes, -1))
        miss = np.isnan(s) & ~np.isnan(F.f_font.values)             # có glyph sản xuất nhưng thiếu glyph HQC
        cov[b] = dict(rows=int(len(F)), cells=int(F.i.nunique()), thieu_hqc=int(miss.sum()))
        F["f_hqc"] = s.astype(np.float32)
        F.to_pickle(W / "cand" / f"{b}.pkl")
        c = W / "cand" / f"{b}_cells.pkl"
        if not c.exists():
            os.symlink(SRC / "cand" / f"{b}_cells.pkl", c)
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="s1000", choices=["s1000", "s3000", "truth", "all"])
    ap.add_argument("--ks", nargs="*", type=int, default=[0])
    ap.add_argument("--pairs", nargs="*", default=None, help="vd TK>L16 L16>TK (mặc định cả 4 cặp)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pairs = [tuple(x.split(">")) for x in a.pairs] if a.pairs else PAIRS
    cov = prep(a.sample, a.ks, pairs)
    print("phủ:", cov, flush=True)
    if any(v["thieu_hqc"] for v in cov.values()):
        print("CẢNH BÁO: thiếu glyph HQC ở một số hàng -> has_hqc=0 ở các hàng đó (không đồng nhất)", flush=True)
    T.OUT = W
    import tn8model as M
    import t06_eval as E
    E.PRED = W / "pred"
    base_sims = list(M.SIMS)
    for run in ("goc", "hqc"):
        M.SIMS[:] = base_sims + (["f_hqc"] if run == "hqc" else [])
        for tr, te in pairs:
            E.run_pair([tr], te, tag=f"{te}__from_{tr}__{run}")
    res = {}
    for tr, te in pairs:
        F = pd.read_pickle(W / "cand" / f"{te}.pkl")
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        rare = set(tru.index[tru.n_self.fillna(0) < 2])
        r = {}
        for run in ("goc", "hqc"):
            P = pd.read_pickle(W / "pred" / f"{te}__from_{tr}__{run}.pkl")
            P = P[P.any_y == 1]
            m = P.i.isin(rare)
            ok = P.y1.values.astype(float); pg = P.page.values
            t, lo, hi = H.boot_mean(ok * 100, pg)
            th, lh, hh = H.boot_mean(ok[m.values] * 100, pg[m.values])
            r[run] = dict(tat_ca=round(t, 2), tat_ca_ci=[round(lo, 2), round(hi, 2)], chu_hiem=round(th, 2), chu_hiem_ci=[round(lh, 2), round(hh, 2)],
                          n=int(len(P)), n_hiem=int(m.sum()))
        # chênh lệch cặp theo ô
        Pg = pd.read_pickle(W / "pred" / f"{te}__from_{tr}__goc.pkl"); Pg = Pg[Pg.any_y == 1].set_index("i")
        Ph = pd.read_pickle(W / "pred" / f"{te}__from_{tr}__hqc.pkl"); Ph = Ph[Ph.any_y == 1].set_index("i")
        ix = Pg.index.intersection(Ph.index)
        d = H.boot_diff(Ph.y1.reindex(ix).values.astype(float), Pg.y1.reindex(ix).values.astype(float), Pg.page.reindex(ix).values)
        r["delta_hqc_goc"] = dict(delta=round(d[0] * 100, 2), ci=[round(d[1] * 100, 2), round(d[2] * 100, 2)], n=d[3])
        res[f"{te}<-{tr}"] = r
        print(f"{te} <- {tr}: goc {r['goc']['tat_ca']} % (hiếm {r['goc']['chu_hiem']}) -> +f_hqc {r['hqc']['tat_ca']} % (hiếm {r['hqc']['chu_hiem']}); Δ {r['delta_hqc_goc']}", flush=True)
    H.jdump(dict(mau=a.sample, ks=a.ks, phu=cov, ket_qua=res), W / f"ket_qua_tn8_hqc{a.tag}.json")


if __name__ == "__main__":
    main()
