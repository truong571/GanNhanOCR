"""t05b_pass2.py — LƯỢT 2 nguyên mẫu cùng sách (tự học, không nhãn người): thêm vào tập "neo" các ô KHÔNG GOLD mà lượt 1
nhận với P ≥ P_MIN (nhãn = top-1 lượt 1), rồi tính lại f_self / n_self / lp_syl / lp_glob (vẫn CHỈ từ khối trang khác).
Lượt 1 của mỗi bộ phải do mô hình KHÔNG thấy nhãn người của bộ đó (B18 <- B34, B34 <- B18, STT <- họ hand, …).
Ra: measure_out/_tn8/cand/<bộ>_p2.pkl. 0 API.
  .venv/bin/python lab/thu_nghiem_kim/TN8_chon_chu/t05b_pass2.py B18:B18__from_B34 B34:B34__from_B18
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t05_feats as F5  # noqa: E402

P_MIN = 0.95


def run(b: str, pred_name: str, tag: str = "_p2"):
    C = T.lex()
    D = T.load_base(b)
    E = np.load(F5.EMB / f"{b}_enc.npy").astype(np.float32)
    F = pd.read_pickle(F5.CAND / f"{b}.pkl")
    cells = pd.read_pickle(F5.CAND / f"{b}_cells.pkl")
    o = pd.read_pickle(T.OUT / "pred" / f"{pred_name}.pkl")
    blk = cells.blk.values
    anc = cells.anchor.values.copy()
    lab = D.label.values.astype(object).copy()
    add = (o.tier.values != "GOLD") & (np.nan_to_num(o.P.values, nan=-1) >= P_MIN) & o.top1.notna().values & ~anc
    lab[add] = o.top1.values[add]
    anc2 = anc | add
    ai = np.nonzero(anc2 & (cells.crop_ok.values > 0))[0]
    alab = lab[ai]; ablk = blk[ai]; asyl = np.array([C.R_key(s) for s in D.syllable.values[ai]], dtype=object)
    tot = defaultdict(lambda: np.zeros(512, np.float64)); totn = defaultdict(int)
    bs = defaultdict(lambda: np.zeros(512, np.float64)); bn = defaultdict(int)
    for i, l, k in zip(ai, alab, ablk):
        tot[l] += E[i]; totn[l] += 1; bs[(k, l)] += E[i]; bn[(k, l)] += 1
    sy_tot = defaultdict(int); sy_blk = defaultdict(int); syc_tot = defaultdict(int); syc_blk = defaultdict(int)
    for l, k, s in zip(alab, ablk, asyl):
        sy_tot[s] += 1; sy_blk[(k, s)] += 1; syc_tot[(s, l)] += 1; syc_blk[(k, s, l)] += 1
    ci = F.i.values; cc = F.c.values
    keys = sorted(set(zip(blk[ci].tolist(), cc.tolist())))
    kid = {k: j for j, k in enumerate(keys)}
    Ps = np.zeros((len(keys), 512), np.float32); js = np.full(len(keys), -1); ns = np.zeros(len(keys), np.int32)
    for (k, c), j in kid.items():
        n = totn.get(c, 0) - bn.get((k, c), 0)
        ns[j] = n
        if n >= 2:
            v = tot[c] - bs.get((k, c), 0)
            Ps[j] = v / max(np.linalg.norm(v), 1e-9); js[j] = j
    kk = np.array([kid[(k, c)] for k, c in zip(blk[ci].tolist(), cc.tolist())])
    F = F.copy()
    F["f_self"] = F5.rowdot(E, ci, Ps, js[kk])
    F["n_self"] = ns[kk]
    rk = np.array([C.R_key(s) for s in D.syllable.values], dtype=object)
    Rm = {s: C.R_of(s) for s in set(D.syllable)}
    nsy = np.array([sy_tot.get(rk[i], 0) - sy_blk.get((blk[i], rk[i]), 0) for i in ci])
    nsc = np.array([syc_tot.get((rk[i], c), 0) - syc_blk.get((blk[i], rk[i], c), 0) for i, c in zip(ci, cc)])
    nR = np.array([max(1, len(Rm[s])) for s in D.syllable.values])[ci]
    F["lp_syl"] = np.log((nsc + 0.5) / (nsy + 0.5 * nR)).astype(np.float32)
    F["n_syl_c"] = nsc
    F["lp_glob"] = np.log1p(ns[kk]).astype(np.float32)
    F["f_self"] = F.f_self.astype(np.float32)
    F.to_pickle(F5.CAND / f"{b}{tag}.pkl")
    print(f"[t05b] {b}: neo {int(anc.sum())} + lượt 1 (P ≥ {P_MIN}) {int(add.sum())} -> {int(anc2.sum())}", flush=True)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        x = a.split(":")
        run(x[0], x[1], x[2] if len(x) > 2 else "_p2")
