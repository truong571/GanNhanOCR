"""K5 · Đối chứng hoán vị cho chuỗi "thanh ghi trượt" (k5_register_shift.py, thr 0,8, chuỗi ≥3).
Hoán vị (argmax, max_prob) giữa các ô TRONG TỪNG CỘT (giữ tỉ lệ ô trượt/cột) rồi đếm lại chuỗi ≥3.
    .venv/bin/python KhoiB/v3/k5_register_shift_null.py [n_lap=3] [seed=0]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 3; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0; THR = 0.8
L = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
O = pd.read_csv(REPO / "KhoiB/v3/p_visual_oof_v3_results/p_visual_oof_v3.csv", dtype=str, keep_default_na=False)
D = L[["book", "page", "column", "syl_idx", "syllable"]].copy(); D["si"] = D.syl_idx.astype(int)
am = O["argmax"].values.copy(); mp = pd.to_numeric(O.max_prob, errors="coerce").values.copy()
key = (D.book + "|" + D.page + "|" + D["column"]).values
syl = dict(zip(zip(key, D.si), D.syllable))
groups = D.groupby(key, sort=False).indices


def count_runs(am, mp):
    h = np.zeros((len(D), 2), bool)
    for n, (k, s, own) in enumerate(zip(key, D.si.values, D.syllable.values)):
        if mp[n] < THR:
            continue
        for j, d in enumerate((+1, -1)):
            nb = syl.get((k, s + d))
            if nb is not None and nb != own and am[n] == nb:
                h[n, j] = True
    n_runs = 0
    for k, idx in groups.items():
        order = idx[np.argsort(D.si.values[idx])]
        for j in (0, 1):
            cur = 0
            for i in order:
                cur = cur + 1 if h[i, j] else 0
                if cur == 3:
                    n_runs += 1
    return int(h.any(1).sum()), n_runs


hit, real = count_runs(am, mp)
print(f"thật: ô trượt đơn lẻ {hit}/{len(D)} = {hit/len(D):.4f} · chuỗi ≥3: {real}")
rng = np.random.default_rng(SEED)
for t in range(NREP):
    am2, mp2 = am.copy(), mp.copy()
    for k, idx in groups.items():
        p = rng.permutation(idx); am2[idx] = am[p]; mp2[idx] = mp[p]
    print(f"đối chứng hoán vị trong cột #{t}: chuỗi ≥3: {count_runs(am2, mp2)[1]}")
