#!/usr/bin/env python
"""r02b_self.py — kênh 's_self': nguyên mẫu crop CÙNG SÁCH, RỜI TRANG (leave-page-out), dựng từ nhãn GOLD của
pipeline (= chữ kim) trên các trang KHÁC. Không dùng GT người (không rò truth); là kênh tự nhất quán kiểu
confident-learning: crop bị kim đọc X→Y sẽ giống mẫu X (đa số crop đúng) hơn mẫu Y.
Cần ≥3 crop ở trang khác. Ra: out/cand_self.pkl (cột s_self, n_self; cùng thứ tự hàng cand.pkl)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; SCR = Path('/private/tmp/claude-501/-Users-truongmdn-TruongMDN-ThS-DoAn-GanNhanOCR/21e87791-d6ef-436a-843b-9070174de500/scratchpad')
OUT = Path("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/measure_out/_thu_nghiem_anh_chu/TN4/v1/rerank_new")
d = pd.read_csv(SCR / "kim_bottleneck/harness/cells_eval.csv", dtype=str, keep_default_na=False)
EA = np.load(OUT / "E_A.f16.npy").astype(np.float32)
cand = pd.read_pickle(OUT / "cand.pkl")
books = d.book.values; pages = d.page.values
m = (d.tier == "GOLD").values & (d.label.str.len() == 1).values
S, T = {}, {}
for i in np.nonzero(m)[0]:
    k = (books[i], pages[i], d.label.iat[i])
    s, n = S.get(k, (0, 0)); S[k] = (s + EA[i], n + 1)
for (b, p, ch), (s, n) in S.items():
    s0, n0 = T.get((b, ch), (0, 0)); T[(b, ch)] = (s0 + s, n0 + n)
val = np.full(len(cand), np.nan); cnt = np.zeros(len(cand), int)
ii = cand.i.values; chs = cand.ch.values
cache = {}
for r in range(len(cand)):
    i = ii[r]; b = books[i]; p = pages[i]; ch = chs[r]
    key = (b, p, ch)
    if key not in cache:
        s, n = T.get((b, ch), (0, 0)); sp, npg = S.get(key, (0, 0))
        s, n = s - sp, n - npg
        cache[key] = (s / (np.linalg.norm(s) + 1e-9), n) if n >= 3 else (None, n)
    v, n = cache[key]
    cnt[r] = n
    if v is not None:
        val[r] = float(EA[i] @ v)
out = pd.DataFrame(dict(s_self=val, n_self=cnt))
out.to_pickle(OUT / "cand_self.pkl")
print(json.dumps(dict(rows=len(out), frac_self=float(np.mean(~np.isnan(val))),
                      frac_self_kim=float(np.mean(~np.isnan(val[cand.is_kim.values == 1]))))))
