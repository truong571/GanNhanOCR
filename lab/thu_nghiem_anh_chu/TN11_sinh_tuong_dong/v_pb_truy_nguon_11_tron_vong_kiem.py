"""v_pb_truy_nguon_11 (04/10) — PHẢN BIỆN: tự tính lại 2 số then chốt của 'vòng tròn': Top-1 f_head/f_fd/f_font so NHÃN pipeline ở stt2 và so chữ NGƯỜI ở B34. 0 API, CPU.
Ra: measure_out/_tn11/verify/pb_truy_nguon/tron_vong_kiem.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"; res = {}
def t1(F, col, y):
    x = F[["i", col, y]].copy(); x[col] = x[col].fillna(-9); return x.loc[x.groupby("i")[col].idxmax()].set_index("i")[y]
for b in ("stt2", "B34"):
    D = T.load_base(b); F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_head", "f_fd", "f_font"]]
    lbl = D.label.values[F.i.values]; F["yl"] = ((F.c.values == lbl) & (lbl != "")).astype(int)
    r = {}
    for nm, y in (("nhan_pipeline", "yl"), ("chu_nguoi", "y")):
        h = F.groupby("i")[y].max(); cells = h[h == 1].index.values
        if len(cells) == 0: continue
        G = F[F.i.isin(cells)]
        r[nm] = dict(n=int(len(cells)), **{m: round(100 * float(t1(G, m, y).mean()), 2) for m in ("f_head", "f_fd", "f_font")})
    res[b] = r; print(b, r, flush=True)
(OUT / "tron_vong_kiem.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
