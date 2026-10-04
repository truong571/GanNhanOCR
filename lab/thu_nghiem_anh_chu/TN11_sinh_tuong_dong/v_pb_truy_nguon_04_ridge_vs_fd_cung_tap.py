"""v_pb_truy_nguon_04 (04/10) — PHẢN BIỆN: người kiểm chứng so f_fd (chỉ trên ô mà chữ đúng CÓ ảnh FD, 82,9/82,6 %) với f_ridge trên TẤT CẢ ô (82,8/78,6 %).
Đó là so khác tập. Ở đây: Top-1 của f_fd, f_ridge, f_font, f_self trên CÙNG tập con (chữ đúng có / thiếu ảnh FD), theo đúng p01 (NaN -> -9).
0 API, CPU. Ra: measure_out/_tn11/verify/pb_truy_nguon/ridge_vs_fd.json
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"

def top1(F, col, cells):
    x = F[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9)
    t = x.loc[x.groupby("i")[col].idxmax()].set_index("i").y
    return t.reindex(cells).values

res = {}
for b in ("L16", "TK", "B34", "B18"):
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "n_self", "f_font", "f_fd", "f_self"]]
    S = pd.read_pickle(REPO / f"measure_out/_tn11/p01_{b}_scores.pkl")
    F = F.merge(S[["i", "c", "f_ridge"]], on=["i", "c"], how="left")
    hy = F.groupby("i").y.max(); cells = hy[hy == 1].index.values
    F = F[F.i.isin(cells)].copy()
    tru = F[F.y == 1].drop_duplicates("i").set_index("i").reindex(cells)
    has = tru.f_fd.notna().values; hasfont = tru.f_font.notna().values
    rare = (tru.n_self.fillna(0) < 2).values
    sig = {c: top1(F, c, cells) for c in ("f_fd", "f_ridge", "f_font", "f_self")}
    def pr(mask):
        return {k: (round(100 * float(v[mask].mean()), 2) if mask.sum() else None) for k, v in sig.items()} | {"n": int(mask.sum())}
    r = dict(n=int(len(cells)), co_FD=pr(has), thieu_FD=pr(~has), tat_ca=pr(np.ones(len(cells), bool)),
             co_FD_va_chu_hiem=pr(has & rare), thieu_FD_va_chu_hiem=pr(~has & rare),
             co_FD_va_chu_co_mau=pr(has & ~rare), thieu_FD_va_chu_co_mau=pr(~has & ~rare),
             ty_le_co_FD=round(100 * float(has.mean()), 2), ty_le_thieu_glyph_font=round(100 * float((~hasfont).mean()), 3))
    # chênh ghép cặp ridge - fd trên co_FD
    d = (sig["f_ridge"][has] - sig["f_fd"][has]).astype(float)
    r["ridge_tru_fd_tren_co_FD_diem"] = round(100 * float(d.mean()), 2)
    d2 = (sig["f_ridge"][~has] - sig["f_fd"][~has]).astype(float)
    r["ridge_tru_fd_tren_thieu_FD_diem"] = round(100 * float(d2.mean()), 2)
    res[b] = r
    print(b, json.dumps(r, ensure_ascii=False), flush=True)
(OUT / "ridge_vs_fd.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
