"""v_pb_truy_nguon_10 (04/10) — PHẢN BIỆN: tính lại chênh ghép cặp 'thêm f_map (Ridge) vào bộ chọn TN8' độc lập (đọc pred pkl của p05_bo_chon_them_f_map),
đếm ô được/mất, CI bootstrap cụm trang, và kiểm đúng cột y1/any_y. 0 API, CPU.
Ra: measure_out/_tn11/verify/pb_truy_nguon/map_trong_bo_chon.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T
OUT = REPO / "measure_out/_tn11/verify/pb_truy_nguon"; W = REPO / "measure_out/_tn11/tn8_map/pred"
PAIR = {"B18": "B34", "B34": "B18", "L16": "TK", "TK": "L16"}
res = {}
for b, tr in PAIR.items():
    D = T.load_base(b)
    G = pd.read_pickle(W / f"{b}__from_{tr}__goc.pkl"); M = pd.read_pickle(W / f"{b}__from_{tr}__map.pkl")
    O = pd.read_pickle(T.OUT / "pred" / f"{b}__from_{tr}.pkl")
    cols = list(G.columns)
    G = G[G.any_y == 1].set_index("i"); M = M[M.any_y == 1].set_index("i"); O = O[O.any_y == 1].set_index("i")
    com = np.array(sorted(set(G.index) & set(M.index))); g = G.y1.reindex(com).values.astype(float); m = M.y1.reindex(com).values.astype(float)
    pg = D.page.values[com]; d = m - g
    up, inv = np.unique(pg, return_inverse=True); k = np.bincount(inv, weights=d, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    idx = np.random.default_rng(T.BOOT_SEED).integers(0, len(up), size=(T.BOOT_B, len(up))); r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    res[b] = dict(cols=cols, n=int(len(com)), goc=round(100 * g.mean(), 2), map=round(100 * m.mean(), 2), delta=round(100 * d.mean(), 3),
                  ci=[round(100 * float(np.quantile(r, .025)), 3), round(100 * float(np.quantile(r, .975)), 3)],
                  duoc=int(((m == 1) & (g == 0)).sum()), mat=int(((m == 0) & (g == 1)).sum()), rong=int(d.sum()),
                  goc_vs_tn8_ban_dau=round(100 * float(O.y1.mean()), 2), n_ban_dau=int(len(O)))
    print(b, {k_: v for k_, v in res[b].items() if k_ != "cols"}, flush=True)
(OUT / "map_trong_bo_chon.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
