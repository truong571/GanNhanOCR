"""t17_check_export.py — mô hình xuất ra models/chon_chu tái lập ĐÚNG dự đoán TN8 (t06: <bộ thử>__from_<bộ học>.pkl) khi áp
lên chính bảng đặc trưng TN8. 0 API."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa
from pipeline.chon_chu import model as M, export_assets as X  # noqa

for te, tr, name in (("B18", "B34", "hand_B34"), ("B34", "B18", "hand_B18"), ("L16", "TK", "print_TK"), ("TK", "L16", "print_L16")):
    Bk = X.load_tn8(te)
    mods, meta = M.unpack(np.load(T.REPO / "models/chon_chu" / f"chooser_{name}.npz"))
    P = np.full(len(Bk["D"]), np.nan); top = np.array([""] * len(Bk["D"]), dtype=object)
    for part in ("in", "out"):
        m1, m2 = mods[part]
        p = m1.predict(Bk["Xc"], Bk["F"].i.values)
        Tt = M.top_table(Bk["F"], p, Bk["Xc"], Bk["names"])
        idx = np.nonzero(Bk["part"] == part)[0]; idx = idx[np.isin(idx, Tt.index.values)]
        X2, _ = M.cell_X(Bk["D"], Bk["cells"], Bk["grp"], Tt, idx)
        P[idx] = m2.predict(X2); top[idx] = Tt.top1.reindex(idx).values
    o = pd.read_pickle(T.OUT / "pred" / f"{te}__from_{tr}.pkl")
    m = ~np.isnan(P) & ~np.isnan(o.P.values)
    print(f"[t17] {te} <- {name}: ô có P {int(m.sum())}, |ΔP| max {np.nanmax(np.abs(P[m] - o.P.values[m])):.2e}, "
          f"top1 khác {int((top[m] != o.top1.values[m]).sum())}")
