"""t09c_box_baseline.py — nền của thước CNN (t09): trên ô GOLD mà hai bộ giải hộp TRÙNG (IoU ≥ 0,5), tỉ lệ lề > 0
(nhãn thắng nhãn láng giềng) và kim_geo — để biết 30–45 % "lề > 0" của legacy trên ô lệch là thấp bất thường hay bình thường.
0 API. Ra measure_out/_tn9/box_baseline.json
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
import t09_box_arbiter as X  # noqa: E402


def main():
    nets, uid = X.nets_tables(["hand_T_Kinh", "hand_L_Kinh", "hand_T_DungLy", "hand_L_DungLy"])
    LA = T.rd(T.OUT / "stt_builds/l1skel_l2__legacy/dataset_out/labels_final.csv")
    LB = T.rd(T.OUT / "stt_builds/l1skel_l2__visual_dp/dataset_out/labels_final.csv")
    geo = pd.read_pickle(T.OUT / "kimgeo/l1skel_l2__legacy.pkl")
    out = {}
    rng = np.random.default_rng(0)
    for bk in T.STT:
        A = LA[LA.book == bk].reset_index(drop=True); B = LB[LB.book == bk].reset_index(drop=True)
        for Z in (A, B):
            Z["key"] = Z.page + "/" + Z.column + "/" + Z.syl_idx
        Ag = A[A.tier == "GOLD"].drop_duplicates("key").set_index("key"); Bg = B[B.tier == "GOLD"].drop_duplicates("key").set_index("key")
        k = Ag.index.intersection(Bg.index)
        io = np.array([X.iou(X.T8b(a), X.T8b(b)) for a, b in zip(Ag.bbox.reindex(k), Bg.bbox.reindex(k))])
        same = k[io >= 0.5]
        same = np.sort(rng.choice(same, min(1500, len(same)), replace=False))
        kA = dict(zip(A.key, A.index))
        nb = X.neighbors(A)
        idx = {kA[x] for x in same}
        cr = X.crops_for(A, idx, T.REPO / "prepared" / T.STT_FULL[bk], False)
        ch = {kA[x]: {A.at[kA[x], "label"]} | nb(A.at[kA[x], "page"], A.at[kA[x], "column"], A.at[kA[x], "syl_idx"]) for x in same}
        S = X.scores(cr, ch, nets, uid)
        m = []
        for x in same:
            i = kA[x]; lab = A.at[i, "label"]; d = S.get(i, {})
            if lab in d:
                m.append(d[lab] - max([v for c, v in d.items() if c != lab] or [-9]))
        g = geo[geo.book == bk]
        gd = dict(zip(g.page + "/" + g.column.astype(str) + "/" + g.syl_idx.astype(str), g.geo_own == g.ocr_char))
        out[bk] = dict(n=len(m), margin_pos=round(float((np.array(m) > 0).mean()), 4),
                       geo=round(float(np.mean([bool(gd.get(x)) for x in same])), 4))
        print(f"[t09c] {bk} ô trùng hộp: {out[bk]}", flush=True)
    T.jdump(out, T.OUT / "box_baseline.json")


if __name__ == "__main__":
    main()
