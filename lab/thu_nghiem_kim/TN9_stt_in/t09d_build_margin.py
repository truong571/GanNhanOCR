"""t09d_build_margin.py — thước ẢNH sạch cho cả bản dựng: trên mẫu 1.500 ô GOLD/sách (seed cố định), tỉ lệ "lề > 0" của CNN sạch
với STT (hand_T/L_Kinh + hand_T/L_DungLy, học Borg; t09): nhãn thắng nhãn các ô kề ±1, ±2 cùng cột. Crop đúng chữ nhãn ⇒ lề > 0
(nền ô hai bộ giải trùng hộp: 86,7–88,1 %, t09c); crop trượt/nhãn lệch chỗ ⇒ thấp (Borg crop sai khe: 19 %).
So bản dựng theo nguồn đọc (lt1 / l1skel_l2 / twopass / lt2) và bộ giải hộp. 0 API. Ra measure_out/_tn9/build_margin.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
import t09_box_arbiter as X  # noqa: E402


def main():
    names = sys.argv[1:] or ["lt1__legacy", "l1skel_l2__legacy", "twopass__legacy", "lt2__legacy",
                             "l1skel_l2__visual_dp", "twopass__visual_dp"]
    nets, uid = X.nets_tables(["hand_T_Kinh", "hand_L_Kinh", "hand_T_DungLy", "hand_L_DungLy"])
    f = T.OUT / "build_margin.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    for nm in names:
        L = T.rd(T.OUT / "stt_builds" / nm / "dataset_out/labels_final.csv")
        res = {}
        for bk in T.STT:
            A = L[L.book == bk].reset_index(drop=True)
            gi = np.nonzero((A.tier == "GOLD").values)[0]
            rng = np.random.default_rng(20260930)
            sel = set(rng.choice(gi, min(1500, len(gi)), replace=False).tolist())
            nb = X.neighbors(A)
            cr = X.crops_for(A, sel, T.REPO / "prepared" / T.STT_FULL[bk], False)
            ch = {i: {A.at[i, "label"]} | nb(A.at[i, "page"], A.at[i, "column"], A.at[i, "syl_idx"]) for i in sel}
            S = X.scores(cr, ch, nets, uid)
            m = []
            for i in sel:
                lab = A.at[i, "label"]; d = S.get(i, {})
                if lab in d:
                    m.append(d[lab] - max([v for c, v in d.items() if c != lab] or [-9]))
            m = np.array(m)
            res[bk] = dict(n=int(len(m)), margin_pos=round(float((m > 0).mean()), 4))
        out[nm] = res
        print(f"[t09d] {nm}: {res}", flush=True)
        T.jdump(out, f)


if __name__ == "__main__":
    main()
