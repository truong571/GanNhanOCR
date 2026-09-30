"""t09b_box_strata.py — tách kết quả trọng tài hộp (t09) theo tầng: cột đếm khớp (n_ocr == n_qn == n_det) hay không, nguồn hộp
legacy/visual_dp. Nếu CNN sạch vẫn nghiêng visual_dp ngay ở cột đếm khớp (nơi gán theo chỉ số gần như chắc đúng) thì nghi thước
CNN (thiên về crop "gọn"), ngược lại thì củng cố visual_dp. 0 API. Ra measure_out/_tn9/box_strata.json
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
    geoA = pd.read_pickle(T.OUT / "kimgeo/l1skel_l2__legacy.pkl"); geoB = pd.read_pickle(T.OUT / "kimgeo/l1skel_l2__visual_dp.pkl")
    kk = lambda G: G.page + "/" + G.column.astype(str) + "/" + G.syl_idx.astype(str)  # noqa: E731
    out = {}
    allR = []
    for bk in T.STT:
        A = LA[LA.book == bk].reset_index(drop=True); B = LB[LB.book == bk].reset_index(drop=True)
        R, _, _ = X.compare(A, B, T.REPO / "prepared" / T.STT_FULL[bk], False, nets, uid, None, max_n=10 ** 9)
        Ai = A.assign(key=kk(A)).drop_duplicates("key").set_index("key")
        Bi = B.assign(key=kk(B)).drop_duplicates("key").set_index("key")
        R["cons"] = ((Ai.n_ocr == Ai.n_qn) & (Ai.n_qn == Ai.n_det)).reindex(R.key).values
        R["cs"] = Ai.count_source.reindex(R.key).values
        R["bsB"] = Bi.box_source.reindex(R.key).values
        gA = geoA[geoA.book == bk]; gB = geoB[geoB.book == bk]
        gA = dict(zip(kk(gA), gA.geo_own == gA.ocr_char)); gB = dict(zip(kk(gB), gB.geo_own == gB.ocr_char))
        R["geoA"] = [bool(gA.get(k)) for k in R.key]; R["geoB"] = [bool(gB.get(k)) for k in R.key]
        R["book"] = bk
        allR.append(R)
    R = pd.concat(allR)
    R.to_pickle(T.OUT / "box_strata.pkl")
    for nm, sub in [("all", R), ("counts_agree", R[R.cons == True]), ("counts_disagree", R[R.cons == False])] + \
            [(f"cs={c}", R[R.cs == c]) for c in ("equal_qn", "equal_ocr", "conflict")] + \
            [(f"vdp={c}", R[R.bsB == c]) for c in ("vdp_det", "vdp_low", "vdp_agree")]:
        if len(sub) == 0:
            continue
        out[nm] = dict(n=int(len(sub)), vdp_higher_s=round(float((sub.sB > sub.sA).mean()), 4),
                       margin_pos_legacy=round(float((sub.mA > 0).mean()), 4), margin_pos_vdp=round(float((sub.mB > 0).mean()), 4),
                       geo_legacy=round(float(sub.geoA.mean()), 4), geo_vdp=round(float(sub.geoB.mean()), 4))
        print(f"[t09b] {nm:16s} {out[nm]}", flush=True)
    T.jdump(out, T.OUT / "box_strata.json")


if __name__ == "__main__":
    main()
