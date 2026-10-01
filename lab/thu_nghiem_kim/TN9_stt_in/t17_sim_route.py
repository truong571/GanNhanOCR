"""t17_sim_route.py — mô phỏng ĐƯỜNG STT CHẶT sẽ cài (hai lượt + hợp theo ô + cổng R4 + hộp visual_dp + cổng vdp lệch legacy)
trên các bản dựng hộp cát, làm ĐÍCH nghiệm thu cho bản cài trong pipeline. 0 API.
A = lt1__<hộp>, B = l1skel_l2__<hộp>; lớp/cổng như t13 (R4); cổng vdp: ô GOLD sau R4 có box_source ∈ {vdp_low, vdp_virtual,
vdp_fallback} và IoU(hộp, hộp legacy cùng khoá cùng bản đọc) < 0,5 -> REVIEW. Ra measure_out/_tn9/sim_route.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
from t13_stt_recipe import load  # noqa: E402

WEAK = ("vdp_low", "vdp_virtual", "vdp_fallback")


def iou(a, b):
    try:
        a, b = json.loads(a), json.loads(b)
    except Exception:  # noqa: BLE001
        return np.nan
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I)


def main():
    C = T.lex()
    lat = json.loads((T.OUT / "latent2.json").read_text())
    inr = json.loads((T.OUT / "inR_model.json").read_text())
    out = {}
    for box in ("legacy", "visual_dp"):
        A, B = load(f"lt1__{box}"), load(f"l1skel_l2__{box}")
        AL, BL = load("lt1__legacy"), load("l1skel_l2__legacy")
        res = {}
        for bk in T.STT:
            D = T.load_stt(bk)
            D["key"] = bk + "/" + D.page + "/" + D.column.astype(str) + "/" + D.syl_idx.astype(str)
            D = D.drop_duplicates("key").set_index("key")
            keys = A.index[A.book == bk].union(B.index[B.book == bk])
            l1 = D.ocr_char.reindex(keys).fillna("").values
            l2 = D.lt2.reindex(keys).fillna("").values
            onlyB = ~keys.isin(D.index)
            l2 = np.where(onlyB, B.ocr_char.reindex(keys).fillna("").values, l2)
            syl = pd.Series(A.syllable.reindex(keys).values, index=keys).fillna(pd.Series(B.syllable.reindex(keys).values, index=keys)).fillna("").values
            R = {s: C.R_of(s) for s in set(syl)}
            gA = A.tier.reindex(keys).fillna("").values == "GOLD"; gB = B.tier.reindex(keys).fillna("").values == "GOLD"
            lab = np.where(gB, B.label.reindex(keys).fillna("").values, A.label.reindex(keys).fillna("").values)
            rule = np.where(gB, B.rule.reindex(keys).fillna("").values, A.rule.reindex(keys).fillna("").values)
            veq = C.var_eq_plus
            cls = []
            for x, y, s, lb, ru in zip(l1, l2, syl, lab, rule):
                xi, yi = bool(x) and x in R[s], bool(y) and y in R[s]
                if ru.startswith("self_training_rescue"):
                    c = "rescue"
                elif x and y and veq(x, y) and xi and veq(lb, x):
                    c = "agree"
                elif xi and yi and not veq(x, y):
                    c = "conflict"
                elif yi and not xi and veq(lb, y):
                    c = "lt2only"
                elif xi and not y and veq(lb, x):
                    c = "lt1_nolt2"
                elif xi and y and not yi and veq(lb, x):
                    c = "lt1_l2out"
                elif ru.startswith("s1_inter_s2_similar"):
                    c = "similar"
                else:
                    c = "other"
                cls.append(c)
            cls = np.array(cls, dtype=object)
            g4 = (gA | gB) & np.isin(cls, ["agree", "lt2only", "lt1_nolt2"])
            # cổng vdp
            src_bs = np.where(gB, B.box_source.reindex(keys).fillna("").values, A.box_source.reindex(keys).fillna("").values)
            bb = np.where(gB, B.bbox.reindex(keys).fillna("").values, A.bbox.reindex(keys).fillna("").values)
            bl = np.where(gB, BL.bbox.reindex(keys).fillna("").values, AL.bbox.reindex(keys).fillna("").values)
            weak = np.isin(src_bs, WEAK)
            io = np.array([iou(x, y) if (w and x and y) else 1.0 for x, y, w in zip(bb, bl, weak)])
            vdpdem = g4 & weak & (io < 0.5)
            g5 = g4 & ~vdpdem
            rng = lat[bk]["range"]
            e = inr["stt"][bk]["lt1"]["est"]
            prec = {"agree": rng["agree_inR (lt1≡lt2 ∈ R)"], "lt2only": rng["lt2∈R, lt1∉R: lt2 đúng"],
                    "lt1_nolt2": [min(v["prec_inR"] for k, v in e.items() if k in ("q=0.089", "q=0.117", "q=0.145")),
                                  max(v["prec_inR"] for k, v in e.items() if k in ("q=0.089", "q=0.117", "q=0.145"))]}
            comp = pd.Series(cls[g5]).value_counts().to_dict()
            lo = sum(n * prec[c][0] for c, n in comp.items()) / max(1, g5.sum())
            hi = sum(n * prec[c][1] for c, n in comp.items()) / max(1, g5.sum())
            res[bk] = dict(keys=int(len(keys)), gold_R4=int(g4.sum()), share_R4=round(float(g4.mean()), 4),
                           vdp_demoted=int(vdpdem.sum()), gold_final=int(g5.sum()), share_final=round(float(g5.mean()), 4),
                           comp=comp, prec_text_SD=[round(lo, 4), round(hi, 4)])
            print(f"[t17] {box} {bk}: {res[bk]}", flush=True)
        out[box] = res
    T.jdump(out, T.OUT / "sim_route.json")


if __name__ == "__main__":
    main()
