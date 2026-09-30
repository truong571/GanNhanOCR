"""t13_levers.py — độ đúng CỦA Ô THÊM theo từng đòn bẩy (mỗi đòn một mình, ngưỡng LOBO biên 0,80) trên 4 bộ có sự thật,
cả biến thể 'nhận hết với nhãn kim' (không bộ chọn) — chữ (mọi ô có gt) và hai vế (ô có vị trí), CI bootstrap cụm trang.
Ra: measure_out/_tn8/levers.json. 0 API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t12_final as F  # noqa: E402

GROUPS = {"L1": ["direct_crop"], "L2": ["direct_qn"], "L2b": ["direct_other", "similar", "other"],
          "L4": ["syl", "nocontext", "lowpost", "syl_cropbad"]}


def prec(o, m, lab_col):
    C = T.lex()
    lab = o[lab_col].fillna("").values
    ok = np.array([bool(l) and bool(g) and C.var_eq_plus(l, g) for l, g in zip(lab, o.gtc.values)])
    has = o.gtc.values != ""
    r = {}
    t = m & has
    if t.sum():
        pt, lo, hi = T.boot_ci_pages(ok[t], o.page.values[t])
        r["text"] = [round(pt, 4), round(lo, 4), round(hi, 4), int(t.sum())]
    b = t & o.pos_known.values
    if b.sum():
        pt, lo, hi = T.boot_ci_pages((ok & o.pos_ok.values)[b], o.page.values[b])
        r["both"] = [round(pt, 4), round(lo, 4), round(hi, 4), int(b.sum())]
    return r


def main():
    S = json.load(open(T.OUT / "summary_t12_b80.json", encoding="utf-8"))
    out = {}
    for te, tr, fam in (("B18", "B34", "hand"), ("B34", "B18", "hand"), ("L16", "TK", "print"), ("TK", "L16", "print")):
        o = pd.read_pickle(T.OUT / "pred" / f"{te}__from_{tr}.pkl")
        ti, to = S["books"][te]["tau_lobo"]
        gold = o.tier.values == "GOLD"
        res = {}
        for L, gs in GROUPS.items():
            m_all = np.isin(o.grp.values, gs) & ~gold & ~np.isin(o.gate.values, F.P.EXCL_GATES)
            a = F.acc_of(o, "hand" if fam == "hand" else fam, ti, to, gs if fam == "hand" else [g for g in gs if g in F.PROMO[fam]])
            if fam == "print" and L in ("L2", "L4"):
                a = np.zeros(len(o), bool)
            r = dict(n_nhom=int(m_all.sum()), n_nhan=int(a.sum()), share_mot_minh=round((gold.sum() + a.sum()) / len(o), 4),
                     o_them_bo_chon=prec(o, a, "top1"))
            if L in ("L1", "L2", "L2b"):
                k = m_all & (o.label.values != "")
                r["nhan_het_nhan_kim"] = dict(n=int(k.sum()), share=round((gold.sum() + k.sum()) / len(o), 4), **prec(o, k, "label"))
            if L == "L4":
                k = m_all & np.isin(o.part.values, ["out"])
                r["nhan_het_top1"] = dict(n=int(k.sum()), **prec(o, k, "top1"))
            res[L] = r
            print(f"[t13] {te} {L}: nhóm {r['n_nhom']}, nhận {r['n_nhan']} -> {r['share_mot_minh']} | ô thêm {r['o_them_bo_chon']}"
                  + (f" | nhận hết nhãn kim {r['nhan_het_nhan_kim']}" if "nhan_het_nhan_kim" in r else "")
                  + (f" | nhận hết top1 {r['nhan_het_top1']}" if "nhan_het_top1" in r else ""), flush=True)
        out[te] = res
    T.jdump(out, T.OUT / "levers.json")


if __name__ == "__main__":
    main()
