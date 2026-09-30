"""t01_oracle.py — tiềm năng theo NHÓM ô trên bộ có sự thật (B18, B34, L16, TK): nhãn đúng, kim đúng, vị trí đúng,
chữ người ∈ R(âm) (trần của bộ chọn trong R), đúng hai vế. 0 API, CPU. Ra: measure_out/_tn8/oracle.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402


def cat_of(D: pd.DataFrame) -> pd.Series:
    r = D.rule.astype(str)
    g = D.gate_reason.astype(str).str.split(":").str[0]
    base = r.str.split("|").str[0].str.split(":").str[0]
    return D.tier + "|" + base + np.where(g != "", "|" + g, "")


def main():
    C = T.lex()
    out = {}
    for b in ("B18", "B34", "L16", "TK"):
        D = T.load_base(b)
        D["cat"] = cat_of(D)
        R = {s: C.R_of(s) for s in D.syllable.unique()}
        has = D.gt_char != ""
        D["lab_ok"] = [bool(l) and bool(g) and C.var_eq_plus(l, g) for l, g in zip(D.label, D.gt_char)]
        D["kim_ok"] = [bool(k) and bool(g) and C.var_eq_plus(k, g) for k, g in zip(D.ocr_char, D.gt_char)]
        D["gt_inR"] = [bool(g) and C.var_in(g, R[s]) for g, s in zip(D.gt_char, D.syllable)]
        D["kim_inR"] = [bool(k) and k in R[s] for k, s in zip(D.ocr_char, D.syllable)]
        res = {}
        for cat, g in D.groupby("cat"):
            h = g[g.gt_char != ""]
            pk = g[g.pos_known & (g.gt_char != "")]
            res[cat] = dict(n=int(len(g)), n_gt=int(len(h)), gt_frac=round(len(h) / len(g), 3),
                            lab_ok=round(float(h.lab_ok.mean()), 4) if len(h) else None,
                            kim_ok=round(float(h.kim_ok.mean()), 4) if len(h) else None,
                            gt_inR=round(float(h.gt_inR.mean()), 4) if len(h) else None,
                            n_pos=int(len(pk)), pos_ok=round(float(pk.pos_ok.mean()), 4) if len(pk) else None,
                            both_lab=round(float((pk.lab_ok & pk.pos_ok).mean()), 4) if len(pk) else None,
                            both_gtinR=round(float((pk.gt_inR & pk.pos_ok).mean()), 4) if len(pk) else None)
        out[b] = dict(n=int(len(D)), n_gt=int(has.sum()), cats=dict(sorted(res.items(), key=lambda kv: -kv[1]["n"])))
        print(f"== {b}: {len(D)} ô, gt {int(has.sum())}")
        print(f"  {'nhóm':55s} {'n':>6s} {'gt%':>5s} {'nhãn':>6s} {'kim':>6s} {'gt∈R':>6s} {'n_pos':>6s} {'vtrí':>6s} {'2vế':>6s} {'2vếR':>6s}")
        for cat, v in list(out[b]["cats"].items())[:16]:
            f = lambda x: "  -   " if x is None else f"{100 * x:6.1f}"  # noqa: E731
            print(f"  {cat[:55]:55s} {v['n']:6d} {100 * v['gt_frac']:5.0f} {f(v['lab_ok'])} {f(v['kim_ok'])} {f(v['gt_inR'])} "
                  f"{v['n_pos']:6d} {f(v['pos_ok'])} {f(v['both_lab'])} {f(v['both_gtinR'])}")
    T.jdump(out, T.OUT / "oracle.json")


if __name__ == "__main__":
    main()
