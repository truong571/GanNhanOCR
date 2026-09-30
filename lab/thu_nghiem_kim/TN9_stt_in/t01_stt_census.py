"""t01_stt_census.py — STT: mỗi ô rơi vào đâu theo hai lần đọc kim (lt1 Hán, lt2 Nôm) và âm QN (R chính xác / R bỏ thanh /
R bỏ mọi dấu). 0 API, CPU. Ra measure_out/_tn9/stt_census.json + stt_cells/<bộ>.pkl (cờ dùng lại ở bước sau).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402
import tn8model as M8  # noqa: E402


def flags(D):
    C = T.lex()
    veq = C.var_eq_plus
    syl = D.syllable.values
    R = {s: C.R_of(s) for s in set(syl)}
    Rt = {s: T.R_tone(s) for s in set(syl)}
    Rf = {s: T.R_fold(s) for s in set(syl)}
    l1, l2 = D.ocr_char.values, D.lt2.values
    f = pd.DataFrame(index=D.index)
    f["l1in"] = [bool(a) and a in R[s] for a, s in zip(l1, syl)]
    f["l2in"] = [bool(a) and a in R[s] for a, s in zip(l2, syl)]
    f["l1tone"] = [bool(a) and a in Rt[s] for a, s in zip(l1, syl)]
    f["l2tone"] = [bool(a) and a in Rt[s] for a, s in zip(l2, syl)]
    f["l1fold"] = [bool(a) and a in Rf[s] for a, s in zip(l1, syl)]
    f["l2fold"] = [bool(a) and a in Rf[s] for a, s in zip(l2, syl)]
    f["eq12"] = [bool(a) and bool(b) and veq(a, b) for a, b in zip(l1, l2)]
    f["has2"] = D.lt2.values != ""
    f["nR"] = [len(R[s]) for s in syl]
    f["syl_known"] = f.nR > 0
    f["grp"] = M8.group_of(D)
    return f


def main():
    out = {}
    od = T.OUT / "stt_cells"; od.mkdir(parents=True, exist_ok=True)
    for b in T.STT:
        D = T.load_stt(b)
        f = flags(D)
        f.to_pickle(od / f"{b}.pkl")
        N = len(D)
        gold = D.tier.values == "GOLD"
        r = dict(N=N, gold=int(gold.sum()), gold_share=round(gold.mean(), 4))
        r["tiers"] = D.tier.value_counts().to_dict()
        r["grp"] = f.grp.value_counts().to_dict()
        anyin = f.l1in | f.l2in
        r["share_l1in"] = round(f.l1in.mean(), 4)
        r["share_l2in"] = round(f.l2in.mean(), 4)
        r["share_any_in"] = round(anyin.mean(), 4)
        r["share_any_tone"] = round((anyin | f.l1tone | f.l2tone).mean(), 4)
        r["share_any_fold"] = round((anyin | f.l1tone | f.l2tone | f.l1fold | f.l2fold).mean(), 4)
        r["share_has2"] = round(f.has2.mean(), 4)
        r["share_eq12"] = round(f.eq12.mean(), 4)
        # GOLD thành phần
        G = D[gold]; fg = f[gold]
        r["gold_rules"] = G.rule.str.split(":").str[0].value_counts().to_dict()
        r["gold_l1in"] = round(fg.l1in.mean(), 4)
        r["gold_eq12_given_has2"] = round(fg.eq12[fg.has2].mean(), 4)
        r["gold_label_eq_l1"] = round(float(np.mean([bool(a) and a == c for a, c in zip(G.label, G.ocr_char)])), 4)
        # ô KHÔNG GOLD, không notplaus/quarantine
        ng = ~gold & ~f.grp.isin(["notplaus", "quarantine"]).values
        sub = f[ng]
        cat = np.select(
            [sub.l1in & sub.l2in, sub.l1in & ~sub.l2in, ~sub.l1in & sub.l2in,
             ~sub.l1in & ~sub.l2in & (sub.l1tone | sub.l2tone),
             ~sub.l1in & ~sub.l2in & ~(sub.l1tone | sub.l2tone) & (sub.l1fold | sub.l2fold),
             ~sub.l1in & ~sub.l2in & sub.eq12 & ~sub.syl_known,
             ~sub.l1in & ~sub.l2in & sub.eq12,
             ~sub.l1in & ~sub.l2in & ~sub.has2,
             ],
            ["l1in_l2in", "l1in_only", "l2in_only", "tone_fuzzy", "fold_fuzzy", "eq12_syl_unknown", "eq12_notR",
             "neither_no_lt2"], "neither_diff")
        r["nongold_n"] = int(ng.sum())
        r["nongold_cat"] = pd.Series(cat).value_counts().to_dict()
        r["nongold_cat_by_grp"] = pd.crosstab(pd.Series(cat, name="cat"), sub.grp.values).to_dict()
        # âm QN 'lạ' (không có trong từ điển) trong ô không GOLD
        r["nongold_syl_unknown"] = int((~sub.syl_known).sum())
        out[b] = r
        print(f"[t01] {b}: N {N} GOLD {r['gold_share']} | l1∈R {r['share_l1in']} l2∈R {r['share_l2in']} any {r['share_any_in']}"
              f" +thanh {r['share_any_tone']} +dấu {r['share_any_fold']} | có lt2 {r['share_has2']} lt1==lt2 {r['share_eq12']}",
              flush=True)
        print(f"      GOLD: l1∈R {r['gold_l1in']} lt1==lt2|có lt2 {r['gold_eq12_given_has2']} rules {r['gold_rules']}", flush=True)
        print(f"      không GOLD {r['nongold_n']}: {r['nongold_cat']} | âm lạ {r['nongold_syl_unknown']}", flush=True)
    T.jdump(out, T.OUT / "stt_census.json")


if __name__ == "__main__":
    main()
