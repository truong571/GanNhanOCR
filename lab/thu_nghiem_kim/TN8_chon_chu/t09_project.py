"""t09_project.py — DỰ PHÓNG cho bộ không có sự thật (STT, Chr, L83, KVK): tỉ lệ GOLD mới + độ đúng ƯỚC LƯỢNG / SUY ĐOÁN.

Ngưỡng: tiêu chí biên (t07.pick_tau) trên dự đoán NGOÀI-KHỐI của hai bộ có sự thật cùng họ (models/<họ>.pkl), đích biên
0,80 và 0,90 — giống hệt quy tắc đã kiểm LOBO trên B18/B34/L16/TK.
Độ đúng ô MỚI (không phải số đo):
  mean_P      trung bình P tầng 2 của ô được nhận (P đã kiểm hiệu chuẩn trên bộ thử LOBO — t06/q_calib);
  analog      độ đúng ĐO ĐƯỢC trên bộ thử LOBO cùng họ ở cùng ngưỡng, theo phần (in/out), trộn theo tỉ lệ phần của bộ này;
  lt2_agree   (STT, chế độ lt1) tỉ lệ top-1 == chữ lt2 ở ô mới có lt2 ∈ R — kiểm chéo độc lập một phần (lt2 không vào mô hình);
  ref_agree   (L83/KVK) tỉ lệ top-1 == chữ dị bản ở ô mới, so với cùng tỉ lệ của GOLD hiện tại (nền dị bản).
Ra: measure_out/_tn8/project.json
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t07_policy as P  # noqa: E402

LOBO_PAIRS = {"hand": [("B18", "B34"), ("B34", "B18")], "print": [("L16", "TK"), ("TK", "L16")]}


def analog(fam, ti, to):
    """Độ đúng đo được (hai vế nếu đủ, không thì chữ) của ô mới theo phần trên bộ thử LOBO cùng họ ở ngưỡng (ti, to)."""
    out = {}
    for part, tau in (("in", ti), ("out", to)):
        k = n = 0
        for te, tr in LOBO_PAIRS[fam]:
            o = pd.read_pickle(T.OUT / "pred" / f"{te}__from_{tr}.pkl")
            sel = (o.tier.values != "GOLD") & (o.part.values == part) & (np.nan_to_num(o.P.values, nan=-1) >= tau)
            has = sel & (o.gtc.values != "")
            y = o.y1.fillna(0).values.astype(bool)
            if te in ("L16", "TK"):
                y = y & o.pos_ok.values
            k += int(y[has].sum()); n += int(has.sum())
        out[part] = (k / n if n else np.nan, n)
    return out


def project(b: str, fam: str, mode: str, target: float) -> dict:
    M = pickle.load(open(T.OUT / "models" / f"{fam}.pkl", "rb"))
    oof = M["oof"]
    ti, to = P.pick_tau(oof, "in", target), P.pick_tau(oof, "out", target)
    o = pd.read_pickle(T.OUT / "pred" / f"{b}__from_{fam}__{mode}.pkl")
    gold = o.tier.values == "GOLD"
    acc = P.accept(o, ti, to)
    N = len(o)
    an = analog(fam, ti, to)
    new_in, new_out = acc & (o.part.values == "in"), acc & (o.part.values == "out")
    mp = float(np.nanmean(o.P.values[acc])) if acc.sum() else np.nan
    mix = (an["in"][0] * new_in.sum() + (an["out"][0] if an["out"][1] else 0) * new_out.sum()) / max(1, acc.sum())
    r = dict(book=b, fam=fam, mode=mode, target=target, tau_in=ti, tau_out=to, N=N, gold_now=int(gold.sum()),
             share_now=round(gold.mean(), 4), n_new=int(acc.sum()), n_new_in=int(new_in.sum()), n_new_out=int(new_out.sum()),
             share_new=round((gold.sum() + acc.sum()) / N, 4), new_mean_P=round(mp, 4), new_analog=round(float(mix), 4),
             analog_by_part={k: [round(v[0], 4) if v[0] == v[0] else None, v[1]] for k, v in an.items()},
             groups={g: dict(n=int((o.grp.values == g).sum()), nhan=int((acc & (o.grp.values == g)).sum()))
                     for g in sorted(set(o.grp.values))})
    if b.startswith("stt"):
        L2 = pd.read_pickle(T.OUT / "stt_lt2" / f"{b}.pkl")
        C = T.lex()
        D = T.load_base(b)
        R = {s: C.R_of(s) for s in set(D.syllable)}
        l2 = L2.lt2.fillna("").values
        l2in = np.array([bool(x) and x in R[s] for x, s in zip(l2, D.syllable.values)])
        top = o.top1.fillna("").values
        m = acc & l2in
        r["lt2_agree_new"] = [round(float(np.mean([C.var_eq_plus(a, x) for a, x in zip(top[m], l2[m])])), 4) if m.sum() else None,
                              int(m.sum())]
        g2 = gold & l2in & (D.label.values != "")
        r["lt2_agree_gold"] = [round(float(np.mean([C.var_eq_plus(a, x) for a, x in zip(D.label.values[g2], l2[g2])])), 4),
                               int(g2.sum())]
    if b in ("L83", "KVK"):
        hasref = o.ref.values != ""
        m = acc & hasref
        r["ref_agree_new"] = [round(float(o.yref1.values[m].mean()), 4) if m.sum() else None, int(m.sum())]
        C = T.lex()
        g = gold & hasref
        yg = [any(C.var_eq_plus(l, x) for x in rf.split("|")) for l, rf in zip(o.label.values[g], o.ref.values[g])]
        r["ref_agree_gold"] = [round(float(np.mean(yg)), 4), int(g.sum())]
    return r


def main():
    jobs = [a.split(":") for a in (sys.argv[1:] or ["stt2:kim", "stt2:lt2", "stt2:union", "stt4:kim", "stt4:lt2", "stt4:union",
                                                    "stt11:kim", "stt11:lt2", "stt11:union", "Chr:kim", "L83:kim", "KVK:kim"])]
    res = []
    for b, mode in jobs:
        fam = "hand" if b.startswith("stt") else "print"
        for target in (0.80, 0.90):
            r = project(b, fam, mode, target)
            res.append(r)
            ex = ""
            if "lt2_agree_new" in r:
                ex = f" | lt2 trùng top1 ô mới {r['lt2_agree_new']} (GOLD {r['lt2_agree_gold']})"
            if "ref_agree_new" in r:
                ex = f" | dị bản trùng ô mới {r['ref_agree_new']} (GOLD {r['ref_agree_gold']})"
            print(f"[t09] {b:5s} {mode:5s} biên {target}: GOLD {r['share_now']:.3f} -> {r['share_new']:.3f} (+{r['n_new']}: in "
                  f"{r['n_new_in']}, out {r['n_new_out']}); ô mới mean_P {r['new_mean_P']}, analog {r['new_analog']}{ex}",
                  flush=True)
    T.jdump(res, T.OUT / "project.json")


if __name__ == "__main__":
    main()
