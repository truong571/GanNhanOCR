"""t05_inR_model.py — ước lượng độ đúng nhãn "kim ∈ R(âm)" của STT từ tỉ lệ ∈ R, hiệu chuẩn trên Borg (có người). 0 API.

Mô hình (một lần đọc kim, ô có chữ kim):
   r = P(kim ∈ R) = a·s + (1 − a)·q      a = P(kim đúng), s = P(chữ đúng ∈ R | kim đúng), q = P(kim ∈ R | kim sai)
   P(đúng | ∈ R) = a·s / r
Borg (kim = lt2, chữ người) cho a, s, q, r và P(đúng | ∈ R) ĐO được -> kiểm mô hình; STT chỉ biết r (lt1, lt2, lt1==lt2) ->
suy a và P(đúng | ∈ R) với (s, q) mượn từ Borg (GIẢ ĐỊNH chuyển giao; q là tham số nhạy — báo cả dải q của hai sách Borg).
Thêm: theo khối Unicode của chữ người (ExtB+ = chữ Nôm riêng) — kim đọc sai chữ Nôm riêng thành chữ Hán thành phần/âm mượn
(∈ R) là nguồn sai hệ thống. Ra measure_out/_tn9/inR_model.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402


def blk(c):
    if not c:
        return "none"
    o = ord(c[0])
    if 0xE000 <= o <= 0xF8FF or o >= 0xF0000:
        return "PUA"
    if o >= 0x20000:
        return "ExtB+"
    if 0x3400 <= o <= 0x4DBF:
        return "ExtA"
    return "URO" if 0x4E00 <= o <= 0x9FFF else "other"


def main():
    C = T.lex()
    out = {"borg": {}, "stt": {}}
    for b in ("B18", "B34"):
        D = pd.read_pickle(T.T8 / "base" / f"{b}.pkl")
        D = D[(D.gt_char != "") & (D.ocr_char != "")]
        R = {s: C.R_of(s) for s in set(D.syllable)}
        k, g, s = D.ocr_char.values, D.gt_char.values, D.syllable.values
        right = np.array([C.var_eq_plus(x, y) for x, y in zip(k, g)])
        inR = np.array([x in R[y] for x, y in zip(k, s)])
        gtin = np.array([C.var_in(x, R[y]) for x, y in zip(g, s)])
        a = right.mean(); r = inR.mean()
        sv = inR[right].mean(); q = inR[~right].mean()
        prec = right[inR].mean()
        gb = np.array([blk(x) for x in g])
        byb = {}
        for bb in ("URO", "ExtA", "ExtB+"):
            m = gb == bb
            if m.sum():
                byb[bb] = dict(n=int(m.sum()), kim_right=round(float(right[m].mean()), 4),
                               kim_inR=round(float(inR[m].mean()), 4),
                               wrong_inR_share_of_errors_inR=round(float((m & ~right & inR).sum() / max(1, (~right & inR).sum())), 4))
        out["borg"][b] = dict(n=int(len(D)), a=round(a, 4), r=round(r, 4), s=round(sv, 4), q=round(q, 4),
                              prec_inR_measured=round(float(prec), 4), prec_inR_model=round(float(a * sv / r), 4),
                              gt_inR=round(float(gtin.mean()), 4), by_truth_block=byb)
        print(f"[t05] {b}: {out['borg'][b]}", flush=True)
    s_b = np.mean([out["borg"][b]["s"] for b in out["borg"]])
    qs = [out["borg"][b]["q"] for b in out["borg"]]
    for b in T.STT:
        D = T.load_stt(b)
        R = {s: C.R_of(s) for s in set(D.syllable)}
        res = {}
        for nm, col in (("lt1", D.ocr_char.values), ("lt2", D.lt2.values)):
            has = np.array([bool(x) for x in col])
            inR = np.array([bool(x) and x in R[y] for x, y in zip(col, D.syllable.values)])
            r = inR[has].mean()
            est = {}
            for q in sorted(set([min(qs), float(np.mean(qs)), max(qs), 0.25, 0.30])):
                a = (r - q) / (s_b - q)
                est[f"q={q:.3f}"] = dict(a=round(float(a), 4), prec_inR=round(float(a * s_b / r), 4))
            res[nm] = dict(n_has=int(has.sum()), r=round(float(r), 4), est=est)
        # hai lần đọc đồng ý
        eq = np.array([bool(x) and bool(y) and C.var_eq_plus(x, y) for x, y in zip(D.ocr_char.values, D.lt2.values)])
        both = D.lt2.values != ""
        res["eq12_given_both"] = round(float(eq[both].mean()), 4)
        out["stt"][b] = res
        print(f"[t05] {b}: {res}", flush=True)
    T.jdump(out, T.OUT / "inR_model.json")


if __name__ == "__main__":
    main()
