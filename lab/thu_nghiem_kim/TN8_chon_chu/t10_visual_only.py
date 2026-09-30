"""t10_visual_only.py — bộ chọn CHỈ ẢNH (+ tiên nghiệm âm, KHÔNG đặc trưng kim/lt2) để KIỂM CHÉO chuyển giao sang STT.

Học (tầng 1) trên Borg có sự thật; đo top-1:
  Borg LOBO (B34 -> B18, B18 -> B34): độ đúng thật theo nhóm ô;
  STT (mô hình học trên B18+B34): tỉ lệ top-1 == nhãn trên ô "chắc" (GOLD, lt1 == lt2 V1+, cả hai ∈ R(âm)) — hai lần đọc kim
  độc lập đồng ý nên nhãn gần như đúng -> tỉ lệ này ≈ độ đúng top-1 của bộ chọn chỉ-ảnh trên chữ viết tay STT.
So hai con số trên CÙNG loại ô (GOLD, kim ∈ R) cho biết bộ chọn ảnh có chuyển được sang tay chép STT không. 0 API.
Ra: measure_out/_tn8/visual_only.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import tn8model as M  # noqa: E402
import t06_eval as E  # noqa: E402

KIM_COLS = ("is_kim", "kim_inR", "is_lt2")


def strip(Bk):
    keep = [j for j, n in enumerate(Bk["names"]) if n not in KIM_COLS]
    return Bk["Xc"][:, keep]


def fit(train, max_cells=60000, seed=0):
    rng = np.random.default_rng(seed)
    Xs, Is, Ys = [], [], []
    off = 0
    for Bk in train:
        D, F = Bk["D"], Bk["F"]
        anyy = F.groupby("i").y.max().reindex(np.arange(len(D))).fillna(0).values > 0
        nc = F.groupby("i").size().reindex(np.arange(len(D))).fillna(0).values
        cells = np.nonzero(anyy & (nc >= 2) & (D.gt_char.values != ""))[0]
        if len(cells) > max_cells:
            cells = np.sort(rng.choice(cells, max_cells, replace=False))
        sel = np.isin(F.i.values, cells)
        Xs.append(strip(Bk)[sel]); Is.append(F.i.values[sel] + off); Ys.append(F.y.values[sel])
        off += len(D) + 1
    return M.CLogit().fit(np.concatenate(Xs), np.concatenate(Is), np.concatenate(Ys))


def top1(Bk, m):
    F = Bk["F"]
    p = m.predict(strip(Bk), F.i.values)
    G = pd.DataFrame({"i": F.i.values, "c": F.c.values, "p": p, "y": F.y.values}).sort_values(["i", "p"], ascending=[True, False])
    return G.groupby("i").head(1).set_index("i")


def main():
    C = T.lex()
    res = {}
    B = {b: E.load_book(b) for b in ("B18", "B34")}
    for te, tr in (("B18", "B34"), ("B34", "B18")):
        m = fit([B[tr]])
        t = top1(B[te], m)
        D = B[te]["D"]
        grp = B[te]["grp"]
        ok = D.gt_char.values != ""
        r = {}
        for g in ("GOLD", "direct_qn", "syl", "nocontext"):
            idx = np.nonzero(ok & (grp == g))[0]
            idx = idx[np.isin(idx, t.index.values)]
            r[g] = [round(float(t.y.reindex(idx).mean()), 4), int(len(idx))]
        # ô "chắc" kiểu STT: GOLD có nhãn đúng? (không có lt2 ở Borg) -> dùng GOLD mà kim đúng theo người
        gi = np.nonzero(ok & (grp == "GOLD"))[0]
        lab_ok = np.array([C.var_eq_plus(l, g_) for l, g_ in zip(D.label.values[gi], D.gt_char.values[gi])])
        gi = gi[lab_ok]
        gi = gi[np.isin(gi, t.index.values)]
        r["GOLD_nhan_dung__top1_eq_label"] = [round(float(np.mean([C.var_eq_plus(a, l) for a, l in
                                                                   zip(t.c.reindex(gi).values, D.label.values[gi])])), 4), int(len(gi))]
        res[f"{te}<-{tr}"] = r
        print(f"[t10] {te} <- {tr}: top-1 chỉ-ảnh đúng theo nhóm {r}", flush=True)
    m = fit([B["B18"], B["B34"]])
    for b in ("stt2", "stt4", "stt11"):
        f = T.OUT / "cand" / f"{b}.pkl"
        if not f.exists():
            continue
        Bk = E.load_book(b)
        t = top1(Bk, m)
        D = Bk["D"]
        L2 = pd.read_pickle(T.OUT / "stt_lt2" / f"{b}.pkl")
        l2 = L2.lt2.fillna("").values
        R = {s: C.R_of(s) for s in set(D.syllable)}
        sure = np.array([(tier == "GOLD") and bool(lab) and bool(x) and lab in R[s] and C.var_eq_plus(lab, x)
                         for tier, lab, x, s in zip(D.tier.values, D.label.values, l2, D.syllable.values)])
        idx = np.nonzero(sure)[0]; idx = idx[np.isin(idx, t.index.values)]
        acc_sure = float(np.mean([C.var_eq_plus(a, l) for a, l in zip(t.c.reindex(idx).values, D.label.values[idx])]))
        # ô lt1 ∉ R nhưng lt2 ∈ R: top-1 chỉ-ảnh == lt2 ?
        grp = Bk["grp"]
        l1in = np.array([bool(k) and k in R[s] for k, s in zip(D.ocr_char.values, D.syllable.values)])
        l2in = np.array([bool(x) and x in R[s] for x, s in zip(l2, D.syllable.values)])
        j = np.nonzero(~l1in & l2in & (D.tier.values != "GOLD"))[0]; j = j[np.isin(j, t.index.values)]
        agree_l2 = float(np.mean([C.var_eq_plus(a, x) for a, x in zip(t.c.reindex(j).values, l2[j])])) if len(j) else np.nan
        k = np.nonzero(l1in & l2in & (D.tier.values == "GOLD") & ~sure)[0]; k = k[np.isin(k, t.index.values)]
        top_l1 = float(np.mean([C.var_eq_plus(a, x) for a, x in zip(t.c.reindex(k).values, D.label.values[k])])) if len(k) else np.nan
        top_l2 = float(np.mean([C.var_eq_plus(a, x) for a, x in zip(t.c.reindex(k).values, l2[k])])) if len(k) else np.nan
        res[b] = dict(sure_top1_eq_label=[round(acc_sure, 4), int(len(idx))],
                      new_l2only_top1_eq_lt2=[round(agree_l2, 4), int(len(j))],
                      gold_conflict_top1_eq_lt1=[round(top_l1, 4), int(len(k))],
                      gold_conflict_top1_eq_lt2=[round(top_l2, 4), int(len(k))])
        print(f"[t10] {b}: {res[b]}", flush=True)
    T.jdump(res, T.OUT / "visual_only.json")


if __name__ == "__main__":
    main()
