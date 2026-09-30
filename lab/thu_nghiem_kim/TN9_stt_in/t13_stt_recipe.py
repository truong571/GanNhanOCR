"""t13_stt_recipe.py — STT: dựng các CÔNG THỨC nhãn từ hai bản dựng hộp cát (A = lt1__legacy, B = l1skel_l2__legacy) + bản
giao nộp (prod, có rescue) và ƯỚC LƯỢNG độ đúng theo lớp bằng chứng. 0 API. [SĐ]

Lớp bằng chứng của một ô GOLD (theo nhãn chọn, lt1 = chữ kim Hán của ô, lt2 = chữ kim Nôm ánh xạ về ô (TN8 t03)):
  agree      nhãn ≡ lt1 ≡ lt2 ∈ R(âm)                       t08: 95,9–98,7 %
  lt2only    nhãn ≡ lt2 ∈ R, lt1 ∉ R                         t08: 83,9–93,8 %
  lt1_nolt2  nhãn ≡ lt1 ∈ R, KHÔNG có lt2 ở ô                 t05 (một lần đọc, q Borg): 87,3–93,3 %
  lt1_l2out  nhãn ≡ lt1 ∈ R, lt2 có mà ∉ R                   t08: 59,4–73,4 %
  conflict   lt1, lt2 cùng ∈ R, khác nhau (nhãn = một trong hai)  t08: nhãn lt2 59,7–81,9 %, nhãn lt1 11,7–28,9 %
  similar / rescue / other (luật khác, nhãn ≠ cả hai lần đọc)  Borg similar 59,6–62,8 % (đo); rescue: không đo được —
             lấy dải 40–70 % (đồng thuận lt2 chỉ 40–48 % so với 89–91 % ở direct) [SĐ]
Vị trí: kim_geo (t06: tâm crop ∈ hộp chữ kim đã ghép) — ô ≠ coi là SAI vị trí (bi quan) / bỏ qua (lạc quan).
Công thức:
  R0_prod       bản giao nộp (lt1, legacy, + rescue)
  R0_lt1        A (trước rescue)
  R1_l2         B (chữ lt2 trên khung lt1)
  R2_union      A ∪ B theo ô; xung đột lấy nhãn B (lt2)
  R3_union_resc R2 + ô rescue của prod chưa GOLD
  R4_prec       R2 nhưng BỎ lt1_l2out, xung đột, similar/rescue/other (chỉ giữ agree, lt2only, lt1_nolt2)
  R5_agree      chỉ agree (hai lần đọc đồng ý)
Ra measure_out/_tn9/stt_recipes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402

B_ = T.OUT / "stt_builds"
PREC = {  # (thấp, cao) theo lớp — nguồn: t08_latent2 / t05_inR_model / Borg; cập nhật theo sách ở main()
    "agree": None, "lt2only": None, "lt1_nolt2": None, "lt1_l2out": None, "conflict_l2": None, "conflict_l1": None,
    "similar": (0.596, 0.628), "rescue": (0.40, 0.70), "other": (0.60, 0.90)}


def load(name):
    p = (T.REPO / "dataset_out/labels_final.csv") if name == "prod" else (B_ / name / "dataset_out/labels_final.csv")
    L = T.rd(p)
    L = L[L.book.isin(T.STT)].copy()
    L["key"] = L.book + "/" + L.page + "/" + L.column + "/" + L.syl_idx
    return L.drop_duplicates("key").set_index("key")


def main():
    C = T.lex()
    lat = json.loads((T.OUT / "latent2.json").read_text())
    inr = json.loads((T.OUT / "inR_model.json").read_text())
    geoA = pd.read_pickle(T.OUT / "kimgeo" / "lt1__legacy.pkl")
    geoB = pd.read_pickle(T.OUT / "kimgeo" / "l1skel_l2__legacy.pkl")
    geoP = pd.read_pickle(T.OUT / "kimgeo" / "prod.pkl")
    gk = lambda G: dict(zip(G.book + "/" + G.page + "/" + G.column.astype(str) + "/" + G.syl_idx.astype(str),  # noqa: E731
                            [bool(a) and a == o for a, o in zip(G.geo_own if "geo_own" in G else G.geo1, G.ocr_char)]))
    GA, GB, GP = gk(geoA), gk(geoB), gk(geoP)
    A, Bb, P = load("lt1__legacy"), load("l1skel_l2__legacy"), load("prod")
    EXTRA = {}
    for nm in ("twopass__legacy", "lt2__legacy"):
        if (B_ / nm / "dataset_out/labels_final.csv").exists() and (T.OUT / "kimgeo" / f"{nm}.pkl").exists():
            EXTRA[nm] = (load(nm), gk(pd.read_pickle(T.OUT / "kimgeo" / f"{nm}.pkl")))
    out = {}
    for bk in T.STT:
        rng = lat[bk]["range"]
        prec = dict(PREC)
        prec["agree"] = tuple(rng["agree_inR (lt1≡lt2 ∈ R)"])
        prec["lt2only"] = tuple(rng["lt2∈R, lt1∉R: lt2 đúng"])
        prec["lt1_l2out"] = tuple(rng["lt1∈R, lt2∉R: lt1 đúng"])
        prec["conflict_l2"] = tuple(rng["conflict: lt2 đúng"])
        prec["conflict_l1"] = tuple(rng["conflict: lt1 đúng"])
        e = inr["stt"][bk]["lt1"]["est"]
        prec["lt1_nolt2"] = (min(v["prec_inR"] for k, v in e.items() if k in ("q=0.089", "q=0.145", "q=0.117")),
                             max(v["prec_inR"] for k, v in e.items() if k in ("q=0.089", "q=0.145", "q=0.117")))
        D = T.load_stt(bk)                       # TN8 base (== A) + lt2 ánh xạ
        D["key"] = bk + "/" + D.page + "/" + D.column.astype(str) + "/" + D.syl_idx.astype(str)
        D = D.drop_duplicates("key").set_index("key")
        keys = A.index[A.book == bk].union(Bb.index[Bb.book == bk])
        for nm, (X, _) in EXTRA.items():
            keys = keys.union(X.index[X.book == bk])
        l1 = D.ocr_char.reindex(keys).fillna("").values
        l2 = D.lt2.reindex(keys).fillna("").values
        # ô chỉ có ở B: lấy chữ B làm lt2 (xấp xỉ), lt1 rỗng
        onlyB = ~keys.isin(D.index)
        l2 = np.where(onlyB, Bb.ocr_char.reindex(keys).fillna("").values, l2)
        syl = pd.Series(A.syllable.reindex(keys).values, index=keys).fillna(pd.Series(Bb.syllable.reindex(keys).values, index=keys))
        for nm, (X, _) in EXTRA.items():
            syl = syl.fillna(pd.Series(X.syllable.reindex(keys).values, index=keys))
        syl = syl.fillna("").values
        l2 = np.where(~keys.isin(D.index) & (l2 == ""), next(iter([X.ocr_char.reindex(keys).fillna("").values for X, _ in EXTRA.values()]), l2), l2)
        R = {s: C.R_of(s) for s in set(syl)}
        tA = A.tier.reindex(keys).fillna("").values; tB = Bb.tier.reindex(keys).fillna("").values
        tP = P.tier.reindex(keys).fillna("").values
        labA = A.label.reindex(keys).fillna("").values; labB = Bb.label.reindex(keys).fillna("").values
        labP = P.label.reindex(keys).fillna("").values
        ruleP = P.rule.reindex(keys).fillna("").values
        ruleA = A.rule.reindex(keys).fillna("").values; ruleB = Bb.rule.reindex(keys).fillna("").values
        N = len(keys)
        veq = C.var_eq_plus

        def cls_of(lab, rule):
            out = []
            for x, y, s, lb, ru in zip(l1, l2, syl, lab, rule):
                x_in, y_in = bool(x) and x in R[s], bool(y) and y in R[s]
                if ru.startswith("self_training_rescue"):
                    c = "rescue"
                elif x and y and veq(x, y) and x_in and veq(lb, x):
                    c = "agree"
                elif x_in and y_in and not veq(x, y):
                    c = "conflict_l2" if veq(lb, y) else ("conflict_l1" if veq(lb, x) else "other")
                elif y_in and not x_in and veq(lb, y):
                    c = "lt2only"
                elif x_in and not y and veq(lb, x):
                    c = "lt1_nolt2"
                elif x_in and y and not y_in and veq(lb, x):
                    c = "lt1_l2out"
                elif ru.startswith("s1_inter_s2_similar"):
                    c = "similar"
                else:
                    c = "other"
                out.append(c)
            return np.array(out, dtype=object)

        rec = {}
        gold_A, gold_B, gold_P = tA == "GOLD", tB == "GOLD", tP == "GOLD"
        # nhãn union: B nếu B GOLD (xung đột -> B), ngược lại A
        lab_U = np.where(gold_B, labB, labA)
        rule_U = np.where(gold_B, ruleB, ruleA)
        geo_U = np.array([GB.get(k, False) if g else GA.get(k, False) for k, g in zip(keys, gold_B)])
        cls_U = cls_of(lab_U, rule_U)
        cls_P = cls_of(labP, ruleP); cls_A = cls_of(labA, ruleA); cls_B = cls_of(labB, ruleB)
        geo_P = np.array([GP.get(k, False) for k in keys]); geo_A = np.array([GA.get(k, False) for k in keys])
        geo_B = np.array([GB.get(k, False) for k in keys])
        resc = np.array([r.startswith("self_training_rescue") for r in ruleP])
        recipes = {
            "R0_prod": (gold_P, cls_P, geo_P),
            "R0_lt1": (gold_A, cls_A, geo_A),
            "R1_l2": (gold_B, cls_B, geo_B),
            "R2_union": (gold_A | gold_B, cls_U, geo_U),
            "R3_union_resc": (gold_A | gold_B | (gold_P & resc), np.where(gold_A | gold_B, cls_U, cls_P),
                              np.where(gold_A | gold_B, geo_U, geo_P)),
            "R4_prec": ((gold_A | gold_B) & np.isin(cls_U, ["agree", "lt2only", "lt1_nolt2"]), cls_U, geo_U),
            "R5_agree": ((gold_A | gold_B) & (cls_U == "agree"), cls_U, geo_U),
        }
        for nm, (X, GX) in EXTRA.items():
            tX = X.tier.reindex(keys).fillna("").values; labX = X.label.reindex(keys).fillna("").values
            ruleX = X.rule.reindex(keys).fillna("").values
            gold_X = tX == "GOLD"
            cls_X = cls_of(labX, ruleX)
            geo_X = np.array([GX.get(k, False) for k in keys])
            lab_UX = np.where(gold_X, labX, labA); rule_UX = np.where(gold_X, ruleX, ruleA)
            cls_UX = cls_of(lab_UX, rule_UX)
            geo_UX = np.array([GX.get(k, False) if g else GA.get(k, False) for k, g in zip(keys, gold_X)])
            tag = nm.split("__")[0]
            recipes[f"R1_{tag}"] = (gold_X, cls_X, geo_X)
            recipes[f"R2_union_{tag}"] = (gold_A | gold_X, cls_UX, geo_UX)
            recipes[f"R4_prec_{tag}"] = ((gold_A | gold_X) & np.isin(cls_UX, ["agree", "lt2only", "lt1_nolt2"]), cls_UX, geo_UX)
        for nm, (g, cl, geo) in recipes.items():
            comp = pd.Series(cl[g]).value_counts().to_dict()
            lo = sum(n * prec[c][0] for c, n in comp.items()) / max(1, g.sum())
            hi = sum(n * prec[c][1] for c, n in comp.items()) / max(1, g.sum())
            pos = float(geo[g].mean()) if g.sum() else float("nan")
            rec[nm] = dict(gold=int(g.sum()), share=round(float(g.mean()), 4), comp=comp,
                           prec_text=[round(lo, 4), round(hi, 4)], geo_ok=round(pos, 4),
                           prec_both_pess=[round(lo * pos, 4), round(hi * pos, 4)])
        out[bk] = dict(N=int(N), prec_by_class={k: [round(v[0], 4), round(v[1], 4)] for k, v in prec.items()}, recipes=rec)
        print(f"== {bk}: N {N}", flush=True)
        for nm, v in rec.items():
            print(f"   {nm:14s} GOLD {v['gold']:6d} ({v['share']:.3f}) chữ~{v['prec_text']} geo {v['geo_ok']} hai-vế(bi quan)~{v['prec_both_pess']}",
                  flush=True)
    T.jdump(out, T.OUT / "stt_recipes.json")


if __name__ == "__main__":
    main()
