"""t03_compare_builds.py — so các bản dựng STT trong hộp cát (t02) với nhau và với bản giao nộp dataset_out/. 0 API.

1) KIỂM tái lập: lt1__legacy/labels.csv phải trùng dataset_out/labels.csv (phần STT) trên các cột khoá (bảo đảm hộp cát
   == đường sản xuất; chênh = mã/đầu vào khác).
2) Mỗi bản dựng: số ô, GOLD, tỉ lệ GOLD / ô, / âm QN (mẫu số bất biến = tổng n_qn các cột có ô), cặp khoá (trang, cột, syl_idx).
3) Hợp theo ô (khoá trang+cột+syl_idx) giữa hai bản dựng: ô GOLD ở bản nào thì lấy bản đó; hai bản GOLD khác nhãn = xung đột.
Ra measure_out/_tn9/stt_builds/compare.json.
  .venv/bin/python lab/thu_nghiem_kim/TN9_stt_in/t03_compare_builds.py lt1__legacy l1skel_l2__legacy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402

B = T.OUT / "stt_builds"
KEYC = ["book", "page", "column", "nom_idx", "syl_idx", "ocr_char", "syllable", "label", "tier", "rule", "bbox"]


def load(name: str, stage: str = "labels_final.csv") -> pd.DataFrame:
    p = (T.REPO / "dataset_out" / stage) if name == "prod" else (B / name / "dataset_out" / stage)
    L = T.rd(p)
    L["key"] = L.book + "/" + L.page + "/" + L.column + "/" + L.syl_idx
    return L


def iou(a, b):
    try:
        a, b = json.loads(a), json.loads(b)
    except Exception:  # noqa: BLE001
        return np.nan
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I)


def stats(L: pd.DataFrame) -> dict:
    out = {}
    for bk, g in L.groupby("book"):
        cols = g.groupby(["page", "column"]).n_qn.first()
        nq = pd.to_numeric(cols, errors="coerce").fillna(0).sum()
        gold = (g.tier == "GOLD")
        out[bk] = dict(n=int(len(g)), gold=int(gold.sum()), share=round(float(gold.mean()), 4), n_qn=int(nq),
                       share_of_qn=round(float(gold.sum() / max(1, nq)), 4), tiers=g.tier.value_counts().to_dict(),
                       rules=g.rule.str.split(":").str[0].value_counts().head(8).to_dict())
    return out


def main():
    names = sys.argv[1:] or ["lt1__legacy"]
    res = {}
    # 1) tái lập
    if "lt1__legacy" in names and (B / "lt1__legacy/dataset_out/labels.csv").exists():
        P = load("prod", "labels.csv"); S = load("lt1__legacy", "labels.csv")
        P = P[P.book.isin(T.STT)].reset_index(drop=True)
        same_n = len(P) == len(S)
        eq = {}
        if same_n:
            for c in KEYC:
                eq[c] = round(float((P[c].values == S[c].values).mean()), 6)
        res["repro_labels_csv"] = dict(n_prod=len(P), n_sandbox=len(S), col_eq=eq)
        print(f"[t03] tái lập labels.csv: prod {len(P)} vs hộp cát {len(S)}; trùng cột {eq}", flush=True)
        Pf = load("prod", "labels_remediated.csv"); Sf = load("lt1__legacy", "labels_remediated.csv")
        Pf = Pf[Pf.book.isin(T.STT)].reset_index(drop=True)
        if len(Pf) == len(Sf):
            res["repro_remediated"] = {c: round(float((Pf[c].values == Sf[c].values).mean()), 6) for c in KEYC}
            print(f"[t03] tái lập labels_remediated: {res['repro_remediated']}", flush=True)
    # 2) thống kê
    Ls = {}
    for nm in names:
        if not (B / nm / "dataset_out/labels_final.csv").exists():
            print(f"[t03] thiếu {nm}"); continue
        Ls[nm] = load(nm)
        res[nm] = stats(Ls[nm])
        for bk, v in res[nm].items():
            print(f"[t03] {nm:24s} {bk:6s} ô {v['n']:6d} GOLD {v['gold']:6d} ({v['share']:.4f}) /âmQN {v['share_of_qn']:.4f} "
                  f"tầng {v['tiers']}", flush=True)
    # 3) hợp theo ô
    if len(Ls) >= 2:
        a, b = list(Ls)[:2]
        A, Bb = Ls[a].set_index("key"), Ls[b].set_index("key")
        A = A[~A.index.duplicated()]; Bb = Bb[~Bb.index.duplicated()]
        keys = A.index.union(Bb.index)
        J = pd.DataFrame(index=keys)
        for nm, X in ((a, A), (b, Bb)):
            J[f"tier_{nm}"] = X.tier.reindex(keys).fillna("")
            J[f"lab_{nm}"] = X.label.reindex(keys).fillna("")
            J[f"bbox_{nm}"] = X.bbox.reindex(keys).fillna("")
        J["book"] = [k.split("/")[0] for k in keys]
        ga, gb = J[f"tier_{a}"] == "GOLD", J[f"tier_{b}"] == "GOLD"
        both = ga & gb
        J["same_lab"] = J[f"lab_{a}"] == J[f"lab_{b}"]
        J["iou"] = [iou(x, y) if x and y else np.nan for x, y in zip(J[f"bbox_{a}"], J[f"bbox_{b}"])]
        u = {}
        for bk, g in J.groupby("book"):
            gA, gB = g[f"tier_{a}"] == "GOLD", g[f"tier_{b}"] == "GOLD"
            bo = gA & gB
            u[bk] = dict(keys=int(len(g)), in_a=int((g[f"tier_{a}"] != "").sum()), in_b=int((g[f"tier_{b}"] != "").sum()),
                         gold_a=int(gA.sum()), gold_b=int(gB.sum()), gold_both=int(bo.sum()),
                         both_same_label=int((bo & g.same_lab).sum()), both_conflict=int((bo & ~g.same_lab).sum()),
                         only_a=int((gA & ~gB).sum()), only_b=int((gB & ~gA).sum()),
                         union=int((gA | gB).sum()), union_noconf=int(((gA | gB) & ~(bo & ~g.same_lab)).sum()),
                         share_union_over_keys=round(float((gA | gB).mean()), 4),
                         bbox_iou_ge05_on_common=round(float((g.iou.dropna() >= 0.5).mean()), 4),
                         bbox_identical_on_common=round(float((g[f"bbox_{a}"] == g[f"bbox_{b}"])[(g[f"bbox_{a}"] != "") & (g[f"bbox_{b}"] != "")].mean()), 4))
            print(f"[t03] hợp {a} ∪ {b} {bk}: {u[bk]}", flush=True)
        res[f"union::{a}::{b}"] = u
        J.to_pickle(B / f"join__{a}__{b}.pkl")
    T.jdump(res, B / f"compare__{'__'.join(names)}.json")


if __name__ == "__main__":
    main()
