"""TN11 p05 (03/10) — thêm tín hiệu "glyph đã ánh xạ sang nét của cuốn" (f_map = f_ridge của p01) vào BỘ CHỌN CHỮ TN8, đo LOBO.

Chạy lại đúng t06_eval của TN8 (cùng mã, cùng cặp LOBO: B34->B18, B18->B34, TK->L16, L16->TK) trong thư mục riêng
measure_out/_tn11/tn8_map/ (không ghi đè _tn8): lượt "goc" = đặc trưng TN8 như cũ; lượt "map" = thêm f_map vào tầng 1.
So Top-1 đúng chữ trên mọi ô có sự thật + trên ô chữ hiếm (chữ đúng < 2 ô neo cùng sách ở khối khác). 0 API.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p05_bo_chon_them_f_map.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

SRC = T.OUT
W = REPO / "measure_out" / "_tn11" / "tn8_map"
PAIRS = [("B34", "B18"), ("B18", "B34"), ("TK", "L16"), ("L16", "TK")]


def prep():
    (W / "cand").mkdir(parents=True, exist_ok=True)
    for d in ("base", "emb"):
        if not (W / d).exists():
            os.symlink(SRC / d, W / d)
    for b in {x for p in PAIRS for x in p}:
        F = pd.read_pickle(SRC / "cand" / f"{b}.pkl")
        S = pd.read_pickle(REPO / "measure_out" / "_tn11" / f"p01_{b}_scores.pkl")
        F = F.merge(S[["i", "c", "f_ridge"]].rename(columns={"f_ridge": "f_map"}), on=["i", "c"], how="left")
        F.to_pickle(W / "cand" / f"{b}.pkl")
        c = W / "cand" / f"{b}_cells.pkl"
        if not c.exists():
            os.symlink(SRC / "cand" / f"{b}_cells.pkl", c)


def main():
    prep()
    T.OUT = W
    import tn8model as M
    import t06_eval as E
    E.PRED = W / "pred"
    base_sims = list(M.SIMS)
    for run in ("goc", "map"):
        M.SIMS[:] = base_sims + (["f_map"] if run == "map" else [])
        for tr, te in PAIRS:
            E.run_pair([tr], te, tag=f"{te}__from_{tr}__{run}")
    res = {}
    for tr, te in PAIRS:
        F = pd.read_pickle(W / "cand" / f"{te}.pkl")
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        rare = set(tru.index[tru.n_self.fillna(0) < 2])
        r = {}
        for run in ("goc", "map"):
            P = pd.read_pickle(W / "pred" / f"{te}__from_{tr}__{run}.pkl")
            P = P[P.any_y == 1]
            m = P.i.isin(rare)
            r[run] = dict(tat_ca=round(float(P.y1.mean()) * 100, 2), chu_hiem=round(float(P[m].y1.mean()) * 100, 2),
                          n=int(len(P)), n_hiem=int(m.sum()))
        res[f"{te}<-{tr}"] = r
        print(f"{te} <- {tr}: gốc {r['goc']['tat_ca']} % (chữ hiếm {r['goc']['chu_hiem']} %) -> thêm f_map "
              f"{r['map']['tat_ca']} % (chữ hiếm {r['map']['chu_hiem']} %), n {r['goc']['n']} / hiếm {r['goc']['n_hiem']}", flush=True)
    (REPO / "measure_out" / "_tn11" / "p05_ket_qua.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
