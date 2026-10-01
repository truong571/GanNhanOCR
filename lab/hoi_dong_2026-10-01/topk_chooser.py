#!/usr/bin/env python3
"""Top-1 / Top-5 của bộ chọn chữ TN8 tầng 1 (logit có điều kiện trên ứng viên R(âm) ∪ {kim}) so chữ NGƯỜI, LOBO theo sách.
Đọc bảng ứng viên đã dựng `measure_out/_tn8/cand/*.pkl` (0 API, không ghi gì vào repo). Ba cách xếp hạng ứng viên:
  full    : tầng 1 đầy đủ (ảnh + tiên nghiệm âm + cờ kim) — như t06_eval.fit_models
  no_kim  : bỏ 3 đặc trưng kim (is_kim, kim_inR, is_lt2) — chỉ ảnh + tiên nghiệm
  prior   : chỉ tiên nghiệm lp_syl = log P(chữ | âm) từ ô neo khối khác (không ảnh, không kim)
và mốc kim: top-1 = chữ kim (nếu là ứng viên). Ô đo: có chữ người, có ≥ 1 ứng viên, phần in/out (bỏ notplaus/quarantine).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import t06_eval as E6  # noqa: E402
import tn8model as M  # noqa: E402

KIM = ("is_kim", "kim_inR", "is_lt2")
RNG = np.random.default_rng(0)


def cols_keep(names, variant):
    if variant == "no_kim":
        return [j for j, n in enumerate(names) if n not in KIM]
    return list(range(len(names)))


def fit_m1(train, part, variant):
    X1s, i1s, y1s = [], [], []
    off = 0
    for Bk in train:
        D, F = Bk["D"], Bk["F"]
        ok = (Bk["part"] == part) & (D.gt_char.values != "")
        anyy = F.groupby("i").y.max().reindex(np.arange(len(D))).fillna(0).values > 0
        nc = F.groupby("i").size().reindex(np.arange(len(D))).fillna(0).values
        cells = np.nonzero(ok & anyy & (nc >= 2))[0]
        if len(cells) > E6.MAX_TRAIN_CELLS:
            cells = np.sort(RNG.choice(cells, E6.MAX_TRAIN_CELLS, replace=False))
        sel = np.isin(F.i.values, cells)
        ck = cols_keep(Bk["names"], variant)
        X1s.append(Bk["Xc"][sel][:, ck]); i1s.append(F.i.values[sel] + off); y1s.append(F.y.values[sel])
        off += len(D) + 1
    return M.CLogit().fit(np.concatenate(X1s), np.concatenate(i1s), np.concatenate(y1s))


def topk(F, score, cells, ks=(1, 5)):
    G = pd.DataFrame({"i": F.i.values, "s": score, "y": F.y.values})
    G = G[G.i.isin(cells)]
    G["rk"] = G.groupby("i").s.rank(ascending=False, method="first")
    out = {}
    n = len(cells)
    for k in ks:
        hit = G[(G.rk <= k) & (G.y == 1)].i.nunique()
        out[f"top{k}"] = round(hit / n, 4)
    out["oracle_any"] = round(G.groupby("i").y.max().sum() / n, 4)
    out["n"] = int(n)
    return out


def run(train_codes, test_code):
    train = [E6.load_book(b) for b in train_codes]
    test = E6.load_book(test_code)
    D, F = test["D"], test["F"]
    has_c = np.zeros(len(D), bool); has_c[np.unique(F.i.values)] = True
    res = {}
    for part in ("in", "out", "all"):
        parts = ("in", "out") if part == "all" else (part,)
        cells = np.nonzero(np.isin(test["part"], parts) & (D.gt_char.values != "") & has_c)[0]
        r = {}
        # mốc kim: chữ kim là ứng viên và đúng
        kimc = F[(F.is_kim.values == 1)]
        kim_ok = kimc[kimc.i.isin(cells)].groupby("i").y.max()
        r["kim_top1"] = round(float(kim_ok.reindex(cells).fillna(0).sum()) / len(cells), 4)
        r["prior"] = topk(F, F.lp_syl.values.astype(float) + 1e-9 * RNG.random(len(F)), cells)
        res[part] = r
    for variant in ("full", "no_kim"):
        sc = np.full(len(F), np.nan)
        for part in ("in", "out"):
            m1 = fit_m1(train, part, variant)
            ck = cols_keep(test["names"], variant)
            p = m1.predict(test["Xc"][:, ck], F.i.values)
            rows = np.isin(F.i.values, np.nonzero(test["part"] == part)[0])
            sc[rows] = p[rows]
        sc = np.nan_to_num(sc, nan=-1.0)
        for part in ("in", "out", "all"):
            parts = ("in", "out") if part == "all" else (part,)
            cells = np.nonzero(np.isin(test["part"], parts) & (D.gt_char.values != "") & has_c)[0]
            res[part][variant] = topk(F, sc, cells)
    return res


if __name__ == "__main__":
    out = {}
    for tr, te in (("B34", "B18"), ("B18", "B34"), ("TK", "L16"), ("L16", "TK")):
        out[f"{te}<-{tr}"] = run([tr], te)
        print(te, "<-", tr, json.dumps(out[f"{te}<-{tr}"], ensure_ascii=False), flush=True)
    Path(__file__).with_name("topk_chooser.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
