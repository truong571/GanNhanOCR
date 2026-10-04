"""v_pb_nguon_goc_17 (04/10) — PHẢN BIỆN: như 07 (HQC2 so HQC1, giao thức p05) nhưng trên MẪU LỚN N ô hợp lệ (rẻ hơn toàn quần thể) cho bộ lớn (B18).

Chọn N ô hợp lệ ngẫu nhiên (seed 777) trong các ô có chữ đúng trong top-3 theo f_vW; chấm HQC1/HQC2 đúng như p05:259-267 (nhúng glyph norm=False,
vector crop = E_full sản xuất); McNemar chính xác + Wilson. CPU, 0 API.
Ra: measure_out/_tn11/verify/pb_nguon_goc/quan_the_mau_<tag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
sp = importlib.util.spec_from_file_location("v02", str(HERE / "v_pb_nguon_goc_02_tim_seed.py"))
V2 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(V2)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    M = V2.load_p05()
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    enc = SI.Scorers(Assets(), "cpu", M.EMB_DIR, False, lambda m: None).enc()
    res = {}
    for book in a.books:
        t0 = time.time()
        T = M.T
        D = T.load_base(book)
        E = M.nrm(np.load(M.EMB_DIR / f"{book}_enc.npy").astype(np.float32))
        F = pd.read_pickle(M.CAND_DIR / f"{book}.pkl")
        if "y" in F.columns and F.y.sum() > 0:
            tru = F[F.y == 1].drop_duplicates("i").set_index("i"); target = list(tru.index); human = True
        else:
            lbl = D.label.values; tgt = lbl[F.i.values]; vm = (F.c.values == tgt) & (tgt != "")
            target = sorted(set(F.i.values[vm])); F["y"] = np.where(vm, 1, 0); human = False
        Ft = F[F.i.isin(set(target))].copy()
        Ft["rk"] = Ft.f_vW.fillna(-9) if "f_vW" in Ft.columns else Ft.f_font.fillna(-9)
        S = Ft.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
        ht = S.groupby("i").y.max()
        valid_all = np.array(sorted(ht[ht == 1].index))
        sel = np.sort(np.random.default_rng(777).choice(valid_all, size=min(a.n, len(valid_all)), replace=False))
        S = S[S.i.isin(set(int(x) for x in sel))]
        paper = M.extract_book_paper_model(book)
        chars = sorted(set(S.c))
        raw = [M.render_font_raw(c, 128) for c in chars]
        V1 = dict(zip(chars, M.nrm(V2.embed_chunks(enc, raw))))
        Vb = dict(zip(chars, M.nrm(V2.embed_chunks(enc, [M.blend_on_paper(r, paper) for r in raw]))))
        a1, a2 = [], []
        for i, g in S.groupby("i"):
            cs = list(g.c); ys = list(g.y)
            gt = cs[[k for k, y in enumerate(ys) if y == 1][0]]
            for V, A in ((V1, a1), (Vb, a2)):
                s = {c: float(E[i] @ V[c]) for c in cs}
                A.append(int(max(s, key=s.get) == gt))
        a1 = np.array(a1); a2 = np.array(a2)
        b = int(((a1 == 1) & (a2 == 0)).sum()); c = int(((a1 == 0) & (a2 == 1)).sum())
        p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        res[book] = {"human": human, "n_valid_pop": int(len(valid_all)), "n_mau": int(len(a1)),
                     "HQC1": [round(float(a1.mean()) * 100, 2), wilson(int(a1.sum()), len(a1))], "HQC2": [round(float(a2.mean()) * 100, 2), wilson(int(a2.sum()), len(a2))],
                     "HQC2_tru_HQC1_pct": round(float((a2.mean() - a1.mean()) * 100), 2), "HQC1dung_HQC2sai": b, "HQC1sai_HQC2dung": c, "p_mcnemar": p,
                     "paper": {"bg": paper["bg_median"], "ink": paper["ink_median"], "bg_std": paper["bg_std"]}, "sec": round(time.time() - t0, 1)}
        print(book, json.dumps(res[book], ensure_ascii=False), flush=True)
        (OUT / f"quan_the_mau_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
