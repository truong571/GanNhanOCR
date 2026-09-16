"""K5 · Chuỗi "thanh ghi trượt" trong bộ giao nộp, đo bằng OOF (p_visual_oof_v3.csv) — chỉ đo.

Ô i "trượt +1" nếu argmax(crop i) == âm của ô syl_idx+1 (cùng cột) ∧ max_prob ≥ thr ∧ âm kề ≠ âm mình;
"trượt −1" tương tự với syl_idx−1. Chuỗi = ≥ L ô liên tiếp (theo syl_idx) cùng chiều trượt trong 1 cột.
Mô hình nhầm ngẫu nhiên không tạo chuỗi cùng chiều; chuỗi dài = hộp lệch so với chữ OCR (ví dụ đã nhìn bằng mắt:
stt11/page_0126/c6 nom 5–12, cột legacy_locked_col, ô QĐ-01 nom 5 crop là 羅 'là' chứ không phải 𠊚).

    .venv/bin/python KhoiB/v3/k5_register_shift.py [thr=0.8] [min_len=3]  -> KhoiB/v3/k5_register_shift_thr{thr}_len{L}.{json,csv}
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
MINLEN = int(sys.argv[2]) if len(sys.argv) > 2 else 3
TAG = f"_thr{THR}_len{MINLEN}"
L = pd.read_csv(REPO / "dataset_out/labels_final.csv", dtype=str, keep_default_na=False)
O = pd.read_csv(REPO / "KhoiB/v3/p_visual_oof_v3_results/p_visual_oof_v3.csv", dtype=str, keep_default_na=False)
assert len(L) == len(O) and (L.syllable.values == O.syllable_v3.values).all()
D = L[["book", "page", "column", "nom_idx", "syl_idx", "ocr_char", "syllable", "tier", "tier_v3", "box_source", "qd01_locked", "n_ocr", "n_qn", "n_det"]].copy()
D["argmax"] = O.argmax.values; D["max_prob"] = pd.to_numeric(O.max_prob, errors="coerce").values
D["p_syl"] = pd.to_numeric(O.p_syl_v3, errors="coerce").values
D["syl_idx_i"] = D.syl_idx.astype(int)
runs = []
for (b, p, c), g in D.groupby(["book", "page", "column"], sort=False):
    g = g.sort_values("syl_idx_i")
    syl = dict(zip(g.syl_idx_i, g.syllable))
    for d in (+1, -1):
        cur = []
        for _, r in g.iterrows():
            nb = syl.get(r.syl_idx_i + d)
            hit = nb is not None and nb != r.syllable and r["argmax"] == nb and r.max_prob >= THR
            if hit:
                cur.append(r)
            else:
                if len(cur) >= MINLEN:
                    runs.append((b, p, c, d, cur))
                cur = []
        if len(cur) >= MINLEN:
            runs.append((b, p, c, d, cur))
rows = []
for b, p, c, d, cur in runs:
    for r in cur:
        rows.append({"book": b, "page": p, "column": c, "shift": d, "run_len": len(cur), "nom_idx": r.nom_idx, "syl_idx": r.syl_idx,
                     "ocr_char": r.ocr_char, "syllable": r.syllable, "argmax": r["argmax"], "max_prob": r.max_prob, "p_syl": r.p_syl,
                     "tier": r.tier, "tier_v3": r.tier_v3, "box_source": r.box_source, "qd01_locked": r.qd01_locked,
                     "n_ocr": r.n_ocr, "n_qn": r.n_qn, "n_det": r.n_det})
R = pd.DataFrame(rows)
R.to_csv(REPO / f"KhoiB/v3/k5_register_shift{TAG}.csv", index=False)
cols = {(b, p, c) for b, p, c, _, _ in runs}
rep = {"thr": THR, "min_run": MINLEN, "n_runs": len(runs), "n_cols": len(cols), "n_cells": int(len(R)),
       "run_len_hist": dict(Counter(len(cur) for *_, cur in runs)),
       "cells_by_tier": dict(Counter(R.tier)) if len(R) else {}, "cells_by_tier_v3": dict(Counter(R.tier_v3)) if len(R) else {},
       "cells_usable": int(R.tier.isin(["GOLD", "SYLLABLE"]).sum()) if len(R) else 0,
       "cells_qd01_locked": int((R.qd01_locked == "1").sum()) if len(R) else 0,
       "cells_by_box_source": dict(Counter(R.box_source)) if len(R) else {},
       "cols_by_box_source_majority": dict(Counter(R.groupby(["book", "page", "column"]).box_source.agg(lambda s: s.mode().iat[0]))) if len(R) else {},
       "cols_n_det_ne_n_ocr": int(sum(1 for (b, p, c) in cols if (lambda g: (g.n_det.iat[0] != g.n_ocr.iat[0]))(D[(D.book == b) & (D.page == p) & (D["column"] == c)]))),
       "cells_by_book": dict(Counter(R.book)) if len(R) else {},
       "vi_du_stt11_page_0126_c6": R[(R.book == "stt11") & (R.page == "page_0126") & (R["column"] == "6")][["nom_idx", "syllable", "argmax", "max_prob", "tier"]].to_dict("records") if len(R) else []}
json.dump(rep, open(REPO / f"KhoiB/v3/k5_register_shift{TAG}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
