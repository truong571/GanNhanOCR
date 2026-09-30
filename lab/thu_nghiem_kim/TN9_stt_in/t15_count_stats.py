"""t15_count_stats.py — câu hỏi của đề xuất "đọc hai lượt": đủ SỐ CHỮ có giúp GIÓNG không? 0 API.
Mỗi bản dựng STT (hộp cát t02): theo cột — n_ocr (số chữ kim của cột), n_qn, n_det; tỉ lệ cột n_ocr == n_qn, Σn_ocr/Σn_qn,
phân bố count_source (equal_qn / equal_ocr / conflict), số âm QN không có ô (khe DP) và GOLD. Ra measure_out/_tn9/count_stats.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402


def main():
    names = sys.argv[1:] or ["lt1__legacy", "l1skel_l2__legacy", "twopass__legacy", "lt2__legacy"]
    out = {}
    for nm in names:
        f = T.OUT / "stt_builds" / nm / "dataset_out/labels_final.csv"
        if not f.exists():
            continue
        L = T.rd(f)
        L = L[L.book.isin(T.STT)]
        res = {}
        for bk, g in L.groupby("book"):
            col = g.groupby(["page", "column"]).agg(n_ocr=("n_ocr", "first"), n_qn=("n_qn", "first"), n_det=("n_det", "first"),
                                                    cells=("syl_idx", "size"), cs=("count_source", "first"))
            for c in ("n_ocr", "n_qn", "n_det"):
                col[c] = pd.to_numeric(col[c], errors="coerce")
            res[bk] = dict(cols=int(len(col)), frac_ocr_eq_qn=round(float((col.n_ocr == col.n_qn).mean()), 4),
                           frac_all3_eq=round(float(((col.n_ocr == col.n_qn) & (col.n_det == col.n_qn)).mean()), 4),
                           ratio_ocr_qn=round(float(col.n_ocr.sum() / col.n_qn.sum()), 4),
                           mean_abs_ocr_minus_qn=round(float((col.n_ocr - col.n_qn).abs().mean()), 3),
                           syl_without_cell=int(col.n_qn.sum() - col.cells.sum()),
                           count_source=g.count_source.value_counts(normalize=True).round(4).to_dict(),
                           gold=int((g.tier == "GOLD").sum()), cells=int(len(g)),
                           gold_share=round(float((g.tier == "GOLD").mean()), 4),
                           gold_over_qn=round(float((g.tier == "GOLD").sum() / col.n_qn.sum()), 4))
            print(f"[t15] {nm:20s} {bk:6s} {res[bk]}", flush=True)
        out[nm] = res
    T.jdump(out, T.OUT / "count_stats.json")


if __name__ == "__main__":
    main()
