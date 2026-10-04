"""v_thu_hqc_09_giao_thuc_k09.py — dựng lại GIAO THỨC của thí nghiệm p05_full_corpus_he_quy_chieu.py (K09) trên cột sản xuất, 0 mô hình:
  * chọn top-3 ứng viên theo f_vW (p05:196-197), giữ ô mà chữ đúng nằm trong top-3 (p05:199-200);
  * Top-1 của f_font (HQC1 sản xuất) CHỈ trong 3 ứng viên đó; so với Top-1 trên TOÀN BỘ ứng viên của ô (giao thức p01/TN8);
  * độ ngẫu nhiên khi n = 20 ô (p05:356): phân phối Top-1 qua 5000 mẫu 20 ô.
Ra: measure_out/_tn11/verify/thu_hqc/giao_thuc_k09.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v_thu_hqc_lib as H  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T = H.T


def one(b):
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_vW", "f_font", "f_fd"]]
    hy = F.groupby("i").y.max()
    cells = hy[hy == 1].index.values
    F = F[F.i.isin(cells)].reset_index(drop=True)
    r = dict(book=b, n_cells=int(len(cells)))
    for col in ("f_font", "f_fd"):
        m = H.cell_metrics(F.i.values, F[col].values, F.y.values)
        r[f"top1_toan_bo_ung_vien_{col}"] = round(float(m.ok.mean()) * 100, 2)
    F["rk"] = F.f_vW.fillna(-9)
    top3 = F.sort_values(["i", "rk"], ascending=[True, False], kind="stable").groupby("i").head(3)
    has = top3.groupby("i").y.max()
    valid = has[has == 1].index.values
    r["ti_le_o_co_chu_dung_trong_top3_theo_f_vW"] = round(float(len(valid) / len(cells)) * 100, 2)
    t3 = top3[top3.i.isin(set(valid))].sort_values(["i", "c"], kind="stable").reset_index(drop=True)
    for col in ("f_font", "f_fd"):
        m = H.cell_metrics(t3.i.values, t3[col].values, t3.y.values)
        r[f"top1_trong_3_ung_vien_{col}"] = round(float(m.ok.mean()) * 100, 2)
        r[f"n_o_top3_{col}"] = int(len(m))
        ok = m.ok.values.astype(float)
        rng = np.random.default_rng(H.SEED + 71)
        acc = np.array([ok[rng.choice(len(ok), 20, replace=False)].mean() for _ in range(5000)]) * 100
        r[f"n20_{col}"] = dict(sd=round(float(acc.std()), 1), p025=round(float(np.quantile(acc, .025)), 1), p975=round(float(np.quantile(acc, .975)), 1))
    r["so_ung_vien_trung_binh_moi_o"] = round(float(F.groupby("i").size().mean()), 1)
    r["ngau_nhien_1_tren_3"] = 33.3
    return r


def main():
    import json
    import subprocess
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        H.jdump(one(sys.argv[2]), H.OUT / f"giao_thuc_k09_{sys.argv[2]}.json"); return
    res = {}
    for b in H.BOOKS:
        subprocess.run([sys.executable, __file__, "--one", b], check=True)
        res[b] = json.loads((H.OUT / f"giao_thuc_k09_{b}.json").read_text(encoding="utf-8"))
        H.log(f"{b}: {res[b]}")
    H.jdump(res, H.OUT / "giao_thuc_k09.json")


if __name__ == "__main__":
    main()
