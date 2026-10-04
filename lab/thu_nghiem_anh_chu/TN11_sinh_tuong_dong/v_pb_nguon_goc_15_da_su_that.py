"""v_pb_nguon_goc_15 (04/10) — PHẢN BIỆN: tỉ lệ ô hợp lệ (chữ đúng trong top-3 theo f_vW) có >= 2 ứng viên y==1 (dị thể) — p05:255 chỉ nhận y==1 đầu tiên.

Chỉ đọc measure_out/_tn8/cand/<bộ>.pkl (không mô hình). Với 4 bộ nhãn người: trong các ô hợp lệ của giao thức p05 (top-3 theo f_vW, NaN->-9, groupby head(3)):
  - % ô có >= 2 ứng viên y == 1; % ô chữ y==1 ĐẦU TIÊN không phải hạng 1 theo f_vW nhưng một y==1 khác là hạng 1 (p05 sẽ chấm sai một ô mà TN8 chấm đúng).
Ra: measure_out/_tn11/verify/pb_nguon_goc/da_su_that.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
res = {}
for b in ("B34", "B18", "L16", "TK"):
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_vW"]]
    hy = F.groupby("i").y.max()
    F = F[F.i.isin(hy[hy == 1].index)].copy()
    F["rk"] = F.f_vW.fillna(-9)
    S = F.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(3)
    ht = S.groupby("i").y.max()
    S = S[S.i.isin(ht[ht == 1].index)]
    g = S.groupby("i")
    n = g.ngroups
    multi = (g.y.sum() >= 2)
    first_ys = g.apply(lambda d: list(d.y))
    # p05: gt = hàng y==1 đầu tiên (theo thứ tự f_vW giảm); top1 theo f_vW = hàng đầu
    top_is_truth = first_ys.apply(lambda ys: ys[0] == 1)
    gt_first_is_top = first_ys.apply(lambda ys: [k for k, y in enumerate(ys) if y == 1][0] == 0)
    res[b] = {"n_o_hop_le": int(n), "pct_o_co_>=2_y1": round(float(multi.mean()) * 100, 1),
              "pct_o_top1_vW_la_truth(TN8)": round(float(top_is_truth.mean()) * 100, 1),
              "pct_o_top1_vW_la_gt_dau_tien(p05)": round(float(gt_first_is_top.mean()) * 100, 1)}
    print(b, res[b], flush=True)
(OUT / "da_su_that.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
