"""v_pb_nguon_goc_08 (04/10) — PHẢN BIỆN: "trước" của L16 (49,8) và TK (48,2) trong Bảng IV = Top-1 f_fd của p01. Số thấp do domain gap hay do THIẾU ẢNH FD?

Đọc trực tiếp measure_out/_tn8/cand/<bộ>.pkl (cột f_fd; NaN = không có ảnh FD cho chữ ứng viên) — KHÔNG dùng mã của người kiểm chứng.
Với các ô có sự thật người (có ứng viên y==1; cùng định nghĩa top1 của p01: argmax cột với NaN→-9, đúng nếu y của ứng viên đó == 1):
  - Top-1 f_fd tất cả (so với p01_<bộ>.json);
  - phần ô mà chữ đúng CÓ giá trị f_fd; Top-1 f_fd trong hai nhóm;
  - số ứng viên/ô có f_fd.
Ra: measure_out/_tn11/verify/pb_nguon_goc/fd_phu.json
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
    F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_font", "f_fd"]]
    hy = F.groupby("i").y.max()
    cells = hy[hy == 1].index
    F = F[F.i.isin(cells)].copy()

    def top1(col):
        x = F[["i", col, "y"]].copy()
        x[col] = x[col].fillna(-9)
        return x.loc[x.groupby("i")[col].idxmax(), ["i", "y"]].set_index("i").y

    t_fd = top1("f_fd"); t_font = top1("f_font")
    truth_has = F[F.y == 1].groupby("i").f_fd.apply(lambda s: bool(s.notna().any()))
    has = truth_has.reindex(t_fd.index).fillna(False).astype(bool)
    p01 = json.loads((REPO / "measure_out/_tn11" / f"p01_{b}.json").read_text(encoding="utf-8"))["top1"]
    res[b] = {"n_o": int(len(t_fd)), "top1_f_fd_tat_ca": round(float(t_fd.mean()) * 100, 1), "p01_json_f_fd": p01["f_fd"]["tat_ca"],
              "top1_f_font_tat_ca": round(float(t_font.mean()) * 100, 1), "p01_json_f_font": p01["f_font"]["tat_ca"],
              "pct_o_chu_dung_co_anh_FD": round(float(has.mean()) * 100, 1),
              "top1_f_fd_khi_co_anh": round(float(t_fd[has].mean()) * 100, 1), "top1_f_fd_khi_khong_anh": round(float(t_fd[~has].mean()) * 100, 1),
              "top1_f_font_khi_co_anh": round(float(t_font[has].mean()) * 100, 1), "top1_f_font_khi_khong_anh": round(float(t_font[~has].mean()) * 100, 1),
              "pct_ung_vien_co_f_fd": round(float(F.f_fd.notna().mean()) * 100, 1)}
    print(b, json.dumps(res[b], ensure_ascii=False), flush=True)
(OUT / "fd_phu.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
