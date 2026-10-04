"""v_truy_nguon_09 (04/10) — "Trước" = f_fd (kho FontDiffuser sinh sẵn) ở L16/TK yếu vì ĐIỂM SỐ hay vì KHO THIẾU GLYPH? 0 API, CPU, chỉ ĐỌC.

Đo trên 4 bộ có nhãn người (cand/<bộ>.pkl; ô = mọi ô có chữ đúng trong ứng viên, như p01):
  - tỉ lệ chữ đúng CÓ ảnh FD (f_fd không NaN) và tỉ lệ cặp ứng viên có ảnh FD;
  - Top-1 của f_fd (p01: NaN -> -9) tách theo chữ đúng có / thiếu ảnh FD;
  - độ nhạy của Top-1 trong top-3 (giao thức p05) theo cách xử lý NaN (-9 / 0 / bỏ ứng viên NaN) cho f_fd và f_font.
Ra: measure_out/_tn11/verify/truy_nguon/phu_fd.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

OUT = REPO / "measure_out" / "_tn11" / "verify" / "truy_nguon"


def main():
    res = {}
    for b in ("L16", "TK", "B18", "B34"):
        F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_vW", "f_font", "f_fd"]]
        hy = F.groupby("i").y.max(); cells = hy[hy == 1].index.values
        F = F[F.i.isin(cells)].copy()
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        has = tru.f_fd.notna().reindex(cells).values
        x = F[["i", "f_fd", "y"]].copy(); x["f_fd"] = x.f_fd.fillna(-9)
        t = x.loc[x.groupby("i")["f_fd"].idxmax()].set_index("i").y.reindex(cells).values
        r = dict(n=int(len(cells)), chu_dung_co_anh_FD_pct=round(100 * float(has.mean()), 2),
                 cap_ung_vien_co_anh_FD_pct=round(100 * float(F.f_fd.notna().mean()), 2),
                 f_fd_top1_tat_ca=round(100 * float(t.mean()), 2),
                 f_fd_top1_khi_chu_dung_co_FD=round(100 * float(t[has].mean()), 2),
                 f_fd_top1_khi_chu_dung_thieu_FD=round(100 * float(t[~has].mean()), 2))
        # độ nhạy giao thức top-3
        F["rk"] = F.f_vW.fillna(-9)
        S = F.sort_values(["i", "rk"], ascending=[True, False], kind="stable").groupby("i").head(3)
        keep = S.groupby("i").y.max(); valid = keep[keep == 1].index.values; S = S[S.i.isin(valid)]
        sens = {}
        for col in ("f_font", "f_fd"):
            d = dict(nan_trong_top3_pct=round(100 * float(S[col].isna().mean()), 2))
            for pol, fill in (("NaN=-9", -9.0), ("NaN=0", 0.0)):
                z = S[["i", col, "y"]].copy(); z[col] = z[col].fillna(fill)
                d[pol] = round(100 * float(z.loc[z.groupby("i", sort=False)[col].idxmax()].set_index("i").y.mean()), 2)
            z = S[["i", col, "y"]].dropna(subset=[col])
            d["bo_ung_vien_NaN"] = round(100 * float(z.loc[z.groupby("i", sort=False)[col].idxmax()].set_index("i").y.reindex(valid).fillna(0).mean()), 2)
            sens[col] = d
        r["top3_do_nhay_cach_xu_ly_NaN"] = sens
        res[b] = r
        print(b, {k: v for k, v in r.items() if k != "top3_do_nhay_cach_xu_ly_NaN"})
        print("   top-3:", sens)
    (OUT / "phu_fd.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
