"""TN11 v_nguon_goc_26_vs_3 (03/10) — CÙNG tín hiệu, CÙNG ô: Top-1 trong ~26 ứng viên (giao thức TN8/p01) so với trong top-3 theo f_vW (giao thức p05).
0 mô hình, 0 API, chỉ ĐỌC measure_out/_tn8/cand/<bộ>.pkl và danh sách ô đã lấy mẫu của v_nguon_goc_nhieu_mau.py (nhieu_mau_<tag>.json).

Với các ô DUY NHẤT đã qua lọc của lượt chạy hạt giống:
  - f_font (cos crop–glyph phông sản xuất): Top-1 trên MỌI ứng viên (≈26) và Top-1 trong top-3 theo f_vW (đúng tập p05)
  - f_vW (CNN kiểm đã học, dùng để rút top-3): Top-1 trên mọi ứng viên và là hạng 1 trong top-3
  - f_fd: như trên (NaN -> -9 như p01)
  - số ứng viên trung bình / ô; xác suất đoán ngẫu nhiên
Ra: measure_out/_tn11/verify/nguon_goc/so_26_voi_3_<tag>.json
    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_nguon_goc_26_vs_3.py s0_9
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CAND = REPO / "measure_out/_tn8/cand"
OUT = REPO / "measure_out/_tn11/verify/nguon_goc"


def top1(F, col):
    x = F[["i", col, "y"]].copy()
    x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax(), ["i", "y"]].set_index("i").y


def main(tag):
    runs_all = json.loads((OUT / f"nhieu_mau_{tag}.json").read_text(encoding="utf-8"))
    res = {}
    for b, runs in runs_all.items():
        cells = sorted({rec["cell_id"] for r in runs for rec in r["records"]})
        F = pd.read_pickle(CAND / f"{b}.pkl")
        F = F[F.i.isin(cells)][["i", "c", "y", "f_font", "f_fd", "f_vW"]].copy()
        hy = F.groupby("i").y.max()
        F = F[F.i.isin(hy[hy == 1].index)]
        ncand = float(F.groupby("i").size().mean())
        row = {"n_o_duy_nhat": int(F.i.nunique()), "ung_vien_tb_moi_o": round(ncand, 1), "ngau_nhien_26_pct": round(100 / ncand, 1), "ngau_nhien_top3_pct": 33.3}
        # top-3 theo f_vW (như p05)
        S = F.sort_values(["i", "f_vW"], ascending=[True, False]).groupby("i").head(3)
        in3 = S.groupby("i").y.max()
        F3 = S[S.i.isin(in3[in3 == 1].index)]
        row["o_chu_dung_trong_top3"] = int(F3.i.nunique())
        for col in ("f_font", "f_fd", "f_vW"):
            t_all = top1(F, col)
            t_in3 = top1(F3, col)
            row[col] = {"top1_trong_~26_tren_cung_o_pct": round(float(t_all.mean() * 100), 1), "top1_trong_26_chi_o_trong_top3_pct": round(float(t_all[t_in3.index].mean() * 100), 1),
                        "top1_trong_top3_pct": round(float(t_in3.mean() * 100), 1)}
        res[b] = row
        print(f"[26 vs 3 | {tag}] {b:4s} {row['n_o_duy_nhat']} ô duy nhất ({row['o_chu_dung_trong_top3']} có chữ đúng trong top-3), {row['ung_vien_tb_moi_o']} ứng viên/ô: " +
              " | ".join(f"{c}: 26cv {row[c]['top1_trong_~26_tren_cung_o_pct']}% -> top3 {row[c]['top1_trong_top3_pct']}%" for c in ("f_font", "f_fd", "f_vW")), flush=True)
    (OUT / f"so_26_voi_3_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "s0_9")
