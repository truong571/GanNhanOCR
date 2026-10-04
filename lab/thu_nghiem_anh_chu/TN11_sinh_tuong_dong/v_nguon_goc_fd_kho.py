"""TN11 v_nguon_goc_fd_kho (03/10) — cột 'trước' của K12 cho L16/TK là f_fd (kho FD sinh sẵn) 49,8 / 48,2 %: đó là chất lượng ảnh hay là ĐỘ PHỦ kho?
0 mô hình, 0 API, chỉ ĐỌC measure_out/_tn8/cand/<bộ>.pkl (cột f_fd, NaN khi chữ không có ảnh FD thật; p01 dùng top1() với fillna(-9)).

Với 4 bộ có nhãn người, trên mọi ô có chữ đúng trong ứng viên (cùng tập với p01):
  - độ phủ: tỉ lệ ô mà chữ ĐÚNG có ảnh FD (f_fd không NaN)
  - Top-1 của f_fd (đúng công thức p01: argmax với NaN -> -9): tất cả / chỉ ô chữ đúng CÓ ảnh FD / chỉ ô chữ đúng KHÔNG có ảnh FD
  - đối chiếu f_font (độ phủ ~87 %)
Ra: measure_out/_tn11/verify/nguon_goc/fd_kho.json
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


def top1_per_cell(F, col):
    x = F[["i", col, "y"]].copy()
    x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax(), ["i", "y"]].set_index("i").y


def main():
    res = {}
    p01 = {b: json.loads((REPO / f"measure_out/_tn11/p01_{b}.json").read_text(encoding="utf-8"))["top1"] for b in ("B18", "B34", "L16", "TK")}
    for b in ("B18", "B34", "L16", "TK"):
        F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "f_font", "f_fd"]]
        hy = F.groupby("i").y.max()
        cells = hy[hy == 1].index
        F = F[F.i.isin(cells)].copy()
        truth_rows = F[F.y == 1].drop_duplicates("i").set_index("i")
        cov = truth_rows.f_fd.notna()
        covf = truth_rows.f_font.notna()
        t_fd = top1_per_cell(F, "f_fd")
        t_ft = top1_per_cell(F, "f_font")
        res[b] = {"n_o_co_su_that": int(len(cells)), "do_phu_FD_chu_dung_pct": round(float(cov.mean() * 100), 1), "do_phu_font_chu_dung_pct": round(float(covf.mean() * 100), 1),
                  "f_fd_top1_tat_ca_pct": round(float(t_fd.mean() * 100), 1), "f_fd_top1_khi_chu_dung_CO_anh_FD_pct": round(float(t_fd[cov[cov].index].mean() * 100), 1),
                  "f_fd_top1_khi_chu_dung_KHONG_co_anh_FD_pct": round(float(t_fd[cov[~cov].index].mean() * 100), 1),
                  "f_font_top1_tat_ca_pct": round(float(t_ft.mean() * 100), 1), "p01_f_fd_tat_ca_khop": bool(abs(float(t_fd.mean() * 100) - p01[b]["f_fd"]["tat_ca"]) < 0.06),
                  "ti_le_hang_ung_vien_NaN_f_fd_pct": round(float(F.f_fd.isna().mean() * 100), 1)}
        print(f"[FD kho] {b:4s} độ phủ FD (chữ đúng có ảnh) {res[b]['do_phu_FD_chu_dung_pct']}% | f_fd Top-1: tất cả {res[b]['f_fd_top1_tat_ca_pct']}% (p01 {p01[b]['f_fd']['tat_ca']}), "
              f"khi chữ đúng CÓ ảnh {res[b]['f_fd_top1_khi_chu_dung_CO_anh_FD_pct']}%, KHÔNG có ảnh {res[b]['f_fd_top1_khi_chu_dung_KHONG_co_anh_FD_pct']}% | f_font Top-1 {res[b]['f_font_top1_tat_ca_pct']}%", flush=True)
    (OUT / "fd_kho.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {OUT / 'fd_kho.json'}")


if __name__ == "__main__":
    main()
