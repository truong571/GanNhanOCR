"""TN11 p08 (04/10) — SO CÙNG TẬP ỨNG VIÊN: gỡ hiệu ứng "độ phủ ảnh FD" khỏi so sánh font ↔ FD (và ↔ ảnh sinh). 0 API, CPU.

Vì sao: pipeline chỉ nạp được 1.646 ảnh FD thật (89.813/89.898 tệp trong gannhanocr-fd là con trỏ Git-LFS; Glyphs bỏ tệp < 1 KB),
nên f_fd = NaN ở phần lớn ứng viên; p01/p02 xếp NaN = −9 ⇒ ứng viên không có ảnh FD luôn thua, và chữ có ảnh FD là chữ THƯỜNG GẶP
(tiên nghiệm tần suất), không phải bằng chứng "ảnh FD giống crop hơn". So đúng: chỉ giữ ứng viên CÓ ảnh FD, ô còn chữ đúng và ≥ 2 ứng viên.

  .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p08_so_cung_tap.py [--fd-col f_fd]
Ra: in bảng + measure_out/_tn11/p08_cung_tap.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

BOOKS = ["B18", "B34", "L16", "TK"]


def top1_per_cell(F: pd.DataFrame, col: str) -> pd.Series:
    x = F[["i", col, "y"]]
    idx = x.groupby("i")[col].idxmax()
    return x.loc[idx].set_index("i").y.astype(float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dir", default=str(T.OUT / "cand"))
    ap.add_argument("--fd-col", default="f_fd")
    a = ap.parse_args(argv)
    out = {}
    print(f"{'bộ':4s} | {'ô':>6s} {'ƯV/ô':>5s} | phủ FD: chữ đúng / mọi ứng viên | cùng tập: font   FD   FD−font [CI95 cụm trang] | naive FD−font")
    for b in BOOKS:
        F = pd.read_pickle(Path(a.cand_dir) / f"{b}.pkl")[["i", "c", "y", a.fd_col, "f_font"]]
        D = T.load_base(b)
        pages = D.page.to_numpy()
        hy = F.groupby("i").y.max(); cells = hy[hy == 1].index
        Fc = F[F.i.isin(cells)]
        cov_true = float(Fc[Fc.y == 1][a.fd_col].notna().mean()) * 100
        cov_all = float(Fc[a.fd_col].notna().mean()) * 100
        nv = Fc.copy(); nv["fd9"] = nv[a.fd_col].fillna(-9); nv["fo9"] = nv["f_font"].fillna(-9)
        naive_fd = top1_per_cell(nv, "fd9"); naive_fo = top1_per_cell(nv, "fo9")
        M = Fc[Fc[a.fd_col].notna() & Fc.f_font.notna()]
        k = M.groupby("i").size(); has_true = M.groupby("i").y.max()
        keep = k[(k >= 2) & (has_true.reindex(k.index) == 1)].index
        M = M[M.i.isin(keep)]
        t_fo = top1_per_cell(M, "f_font"); t_fd = top1_per_cell(M, a.fd_col)
        d = (t_fd - t_fo).to_numpy()
        pg = pages[t_fd.index.to_numpy()]
        _, lo, hi = T.boot_ci_pages(d, pg)
        res = dict(n_o=int(len(cells)), n_o_cung_tap=int(len(keep)), uv_mot_o=round(float(M.groupby('i').size().mean()), 2),
                   phu_fd_chu_dung=round(cov_true, 1), phu_fd_moi_uv=round(cov_all, 1),
                   font_cung_tap=round(float(t_fo.mean()) * 100, 1), fd_cung_tap=round(float(t_fd.mean()) * 100, 1),
                   fd_tru_font_cung_tap=round(float(d.mean()) * 100, 1), ci=[round(float(lo) * 100, 1), round(float(hi) * 100, 1)],
                   naive_fd_tru_font=round((float(naive_fd.mean()) - float(naive_fo.mean())) * 100, 1))
        out[b] = res
        print(f"{b:4s} | {res['n_o']:6d} {len(Fc)/len(cells):5.1f} | {res['phu_fd_chu_dung']:5.1f} % / {res['phu_fd_moi_uv']:5.1f} %"
              f"           | n={res['n_o_cung_tap']:5d} ({res['uv_mot_o']:.1f} ƯV/ô)  {res['font_cung_tap']:5.1f} {res['fd_cung_tap']:5.1f}  "
              f"{res['fd_tru_font_cung_tap']:+5.1f} [{res['ci'][0]:+.1f}, {res['ci'][1]:+.1f}] | {res['naive_fd_tru_font']:+5.1f}")
    Path(REPO / "measure_out/_tn11/p08_cung_tap.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
