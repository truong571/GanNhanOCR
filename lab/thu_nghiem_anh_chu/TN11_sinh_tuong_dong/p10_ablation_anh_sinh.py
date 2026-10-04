"""TN11 p10 (04/10) — ĐÓNG GÓP THẬT của tín hiệu "ảnh sinh" (glyph phông + ảnh FD) trong bộ chọn chữ TN8: tắt từng loại, đo LOBO. 0 API, CPU.

Câu hỏi: phần "sinh ảnh mẫu rồi so ảnh" ĐANG chạy trong pipeline (glyph phông render + 1.646 ảnh FD) đóng góp bao nhiêu điểm Top-1 cho bộ chọn
chữ? (KHAO_SAT_TAI_LIEU §5.8: "sinh glyph chưa được đo riêng"). Cách tắt: đặt cột f_font / f_fd của bảng ứng viên = NaN (đặc trưng thành hằng số,
chuẩn hoá về 0 — cả tầng 1 lẫn tầng 2 — nên bằng loại đặc trưng đó). Khung = TN8 t06_eval (cùng mã, 4 cặp LOBO B34→B18, B18→B34, TK→L16, L16→TK).
Biến thể: goc (dùng lại pred của TN12 đã hiệu chuẩn == TN8) · khong_FD · khong_phong · khong_ca_hai (không glyph ảnh sinh nào; còn nguyên mẫu ảnh thật cùng
sách/người, head v1, CNN kiểm, tiên nghiệm). Không ghi vào measure_out/_tn8.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/p10_ablation_anh_sinh.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lab/thu_nghiem_kim/TN8_chon_chu"))
import tn8lib as T  # noqa: E402

SRC = T.OUT
W = REPO / "measure_out" / "_tn11" / "p10_ablation"
PAIRS = [("B34", "B18"), ("B18", "B34"), ("TK", "L16"), ("L16", "TK")]
BOOKS = sorted({b for p in PAIRS for b in p})
VARIANTS = {"khong_FD": ["f_fd"], "khong_phong": ["f_font"], "khong_ca_hai": ["f_font", "f_fd"]}
GOC_PRED = REPO / "measure_out" / "_tn11" / "tn12_fd_day_du" / "pred_goc"      # TN12 'goc' == TN8 (hiệu chuẩn 94,65/94,09/97,48/98,44)


def prep(var, cols):
    wd = W / var
    (wd / "cand").mkdir(parents=True, exist_ok=True)
    for d in ("base", "emb"):
        if not (wd / d).exists():
            os.symlink(SRC / d, wd / d)
    for b in BOOKS:
        F = pd.read_pickle(SRC / "cand" / f"{b}.pkl")
        for c in cols:
            F[c] = np.float32(np.nan)
        F.to_pickle(wd / "cand" / f"{b}.pkl")
        cp = wd / "cand" / f"{b}_cells.pkl"
        if not cp.exists():
            os.symlink(SRC / "cand" / f"{b}_cells.pkl", cp)
    return wd


def top1(P, rare):
    P = P[P.any_y == 1]
    m = P.i.isin(rare)
    return P.set_index("i").y1.astype(float), m.to_numpy()


def main():
    W.mkdir(parents=True, exist_ok=True)
    import tn8model as M  # noqa: F401
    import t06_eval as E
    for var, cols in VARIANTS.items():
        wd = prep(var, cols)
        T.OUT = wd
        E.PRED = wd / "pred"
        for tr, te in PAIRS:
            E.run_pair([tr], te, tag=f"{te}__from_{tr}")
    T.OUT = SRC
    res = {}
    print(f"{'cặp':10s} | {'biến thể':14s} | Top-1 mọi ô | chữ hiếm | Δ mọi ô so với gốc [CI95 cụm trang] | Δ hiếm")
    for tr, te in PAIRS:
        F = pd.read_pickle(SRC / "cand" / f"{te}.pkl")
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        rare = set(tru.index[tru.n_self.fillna(0) < 2])
        pages = T.load_base(te).page.to_numpy()
        base = pd.read_pickle(GOC_PRED / f"{te}__from_{tr}.pkl")
        yb, mb = top1(base, rare)
        r = {"goc": dict(tat_ca=round(float(yb.mean()) * 100, 2), chu_hiem=round(float(yb[mb].mean()) * 100, 2), n=int(len(yb)))}
        print(f"{te}<-{tr:4s} | {'goc':14s} | {r['goc']['tat_ca']:11.2f} | {r['goc']['chu_hiem']:8.2f} |")
        for var in VARIANTS:
            v = pd.read_pickle(W / var / "pred" / f"{te}__from_{tr}.pkl")
            yv, mv = top1(v, rare)
            yv = yv.reindex(yb.index)
            d = (yv - yb).to_numpy()
            _, lo, hi = T.boot_ci_pages(d, pages[yb.index.to_numpy()])
            dh = d[mb]
            r[var] = dict(tat_ca=round(float(yv.mean()) * 100, 2), chu_hiem=round(float(yv[mv].mean()) * 100, 2),
                          delta=round(float(d.mean()) * 100, 2), ci=[round(lo * 100, 2), round(hi * 100, 2)], delta_hiem=round(float(dh.mean()) * 100, 2))
            print(f"{te}<-{tr:4s} | {var:14s} | {r[var]['tat_ca']:11.2f} | {r[var]['chu_hiem']:8.2f} | {r[var]['delta']:+6.2f} [{r[var]['ci'][0]:+.2f}, {r[var]['ci'][1]:+.2f}] | {r[var]['delta_hiem']:+6.2f}", flush=True)
        res[f"{te}<-{tr}"] = r
    (REPO / "measure_out" / "_tn11" / "p10_ablation_ket_qua.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
