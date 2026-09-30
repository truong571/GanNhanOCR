"""TN10 t04 (30/09) — biên SỐ Ô GIỮ ↔ ĐỘ ĐÚNG VỊ TRÍ cho bộ chữ người Borg (0 API), so với chuẩn 90/90 của user.

Độ đúng vị trí ước từ hồ sơ Paddle đóng băng (độc lập với encoder và kim):
  theta  = trượt ±1 theo paddle.est (mô hình nhiễu far đo TRÊN CHÍNH tập — lạc quan ở ô khó vì lệch ±2/±3 thật bị coi là nhiễu)
  e_tong = sai số TỔNG thận trọng = (a0 − P0)/(a0 − far0), nhiễu Paddle far0 lấy từ tập keep gốc (ô chắc), a0 = 1 − 6·far0
Luật giữ: base (hộp thật, det ≥ 0,2, trang bỏ chữ < 10 %) & min(post, post2) ≥ t [& agree2], có/không cộng ô kim xác nhận.
Tỉ lệ giữ tính trên MỌI ô chữ người (143.043, gồm ô không có hộp).
    .venv/bin/python lab/thu_nghiem_kim/TN10_borg_nguoi/t04_frontier.py --works work_b0 work_b1 work_b2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
from t03_curve import kim_confirm  # noqa: E402
T10 = REPO / "measure_out" / "_tn10"

def stats(t, m, far0):
    x = t[m & (t != "")]
    n = len(x)
    if not n:
        return dict(n_paddle=0, p0=np.nan, theta=np.nan, e_tong=np.nan)
    off = np.array([0 if v == "ok" else int(v[4:]) for v in x])
    cnt = {d: float((off == d).mean()) for d in range(-3, 4)}
    far = np.mean([cnt[-3], cnt[-2], cnt[2], cnt[3]]); a = 1 - 6 * far
    theta = (cnt[-1] + cnt[1] - 2 * far) / max(a - far, 1e-6)
    a0 = 1 - 6 * far0
    e = (a0 - cnt[0]) / (a0 - far0)
    return dict(n_paddle=n, p0=round(cnt[0], 4), theta=round(theta, 4), e_tong=round(max(0.0, e), 4))

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--works", nargs="+", default=["work_b0"]); ap.add_argument("--src", default="lt2")
    a = ap.parse_args(argv)
    rows = []
    for w in a.works:
        C = pd.read_csv(T10 / w / "cells.csv.gz")
        N = len(C)
        t = C.paddle_test.fillna("").astype(str).to_numpy()
        keep0 = (C.keep == 1).to_numpy()
        x0 = t[keep0 & (t != "")]; off0 = np.array([0 if v == "ok" else int(v[4:]) for v in x0])
        far0 = float(np.mean([(off0 == d).mean() for d in (-3, -2, 2, 3)]))
        base = ((C.kind == "real") & (C.det_score >= 0.2) & (C.page_skip < 0.1)).to_numpy()
        real = (C.kind == "real").to_numpy()
        mn = np.minimum(C.post, C.post2).fillna(0).to_numpy(); ag = (C.agree2 == 1).to_numpy()
        K = kim_confirm(C, a.src)
        for t_ in (0.999, 0.995, 0.99, 0.95, 0.9, 0.7, 0.5, 0.2, 0.0):
            for use_ag in (True, False):
                m = base & (mn >= t_) & (ag if use_ag else True)
                for kim in (False, True):
                    mm = m | (base & K) if kim else m
                    rows.append(dict(work=w, t=t_, agree2=use_ag, kim=kim, n=int(mm.sum()), ty_le=round(mm.sum() / N, 4),
                                     **stats(t, mm, far0)))
        mm = real | (C.kind != "skip").to_numpy()
        rows.append(dict(work=w, t="moi_o_co_hop", agree2=False, kim=False, n=int(mm.sum()), ty_le=round(mm.sum() / N, 4),
                         **stats(t, mm, far0)))
    R = pd.DataFrame(rows); R.to_csv(T10 / "t04_frontier.csv", index=False)
    R["dung_vi_tri"] = (1 - R.e_tong).round(4)
    for cap in (0.99, 0.98, 0.95, 0.9):
        ok = R[R.dung_vi_tri >= cap]
        if ok.empty:
            print(f"đúng vị trí ≥ {cap:.0%}: không cấu hình nào đạt"); continue
        b = ok.loc[ok.n.idxmax()]
        print(f"đúng vị trí ≥ {cap:.0%}: giữ tối đa {b.n} ô = {b.ty_le:.1%} (work {b.work}, t {b.t}, agree2 {b.agree2}, kim {b.kim}; "
              f"đúng {b.dung_vi_tri:.1%}, θ±1 {b.theta:.1%}, Paddle n {b.n_paddle})")
    print(R[(R.work == a.works[0]) & (R.agree2) & (~R.kim)][["t", "n", "ty_le", "p0", "theta", "dung_vi_tri"]].to_string(index=False))
if __name__ == "__main__":
    raise SystemExit(main())
