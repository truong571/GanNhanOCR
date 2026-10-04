"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập: độ PHỦ của f_fd và so FD−font "cùng tập", làm lại bằng cách khác với người kiểm chứng.

Người kiểm chứng (v_kythuat_fd_phu.py) kết luận: kho FD cục bộ chỉ có 1.646 ảnh thật (89.813 con trỏ LFS), p01 điền -9 cho ứng viên
thiếu ảnh FD => Top-1 f_fd lệch vì độ phủ; cùng tập FD−font = +6,7/+6,7/−4,7/−3,9 (không phải +22/+26/−13/−14).
Script này (CPU, chỉ đọc pickle TN8 + thống kê tệp, KHÔNG sửa tệp nào) kiểm lại:
  I1  tái lập p01: Top-1 kiểu p01 (điền -9) của f_font/f_fd == p01_<bộ>.json (43,0/34,6/63,3/62,5 và 65,4/60,4/49,8/48,2);
  I2  "xổ số độ phủ": kỳ vọng Top-1 f_fd nếu chấm NGẪU NHIÊN trong ứng viên có FD (= P[chữ đúng có FD] × E[1/k_FD]) — phần Top-1 f_fd
      chỉ do độ phủ, không do ảnh FD;
  S   cùng tập với ngưỡng m ∈ {2,3,5} ứng viên có FD/ô (người kiểm chứng chỉ thử m=3) + CI bootstrap 95 % theo ô + kỳ vọng ngẫu nhiên;
  R   tách theo chữ hiếm (n_self<2) / chữ có mẫu (n_self>=2) ở tập cùng tập;
  D   đếm độc lập số tệp kho FD: ảnh thật / con trỏ LFS (kích thước < 1 KB), hợp chữ có ảnh thật.
Ra: measure_out/_tn11/verify/pb_kythuat/fd_phu.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_fd_phu.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CAND = REPO / "measure_out" / "_tn8" / "cand"
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
OUT.mkdir(parents=True, exist_ok=True)
P01 = REPO / "measure_out" / "_tn11"


def top1_fill(x, col):
    x = x[["i", col, "y"]].copy()
    x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax()].set_index("i").y


def boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(n)]
    return [round(float(d.mean() * 100), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


def main():
    res = {"kho": {}, "bo": {}}
    # --- D: đếm kho FD độc lập (không dùng mã của người kiểm chứng)
    real, ptr, sizes = set(), set(), []
    for d in ("gannhanocr-fd", "ArcFace/data/glyphs"):
        for r, _, fs in os.walk(REPO / d):
            if os.sep + ".git" in r:
                continue
            for f in fs:
                if f.startswith("U+") and f.endswith(".png"):
                    ch = int(f[2:-4], 16)
                    s = os.path.getsize(os.path.join(r, f))
                    sizes.append(s)
                    (real if s >= 1024 else ptr).add(ch)
    sz = np.array(sizes)
    res["kho"] = dict(so_tep_png_ten_U=int(len(sz)), tep_duoi_1KB=int((sz < 1024).sum()), tep_tu_1KB=int((sz >= 1024).sum()),
                      kich_thuoc_con_tro_trung_vi_byte=float(np.median(sz[sz < 1024])), chu_co_anh_that=len(real),
                      chu_chi_con_tro=len(ptr - real))
    print("kho", res["kho"], flush=True)

    for b in ("B18", "B34", "L16", "TK"):
        F = pd.read_pickle(CAND / f"{b}.pkl")[["i", "c", "y", "f_font", "f_fd", "n_self"]]
        hy = F.groupby("i").y.max()
        F = F[F.i.isin(hy[hy == 1].index)].copy()
        ref = json.load(open(P01 / f"p01_{b}.json"))["top1"]
        # I1 tái lập p01
        t_font = top1_fill(F, "f_font"); t_fd = top1_fill(F, "f_fd")
        inv = dict(f_font=round(float(t_font.mean()) * 100, 1), f_fd=round(float(t_fd.mean()) * 100, 1),
                   p01_f_font=ref["f_font"]["tat_ca"], p01_f_fd=ref["f_fd"]["tat_ca"])
        inv["PASS"] = bool(abs(inv["f_font"] - inv["p01_f_font"]) <= 0.15 and abs(inv["f_fd"] - inv["p01_f_fd"]) <= 0.15)
        # I2 xổ số độ phủ
        cov = F[F.f_fd.notna()]
        kcov = cov.groupby("i").size()
        true_cov = set(cov[cov.y == 1].i)
        cells = F.i.nunique()
        lot = sum(1.0 / kcov[i] for i in true_cov) / cells * 100
        p_true_cov = len(true_cov) / cells * 100
        # kỳ vọng "chữ đúng không có FD" => f_fd thua chắc chắn nếu có ƯV khác có FD, hoặc hoà -9 (idxmax lấy dòng đầu) nếu không ai có FD
        # (ghi chú: ô mà KHÔNG ứng viên nào có FD: idxmax chọn dòng đầu của nhóm -> kết quả phụ thuộc thứ tự dòng, không phải ảnh)
        n_nocov = int(cells - cov.i.nunique())
        d = dict(o=int(cells), invariant_p01=inv, p_chu_dung_co_FD=round(p_true_cov, 1),
                 ky_vong_top1_neu_cham_NGAU_NHIEN_trong_ƯV_co_FD=round(lot, 1),
                 o_khong_ƯV_nao_co_FD=n_nocov, k_FD_tb_tren_o_co_FD=round(float(kcov.mean()), 2))
        d["top1_f_fd_p01_tru_xo_so_do_phu"] = round(inv["f_fd"] - lot, 1)
        # S cùng tập, nhiều ngưỡng
        G = F[F.f_fd.notna() & F.f_font.notna()]
        cnt = G.groupby("i").size()
        tr = set(G[G.y == 1].i)
        d["cung_tap"] = {}
        for m in (2, 3, 5):
            ok = set(cnt[cnt >= m].index) & tr
            H = G[G.i.isin(ok)]
            pf = H.loc[H.groupby("i").f_fd.idxmax()].set_index("i").y
            pt = H.loc[H.groupby("i").f_font.idxmax()].set_index("i").y
            diff = (pf - pt).to_numpy(dtype=float)
            rnd = float((1.0 / H.groupby("i").size()).mean()) * 100
            d["cung_tap"][f"m>={m}"] = dict(o=int(len(ok)), FD=round(float(pf.mean()) * 100, 1), font=round(float(pt.mean()) * 100, 1),
                                           ngau_nhien=round(rnd, 1), FD_tru_font_CI95=boot_ci(diff))
        # R tách chữ hiếm/có mẫu ở m>=3
        ok = set(cnt[cnt >= 3].index) & tr
        H = G[G.i.isin(ok)]
        ns = F[F.y == 1].drop_duplicates("i").set_index("i").n_self.fillna(0)
        rare = set(ns[ns < 2].index)
        d["cung_tap_m3_tach_hiem"] = {}
        for tag, sel in (("hiem", lambda i: i in rare), ("co_mau", lambda i: i not in rare)):
            ids = [i for i in ok if sel(i)]
            if len(ids) < 50:
                d["cung_tap_m3_tach_hiem"][tag] = dict(o=len(ids), ghi_chu="quá ít ô")
                continue
            HH = H[H.i.isin(ids)]
            pf = HH.loc[HH.groupby("i").f_fd.idxmax()].set_index("i").y
            pt = HH.loc[HH.groupby("i").f_font.idxmax()].set_index("i").y
            d["cung_tap_m3_tach_hiem"][tag] = dict(o=len(ids), FD=round(float(pf.mean()) * 100, 1), font=round(float(pt.mean()) * 100, 1),
                                                   FD_tru_font_CI95=boot_ci((pf - pt).to_numpy(dtype=float)))
        res["bo"][b] = d
        print(b, json.dumps(d, ensure_ascii=False), flush=True)
    (OUT / "fd_phu.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
