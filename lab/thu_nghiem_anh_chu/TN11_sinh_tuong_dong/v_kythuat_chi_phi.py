"""v_kythuat (03/10): TÍNH LẠI chi phí "sinh ảnh pixel cho toàn kho" (CPU, chỉ đọc pickle TN8 + liệt kê tên tệp kho FD).

Báo cáo nêu: "275.136 ô x trung bình 3-4 ứng viên = gần 1 triệu cặp ... hơn 3.000 giờ GPU". Script đếm:
  (a) số ô, số dòng (ô × ứng viên) THẬT, ứng viên/ô (TN8 cand/<bộ>.pkl),
  (b) số CHỮ KHÁC NHAU (glyph chỉ phụ thuộc chữ + phong cách, không phụ thuộc ô) = số ảnh cần sinh cho MỘT phong cách/sách,
  (c) số chữ cần sinh nếu chỉ sinh cho ô "chữ hiếm" (chữ đúng < 2 ô neo cùng sách) với top-K ứng viên theo f_vW (thiết kế p02),
  (d) độ phủ kho FD sinh sẵn (gannhanocr-fd) trên các chữ ứng viên,
rồi nhân với tốc độ s/ảnh đo trong log (logs/tn11_p02_gen.log) và tốc độ T4 ghi trong lab/kaggle_diffusion/README.md.
Ra: measure_out/_tn11/verify/kythuat/chi_phi_sinh_anh.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_kythuat_chi_phi.py [--topk 8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CAND = REPO / "measure_out" / "_tn8" / "cand"
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
ORDER = ["stt2", "stt4", "stt11", "Chr", "L83", "KVK", "L16", "TK", "B18", "B34"]
HAS_TRUTH = {"B18", "B34", "L16", "TK"}

# tốc độ s/ảnh: đọc từ log thật (không gõ tay)
LOG = REPO / "logs" / "tn11_p02_gen.log"


def log_speeds():
    pat = re.compile(r"\((\d+\.\d+)s/img")
    run = [float(m.group(1)) for line in LOG.read_text(errors="ignore").splitlines() for m in [pat.search(line)] if m]
    fin = [float(m.group(1)) for line in LOG.read_text(errors="ignore").splitlines()
           for m in [re.search(r"FontDiffusion: \d+ images in [\d.]+s \((\d+\.\d+)s/img\)", line)] if m]
    return dict(n_dong_chay=len(run), min_chay=min(run), max_chay=max(run), tb_cuoi=fin[-1] if fin else None,
                dong_dau=run[0])


def fd_store_chars():
    """(chữ có tệp tên U+XXXX.png, chữ có ẢNH THẬT >= 1 KB) trong gannhanocr-fd + ArcFace/data/glyphs.
    Tệp < 1 KB là con trỏ git-LFS chưa tải (mã Glyphs/signals_img bỏ qua) -> KHÔNG phải ảnh."""
    named, real = set(), set()
    for d in ("gannhanocr-fd", "ArcFace/data/glyphs"):
        for root, _, fs in os.walk(REPO / d):
            for fn in fs:
                if fn.startswith("U+") and fn.endswith(".png"):
                    try:
                        ch = chr(int(fn[2:-4], 16))
                    except ValueError:
                        continue
                    named.add(ch)
                    if os.path.getsize(os.path.join(root, fn)) >= 1024:
                        real.add(ch)
    return named, real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sp = log_speeds()
    named, store = fd_store_chars()
    res = dict(toc_do_log=sp, kho_fd_chu_co_ten_tep=len(named), kho_fd_chu_co_anh_that=len(store), topk=a.topk, bo={})
    all_chars = set()
    tot = dict(o=0, dong=0, chu_khac_nhau_tong_theo_bo=0, chu_hiem_topk_tong=0)
    for b in ORDER:
        cells = pd.read_pickle(CAND / f"{b}_cells.pkl")
        F = pd.read_pickle(CAND / f"{b}.pkl")
        n_cells = int(len(cells))
        per = F.groupby("i").size()
        uc = set(F.c.unique())
        all_chars |= uc
        d = dict(o=n_cells, o_co_dong_ung_vien=int(per.size), dong=int(len(F)),
                 ung_vien_moi_o_tb=round(float(len(F) / n_cells), 2), ung_vien_moi_o_trung_vi=float(per.median()),
                 ung_vien_p10=float(per.quantile(0.10)), ung_vien_p90=float(per.quantile(0.90)),
                 nR_tb=round(float(cells.nR.mean()), 2) if "nR" in cells else None,
                 chu_khac_nhau=len(uc), chu_co_anh_FD_that=len(uc & store),
                 chu_ten_tep_FD_nhung_la_con_tro_LFS=len((uc & named) - store))
        if b in HAS_TRUTH:
            # thiết kế p02: ô có chữ đúng ∈ ứng viên, chữ đúng có < 2 ô neo cùng sách ("chữ hiếm"); top-K theo f_vW
            hy = F.groupby("i").y.max()
            tru = F[F.y == 1].drop_duplicates("i").set_index("i")
            rare = tru.index[tru.n_self.fillna(0) < 2]
            S = F[F.i.isin(rare)].copy()
            S["rk"] = S.f_vW.fillna(-9)
            top = S.sort_values(["i", "rk"], ascending=[True, False]).groupby("i").head(a.topk)
            d["o_chu_hiem"] = int(len(rare))
            d["chu_can_sinh_topk_cho_o_hiem"] = int(top.c.nunique())
            d["dong_topk_cho_o_hiem"] = int(len(top))
            tot["chu_hiem_topk_tong"] += int(top.c.nunique())
        res["bo"][b] = d
        tot["o"] += n_cells
        tot["dong"] += int(len(F))
        tot["chu_khac_nhau_tong_theo_bo"] += len(uc)
        print(f"[{b}] ô {n_cells:>6}  dòng {len(F):>8}  ƯV/ô {len(F) / n_cells:5.1f}  chữ khác nhau {len(uc):>6}  "
              f"(có ảnh FD thật {len(uc & store)}; chỉ tên tệp {len((uc & named) - store)})" + (f"  ô hiếm {d['o_chu_hiem']}, chữ cần sinh top{a.topk} {d['chu_can_sinh_topk_cho_o_hiem']}" if b in HAS_TRUTH else ""), flush=True)
        del F, cells
    res["tong"] = dict(tot, chu_khac_nhau_hop_10_bo=len(all_chars), hop_co_anh_FD_that=len(all_chars & store),
                                           hop_chi_ten_tep_LFS=len((all_chars & named) - store))
    # chi phí (giờ) = số ảnh × s/ảnh / 3600
    sec = {"log_dong_dau_5.1": sp["dong_dau"], "log_tb_cuoi_8.2": sp["tb_cuoi"], "log_dinh_11.0": 11.0,
           "tay_5.7_(phien_hien_tai)": 5.7, "T4_README_1.7": 1.7}
    n_naive = round(tot["o"] * 3.5)
    cp = {}
    for name, s in sec.items():
        cp[name] = dict(s_moi_anh=s,
                        gia_thiet_bao_cao_o_x3_5_UV=round(n_naive * s / 3600),
                        gia_thiet_bao_cao_o_x4_UV=round(tot["o"] * 4 * s / 3600),
                        dong_that_o_x_UV=round(tot["dong"] * s / 3600),
                        chu_khac_nhau_moi_sach_cong_10_bo=round(tot["chu_khac_nhau_tong_theo_bo"] * s / 3600),
                        chu_khac_nhau_hop_10_bo_1_phong_cach=round(len(all_chars) * s / 3600),
                        chu_hiem_topk_4_bo_co_su_that=round(tot["chu_hiem_topk_tong"] * s / 3600, 1))
    res["so_anh"] = dict(bao_cao_o_x3_5=n_naive, bao_cao_o_x4=tot["o"] * 4, dong_that=tot["dong"],
                         chu_khac_nhau_moi_sach_cong_10_bo=tot["chu_khac_nhau_tong_theo_bo"],
                         chu_khac_nhau_hop_10_bo=len(all_chars), chu_hiem_topk_4_bo_co_su_that=tot["chu_hiem_topk_tong"])
    res["gio_gpu"] = cp
    (OUT / "chi_phi_sinh_anh.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(tong=res["tong"], so_anh=res["so_anh"], toc_do=sp), ensure_ascii=False, indent=1))
    print(json.dumps(cp, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
