"""v_truy_nguon_03 (03/10) — SO CÙNG ĐIỀU KIỆN trên 4 bộ có nhãn người (B18, B34: Borg; L16, TK: IHR). 0 API, CPU, chỉ ĐỌC.

Cùng giao thức với p01/TN8: ô = mọi ô có chữ đúng trong tập ứng viên (~26/ô); Top-1 = ứng viên điểm cao nhất của MỘT tín hiệu đúng chữ người;
"chữ hiếm" = chữ đúng có n_self < 2 ô neo cùng sách ở 4 khối trang khác. Tín hiệu: f_font, f_fd, f_ridge (p01), f_self, f_self_or_fd, và các tín
hiệu SẴN CÓ trong pipeline (f_head, f_vW, f_vP; f_hum chỉ Borg) + BỘ CHỌN TN8 đầy đủ (LOBO; lượt "gốc" và lượt "thêm f_map = f_ridge").
CI 95 % = bootstrap cụm trang (tn8lib.boot_ci_pages); chênh lệch ghép cặp = bootstrap cụm trang trên hiệu từng ô.
Kiểm tái lập: số tính lại phải trùng p01_<bộ>.json (|Δ| <= 0,05) và p05_ket_qua.json.
Ra: measure_out/_tn11/verify/truy_nguon/cung_dieu_kien.json
"""
from __future__ import annotations

import json
import math
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
OUT.mkdir(parents=True, exist_ok=True)
TN11 = REPO / "measure_out" / "_tn11"
PAIR = {"B18": "B34", "B34": "B18", "L16": "TK", "TK": "L16"}   # bộ thử -> bộ học (LOBO)
REPORT_SAU = {"B18": (70.3, 78.5), "B34": (61.9, 72.2), "L16": (94.7, None), "TK": (89.5, None)}   # (Sau, "tự học")
REPORT_TRUOC = {"B18": 43.0, "B34": 34.6, "L16": 49.8, "TK": 48.2}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def boot_diff(a, b, pages, B=T.BOOT_B, seed=T.BOOT_SEED):
    d = np.asarray(a, float) - np.asarray(b, float)
    up, inv = np.unique(pages, return_inverse=True)
    k = np.bincount(inv, weights=d, minlength=len(up)); n = np.bincount(inv, minlength=len(up)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(up), size=(B, len(up)))
    r = k[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(d.mean()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def top1_ok(F, col):
    """y của ứng viên có điểm `col` cao nhất mỗi ô (NaN -> -9; hòa: dòng đầu) — đúng như p01.top1."""
    x = F[["i", col, "y"]].copy(); x[col] = x[col].fillna(-9)
    return x.loc[x.groupby("i")[col].idxmax(), ["i", "y"]].set_index("i").y


def fmt(r):
    return f"{r['p']:6.2f} [{r['lo']:5.2f},{r['hi']:5.2f}]"


def rate(ok, pages):
    m, lo, hi = T.boot_ci_pages(np.asarray(ok, float), pages)
    return dict(p=round(100 * m, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2), n=int(len(ok)))


def main():
    p05 = json.loads((TN11 / "p05_ket_qua.json").read_text(encoding="utf-8"))
    res = {}
    for b in ("B18", "B34", "L16", "TK"):
        tr = PAIR[b]
        D = T.load_base(b)
        page_of = D.page.values
        cols = ["i", "c", "y", "n_self", "f_font", "f_fd", "f_self", "f_head", "f_vW", "f_vP", "f_hum"]
        F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[cols]
        S = pd.read_pickle(TN11 / f"p01_{b}_scores.pkl")[["i", "c", "f_shift", "f_ridge"]]
        F = F.merge(S, on=["i", "c"], how="left")
        hy = F.groupby("i").y.max(); cells = hy[hy == 1].index
        F = F[F.i.isin(cells)].copy()
        F["f_self_or_fd"] = F.f_self.where(F.f_self.notna(), F.f_fd)
        tru = F[F.y == 1].drop_duplicates("i").set_index("i")
        rare_idx = set(tru.index[tru.n_self.fillna(0) < 2])
        # --- bộ chọn TN8 (3 phiên bản) ---
        P0 = pd.read_pickle(T.OUT / "pred" / f"{b}__from_{tr}.pkl"); P0 = P0[P0.any_y == 1].set_index("i")
        Pg = pd.read_pickle(TN11 / "tn8_map" / "pred" / f"{b}__from_{tr}__goc.pkl"); Pg = Pg[Pg.any_y == 1].set_index("i")
        Pm = pd.read_pickle(TN11 / "tn8_map" / "pred" / f"{b}__from_{tr}__map.pkl"); Pm = Pm[Pm.any_y == 1].set_index("i")
        common = np.array(sorted(set(cells) & set(P0.index) & set(Pg.index) & set(Pm.index)))
        rare_c = np.array([i in rare_idx for i in common])
        pages_c = page_of[common]
        out = dict(o_co_su_that_p01=int(len(cells)), o_co_su_that_bo_chon=int(len(P0)), o_chung=int(len(common)),
                   chu_hiem_chung=int(rare_c.sum()), tin_hieu={})
        meths = ["f_font", "f_fd", "f_shift", "f_ridge", "f_self", "f_self_or_fd", "f_head", "f_vW", "f_vP"] + (["f_hum"] if b in ("B18", "B34") else [])
        okmap = {}
        for m in meths:
            t = top1_ok(F, m)
            ok = t.reindex(common).values.astype(float)
            okmap[m] = ok
            out["tin_hieu"][m] = dict(tat_ca=rate(ok, pages_c), chu_hiem=rate(ok[rare_c], pages_c[rare_c]),
                                      p01_tat_ca_toan_bo=round(100 * float(t.mean()), 1))
        for name, P in (("bo_chon_TN8_goc_luot_cu", P0), ("bo_chon_TN8_goc_tn8_map", Pg), ("bo_chon_TN8_them_f_map", Pm)):
            ok = P.y1.reindex(common).values.astype(float)
            okmap[name] = ok
            out["tin_hieu"][name] = dict(tat_ca=rate(ok, pages_c), chu_hiem=rate(ok[rare_c], pages_c[rare_c]),
                                         toan_bo_n=int(len(P)), toan_bo_p=round(100 * float(P.y1.mean()), 2))
        # --- chênh ghép cặp ---
        d = {}
        for nm, (a, c) in {"them_f_map_tru_goc": ("bo_chon_TN8_them_f_map", "bo_chon_TN8_goc_tn8_map"),
                           "bo_chon_tru_f_ridge": ("bo_chon_TN8_goc_tn8_map", "f_ridge"),
                           "bo_chon_tru_f_self_or_fd": ("bo_chon_TN8_goc_tn8_map", "f_self_or_fd"),
                           "f_ridge_tru_f_fd": ("f_ridge", "f_fd"), "f_ridge_tru_f_font": ("f_ridge", "f_font"),
                           "f_ridge_tru_f_self": ("f_ridge", "f_self"), "f_self_tru_f_ridge": ("f_self", "f_ridge")}.items():
            m0, lo, hi = boot_diff(okmap[a], okmap[c], pages_c)
            mr, lor, hir = boot_diff(okmap[a][rare_c], okmap[c][rare_c], pages_c[rare_c])
            d[nm] = dict(tat_ca=[round(100 * m0, 2), round(100 * lo, 2), round(100 * hi, 2)],
                         chu_hiem=[round(100 * mr, 2), round(100 * lor, 2), round(100 * hir, 2)])
        out["chenh_ghep_cap_diem_phan_tram"] = d
        # --- đối chiếu số của báo cáo ---
        sel = out["tin_hieu"]["bo_chon_TN8_goc_luot_cu"]["tat_ca"]
        k = {}
        sau, tu_hoc = REPORT_SAU[b]
        k["truoc_bao_cao"] = REPORT_TRUOC[b]
        k["sau_bao_cao"] = sau
        if b in ("B18", "B34"):
            k["sau_la_f_ridge_tai_lap"] = out["tin_hieu"]["f_ridge"]["p01_tat_ca_toan_bo"]
            k["tu_hoc_bao_cao"] = tu_hoc
            k["tu_hoc_la_f_self_or_fd"] = out["tin_hieu"]["f_self_or_fd"]["p01_tat_ca_toan_bo"]
            k["f_self_thuan"] = out["tin_hieu"]["f_self"]["p01_tat_ca_toan_bo"]
        else:
            n05 = 19
            lo5, hi5 = wilson(round(sau * n05 / 100), n05)
            k["n_p05"] = n05; k["wilson95_sau"] = [round(lo5, 1), round(hi5, 1)]
            k["f_ridge_p01_26_ung_vien"] = out["tin_hieu"]["f_ridge"]["p01_tat_ca_toan_bo"]
            k["f_font_p01"] = out["tin_hieu"]["f_font"]["p01_tat_ca_toan_bo"]
        k["bo_chon_tn8_toan_bo_p"] = out["tin_hieu"]["bo_chon_TN8_goc_luot_cu"]["toan_bo_p"]
        k["sau_tru_bo_chon_diem"] = round(sau - k["bo_chon_tn8_toan_bo_p"], 2)
        if tu_hoc:
            k["tu_hoc_tru_bo_chon_diem"] = round(tu_hoc - k["bo_chon_tn8_toan_bo_p"], 2)
        out["doi_chieu_bao_cao"] = k
        # --- kiểm tái lập p01 / p05 ---
        J = json.loads((TN11 / f"p01_{b}.json").read_text(encoding="utf-8"))["top1"]
        out["tai_lap_p01"] = {m: dict(moi=out["tin_hieu"][m]["p01_tat_ca_toan_bo"], p01=J[m]["tat_ca"],
                                      khop=abs(out["tin_hieu"][m]["p01_tat_ca_toan_bo"] - J[m]["tat_ca"]) <= 0.051)
                              for m in ("f_font", "f_fd", "f_shift", "f_ridge", "f_self", "f_self_or_fd") + (("f_hum",) if b in ("B18", "B34") else ())}
        key = f"{b}<-{tr}"
        out["tai_lap_p05"] = dict(goc_p05=p05[key]["goc"]["tat_ca"], goc_tinh_lai=out["tin_hieu"]["bo_chon_TN8_goc_tn8_map"]["toan_bo_p"],
                                  map_p05=p05[key]["map"]["tat_ca"], map_tinh_lai=out["tin_hieu"]["bo_chon_TN8_them_f_map"]["toan_bo_p"])
        res[b] = out
        # --- in ---
        print(f"\n=== {b} (học {tr}): ô có sự thật p01 {out['o_co_su_that_p01']}, bộ chọn {out['o_co_su_that_bo_chon']}, chung {out['o_chung']}, chữ hiếm chung {out['chu_hiem_chung']}")
        print(f"  {'tín hiệu':26s} {'tất cả Top-1 [CI cụm trang]':26s} {'chữ hiếm Top-1 [CI]':26s}")
        for m, v in out["tin_hieu"].items():
            print(f"  {m:26s} {fmt(v['tat_ca']):26s} {fmt(v['chu_hiem']):26s}")
        print("  chênh ghép cặp (điểm %, tất cả | chữ hiếm):")
        for nm, v in d.items():
            print(f"    {nm:26s} {v['tat_ca'][0]:+6.2f} [{v['tat_ca'][1]:+6.2f},{v['tat_ca'][2]:+6.2f}] | {v['chu_hiem'][0]:+6.2f} [{v['chu_hiem'][1]:+6.2f},{v['chu_hiem'][2]:+6.2f}]")
        print("  đối chiếu báo cáo:", k)
        print("  tái lập p01 đủ khớp:", all(x["khop"] for x in out["tai_lap_p01"].values()), "| p05:", out["tai_lap_p05"])
    (OUT / "cung_dieu_kien.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
