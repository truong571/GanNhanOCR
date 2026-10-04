"""v_pb_nguon_goc_09 (04/10) — PHẢN BIỆN: tổng hợp nhiễu lấy mẫu bằng HÀM GỐC p05 ở hạt giống KHÁC (10-19) + so với số của người kiểm chứng (hạt giống 0-9).

Đầu vào: chay_goc_s10_19.json (script 01: hàm gốc, vá hash), ../nguon_goc/nhieu_mau_s0_9.json (người kiểm chứng), bảng gốc p05.
Tính: TB/SD/min/max Top-1 theo HQC; số hạt giống có khoảng max-min >= của bảng gốc; số hạt giống tái hiện thứ tự HQC của bảng gốc;
best-of-3 so với HQC1; gộp ô duy nhất + McNemar chính xác (HQC2/HQC3 so HQC1) cho (a) chỉ hạt giống 10-19, (b) cả 20 hạt giống.
Ra: measure_out/_tn11/verify/pb_nguon_goc/tong_hop_s10_19.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[3]
V = REPO / "measure_out/_tn11/verify"
mine = json.loads((V / "pb_nguon_goc/chay_goc_s10_19.json").read_text(encoding="utf-8"))
theirs = json.loads((V / "nguon_goc/nhieu_mau_s0_9.json").read_text(encoding="utf-8"))
orig = {r["book"]: r for r in json.loads((REPO / "measure_out/_tn11/p05_full_corpus/ket_qua_full_10_bo.json").read_text(encoding="utf-8"))}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def order(v):
    return ">".join(f"HQC{k + 1}" for k in sorted(range(3), key=lambda k: (-v[k], k)))


def mcn(a, b):
    """a, b: dict cell->acc. trả (hiệu a-b tính %, bAdungBsai, AsaiBdung, p)"""
    cells = sorted(set(a) & set(b))
    x = np.array([a[c] for c in cells]); y = np.array([b[c] for c in cells])
    bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
    p = float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0
    return round(float((x.mean() - y.mean()) * 100), 1), bb, cc, round(p, 4), len(cells)


res = {}
cells_mine = {b: {} for b in ("B34", "B18", "L16", "TK")}
for r in mine["runs"]:
    b = r["book"]
    for c in r["cells"]:
        cells_mine[b].setdefault(c["cell_id"], c)
by = {b: [] for b in cells_mine}
for r in mine["runs"]:
    by[r["book"]].append(r)
pool_all = {b: {"1": {}, "2": {}, "3": {}} for b in cells_mine}
for b, runs in by.items():
    runs = sorted(runs, key=lambda r: r["seed"])
    T1 = np.array([[r["res"][k]["top1"] for k in ("hqc1_white_canvas", "hqc2_book_paper_canvas", "hqc3_canonical_normalized")] for r in runs])
    n = [r["res"]["n_cells"] for r in runs]
    o = orig[b]
    ov = [o[k]["top1"] for k in ("hqc1_white_canvas", "hqc2_book_paper_canvas", "hqc3_canonical_normalized")]
    sp_o = max(ov) - min(ov)
    spreads = T1.max(1) - T1.min(1)
    ord_o = order(ov)
    n_ord = sum(order(list(row)) == ord_o for row in T1)
    th = theirs[b]
    T0 = np.array([[x["top1"]["1"], x["top1"]["2"], x["top1"]["3"]] for x in th])
    res[b] = {
        "seeds": [r["seed"] for r in runs], "n_sau_loc_tb": round(float(np.mean(n)), 1),
        "top1_tb_s10_19": [round(float(x), 1) for x in T1.mean(0)], "top1_sd_s10_19": [round(float(x), 1) for x in T1.std(0, ddof=1)],
        "top1_min": [float(x) for x in T1.min(0)], "top1_max": [float(x) for x in T1.max(0)],
        "top1_tb_s0_9_(nguoi_kiem_chung)": [round(float(x), 1) for x in T0.mean(0)], "top1_sd_s0_9": [round(float(x), 1) for x in T0.std(0, ddof=1)],
        "bang_goc": ov, "khoang_chenh_goc": round(sp_o, 1),
        "so_hat_giong_khoang_chenh>=goc_s10_19": int((spreads >= sp_o - 1e-9).sum()), "so_hat_giong_khoang_chenh>=goc_s0_9": int(((T0.max(1) - T0.min(1)) >= sp_o - 1e-9).sum()),
        "thu_tu_goc": ord_o, "so_hat_giong_dung_thu_tu_goc_s10_19": int(n_ord), "so_hat_giong_dung_thu_tu_goc_s0_9": int(sum(order(list(row)) == ord_o for row in T0)),
        "best_of_3_tru_HQC1_s10_19": round(float((T1.max(1) - T1[:, 0]).mean()), 1), "best_of_3_tru_HQC1_s0_9": round(float((T0.max(1) - T0[:, 0]).mean()), 1),
        "max_HQC1_cac_hat_giong_s10_19": float(T1[:, 0].max()),
    }
    # gộp ô duy nhất
    for r in runs:
        for c in r["cells"]:
            for k in "123":
                pool_all[b][k].setdefault(c["cell_id"], c[f"acc_hqc{k}"])
    pa = {k: pool_all[b][k] for k in "123"}
    res[b]["gop_o_duy_nhat_s10_19"] = {"n": len(pa["1"]), "top1": [round(float(np.mean(list(pa[k].values()))) * 100, 1) for k in "123"],
                                        "HQC2-HQC1": mcn(pa["2"], pa["1"]), "HQC3-HQC1": mcn(pa["3"], pa["1"])}
    # gộp cả 20 hạt giống (ô của người kiểm chứng + của tôi)
    pb = {k: dict(pa[k]) for k in "123"}
    for x in th:
        for rec in x["records"]:
            for k in "123":
                pb[k].setdefault(rec["cell_id"], rec[f"acc{k}"])
    res[b]["gop_o_duy_nhat_s0_19"] = {"n": len(pb["1"]), "top1": [round(float(np.mean(list(pb[k].values()))) * 100, 1) for k in "123"],
                                       "HQC2-HQC1": mcn(pb["2"], pb["1"]), "HQC3-HQC1": mcn(pb["3"], pb["1"])}
    pool_all[b] = {"a": pa, "b": pb}

for tag, key in (("s10_19", "a"), ("s0_19", "b")):
    P = {k: {} for k in "123"}
    for b in cells_mine:
        for k in "123":
            for c, v in pool_all[b][key][k].items():
                P[k][(b, c)] = v
    res[f"gop_4_bo_{tag}"] = {"n": len(P["1"]), "top1": [round(float(np.mean(list(P[k].values()))) * 100, 1) for k in "123"],
                              "HQC2-HQC1": mcn(P["2"], P["1"]), "HQC3-HQC1": mcn(P["3"], P["1"]), "HQC3-HQC2": mcn(P["3"], P["2"])}
(V / "pb_nguon_goc/tong_hop_s10_19.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
for b in cells_mine:
    print(b, json.dumps({k: v for k, v in res[b].items() if k in ("n_sau_loc_tb", "top1_tb_s10_19", "top1_sd_s10_19", "top1_tb_s0_9_(nguoi_kiem_chung)", "top1_sd_s0_9", "bang_goc", "so_hat_giong_khoang_chenh>=goc_s10_19", "so_hat_giong_khoang_chenh>=goc_s0_9", "so_hat_giong_dung_thu_tu_goc_s10_19", "so_hat_giong_dung_thu_tu_goc_s0_9", "best_of_3_tru_HQC1_s10_19", "max_HQC1_cac_hat_giong_s10_19")}, ensure_ascii=False))
    print("   gop s10_19", res[b]["gop_o_duy_nhat_s10_19"], "\n   gop s0_19", res[b]["gop_o_duy_nhat_s0_19"])
print("GOP 4 BO s10_19", res["gop_4_bo_s10_19"])
print("GOP 4 BO s0_19", res["gop_4_bo_s0_19"])
