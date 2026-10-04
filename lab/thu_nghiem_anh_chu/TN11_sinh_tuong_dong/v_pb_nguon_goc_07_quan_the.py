"""v_pb_nguon_goc_07 (04/10) — PHẢN BIỆN: số THẬT trên TOÀN QUẦN THỂ mà mẫu 20 ô của p05 đang ước lượng (HQC1/HQC2, giao thức p05: top-3 theo f_vW).

Dùng lại `per_cell_tables` của v_pb_nguon_goc_02 (mã tôi viết): với MỌI ô đích có chữ đúng trong top-3, tính Top-1 HQC1 và HQC2
(đúng công thức p05:259-267). Từ đó:
  - Top-1 quần thể HQC1, HQC2 (không còn nhiễu lấy mẫu) + McNemar chính xác HQC2 vs HQC1 trên cặp ô;
  - phân phối của "mẫu 20 ô kiểu p05" qua 100 000 hạt giống: Top-1 HQC1/HQC2 (TB, SD, phân vị); phân vị của số bảng gốc;
  - nếu có hạt giống gốc đã phục hồi (tim_seed_*.json) -> vị trí của mẫu gốc trong phân phối đó.
HQC3 KHÔNG tính ở đây (crop cắt lại phụ thuộc mẫu). CPU, 0 API.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .../v_pb_nguon_goc_07_quan_the.py --books B34 L16 TK B18 --tag human
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
sp = importlib.util.spec_from_file_location("v02", str(HERE / "v_pb_nguon_goc_02_tim_seed.py"))
V2 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(V2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    M = V2.load_p05()
    from pipeline.gold_exact import signals_img as SI
    from pipeline.gold_exact.common import Assets
    enc = SI.Scorers(Assets(), "cpu", M.EMB_DIR, False, lambda m: None).enc()
    orig = {r["book"]: r for r in json.loads(V2.ORIG.read_text(encoding="utf-8"))}
    recovered = {}
    for fp in glob.glob(str(OUT / "tim_seed_*.json")):
        for b, v in json.loads(Path(fp).read_text(encoding="utf-8")).items():
            ok = [h for h in v["hits"] if h["khop_cos_margin"]]
            if len(ok) >= 1:
                recovered[b] = ok

    from scipy.stats import binomtest
    res = {}
    for b in a.books:
        t0 = time.time()
        tc, vc, tabs, human = V2.per_cell_tables(M, b, enc, lambda s: print(s, flush=True))
        acc1, cos1, m1, acc2, cos2, m2 = tabs
        A1 = np.array([acc1[i] for i in vc]); A2 = np.array([acc2[i] for i in vc])
        n = len(vc)
        bb = int(((A1 == 1) & (A2 == 0)).sum()); cc = int(((A1 == 0) & (A2 == 1)).sum())
        p_mc = float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0
        tarr, row_of, vflag, arr = V2.build_arrays(tc, vc, tabs)
        size = min(20, len(tarr))
        d1 = []; d2 = []; ns = []
        for seed in range(100000):
            sel = np.sort(np.random.default_rng(seed).choice(tarr, size=size, replace=False))
            r = row_of[sel]; v = vflag[r]; rv = r[v]
            if len(rv) == 0:
                d1.append(np.nan); d2.append(np.nan); ns.append(0); continue
            d1.append(arr["A1"][rv].mean() * 100); d2.append(arr["A2"][rv].mean() * 100); ns.append(len(rv))
        d1 = np.array(d1); d2 = np.array(d2)
        o = orig[b]
        o1 = o["hqc1_white_canvas"]["top1"]; o2 = o["hqc2_book_paper_canvas"]["top1"]
        diff = d1 - d2
        res[b] = {
            "is_human": human, "n_target": len(tc), "n_valid_top3": n,
            "top1_quan_the": {"HQC1": round(float(A1.mean() * 100), 2), "HQC2": round(float(A2.mean() * 100), 2)},
            "mcnemar_HQC1dung_HQC2sai_b": bb, "HQC1sai_HQC2dung_c": cc, "p_mcnemar_chinh_xac": p_mc,
            "mau20_kieu_p05_100k_seed": {"HQC1_tb": round(float(np.nanmean(d1)), 2), "HQC1_sd": round(float(np.nanstd(d1)), 2),
                                           "HQC2_tb": round(float(np.nanmean(d2)), 2), "HQC2_sd": round(float(np.nanstd(d2)), 2),
                                           "n_hop_le_tb": round(float(np.mean(ns)), 2)},
            "bang_goc": {"n": o["n_cells"], "HQC1": o1, "HQC2": o2, "HQC3": o["hqc3_canonical_normalized"]["top1"]},
            "phan_vi_mau_goc": {"P(HQC1_mau>=goc)": round(float(np.nanmean(d1 >= o1 - 1e-9)), 4), "P(HQC2_mau>=goc)": round(float(np.nanmean(d2 >= o2 - 1e-9)), 4),
                                 "P(HQC1-HQC2_mau>=goc)": round(float(np.nanmean(diff >= (o1 - o2) - 1e-9)), 4)},
            "seed_goc_phuc_hoi": [h["seed"] for h in recovered.get(b, [])], "sec": round(time.time() - t0, 1),
        }
        print(b, json.dumps(res[b], ensure_ascii=False), flush=True)
        (OUT / f"quan_the_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
