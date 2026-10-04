"""v_truy_nguon_07 (04/10) — BỘ CHỌN CHỮ TN8 ĐANG CHẠY trên ĐÚNG tập ô mà giao thức p05 giữ lại (chữ đúng nằm trong top-3 theo f_vW). 0 API, CPU, chỉ ĐỌC.

p05 chỉ tính các ô mà chữ đúng ∈ top-3 ứng viên theo f_vW (p05_full_corpus_he_quy_chieu.py:196-201) và cho mỗi ô chỉ 3 ứng viên. Bộ chọn TN8 chọn trong
~26 ứng viên (khó hơn) trên mọi ô. Script đo Top-1 của bộ chọn (pred LOBO, lượt gốc) trên (a) các ô p05 giữ lại, (b) các ô p05 loại bỏ (chữ đúng ∉ top-3),
(c) tất cả — để so TRỰC TIẾP với số HQC của báo cáo trên cùng quần thể ô. CI cụm trang.
Ra: measure_out/_tn11/verify/truy_nguon/bo_chon_tren_o_p05.json
"""
from __future__ import annotations

import json
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
PAIR = {"B18": "B34", "B34": "B18", "L16": "TK", "TK": "L16"}
P05_HQC1 = {"B18": 50.0, "B34": 66.7, "L16": 94.7, "TK": 89.5}
P05_N = {"B18": 18, "B34": 18, "L16": 19, "TK": 19}


def rate(ok, pages):
    m, lo, hi = T.boot_ci_pages(np.asarray(ok, float), pages)
    return dict(p=round(100 * m, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2), n=int(len(ok)))


def main():
    res = {}
    for b, tr in PAIR.items():
        D = T.load_base(b)
        F = pd.read_pickle(T.OUT / "cand" / f"{b}.pkl")[["i", "c", "y", "f_vW"]]
        hy = F.groupby("i").y.max(); cells = hy[hy == 1].index.values
        F = F[F.i.isin(cells)].copy()
        F["rk"] = F.f_vW.fillna(-9)
        S3 = F.sort_values(["i", "rk"], ascending=[True, False], kind="stable").groupby("i").head(3)
        keep = S3.groupby("i").y.max(); kept = keep[keep == 1].index.values; dropped = keep[keep == 0].index.values
        P = pd.read_pickle(T.OUT / "pred" / f"{b}__from_{tr}.pkl"); P = P[P.any_y == 1].set_index("i")
        pg = D.page.values
        out = {}
        for name, idx in (("o_p05_giu_lai", kept), ("o_p05_loai_bo", dropped), ("tat_ca", cells)):
            idx = np.array([i for i in idx if i in P.index])
            out[name] = rate(P.y1.reindex(idx).values, pg[idx])
        out["ty_le_giu_lai_pct"] = round(100 * len(kept) / len(cells), 2)
        out["hqc1_p05_bao_cao"] = dict(top1=P05_HQC1[b], n=P05_N[b])
        res[b] = out
        print(f"{b:4s} giữ lại {out['ty_le_giu_lai_pct']} %: bộ chọn TN8 Top-1 trên ô giữ lại {out['o_p05_giu_lai']['p']} [{out['o_p05_giu_lai']['lo']},{out['o_p05_giu_lai']['hi']}] (n={out['o_p05_giu_lai']['n']}) | "
              f"ô loại bỏ {out['o_p05_loai_bo']['p']} (n={out['o_p05_loai_bo']['n']}) | tất cả {out['tat_ca']['p']} | HQC1 p05 {P05_HQC1[b]} (n={P05_N[b]})")
    (OUT / "bo_chon_tren_o_p05.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
