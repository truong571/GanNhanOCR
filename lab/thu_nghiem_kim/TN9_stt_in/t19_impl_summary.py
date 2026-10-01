"""t19_impl_summary.py — số TRƯỚC / SAU khi cài đường STT hai lượt chặt (01/10). 0 API.
Trước = HEAD dataset_out/labels_final.csv (ảnh chụp measure_out/_tn9/before_impl/) + gold_exact trước (đếm ghi tay từ
dataset/_ALL/gold_exact.csv lúc chụp). Sau = dataset_out/labels_final.csv + dataset/_ALL/gold_exact.csv hiện tại.
Độ đúng chữ ước lượng [SĐ] = Σ_lớp n·dải_lớp (t08 lớp ẩn hai lần đọc, t05 một lần đọc) như t13/t17.
Ra measure_out/_tn9/impl_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402

GE_BEFORE = {"stt2": 2737, "stt4": 1828, "stt11": 3677}       # dataset/_ALL/gold_exact.csv lúc 08:3x 01/10 (trước merge mới)


def main():
    B = T.rd(T.OUT / "before_impl/labels_final_HEAD.csv")
    A = T.rd(T.REPO / "dataset_out/labels_final.csv")
    GE = T.rd(T.REPO / "dataset/_ALL/gold_exact.csv")
    GE = GE[GE.book_set == "SachThanhTruyen"]
    lat = json.loads((T.OUT / "latent2.json").read_text()); inr = json.loads((T.OUT / "inR_model.json").read_text())
    out = {}
    for bk in T.STT:
        b, a = B[B.book == bk], A[A.book == bk]
        comp = a[a.tier == "GOLD"].hai_luot.str.split(":").str[1].value_counts().to_dict()
        rng = lat[bk]["range"]; e = inr["stt"][bk]["lt1"]["est"]
        prec = {"agree": rng["agree_inR (lt1≡lt2 ∈ R)"], "lt2only": rng["lt2∈R, lt1∉R: lt2 đúng"],
                "lt1_nolt2": [min(v["prec_inR"] for k, v in e.items() if k in ("q=0.089", "q=0.117", "q=0.145")),
                              max(v["prec_inR"] for k, v in e.items() if k in ("q=0.089", "q=0.117", "q=0.145"))],
                "rescue": [0.40, 0.70]}
        n = sum(comp.values())
        lo = sum(v * prec[c][0] for c, v in comp.items()) / max(1, n); hi = sum(v * prec[c][1] for c, v in comp.items()) / max(1, n)
        ge = GE[GE.set8 == bk].gold_exact.value_counts().to_dict()
        out[bk] = dict(truoc=dict(o=len(b), gold=int((b.tier == "GOLD").sum()), ty_le=round(float((b.tier == "GOLD").mean()), 4),
                                  gold_exact_ok=GE_BEFORE[bk]),
                       sau=dict(o=len(a), gold=int((a.tier == "GOLD").sum()), ty_le=round(float((a.tier == "GOLD").mean()), 4),
                                gold_exact=ge, lop_gold=comp, do_dung_chu_SD=[round(lo, 4), round(hi, 4)]))
        print(f"[t19] {bk}: {out[bk]}", flush=True)
    T.jdump(out, T.OUT / "impl_summary.json")


if __name__ == "__main__":
    main()
