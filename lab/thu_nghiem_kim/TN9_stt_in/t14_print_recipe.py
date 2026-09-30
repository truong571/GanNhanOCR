"""t14_print_recipe.py — sách in: GOLD đạt được theo công thức + độ đúng ước lượng (IHR tương tự, dị bản). 0 API.

Công thức (cộng dồn, trên ẢNH CHỤP nhãn trước khi agent khác dựng lại):
  P0  hiện tại (labels_gated)
  P1  + TN8 L1 (crop_bad/confusion_fix; số ô = TN8 summary.json, bộ chọn ảnh + ngưỡng LOBO)
  P2  + qn_count_unfixed (nhóm direct_qn: kim ∈ R, cột lệch đếm QN) có geo_ok (tâm crop trong hộp chữ kim) — nhãn = kim
  P3  + not_plausible (âm QN rác do OCR QN) có geo_ok — nhãn = kim, âm bỏ trống/đánh dấu    [chỉ khi dị bản xác nhận ≈ GOLD]
Độ đúng:
  IHR tương tự [ĐO trên L16/TK, không học]: NỘI DUNG (crop = chữ nhãn) và BỘ BA (+ âm QN người tại khe crop == âm hàng) của
  cùng nhóm/luật trên L16, TK (t10: print/<bộ>.pkl).
  Dị bản [ƯL] (L83/KVK, t12 theo CHỈ SỐ trong câu): tỉ lệ kim == dị bản của ô thêm / của GOLD.
Ra measure_out/_tn9/print_recipes.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402

TN8_L1 = {"Chr": 154, "L83": 1, "KVK": 41}          # measure_out/_tn8/KET_QUA.md §0/§5 (L1 + confusion_fix)


def ihr_rates():
    out = {}
    for b in ("L16", "TK"):
        D = pd.read_pickle(T.OUT / "print" / f"{b}.pkl")
        h = D.content != ""
        r = {}
        for nm, m in (("GOLD", D.grp == "GOLD"), ("direct_qn&geo", (D.grp == "direct_qn") & D.geo_ok),
                      ("notplaus&geo", (D.grp == "notplaus") & D.geo_ok), ("direct_crop", D.grp == "direct_crop")):
            s = m & h
            r[nm] = dict(n=int(s.sum()),
                         content=T.boot_ci_pages(D.kim_ok_content[s].values, D.page[s].values) if s.sum() else None,
                         triple=round(float(D.kim_triple_ok[s].mean()), 4) if s.sum() else None,
                         slot2=round(float(D.kim_ok_slot2[m].mean()), 4) if m.sum() else None)
        out[b] = r
    return out


def main():
    res = {"ihr_analog": ihr_rates()}
    print("[t14] IHR:", json.dumps(res["ihr_analog"], ensure_ascii=False, default=str)[:900], flush=True)
    for b in ("KVK", "L83", "Chr"):
        f = T.OUT / "print" / f"{b}_var.pkl"
        D = pd.read_pickle(f if f.exists() else T.OUT / "print" / f"{b}.pkl")
        N = len(D)
        g0 = int((D.tier == "GOLD").sum())
        dq = (D.grp == "direct_qn") & D.geo_ok
        npl = (D.grp == "notplaus") & D.geo_ok
        steps = [("P0", g0), ("P1", g0 + TN8_L1[b]), ("P2", g0 + TN8_L1[b] + int(dq.sum())),
                 ("P3", g0 + TN8_L1[b] + int(dq.sum()) + int(npl.sum()))]
        r = {"N": N, "steps": {k: dict(gold=v, share=round(v / N, 4)) for k, v in steps}}
        if "kim_eq_vref" in D:
            hv = D.vref != ""
            gm = (D.tier == "GOLD") & hv
            r["variant"] = dict(GOLD=[round(float(D.kim_eq_vref[gm].mean()), 4), int(gm.sum())],
                                GOLD_diffpos=[round(float(D.kim_eq_vref[gm & ~D.vsame.str.contains("1")].mean()), 4),
                                              int((gm & ~D.vsame.str.contains("1")).sum())],
                                direct_qn_geo=[round(float(D.kim_eq_vref[dq & hv].mean()), 4) if (dq & hv).any() else None, int((dq & hv).sum())],
                                notplaus_geo=[round(float(D.kim_eq_vref[npl & hv].mean()), 4) if (npl & hv).any() else None, int((npl & hv).sum())])
        res[b] = r
        print(f"[t14] {b}: {r}", flush=True)
    T.jdump(res, T.OUT / "print_recipes.json")


if __name__ == "__main__":
    main()
