"""t19_geo_variant.py — ĐO SAU khi cài luật TN9 (qn_geo / np_geo) trong pipeline.chon_chu: tỉ lệ trùng DỊ BẢN số hoá của ô mới
(L83, KVK) so với GOLD cũ — đúng phương pháp TN9 t12 (câu khớp auto_precision.match_verses ≥ 0,75 duy nhất; chữ dị bản theo CHỈ SỐ
trong câu, cả khi âm QN khác), cổng TN9: ô mới ≥ GOLD − 3 điểm. Tỉ lệ GOLD sau / dự phóng TN9 cho 3 sách. 0 API, chỉ đọc.
Ra: measure_out/_tn8/impl/geo_variant.json
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402

FULL = {"L83": "LucVanTien1883", "KVK": "KimVanKieu1884", "Chr": "Chrestomathie1872"}
TN9 = {"KVK": 0.9049, "L83": 0.7722, "Chr": 0.7435}


def main():
    import auto_precision as AP
    C = T.lex()
    out = {}
    for b, full in FULL.items():
        L = T.rd(T.REPO / f"prepared/{full}/dataset_out/labels_gated.csv")
        N = len(L)
        lev = L.chon_chu.values if "chon_chu" in L.columns else np.array([""] * N)
        r = dict(N=N, gold=int((L.tier == "GOLD").sum()), share=round(float((L.tier == "GOLD").mean()), 4), tn9=TN9[b],
                 qn_geo=int((lev == "qn_geo").sum()), np_geo=int((lev == "np_geo").sum()))
        if b in ("L83", "KVK"):
            cfg = dict(AP.CROSS_BOOKS[full]); cfg["refs"] = list(cfg["refs"])
            verses, _ = AP.match_verses(full, cfg, 0.75)
            idx = defaultdict(list)
            for v in verses:
                for i in range(v["n"]):
                    if i < len(v["ref_nom"]):
                        idx[(v["page"], int(v["column"]), v["syl_offset"] + i)].append(v["ref_nom"][i])
            col = [int(float(c)) for c in L.column]
            si = [int(float(s)) if s not in ("", None) else -1 for s in L.syl_idx]
            refs = [idx.get((p, c, s), []) for p, c, s in zip(L.page, col, si)]
            eq = np.array([bool(rr) and bool(l) and any(C.var_eq_plus(l, q) for q in rr) for l, rr in zip(L.label, refs)])
            has = np.array([bool(rr) for rr in refs])
            gold_old = (L.tier.values == "GOLD") & (lev == "")
            for nm, m in (("GOLD_cu", gold_old), ("qn_geo", lev == "qn_geo"), ("np_geo", lev == "np_geo")):
                s = m & has
                if s.sum():
                    pt, lo, hi = T.boot_ci_pages(eq[s], L.page.values[s])
                    r[f"dibien_{nm}"] = [round(pt, 4), [round(lo, 4), round(hi, 4)], int(s.sum())]
            g = r.get("dibien_GOLD_cu", [None])[0]
            r["cong_TN9_o_moi_ge_GOLD_tru_3"] = {nm: (None if f"dibien_{nm}" not in r or g is None else
                                                     bool(r[f"dibien_{nm}"][0] >= g - 0.03)) for nm in ("qn_geo", "np_geo")}
        out[b] = r
        print(f"[t19] {b}: {r}", flush=True)
    T.jdump(out, T.OUT / "impl" / "geo_variant.json")


if __name__ == "__main__":
    main()
