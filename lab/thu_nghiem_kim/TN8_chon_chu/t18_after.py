"""t18_after.py — ĐO SAU khi cài (labels_gated.csv đã qua (c') + chon_chu của lượt dựng lại): tỉ lệ GOLD + độ đúng (hai vế / chữ,
CI bootstrap cụm trang) trên 4 bộ có sự thật, tỉ lệ GOLD 3 sách in không sự thật, so với dự phóng TN8 (summary_t12_b80 + L5).
Sự thật chỉ để ĐO (như t00_base). 0 API. Ra: measure_out/_tn8/impl/after.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn8lib as T  # noqa: E402
import t00_base as B0  # noqa: E402

PROJ = {"B18": (0.9043, 0.9622, 0.9441), "B34": (0.6874, 0.9814, 0.967), "L16": (0.9547, 0.9712, 0.9817),
        "TK": (0.9745, 0.9858, 0.9886), "Chr": (0.5541, None, None), "L83": (0.7538, None, None), "KVK": (0.8791, None, None)}


def metrics(D: pd.DataFrame, C) -> dict:
    gold = D.tier.values == "GOLD"
    ok = np.array([bool(l) and bool(g) and C.var_eq_plus(l, g) for l, g in zip(D.label.values, D.gt_char.values)])
    has = D.gt_char.values != ""
    r = dict(N=int(len(D)), gold=int(gold.sum()), share=round(float(gold.mean()), 4))
    m = gold & has
    if m.sum():
        pt, lo, hi = T.boot_ci_pages(ok[m], D.page.values[m])
        r.update(text=round(pt, 4), text_ci=[round(lo, 4), round(hi, 4)], n_text=int(m.sum()))
    m = gold & has & D.pos_known.values
    if m.sum():
        pt, lo, hi = T.boot_ci_pages((ok & D.pos_ok.values)[m], D.page.values[m])
        r.update(both=round(pt, 4), both_ci=[round(lo, 4), round(hi, 4)], n_both=int(m.sum()))
    return r


def main():
    C = T.lex()
    out = {}
    for b in (sys.argv[1:] or ("B18", "B34", "L16", "TK", "Chr", "L83", "KVK")):
        D = B0.base_of(b)                      # đọc labels_gated.csv HIỆN TẠI (đã qua chon_chu)
        r = metrics(D, C)
        if "chon_chu" in D.columns or True:
            L = T.rd(T.REPO / T.BOOKS[b]["labels"])
            if "chon_chu" in L.columns:
                r["chon_chu"] = L.chon_chu[L.chon_chu != ""].value_counts().to_dict()
            if "crop_recheck" in L.columns:
                r["crop_recheck"] = L.crop_recheck[L.crop_recheck != ""].str.split(":").str[0].value_counts().to_dict()
        p = PROJ[b]
        r["tn8_du_phong"] = dict(share=p[0], both=p[1], text=p[2])
        out[b] = r
        print(f"[t18] {b}: GOLD {r['gold']:,}/{r['N']:,} = {r['share']:.4f} (TN8 {p[0]}) | hai vế {r.get('both')} {r.get('both_ci')} "
              f"(TN8 {p[1]}) | chữ {r.get('text')} {r.get('text_ci')} (TN8 {p[2]}) | {r.get('chon_chu')} | (c') {r.get('crop_recheck')}",
              flush=True)
    f = T.OUT / "impl" / "after.json"
    old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    old.update(out)
    T.jdump(old, f)


if __name__ == "__main__":
    main()
