"""K5 · Mổ xẻ ô đổi giữa build văn bản thuần và build --visual-emission (toàn bộ 448 trang, --no-crops).
Đo: cặp bị phá/tạo theo tier_v3 (bên tương ứng), cột n_ocr≠n_qn, ô usable đổi ÂM (cùng nom_idx, syl_idx khác),
p_visual_syl của cặp mới, cặp từ điển (dict_support) bị phá/tạo, ô QĐ-01 liên quan.
    .venv/bin/python KhoiB/v3/b2_changed_cells.py <dir_txt> <dir_vis>  -> KhoiB/v3/b2_changed_cells.json
"""
import json, sys
from collections import Counter
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
T = pd.read_csv(Path(sys.argv[1]) / "labels.csv", dtype=str, keep_default_na=False)
V = pd.read_csv(Path(sys.argv[2]) / "labels.csv", dtype=str, keep_default_na=False)
K5 = ["book", "page", "column", "nom_idx", "syl_idx"]; K4 = K5[:4]
T["k5"] = T[K5].agg("|".join, axis=1); V["k5"] = V[K5].agg("|".join, axis=1)
T["k4"] = T[K4].agg("|".join, axis=1); V["k4"] = V[K4].agg("|".join, axis=1)
onlyT = T[~T.k5.isin(set(V.k5))]; onlyV = V[~V.k5.isin(set(T.k5))]
usable = lambda D: D.tier.isin(["GOLD", "SYLLABLE"])
T2 = T.set_index("k4"); V2 = V.set_index("k4")
common = T2.index.intersection(V2.index)
resyl = [k for k in common if T2.at[k, "syl_idx"] != V2.at[k, "syl_idx"]]
r_usable_both = [k for k in resyl if T2.at[k, "tier"] in ("GOLD", "SYLLABLE") and V2.at[k, "tier"] in ("GOLD", "SYLLABLE")]
r_usable_txt = [k for k in resyl if T2.at[k, "tier"] in ("GOLD", "SYLLABLE")]
cols = {tuple(k.split("|")[:3]) for k in list(onlyT.k5) + list(onlyV.k5)}
colinfo = T.drop_duplicates(["book", "page", "column"]).set_index(["book", "page", "column"])
n_ne = sum(1 for c in cols if colinfo.at[c, "n_ocr"] != colinfo.at[c, "n_qn"])
pv = pd.to_numeric(onlyV.p_visual_syl, errors="coerce")
rep = {"pairs_only_txt": len(onlyT), "pairs_only_vis": len(onlyV), "cols_changed": len(cols), "cols_n_ocr_ne_n_qn": n_ne,
       "destroyed_by_tier_v3_txt": dict(Counter(onlyT.tier_v3)), "created_by_tier_v3_vis": dict(Counter(onlyV.tier_v3)),
       "destroyed_usable_txt": int(usable(onlyT).sum()), "created_usable_vis": int(usable(onlyV).sum()),
       "destroyed_dict_pair_CHAR_AB": int(onlyT.tier_v3.isin(["CHAR_A", "CHAR_B"]).sum()), "created_dict_pair_CHAR_AB": int(onlyV.tier_v3.isin(["CHAR_A", "CHAR_B"]).sum()),
       "p_visual_new_pairs_p50": round(float(pv.median()), 4), "p_visual_new_pairs_pct_ge_0.5": round(float((pv >= 0.5).mean()), 4),
       "p_visual_new_pairs_pct_ge_0.9": round(float((pv >= 0.9).mean()), 4),
       "cells_same_nom_idx_diff_syl_idx": len(resyl), "cells_resyl_usable_txt": len(r_usable_txt), "cells_resyl_usable_both": len(r_usable_both),
       "cells_resyl_usable_both_by_tier_txt": dict(Counter(T2.loc[r_usable_both, "tier"])),
       "cells_resyl_qd01_locked_txt": int((T2.loc[resyl, "qd01_locked"] == "1").sum()),
       "qd01_locked": {"txt": int((T.qd01_locked == "1").sum()), "vis": int((V.qd01_locked == "1").sum())},
       "qd01_pending_vis": V[V.rule.str.contains("pending")][["book", "page", "column", "nom_idx", "syl_idx", "ocr_char", "syllable", "tier", "p_visual_syl", "visual_argmax", "visual_max_p"]].to_dict("records"),
       "usable": {"txt": int(usable(T).sum()), "vis": int(usable(V).sum())},
       "resyl_usable_both_sample": [{"k": k, "ocr": T2.at[k, "ocr_char"], "syl_txt": T2.at[k, "syllable"], "syl_vis": V2.at[k, "syllable"], "tier_txt": T2.at[k, "tier"], "tier_vis": V2.at[k, "tier"],
                                     "p_vis": V2.at[k, "p_visual_syl"], "argmax": V2.at[k, "visual_argmax"], "max_p": V2.at[k, "visual_max_p"]} for k in r_usable_both[:25]]}
json.dump(rep, open(REPO / "KhoiB/v3/b2_changed_cells.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "resyl_usable_both_sample"}, ensure_ascii=False, indent=1))
