"""So hai bản build TOÀN BỘ (--no-crops): văn bản thuần vs --visual-emission (B-2, mô hình OOF thật).

Đo "trên hộp thật ≤≈50 ô đổi" (DANH_MUC B-2): cặp (book,page,column,nom_idx,syl_idx) chỉ có ở một bên,
tier_v3 của các ô đổi, tier_v3 crosstab hai bên, phân bố p_visual_syl theo tier_v3 (từ labels_trace_visual.csv),
và cột đổi đường ghép (summary.json visual_emission). KHÔNG đổi bộ giao nộp.

    .venv/bin/python KhoiB/v3/b2_compare_builds.py <dir_txt> <dir_vis> [--out KhoiB/v3/b2_build_compare.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

KEY = ["book", "page", "column", "nom_idx", "syl_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt"); ap.add_argument("vis")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "b2_build_compare.json"))
    ap.add_argument("--restrict-to-vis-pages", action="store_true",
                    help="K5: bên txt là bộ đầy đủ (vd dataset_out) — chỉ so trên (book,page) có ở bên vis (build --limit)")
    a = ap.parse_args()
    T = pd.read_csv(Path(a.txt) / "labels.csv", dtype=str, keep_default_na=False)
    V = pd.read_csv(Path(a.vis) / "labels.csv", dtype=str, keep_default_na=False)
    if a.restrict_to_vis_pages:
        pg = set(zip(V.book, V.page)); T = T[[(b, p) in pg for b, p in zip(T.book, T.page)]].reset_index(drop=True)
    sv = json.load(open(Path(a.vis) / "summary.json")).get("visual_emission")
    st = json.load(open(Path(a.txt) / "summary.json"))
    kT = T[KEY].agg("|".join, axis=1); kV = V[KEY].agg("|".join, axis=1)
    only_t = set(kT) - set(kV); only_v = set(kV) - set(kT)
    kc = ["book", "page", "column", "nom_idx"]
    cT = T[kc].agg("|".join, axis=1); cV = V[kc].agg("|".join, axis=1)
    # ô cùng nom_idx nhưng đổi syl_idx (thanh ghi trượt) vs ô mất/xuất hiện
    T2 = T.assign(_k=cT).set_index("_k"); V2 = V.assign(_k=cV).set_index("_k")
    common = T2.index.intersection(V2.index)
    resyl = [k for k in common if T2.at[k, "syl_idx"] != V2.at[k, "syl_idx"]]
    tier_change = [k for k in common if T2.at[k, "tier_v3"] != V2.at[k, "tier_v3"]]
    tier_final_change = [k for k in common if T2.at[k, "tier"] != V2.at[k, "tier"]]
    cols_changed = sorted({tuple(k.split("|")[:3]) for k in list(only_t) + list(only_v)})
    rep = {
        "n_rows": {"txt": len(T), "vis": len(V)},
        "summary_visual_emission": sv,
        "pairs_only_txt": len(only_t), "pairs_only_vis": len(only_v),
        "cells_same_nom_idx_diff_syl_idx": len(resyl),
        "cells_tier_v3_changed": len(tier_change),
        "cells_tier_changed": len(tier_final_change),
        "tier_v3_change_matrix": dict(Counter(f"{T2.at[k, 'tier_v3']}->{V2.at[k, 'tier_v3']}" for k in tier_change)),
        "tier_change_matrix": dict(Counter(f"{T2.at[k, 'tier']}->{V2.at[k, 'tier']}" for k in tier_final_change)),
        "columns_with_pair_change": len(cols_changed),
        "tier_v3_txt": dict(Counter(T.tier_v3)), "tier_v3_vis": dict(Counter(V.tier_v3)),
        "tier_txt": dict(Counter(T.tier)), "tier_vis": dict(Counter(V.tier)),
        "usable_txt": int(T.tier.isin(["GOLD", "SYLLABLE"]).sum()), "usable_vis": int(V.tier.isin(["GOLD", "SYLLABLE"]).sum()),
        "pass1b_txt": st.get("pass1b"), "pass1b_vis": json.load(open(Path(a.vis) / "summary.json")).get("pass1b"),
        "n_pages": {"txt": int(T[["book", "page"]].drop_duplicates().shape[0]), "vis": int(V[["book", "page"]].drop_duplicates().shape[0])},
        "changed_cells_sample": [{"key": k, "tier_v3_txt": T2.at[k, "tier_v3"], "tier_v3_vis": V2.at[k, "tier_v3"],
                                  "syl_txt": T2.at[k, "syllable"], "syl_vis": V2.at[k, "syllable"], "ocr_char": V2.at[k, "ocr_char"],
                                  "p_visual_syl_vis": (V2.at[k, "p_visual_syl"] if "p_visual_syl" in V.columns else "")} for k in (resyl + tier_change)[:40]],
    }
    if "p_visual_syl" in V.columns:
        p = pd.to_numeric(V.p_visual_syl, errors="coerce")
        rep["p_visual_syl_by_tier_v3"] = {t: {"n": int(len(g)), "n_has_p": int(g.notna().sum()),
                                              "p50": (round(float(g.median()), 4) if g.notna().any() else None),
                                              "pct_ge_0.9": (round(float((g >= 0.9).mean()), 4) if g.notna().any() else None),
                                              "pct_lt_0.05": (round(float((g < 0.05).mean()), 4) if g.notna().any() else None)}
                                          for t, g in p.groupby(V.tier_v3)}
        # argmax == âm ∧ p ≥ 0,9 theo tier_v3 (đối chiếu với B-3 từ crop bbox v3: hộp OCR thô thay bbox cuối)
        hit = (V.visual_argmax == V.syllable) & (p >= 0.9)
        rep["argmax_eq_syl_p_ge_0.9_by_tier_v3"] = {t: int(hit[V.tier_v3 == t].sum()) for t in sorted(V.tier_v3.unique())}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[b2 compare] dòng txt {len(T):,} / vis {len(V):,} | cặp chỉ txt {len(only_t)} · chỉ vis {len(only_v)} | "
          f"ô cùng nom_idx đổi syl_idx {len(resyl)} | ô đổi tier_v3 {len(tier_change)} {rep['tier_v3_change_matrix']} | "
          f"ô đổi tier {len(tier_final_change)} | cột có cặp đổi {len(cols_changed)} | usable {rep['usable_txt']:,} -> {rep['usable_vis']:,}")
    print(f"  summary vis: {sv}")
    if "p_visual_syl_by_tier_v3" in rep:
        print("  p_visual_syl theo tier_v3:", rep["p_visual_syl_by_tier_v3"])
        print("  argmax==âm ∧ p≥0,9 theo tier_v3 (hộp OCR thô):", rep["argmax_eq_syl_p_ge_0.9_by_tier_v3"])
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
