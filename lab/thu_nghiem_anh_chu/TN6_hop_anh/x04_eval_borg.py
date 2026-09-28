"""TN6 x04 — đo một bản dựng Borg (labels_gated.csv) trên NHÃN NGƯỜI: đúng vị trí (tâm bbox ∈ hộp người keep_high+ / keep_v5),
nhãn đúng V1+ (borg_endtoend_eval.evaluate, ghi vào thư mục biến thể), số GOLD, đúng hai vế của GOLD. 0 API, không mở ảnh.

  .venv/bin/python lab/thu_nghiem_anh_chu/TN6_hop_anh/x04_eval_borg.py --book SachDungLyHoThan --variant visual_dp
  (--variant current = prepared/_auto/<S>/dataset_out/labels_gated.csv của bản giao hiện hành)
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import tn6lib as T
REPO = T.REPO


def run(book, variant):
    if variant == "current":
        lab = REPO / f"prepared/_auto/{book}/dataset_out/labels_gated.csv"
    else:
        lab = REPO / f"measure_out/_tn6/{book}/{variant}/dataset_out/labels_gated.csv"
    od = REPO / f"measure_out/_tn6/{book}/{variant}/eval"
    od.mkdir(parents=True, exist_ok=True)
    import borg_endtoend_eval as BE
    BE.evaluate(book, lab, od)
    C = T.rd(od / "cells.csv")
    L = T.rd(lab)
    L["cell_uid"] = [f"{book}/{b}/{p}/c{int(float(c))}/n{n}/s{s}" for b, p, c, n, s in zip(L.book, L.page, L.column, L.nom_idx, L.syl_idx)]
    J = T.join(book, L)
    J = J.merge(C[["cell_uid", "gt", "v1p"]], on="cell_uid", how="left")
    J["v1p"] = J.v1p == "True"
    keep = J.has_h & J.keep_level.isin(T.KEEP_OK)
    out = dict(book=book, variant=variant, labels=str(lab.relative_to(REPO)), n_rows=int(len(J)),
               tiers=J.tier.value_counts().to_dict(), box_source=J.box_source.value_counts().to_dict(),
               count_source=J.count_source.value_counts().to_dict())
    for name, m in (("all", pd.Series(True, index=J.index)), ("GOLD", J.tier == "GOLD"), ("SYLLABLE", J.tier == "SYLLABLE"),
                    ("REVIEW", J.tier == "REVIEW"), ("GOLD_text_only", J.tier == "GOLD_text_only")):
        blk = T.slot_block(J, m)
        g = J[m & (J["gt"] != "") & J["gt"].notna()]
        blk["n_gt"] = int(len(g)); blk["v1p"] = round(float(g.v1p.mean()), 4) if len(g) else None
        k2 = J[m & keep & (J["gt"] != "") & J["gt"].notna()]
        blk["n_both"] = int(len(k2)); blk["both"] = round(float((k2.v1p & k2.slot_ok).mean()), 4) if len(k2) else None
        out[name] = blk
    for bs in sorted(J.box_source.unique()):
        out.setdefault("slot_by_box_source", {})[bs] = T.slot_block(J, J.box_source == bs)["slot"]
    json.dump(out, open(od / "tn6_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
    J.to_pickle(od / "join.pkl")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--variant", required=True)
    a = ap.parse_args()
    o = run(a.book, a.variant)
    print(json.dumps({k: o[k] for k in ("book", "variant", "tiers", "all", "GOLD", "slot_by_box_source")}, ensure_ascii=False,
                     default=str))
