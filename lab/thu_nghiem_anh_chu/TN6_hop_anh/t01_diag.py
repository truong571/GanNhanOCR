"""TN6 t01 — chẩn đoán lỗi vị trí của box_decoder hiện hành (pitch) trên Borg: trong cột vs sai cột (0 API, không mở ảnh)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import tn6lib as T
import pandas as pd
OUT = T.REPO / "measure_out/_tn6/diag"; OUT.mkdir(parents=True, exist_ok=True)
res = {}
for book in T.BOOKS:
    L = T.rd(T.REPO / f"prepared/_auto/{book}/dataset_out/labels_gated.csv")
    J = T.join(book, L)
    r = {"all": T.slot_block(J)}
    for tier in ("GOLD", "SYLLABLE", "REVIEW", "QUARANTINE", "GOLD_text_only"):
        r[tier] = T.slot_block(J, J.tier == tier)
    # theo |n_det - n_qn| và count_source
    J["dn"] = (pd.to_numeric(J.n_det, errors="coerce") - pd.to_numeric(J.n_qn, errors="coerce")).abs()
    r["by_dn"] = {str(k): T.slot_block(J, J.dn.clip(upper=4) == k)["slot"] for k in range(5)}
    r["by_count_source"] = {k: T.slot_block(J, J.count_source == k) for k in J.count_source.unique()}
    r["by_box_source"] = {k: T.slot_block(J, J.box_source == k)["slot"] for k in J.box_source.unique()}
    res[book] = r
    J.to_pickle(OUT / f"{book}_pitch_join.pkl")
json.dump(res, open(OUT / "t01.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, default=str)[:3500])
