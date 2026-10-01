"""Kiểm dấu hiệu "kim đã học IHR-NomDB": độ đúng chữ kim (thô, tại ô) trên câu thuộc tập TRAIN vs VAL của IHR-NomDB
(data/<Bộ>/manifest.tsv cột split) — 0 API. Nếu kim học trên train của IHR (hoặc NomNaOCR cùng chia) thì train >> val.
Ghép ô (measure_out/<Bộ>/ihr_endtoend/cells.csv: page_id, column, part) với câu manifest (page_id, col_index, part); chọn độ
lệch cột cho tỉ lệ khớp chữ người cao nhất (kiểm ánh xạ). CI 95 % bootstrap cụm trang cho chênh lệch train − val.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.gold_exact.common import Assets, set_lexicon, var_eq_plus  # noqa: E402
set_lexicon(Assets())

def run(b):
    M = pd.read_csv(REPO / f"data/{b}/manifest.tsv", sep="\t", dtype=str, keep_default_na=False)
    C = pd.read_csv(REPO / f"measure_out/{b}/ihr_endtoend/cells.csv", dtype=str, keep_default_na=False)
    C = C[C["gt"] != ""].copy()
    best = None
    for off in range(-2, 3):
        C["ci"] = (C.column.astype(int) + off).astype(str)
        J = C.merge(M[["page_id", "col_index", "part", "split", "nom_text"]], left_on=["page_id", "ci", "part"],
                    right_on=["page_id", "col_index", "part"], how="left")
        ok = np.mean([isinstance(t, str) and g in t for g, t in zip(J["gt"], J.nom_text)])
        if best is None or ok > best[0]:
            best = (ok, off, J)
    ok, off, J = best
    J = J[J.split.isin(["train", "val"])].copy()
    J["kim_strict"] = (J.ocr_char == J["gt"])
    J["kim_v1p"] = [bool(o) and bool(var_eq_plus(o, g)) for o, g in zip(J.ocr_char, J["gt"])]
    res = {"ghép_khớp_chữ": round(ok, 4), "lệch_cột": off}
    for s in ("train", "val"):
        g = J[J.split == s]
        res[s] = dict(n_o=len(g), n_trang=g.page_id.nunique(), strict=round(g.kim_strict.mean() * 100, 2),
                      v1p=round(g.kim_v1p.mean() * 100, 2))
    rng = np.random.default_rng(0)
    pages = J.page_id.unique()
    by = {p: g for p, g in J.groupby("page_id")}
    d = []
    for _ in range(2000):
        S = pd.concat([by[p] for p in rng.choice(pages, len(pages))])
        a, v = S[S.split == "train"].kim_strict.mean(), S[S.split == "val"].kim_strict.mean()
        d.append((a - v) * 100)
    res["train_tru_val_strict"] = dict(diem=round(res["train"]["strict"] - res["val"]["strict"], 2),
                                       ci95=[round(x, 2) for x in np.percentile(d, [2.5, 97.5])])
    return res

if __name__ == "__main__":
    import json
    out = {b: run(b) for b in ("LucVanTien1916", "TruyenKieu1872")}
    Path(__file__).with_name("kim_train_val_ihr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
