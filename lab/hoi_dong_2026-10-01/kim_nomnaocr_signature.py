"""Dấu vết NomNaOCR: ở các vị trí mà nhãn NomNaOCR (manifest cột nomnaocr_label) KHÁC nhãn IHR (nom_text), kim chép theo bên
nào? Nếu kim theo NomNaOCR nhiều hơn hẳn theo IHR → kim đã học chính các câu này qua NomNaOCR. 0 API.
Ghép ô (cells.csv: page_id, column, part, pos) với câu manifest (page_id, col_index = column − 1, part); ký tự thứ pos.
"""
import json
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
out = {}
for b in ("LucVanTien1916", "TruyenKieu1872"):
    M = pd.read_csv(REPO / f"data/{b}/manifest.tsv", sep="\t", dtype=str, keep_default_na=False)
    C = pd.read_csv(REPO / f"measure_out/{b}/ihr_endtoend/cells.csv", dtype=str, keep_default_na=False)
    C["ci"] = (C.column.astype(int) - 1).astype(str)
    J = C.merge(M[["page_id", "col_index", "part", "nom_text", "nomnaocr_label"]], left_on=["page_id", "ci", "part"],
                right_on=["page_id", "col_index", "part"], how="inner")
    n = theo_nna = theo_ihr = khac_ca_hai = 0
    vd = []
    for r in J.itertuples():
        p = int(r.pos)
        ihr, nna = r.nom_text, r.nomnaocr_label
        if not nna or len(nna) != len(ihr) or p >= len(ihr):
            continue
        a, c = ihr[p], nna[p]
        if a == c:
            continue
        n += 1
        if r.ocr_char == c:
            theo_nna += 1
            if len(vd) < 5:
                vd.append(f"IHR {a} / NomNaOCR {c} / kim {r.ocr_char}")
        elif r.ocr_char == a:
            theo_ihr += 1
        else:
            khac_ca_hai += 1
    out[b] = dict(vi_tri_NomNaOCR_khac_IHR=n, kim_theo_NomNaOCR=theo_nna, kim_theo_IHR=theo_ihr, kim_khac_ca_hai=khac_ca_hai,
                  vi_du=vd)
Path(__file__).with_name("kim_nomnaocr_signature.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
