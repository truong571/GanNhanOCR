"""TN6 x09 — vị trí hộp so với Ô THAM CHIẾU tự động của scripts/measure/box_ref_eval.py (ƯỚC LƯỢNG/proxy, thạch bản/mộc bản
lục bát): hộp TẦNG của kim cắt tại N−1 khe mực yếu nhất (pitch_decode.ink_cut_cells), chỉ tầng "verified" (số chữ kim của tầng
== số âm QN). ok = tâm bbox của ô (trang, cột, syl_idx) nằm trong ô tham chiếu cùng vị trí. Thiên lệch: ô 'ink_cut' của pitch
trùng tham chiếu THEO CẤU TẠO ⇒ báo thêm "honest" (bỏ ô nguồn ink_cut / vdp_pitch_ink_cut). Kiểm độ tin trên L16/TK (có khe
người, x05). 0 API, không mở ảnh bằng LLM.

  .venv/bin/python lab/thu_nghiem_anh_chu/TN6_hop_anh/x09_boxref.py --book LucVanTien1883 --variant visual_dp
"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "scripts/measure")]
import box_ref_eval as BR      # noqa: E402 (chỉ dùng ref_cells_for_page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--variant", required=True)
    a = ap.parse_args()
    lab = (REPO / f"prepared/{a.book}/dataset_out/labels_gated.csv" if a.variant == "current"
           else REPO / f"measure_out/_tn6/{a.book}/{a.variant}/dataset_out/labels_gated.csv")
    L = pd.read_csv(lab, dtype=str, keep_default_na=False)
    L = L[L.bbox.str.startswith("[")].reset_index(drop=True)
    pdir = REPO / "prepared" / a.book
    ok = np.full(len(L), np.nan)
    for pg, idx in L.groupby("page").groups.items():
        gray = cv2.imread(str(pdir / "pages" / f"{pg}.png"), cv2.IMREAD_GRAYSCALE)
        cf = pdir / "detected" / f"{pg}_ocr_cache.json"
        if gray is None or not cf.exists():
            continue
        cache = json.load(open(cf, encoding="utf-8"))
        tp = pdir / "transcriptions" / f"{pg}.json"
        trans = json.load(open(tp, encoding="utf-8")) if tp.exists() else {}
        cp = cache.get("col_pitch")
        cols = BR.ref_cells_for_page(gray, cache, trans, float(cp) if cp else None, kind="litho")
        ref = {}
        for c in cols:
            off = 0
            for t in c["tiers"]:
                if t["verified"] and len(t["cells"]) == t["n_ref"]:
                    for k, cell in enumerate(t["cells"]):
                        ref[(c["col"], off + k)] = cell
                off += t["n_ref"]
        for i in idx:
            try:
                key = (int(float(L.column[i])), int(float(L.syl_idx[i])))
            except ValueError:
                continue
            cell = ref.get(key)
            if cell is None:
                continue
            b = json.loads(L.bbox[i]); cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            ok[i] = float(cell[0] <= cx <= cell[2] and cell[1] <= cy <= cell[3])
    L["ref_ok"] = ok
    hon = ~L.box_source.isin(["ink_cut", "vdp_pitch_ink_cut"])
    out = dict(book=a.book, variant=a.variant, tiers=L.tier.value_counts().to_dict())
    for name, m in (("ALL", pd.Series(True, index=L.index)), ("GOLD", L.tier == "GOLD")):
        v = L.ref_ok[m].dropna(); h = L.ref_ok[m & hon].dropna()
        out[name] = dict(n=int(len(v)), ok=round(float(v.mean()), 4) if len(v) else None, n_honest=int(len(h)),
                         ok_honest=round(float(h.mean()), 4) if len(h) else None)
    od = REPO / f"measure_out/_tn6/{a.book}/{a.variant}/eval"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "tn6_boxref.json", "w"), ensure_ascii=False, indent=1)
    L[["page", "column", "nom_idx", "syl_idx", "tier", "box_source", "ref_ok"]].to_pickle(od / "boxref.pkl")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
