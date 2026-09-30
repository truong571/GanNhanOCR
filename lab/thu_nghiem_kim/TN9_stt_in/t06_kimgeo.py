"""t06_kimgeo.py — thước VỊ TRÍ độc lập ảnh-model: chữ kim tại TÂM crop ("kim_geo") có == nhãn không. 0 API.

kim trả hộp DÒNG + chuỗi; hộp chữ = chia đều hộp dòng (core.ocr.ocr_api.boxes_to_columns). Với ô (bbox crop), kim_geo = chữ
kim có hộp chia đều chứa tâm crop (cùng cột theo x: tâm x trong [x1, x2] của hộp chữ; theo y: trong [y1, y2]); không có -> ''.
Crop đúng chỗ ⇒ kim_geo thường == chữ kim đã ghép với âm (DP) ⇒ == nhãn. Crop trượt ±1 ⇒ kim_geo = chữ láng giềng.
Hiệu chuẩn trên Borg (hộp người, tn6lib.join: slot_ok): P(đúng vị trí | kim_geo == nhãn) và P(đúng vị trí | ≠).
STT: tỉ lệ kim_geo == nhãn trên GOLD theo bản dựng (legacy / visual_dp) và theo lt1 / lt2.
Ra measure_out/_tn9/kimgeo/<tên>.pkl + kimgeo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tn9lib as T  # noqa: E402

OD = T.OUT / "kimgeo"


def page_chars(cache: dict):
    rows = []
    for col in cache.get("columns") or []:
        for c in col:
            b = c.get("bbox")
            if b and c.get("char"):
                rows.append((c["char"], float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    if not rows:
        return None
    ch = np.array([r[0] for r in rows], dtype=object)
    B = np.array([r[1:] for r in rows], float)
    return ch, B


def geo_for(L: pd.DataFrame, cache_path_of) -> np.ndarray:
    """L: page, bbox. -> mảng chữ kim tại tâm crop ('' nếu không), theo thứ tự hàng của L."""
    L = L.reset_index(drop=True)
    out = np.array([""] * len(L), dtype=object)
    for pg, idx in L.groupby("page").groups.items():
        p = cache_path_of(pg)
        if p is None or not Path(p).exists():
            continue
        pc = page_chars(json.load(open(p, encoding="utf-8")))
        if pc is None:
            continue
        ch, B = pc
        for i in idx:
            try:
                b = json.loads(L.at[i, "bbox"])
            except Exception:  # noqa: BLE001
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            m = (B[:, 0] <= cx) & (cx <= B[:, 2]) & (B[:, 1] <= cy) & (cy <= B[:, 3])
            if m.any():
                j = np.nonzero(m)[0]
                # nhiều hộp chứa (cột chồng): lấy hộp có tâm gần nhất
                k = j[np.argmin(np.abs((B[j, 1] + B[j, 3]) / 2 - cy) + np.abs((B[j, 0] + B[j, 2]) / 2 - cx))]
                out[i] = ch[k]
    return out


def borg():
    C = T.lex()
    res = {}
    for b, full in (("B18", "SachKinhThayCaBinh"), ("B34", "SachDungLyHoThan")):
        L = T.rd(T.SNAP / "prepared/_auto" / full / "dataset_out/labels_gated.csv")
        cdir = T.SNAP / "prepared/_auto" / full / "detected"
        L["kim_geo"] = geo_for(L, lambda pg: cdir / f"{pg}_ocr_cache.json")
        B8 = pd.read_pickle(T.T8 / "base" / f"{b}.pkl")          # sự thật vị trí đã ghép sẵn (TN8 t00, cùng hàng)
        assert len(B8) == len(L) and (B8.bbox.values == L.bbox.values).all()
        J = pd.DataFrame({"slot_ok": B8.pos_ok.values.astype(bool)})
        k = B8.pos_known.values.astype(bool)
        cons = np.array([bool(g) and bool(l) and C.var_eq_plus(g, l) for g, l in zip(L.kim_geo, L.ocr_char)])
        r = {}
        for tier in ("GOLD", "ALL"):
            m = k & ((L.tier == "GOLD") if tier == "GOLD" else True)
            cm = m & cons; im = m & ~cons & (L.kim_geo != ""); nb = m & (L.kim_geo == "")
            r[tier] = dict(n=int(m.sum()), slot=round(float(J.slot_ok[m].mean()), 4),
                           frac_cons=round(float(cons[m].mean()), 4),
                           slot_given_cons=round(float(J.slot_ok[cm].mean()), 4) if cm.sum() else None,
                           slot_given_incons=round(float(J.slot_ok[im].mean()), 4) if im.sum() else None,
                           slot_given_nobox=round(float(J.slot_ok[nb].mean()), 4) if nb.sum() else None,
                           frac_incons=round(float(im[m].mean()), 4), frac_nobox=round(float(nb[m].mean()), 4))
        res[b] = r
        L[["page", "column", "nom_idx", "syl_idx", "tier", "ocr_char", "label", "kim_geo"]].assign(
            slot_ok=J.slot_ok.values, pos_known=k).to_pickle(OD / f"{b}.pkl")
        print(f"[t06] Borg {b}: {r}", flush=True)
    return res


def stt(name: str, labels: Path):
    C = T.lex()
    L = T.rd(labels)
    L = L[L.book.isin(T.STT)].reset_index(drop=True)
    res = {}
    geo1 = np.array([""] * len(L), dtype=object); geo2 = geo1.copy(); geoo = geo1.copy()
    read = name.split("__")[0] if name != "prod" else "lt1"
    for bk in T.STT:
        m = (L.book == bk).values
        sub = L[m]
        full = T.STT_FULL[bk]
        d1 = T.REPO / "prepared" / full / "detected"; d2 = T.REPO / "prepared" / full / "kim_raw_lt2"
        do = T.OUT / "stt_prep" / read / full / "detected"
        geo1[m] = geo_for(sub, lambda pg: d1 / f"{pg}_ocr_cache.json")
        geo2[m] = geo_for(sub, lambda pg: d2 / f"{pg}_lt2.json")
        geoo[m] = geo_for(sub, lambda pg: do / f"{pg}_ocr_cache.json")
    L["geo1"], L["geo2"], L["geo_own"] = geo1, geo2, geoo
    for bk in T.STT:
        g = L[(L.book == bk) & (L.tier == "GOLD")]
        c1 = np.array([bool(a) and C.var_eq_plus(a, l) for a, l in zip(g.geo1, g.label)])
        c2 = np.array([bool(a) and C.var_eq_plus(a, l) for a, l in zip(g.geo2, g.label)])
        co = np.array([bool(a) and C.var_eq_plus(a, l) for a, l in zip(g.geo_own, g.ocr_char)])
        res[bk] = dict(n_gold=int(len(g)), geo1_eq_label=round(float(c1.mean()), 4), geo2_eq_label=round(float(c2.mean()), 4),
                       geo1_or_geo2=round(float((c1 | c2).mean()), 4), geo_own_eq_ocr=round(float(co.mean()), 4),
                       geo1_empty=round(float((g.geo1 == "").mean()), 4))
    L[["book", "page", "column", "nom_idx", "syl_idx", "tier", "ocr_char", "label", "bbox", "geo1", "geo2", "geo_own"]].to_pickle(OD / f"{name}.pkl")
    print(f"[t06] STT {name}: {res}", flush=True)
    return res


def main():
    OD.mkdir(parents=True, exist_ok=True)
    out = {}
    args = sys.argv[1:]
    if not args or "borg" in args:
        out["borg"] = borg()
    for nm in [a for a in args if a != "borg"] or ["prod"]:
        p = (T.REPO / "dataset_out/labels_final.csv") if nm == "prod" else (T.OUT / "stt_builds" / nm / "dataset_out/labels_final.csv")
        out[nm] = stt(nm, p)
    f = T.OUT / "kimgeo.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(out)
    T.jdump(old, f)


if __name__ == "__main__":
    main()
