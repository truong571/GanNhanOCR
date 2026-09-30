"""geo.py — "kim_geo": chữ kim có HỘP CHIA ĐỀU chứa TÂM crop của ô (TN9, lab/thu_nghiem_kim/TN9_stt_in/t06_kimgeo.geo_for — bản
chép nguyên, chỉ đổi nơi đặt). 0 API, chỉ đọc prepared/<Sách>/detected/<trang>_ocr_cache.json.

kim trả hộp DÒNG + chuỗi; hộp chữ = chia đều hộp dòng (core.ocr.ocr_api.boxes_to_columns, trường `columns[][].bbox` của cache).
Ô có tâm bbox (cx, cy): hộp chữ chứa tâm (x1 ≤ cx ≤ x2 ∧ y1 ≤ cy ≤ y2); nhiều hộp chứa (cột chồng) -> hộp có tâm gần nhất
(|Δy| + |Δx|); không hộp nào -> ''. Crop đúng chỗ kim đọc ⇒ kim_geo == chữ kim đã ghép với âm (ocr_char).
Dùng cho luật TN9 sách in: P2 (qn_geo) / P3 (np_geo) — config/chon_chu.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def page_chars(cache: dict):
    """(mảng chữ, mảng hộp (n, 4)) của mọi chữ có hộp trong `columns` của cache kim; None nếu trang không có chữ."""
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


def char_at(ch, B, cx: float, cy: float) -> str:
    m = (B[:, 0] <= cx) & (cx <= B[:, 2]) & (B[:, 1] <= cy) & (cy <= B[:, 3])
    if not m.any():
        return ""
    j = np.nonzero(m)[0]
    k = j[np.argmin(np.abs((B[j, 1] + B[j, 3]) / 2 - cy) + np.abs((B[j, 0] + B[j, 2]) / 2 - cx))]
    return str(ch[k])


def geo_for(L: pd.DataFrame, cache_path_of) -> np.ndarray:
    """L: page, bbox (JSON). -> chữ kim tại tâm crop ('' nếu không), theo thứ tự hàng của L (index được đặt lại)."""
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
            out[i] = char_at(ch, B, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return out


def kim_geo(D: pd.DataFrame, prep: Path) -> np.ndarray:
    """kim_geo của mọi ô sách (cache kim prepared/<Sách>/detected/<trang>_ocr_cache.json)."""
    det = Path(prep) / "detected"
    return geo_for(D[["page", "bbox"]], lambda pg: det / f"{pg}_ocr_cache.json")
