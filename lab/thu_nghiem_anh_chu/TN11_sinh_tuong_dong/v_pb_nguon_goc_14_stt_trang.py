"""v_pb_nguon_goc_14 (04/10) — PHẢN BIỆN: "stt2/stt4/stt11 đều 255/60/195" — hằng số mặc định hay đo? Kiểm TRÊN MỌI TRANG của 3 sách (không chỉ trang mẫu p05).

Với mỗi trang PNG trong prepared/SachThanhTruyen{2,4,11}/pages: số mức xám khác nhau, ngưỡng Otsu T, ink_pixels (< T) có rỗng không.
Ra: measure_out/_tn11/verify/pb_nguon_goc/stt_trang.json
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
res = {}
for b, d in (("stt2", "SachThanhTruyen2"), ("stt4", "SachThanhTruyen4"), ("stt11", "SachThanhTruyen11")):
    pages = sorted((REPO / "prepared" / d / "pages").glob("*.png"))
    n2 = nT0 = nEmpty = 0
    levels = []
    for p in pages:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        u = np.unique(g)
        Tv, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n2 += int(len(u) == 2)
        nT0 += int(Tv == 0)
        nEmpty += int(int((g < Tv).sum()) == 0)
        levels.append(len(u))
    res[b] = {"n_trang": len(pages), "trang_2_muc_xam": n2, "otsu_T_bang_0": nT0, "ink_pixels_rong": nEmpty, "so_muc_xam_min_max": [int(min(levels)), int(max(levels))]}
    print(b, res[b], flush=True)
(OUT / "stt_trang.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
