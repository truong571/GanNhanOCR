"""v_pb_nguon_goc_05 (04/10) — PHẢN BIỆN: (i) hồ sơ mẫu giấy p05 (stt "255/60/195" là đo hay mặc định?) ; (ii) dilate p03:181 làm nét dày hay mảnh?

(i) Dùng đúng hàm gốc `extract_book_paper_model` của p05 (import, không sửa) cho 10 bộ; thêm: số mức xám khác nhau của trang mẫu,
    ngưỡng Otsu T, tỉ lệ điểm ảnh < T, có dùng patch nhiễu Gaussian giả lập không (min_ink > 200), và (stt) ink_pixels có rỗng không.
(ii) Với 24 ảnh FD đã sinh (p03_direct/gen_book_style/b34_style): tỉ lệ điểm tối (<128) trước/sau cv2.dilate(3x3, 2 lần) y như p03:160,181.
Ra: measure_out/_tn11/verify/pb_nguon_goc/giay_dilate.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_nguon_goc"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("p05_goc", str(HERE / "p05_full_corpus_he_quy_chieu.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
T = M.T

res = {"giay": {}}
for b in T.ORDER:
    cfg = T.BOOKS[b]
    pages = sorted((T.REPO / cfg["prep"] / "pages").glob("*.png"))
    sp = pages[min(10, len(pages) - 1)]
    g = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
    Tv, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pm = M.extract_book_paper_model(b)
    below = float((g < Tv).mean())
    ink_empty = int((g < Tv).sum()) == 0
    # min_ink của cửa sổ 128x128 tốt nhất (như p05:98-110)
    h, w = g.shape
    best = 10 ** 9
    for y in range(20, h - 148, 32):
        for x in range(20, w - 148, 32):
            n = int((g[y:y + 128, x:x + 128] < Tv).sum())
            best = min(best, n)
            if best == 0:
                break
        if best == 0:
            break
    res["giay"][b] = {"trang": sp.name, "n_trang": len(pages), "otsu_T": float(Tv), "so_muc_xam": int(len(np.unique(g))), "ti_le_diem_<T": round(below, 4),
                      "ink_pixels_rong": ink_empty, "bg": pm["bg_median"], "ink": pm["ink_median"], "contrast": pm["contrast"],
                      "min_ink_cua_so": best, "dung_patch_nhieu_gia_lap": best > 200}
    print(b, res["giay"][b], flush=True)

gd = REPO / "measure_out/_tn11/p03_direct/gen_book_style/b34_style"
k3 = np.ones((3, 3), np.uint8)
rows = []
for p in sorted(gd.glob("U+*.png")):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    d = cv2.dilate(g, k3, iterations=2)
    rows.append({"f": p.name, "shape": list(g.shape), "mean_truoc": float(g.mean()), "dark_truoc": float((g < 128).mean()), "dark_sau": float((d < 128).mean())})
res["dilate"] = {"n_anh": len(rows), "nen_sang_(mean>128)": int(sum(r["mean_truoc"] > 128 for r in rows)),
                 "dark_truoc_tb": round(float(np.mean([r["dark_truoc"] for r in rows])), 4), "dark_sau_tb": round(float(np.mean([r["dark_sau"] for r in rows])), 4),
                 "n_anh_net_mong_di": int(sum(r["dark_sau"] < r["dark_truoc"] for r in rows)), "n_anh_net_day_len": int(sum(r["dark_sau"] > r["dark_truoc"] for r in rows)),
                 "mtime_sinh": sorted(round(os.path.getmtime(p)) for p in gd.glob("U+*.png"))[:3] + ["..."]}
mt = sorted(os.path.getmtime(p) for p in gd.glob("U+*.png"))
gaps = [round(b - a, 1) for a, b in zip(mt, mt[1:]) if b - a > 20]
res["dilate"]["khoang_cach_lo_s"] = gaps
res["dilate"]["tong_s_24_anh_xap_xi"] = round(mt[-1] - mt[0], 1)
print(res["dilate"], flush=True)
(OUT / "giay_dilate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
