#!/usr/bin/env python3
"""Kiểm độ TRUNG THỰC của bản sao đường cắt: crop cơ sở dựng lại == crop giao nộp trong dataset_out (so từng điểm ảnh). Không đo kết quả thí nghiệm.
  .venv/bin/python vision/crop_test/check_fidelity.py [--n 300]"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_lib as cl

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--books", default="L16,TK,KVK,L83,Chr,stt2,stt4,stt11"); a = ap.parse_args()
rng = np.random.default_rng(5)
tot = {}
for key in a.books.split(","):
    d = cl.load_cells(key)
    b = cl.BOOKS[key]
    root = (cl.REPO / "prepared" / b["prep"] / "dataset_out") if "prepared" in b["table"] else (cl.REPO / "dataset_out")
    sel = d[(d["image"] != "") & d.get("chon_chu", "").eq("") if "chon_chu" in d.columns else (d["image"] != "")]
    sel = sel[sel["image"].map(lambda p: (root / p).is_file())]
    if len(sel) == 0:
        print(f"{key}: không có ô có ảnh giao nộp"); continue
    pick = sel.iloc[rng.choice(len(sel), size=min(a.n, len(sel)), replace=False)]
    pc = cl.PageCache(key)
    same = shape_eq = n = 0
    bad = []
    for _, r in pick.sort_values(["page"]).iterrows():
        pg = pc.get(r["page"])
        mine = cl.make_crop(pg, (r.x0, r.y0, r.x1, r.y1), r.prev_bbox, r.next_bbox)
        ref = cv2.imread(str(root / r["image"]), cv2.IMREAD_GRAYSCALE)
        n += 1
        if mine is None or ref is None:
            bad.append((r["image"], "None")); continue
        if mine.shape == ref.shape:
            shape_eq += 1
            if np.array_equal(mine, ref):
                same += 1
            else:
                bad.append((r["image"], f"khác điểm ảnh: max|Δ|={int(np.abs(mine.astype(int) - ref.astype(int)).max())}"))
        else:
            bad.append((r["image"], f"khác kích thước {mine.shape} vs {ref.shape}"))
    tot[key] = (n, shape_eq, same)
    print(f"{key:6s} n={n:4d} cùng kích thước {100 * shape_eq / n:5.1f} % · giống TỪNG ĐIỂM ẢNH {100 * same / n:5.1f} %" + (f" | ví dụ lệch: {bad[:2]}" if bad else ""))
n = sum(v[0] for v in tot.values()); s = sum(v[2] for v in tot.values())
print(f"TỔNG: {s}/{n} = {100 * s / max(1, n):.1f} % giống từng điểm ảnh")
