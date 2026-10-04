"""v_kythuat (03/10): "Morphological Stroke Alignment" (cv2.dilate 3x3, t = 2) làm gì với nét? Đo tỉ lệ mực (CPU, chỉ đọc PNG).

Notebook lab/kaggle_diffusion/diffusion_run.ipynb ghi: ERODE_ITERS=2 -> mực 5,4 % ("làm mảnh quá tay, rụng nét"), ERODE_ITERS=1 -> 13,1 %
(gần phông gốc 10,8 %, crop thật 7,6 %); hàm erode_strokes thực chất gọi cv2.dilate(…, 3x3, iterations=iters) trên ảnh mực TỐI nền sáng.
Script đo trực tiếp:
  (1) kho FD đang dùng (gannhanocr-fd/*/U+*.png): tỉ lệ mực trung bình (điểm ảnh < 128) trên mẫu ngẫu nhiên -> kho là bản ERODE=1 hay =2;
  (2) ảnh p02 sinh bằng FontDiffuser (measure_out/_tn11/p02_B34/gen/s0): tỉ lệ mực thô, sau dilate x1, x2, x3 (+ erode x1, x2 làm đậm);
  (3) crop thật ô neo B34 (nếu có style_*.png của p02): tỉ lệ mực của ảnh phong cách.
Ra: measure_out/_tn11/verify/kythuat/muc_kho_fd.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "kythuat"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
k3 = np.ones((3, 3), np.uint8)


def ink(im):
    return float((im < 128).mean())


def stats(paths, n):
    paths = list(paths)
    sel = [paths[i] for i in rng.choice(len(paths), size=min(n, len(paths)), replace=False)]
    ims = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sel]
    ims = [i for i in ims if i is not None]
    return ims


res = {}
store = [Path(r) / f for r, _, fs in os.walk(REPO / "gannhanocr-fd") for f in fs if f.startswith("U+") and f.endswith(".png")]
ims = stats(store, 600)
res["kho_fd_gannhanocr-fd"] = dict(n_mau=len(ims), kich_thuoc=list(ims[0].shape), muc_tb=float(np.mean([ink(i) for i in ims])),
                                    muc_trung_vi=float(np.median([ink(i) for i in ims])))
agl = [p for p in (REPO / "ArcFace/data/glyphs").glob("**/U+*.png") if p.stat().st_size >= 1024]
ims = stats(agl, 600)
res["ArcFace_data_glyphs(FD thật dùng khi huấn luyện v2 + làm glyph FD)"] = dict(
    n_mau=len(ims), n_tep_that=len(agl), kich_thuoc=list(ims[0].shape), muc_tb=float(np.mean([ink(i) for i in ims])),
    muc_trung_vi=float(np.median([ink(i) for i in ims])))
gen = sorted((REPO / "measure_out/_tn11/p02_B34/gen/s0").glob("U+*.png"))
ims = stats(gen, 300)
d = dict(n_mau=len(ims), kich_thuoc=list(ims[0].shape), muc_tho=float(np.mean([ink(i) for i in ims])))
for t in (1, 2, 3):
    d[f"dilate_x{t}(lam_manh)"] = float(np.mean([ink(cv2.dilate(i, k3, iterations=t)) for i in ims]))
for t in (1, 2):
    d[f"erode_x{t}(lam_dam)"] = float(np.mean([ink(cv2.erode(i, k3, iterations=t)) for i in ims]))
res["p02_B34_gen_s0_(FD_one-shot)"] = d
for j in (0, 1):
    p = REPO / f"measure_out/_tn11/p02_B34/style_{j}.png"
    if p.exists():
        res[f"anh_phong_cach_p02_style_{j}"] = dict(muc=ink(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)))
(OUT / "muc_kho_fd.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
