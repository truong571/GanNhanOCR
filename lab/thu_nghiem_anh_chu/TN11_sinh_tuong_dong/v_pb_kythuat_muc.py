"""v_pb_kythuat (04/10) — PHẢN BIỆN độc lập khẳng định "Morphological Stroke Alignment: giãn nở 3×3, t=2 làm nét đậm nhạt tương đồng nét mực" (CPU, 0 API).

Người kiểm chứng: cv2.dilate trên ảnh mực TỐI / nền SÁNG làm MẢNH nét (không làm đậm); t=2 làm mực từ 25,4 % -> 7,1 % (vs ảnh phong cách thật 6,9–10,9 %, glyph FD
thật 16,1 %); notebook 18/05 bỏ t=2 vì rụng nét. Script này đo lại bằng mã riêng trên: (1) 793 ảnh sinh p02 (gen/s0), (2) glyph FD thật ArcFace/data/glyphs,
(3) ảnh phong cách p02 (style_0/1), (4) 2.000 crop GOLD thật của STT/B34 (crop thật = đích mà nét sinh ra phải giống) — tỉ lệ điểm tối (<128) sau dilate ×0/×1/×2/×3
và erode ×1. Kèm kiểm cực tính (nền sáng, mực tối) bằng trung bình độ sáng. Ra: measure_out/_tn11/verify/pb_kythuat/muc.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/v_pb_kythuat_muc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "measure_out" / "_tn11" / "verify" / "pb_kythuat"
OUT.mkdir(parents=True, exist_ok=True)
K3 = np.ones((3, 3), np.uint8)


def ink(g):
    return float((g < 128).mean())


def stats(paths, n_max=None, tag=""):
    rng = np.random.default_rng(0)
    paths = list(paths)
    if n_max and len(paths) > n_max:
        paths = [paths[i] for i in rng.choice(len(paths), n_max, replace=False)]
    rows = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or g.size == 0:
            continue
        rows.append(dict(sang=float(g.mean()), raw=ink(g), d1=ink(cv2.dilate(g, K3, iterations=1)), d2=ink(cv2.dilate(g, K3, iterations=2)),
                         d3=ink(cv2.dilate(g, K3, iterations=3)), e1=ink(cv2.erode(g, K3, iterations=1))))
    d = pd.DataFrame(rows)
    return dict(n=int(len(d)), do_sang_tb=round(float(d.sang.mean()), 1), muc_tho=round(float(d.raw.mean()), 4), dilate1=round(float(d.d1.mean()), 4),
                dilate2=round(float(d.d2.mean()), 4), dilate3=round(float(d.d3.mean()), 4), erode1=round(float(d.e1.mean()), 4),
                dilate2_tren_tho=round(float((d.d2 / d.raw.clip(lower=1e-6)).mean()), 3))


def main():
    res = {}
    res["anh_sinh_p02_gen_s0"] = stats((REPO / "measure_out/_tn11/p02_B34/gen/s0").glob("U+*.png"))
    res["glyph_FD_that_ArcFace_data_glyphs"] = stats([p for p in (REPO / "ArcFace/data/glyphs").glob("**/U+*.png") if p.stat().st_size >= 1024])
    res["anh_phong_cach_p02"] = stats([REPO / "measure_out/_tn11/p02_B34/style_0.png", REPO / "measure_out/_tn11/p02_B34/style_1.png"])
    man = pd.read_csv(REPO / "ArcFace/data/manifest.csv")
    cr = man[(man.source == "crop") & (man.tier == "GOLD")]
    rng = np.random.default_rng(0)
    pick = cr.iloc[rng.choice(len(cr), 2000, replace=False)]
    res["crop_GOLD_STT_that_2000"] = stats([REPO / "ArcFace/data" / p for p in pick.path])
    (OUT / "muc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
