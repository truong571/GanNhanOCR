"""KIỂM ẢNH CA THỬ (công cụ cho người chạy Kaggle, chạy trên Mac sau `hf download`): `validate` của bộ chạy chỉ kiểm cấu trúc tar + băm + đủ chữ,
KHÔNG biết ảnh có trắng/đen/hỏng hay không. Công cụ này đọc các tar đã tải về và báo: kích thước/chế độ ảnh, tỉ lệ ảnh trắng tinh/đen kịt,
mực % (p1/p50/p99) theo lô, so với khoảng đã đo ở bước 3 (MPS, mô hình thật, 256 ảnh của 2 việc): ink% p1–p99 = 11,4–28,5 %, trung vị 20,8–21,5 % (ngưỡng mặc định 3–45 % rộng hơn nhiều).
Chỉ ĐỌC; ghi JSON vào --out (mặc định: review/chay_thu_dau_cuoi/09_kiem_anh_ca_thu.json). Mã thoát 1 nếu có dấu hiệu hỏng.

    .venv/bin/python lab/thu_nghiem_anh_chu/TN11_sinh_tuong_dong/r_chay_thu_dau_cuoi_09_kiem_anh_ca_thu.py --shards measure_out/_tn11/full/kaggle_ket_qua [--pack measure_out/_tn11/full/kaggle_pack_20261004]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import argparse
import io
import json
import tarfile
from pathlib import Path

import r_chay_thu_dau_cuoi_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True, help="thư mục chứa shards/<cuốn>/<mô hình_phong cách>/NNNNN.tar (kết quả hf download)")
ap.add_argument("--pack", default=str(L.PACK))
ap.add_argument("--out", default=str(L.R / "09_kiem_anh_ca_thu.json"))
ap.add_argument("--ink-lo", type=float, default=3.0, help="ink%% thấp hơn = nghi trắng/mất nét")
ap.add_argument("--ink-hi", type=float, default=45.0, help="ink%% cao hơn = nghi đen kịt")
ap.add_argument("--max-bad-frac", type=float, default=0.02)
a = ap.parse_args()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

K = L.load_runner("kiem_anh_runner")
plan = K.load_plan(Path(a.pack))
root = Path(a.shards)
rows, bad_imgs, n = [], [], 0
errs = []
for t in plan["tasks"]:
    p = root / K.rel_path(t["id"])
    if not p.exists():
        continue
    errs += K.validate_shard(p, t, plan)
    with tarfile.open(p) as tf:
        inks = []
        for m in tf.getmembers():
            if not m.name.startswith("U+"):
                continue
            im = Image.open(io.BytesIO(tf.extractfile(m).read()))
            g = np.array(im.convert("L"))
            ink = float((g < 128).mean() * 100)
            n += 1
            inks.append(ink)
            if im.size != (96, 96) or im.mode != "RGB" or ink < a.ink_lo or ink > a.ink_hi or g.min() > 240:
                bad_imgs.append((t["id"], m.name, im.size, im.mode, round(ink, 1)))
        rows.append(dict(task=t["id"], n=len(inks), ink_p1=round(float(np.percentile(inks, 1)), 1), ink_p50=round(float(np.percentile(inks, 50)), 1), ink_p99=round(float(np.percentile(inks, 99)), 1)))
frac = len(bad_imgs) / max(1, n)
ok = (not errs) and frac <= a.max_bad_frac and n > 0
res = dict(so_tar=len(rows), so_anh=n, loi_cau_truc=errs[:5], anh_nghi_van=len(bad_imgs), ti_le_nghi_van=round(frac, 4), mau_nghi_van=bad_imgs[:8], theo_lo=rows[:12], ket_luan="TỐT" if ok else "HỎNG/NGHI VẤN — dừng, đừng chạy ca thật")
Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("so_tar", "so_anh", "anh_nghi_van", "ti_le_nghi_van", "ket_luan")}, ensure_ascii=False))
sys.exit(0 if ok else 1)
