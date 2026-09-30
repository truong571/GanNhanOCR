"""TN10 t02 (30/09) — kiểm phần "CẮT CHỮ ×2" của đề xuất trên trang Borg (0 API).

Đề xuất: phóng trang ×2 (Lanczos) -> dò hộp trên ảnh ×2 -> chia toạ độ /2 -> cắt trên ảnh gốc, để "tách chữ dính".
Detector của dự án (train_crop/infer_centernet.py) LETTERBOX mọi ảnh về img×img (s = img / max(H, W)), nên phóng ×2 rồi dò ở
img = 1024 chỉ đổi phép nội suy; muốn detector thật sự "thấy" chữ to gấp đôi phải tăng img (2048). Thử 4 cấu hình:
  A  goc_1024        ảnh prepared như bộ _BORG_NHAN_NGUOI (img 1024, linear)            — mốc (khớp det/boxes_v1)
  B  x2_1024         Lanczos ×2 rồi dò img 1024 (đề xuất, detector hiện có)             — toạ độ /2
  C  x2_2048         Lanczos ×2 rồi dò img 2048 (đề xuất, chữ to gấp đôi khi dò)        — toạ độ /2
  D  goc_1536        ảnh gốc, img 1536 (trung gian)
Đo theo trang (so với chữ NGƯỜI phiên và hộp đơn vị đã gióng của bộ hiện tại):
  * số hộp (điểm ≥ 0,2) / số chữ người của trang — gần 1 là tốt (sót < 1, dư/tách > 1);
  * tỉ lệ hộp "cao bất thường" (cao > 1,6 × trung vị cột) — dấu hiệu dính 2 chữ;
  * phủ ô keep_v5 hiện có: tỉ lệ ô keep_v5 có hộp mới IoU ≥ 0,5 (không được làm mất ô đã tốt).

    .venv/bin/python lab/thu_nghiem_kim/TN10_borg_nguoi/t02_det_scale.py [--pages 120]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "train_crop"))
OUT = REPO / "measure_out" / "_tn10"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main(argv=None) -> int:
    import cv2
    from infer_centernet import CenterNetDetector
    from pipeline.borg_human.params import DETECTOR
    from pipeline.borg_human.encoders import device
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=120)
    a = ap.parse_args(argv)
    C = pd.read_csv(REPO / "measure_out" / "_borg_human" / "cells.csv.gz")
    pages = sorted(C.groupby(["book", "page"]).groups)
    step = max(1, len(pages) // a.pages)
    pages = pages[::step][: a.pages]
    dev = device("auto")
    cfgs = {"goc_1024": (1, 1024), "x2_1024": (2, 1024), "x2_2048": (2, 2048), "goc_1536": (1, 1536)}
    dets = {}
    for k, (_, img) in cfgs.items():
        d = CenterNetDetector(str(REPO / DETECTOR["ckpt"]), img=img, thr=DETECTOR["thr"], resize=DETECTOR["resize"], device=dev)
        d.img = img        # checkpoint ghi đè img=1024 -> ÉP kích thước dò (không thì 1536/2048 chạy như 1024)
        dets[k] = d
    rows = []
    for book, page in pages:
        g = C[(C.book == book) & (C.page == page)]
        n_h = int((g["kind"] != "skip").sum())
        kv = g[g.keep_v5 == 1][["x1", "y1", "x2", "y2"]].to_numpy(float)
        png = cv2.imread(str(REPO / "prepared" / book / "pages" / f"{page}.png"), cv2.IMREAD_GRAYSCALE)
        bgr = cv2.cvtColor(png, cv2.COLOR_GRAY2BGR)
        for k, (sc, _) in cfgs.items():
            im = cv2.resize(bgr, (bgr.shape[1] * sc, bgr.shape[0] * sc), interpolation=cv2.INTER_LANCZOS4) if sc != 1 else bgr
            bx = [b for b in dets[k].boxes_for_page(im) if float(b[4]) >= 0.2]
            B = np.array([[b[0] / sc, b[1] / sc, b[2] / sc, b[3] / sc] for b in bx], float) if bx else np.zeros((0, 4))
            h = B[:, 3] - B[:, 1] if len(B) else np.zeros(0)
            tall = float((h > 1.6 * np.median(h)).mean()) if len(h) else 0.0
            cov = float(np.mean([max((iou(u, b) for b in B), default=0.0) >= 0.5 for u in kv])) if len(kv) else np.nan
            rows.append(dict(book=book, page=page, cfg=k, n_human=n_h, n_box=len(B), ratio=len(B) / max(1, n_h),
                             tall=tall, keep_v5_cov=cov))
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "t02_det_scale.csv", index=False)
    S = R.groupby("cfg").agg(trang=("page", "size"), hop_tren_chu_nguoi=("ratio", "median"),
                             lech_so_luong_tb=("ratio", lambda x: float(np.mean(np.abs(x - 1)))),
                             hop_cao_bat_thuong=("tall", "mean"), phu_keep_v5=("keep_v5_cov", "mean")).round(4)
    S = S.loc[[k for k in cfgs if k in S.index]]
    (OUT / "t02_det_scale.json").write_text(S.to_json(force_ascii=False, indent=1), encoding="utf-8")
    print(S.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
