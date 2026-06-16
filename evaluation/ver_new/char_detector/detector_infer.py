"""Inference glue: trained CenterNet detector.pt -> per-column N boxes.

Used by align_production's `--reseg detector` mode (the production path once you
have a trained detector.pt from Kaggle). Runs the detector ONCE per page, then for
each column keeps the boxes whose centre-x falls in the column's x-range and
reconciles them to the known N (= QN syllables) via count_constrained.

  DetectorInfer(ckpt).boxes_for_page(page_bgr) -> [(x1,y1,x2,y2,score)]  (page px)
  DetectorInfer(...).column_boxes(page_boxes, x_range, n) -> exactly n boxes

Dry-run (no detector.pt needed — random weights, real page, verifies the path):
  .venv/bin/python evaluation/ver_new/char_detector/detector_infer.py --smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from count_constrained import constrain_to_count          # noqa: E402


class DetectorInfer:
    def __init__(self, ckpt: str | None = None, img: int = 768, thr: float = 0.3, device=None):
        import torch
        from train_centernet import _make_model, STRIDE
        self.torch = torch
        self.STRIDE = STRIDE
        self.img, self.thr = img, thr
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.net = _make_model()(pretrained=False)
        if ckpt and Path(ckpt).exists():
            d = torch.load(ckpt, map_location="cpu")
            self.net.load_state_dict(d.get("model", d))
            self.img = d.get("img", img)
            self.trained = True
        else:
            self.trained = False
        self.net.eval().to(self.device)

    def boxes_for_page(self, page_bgr):
        import cv2
        from train_centernet import decode
        H, W = page_bgr.shape[:2]
        s = self.img / max(H, W); nh, nw = int(H * s), int(W * s)
        canvas = np.zeros((self.img, self.img, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(page_bgr, (nw, nh))
        x = self.torch.from_numpy((canvas.astype(np.float32) / 255 - 0.5) / 0.5).permute(2, 0, 1)
        with self.torch.no_grad():
            hm, wh, off = self.net(x.unsqueeze(0).to(self.device))
        dets = decode(hm[0:1], wh[0], off[0], k=512, thr=self.thr)
        return [(d[0] / s, d[1] / s, d[2] / s, d[3] / s, d[4]) for d in dets]   # -> original px

    def column_boxes(self, page_boxes, x_range, n, x_margin=0.5):
        """Boxes of one column (centre-x inside x_range±margin·width) -> exactly n."""
        x1, x2 = x_range
        m = (x2 - x1) * x_margin
        col = [b for b in page_boxes if x1 - m <= (b[0] + b[2]) / 2 <= x2 + m]
        col.sort(key=lambda b: (b[1] + b[3]) / 2)
        return constrain_to_count([b[:4] for b in col], n)


def _smoke():
    import glob, cv2
    REPO = HERE.parent.parent.parent
    pages = sorted(glob.glob(str(REPO / "prepared" / "*" / "pages" / "*.png")))
    if not pages:
        print("no page images to smoke on"); return
    det = DetectorInfer(ckpt=None, img=512, thr=0.0)   # random weights; thr=0 so boxes exist
    img = cv2.imread(pages[0], cv2.IMREAD_COLOR)
    boxes = det.boxes_for_page(img)
    cb = det.column_boxes(boxes, (0, img.shape[1]), 9)   # whole page as one column
    print(f"smoke OK (untrained weights) | device {det.device} | page {Path(pages[0]).name} "
          f"{img.shape[1]}x{img.shape[0]} | page-boxes {len(boxes)} | column->count_constrained {len(cb)} (target 9)")
    assert len(cb) == 9 and len(boxes) > 0
    print("  inference path (load arch -> page forward -> decode -> per-column count-constrain) works.")
    print("  Accurate boxes require a TRAINED detector.pt (Kaggle); drop it in and re-run.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=str(HERE / "detector.pt"))
    a = ap.parse_args()
    if a.smoke:
        _smoke()
    else:
        print("use --smoke (no detector.pt yet) or import DetectorInfer in align_production --reseg detector")
