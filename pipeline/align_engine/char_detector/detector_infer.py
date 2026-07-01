"""Inference glue: detector CenterNet MỚI (ResNet34+FPN+Seam, train_crop/) -> N box/cột.

Giữ NGUYÊN interface DetectorInfer cũ (boxes_for_page, column_boxes, .trained) để
align_production `--reseg detector` dùng được KHÔNG cần sửa, nhưng chạy model mới
(train_crop/infer_centernet.CenterNetDetector). N = số âm tiết QN do _pick_reseg truyền.

ROBUST: tự tìm ckpt ở train_crop/ (canonical, gitignored nhưng luôn có) -> sống sót
qua `git checkout evaluation/`. Khác bản ResNet18 cũ: kiến trúc đọc từ ckpt; tách chữ
dính bằng SEAM CARVING. Audit production (N đúng): 0% cắt dính, 99.5% crop sạch.

Dry-run: .venv/bin/python evaluation/ver_new/char_detector/detector_infer.py --smoke
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
_TRAIN_CROP = REPO / "train_crop"


def _find_ckpt() -> Path | None:
    """Ckpt theo thứ tự: env > train_crop/ (canonical) > char_detector/ (nếu copy về)."""
    cands = [os.environ.get("NOM_DETECTOR_CKPT", ""),
             _TRAIN_CROP / "detector_r34.best.pt",
             Path(__file__).resolve().parent / "detector_r34.best.pt"]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def _load_centernet():
    sys.path.insert(0, str(_TRAIN_CROP))
    for _m in ("infer_centernet", "model_centernet", "train_centernet", "data_centernet"):
        sys.modules.pop(_m, None)                 # lấy bản train_crop, tránh clash tên
    from infer_centernet import CenterNetDetector
    return CenterNetDetector


class DetectorInfer:
    def __init__(self, ckpt: str | None = None, img: int = 1024, thr: float = 0.2,
                 device=None):
        if not ckpt:
            c = _find_ckpt()
            ckpt = str(c) if c else None
        if not ckpt:
            self.det = None; self.trained = False
            self.img = img; self.device = "cpu"; self._gray = None
            return
        CenterNetDetector = _load_centernet()
        self.det = CenterNetDetector(ckpt, img=img, thr=thr, split_method="seam", device=device)
        self.trained = self.det.trained
        self.img = self.det.img
        self.device = self.det.device
        self._gray = None

    def boxes_for_page(self, page_bgr):
        """Detector 1 lần/trang -> [(x1,y1,x2,y2,score)] px gốc. Lưu ảnh xám cho seam."""
        import cv2
        self._gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        return self.det.boxes_for_page(page_bgr)

    def column_boxes(self, page_boxes, x_range, n, x_margin=0.5):
        """Box 1 cột -> ĐÚNG n hộp [x1,y1,x2,y2] (top-N tin cậy + seam-split). n = #âm tiết."""
        return self.det.column_boxes(page_boxes, x_range, n,
                                     gray_image=self._gray, x_margin=x_margin)


def _smoke():
    import glob, cv2
    det = DetectorInfer(thr=0.2)
    print(f"DetectorInfer | ckpt={_find_ckpt()} | trained={det.trained} | img={det.img}")
    if not det.trained:
        print("  (chưa có ckpt -> pipeline tự fallback midpoint)")
        return
    pages = sorted(glob.glob(str(REPO / "prepared" / "*" / "pages" / "*.png")))
    if not pages:
        print("  (load OK, không có trang để test)")
        return
    img = cv2.imread(pages[0], cv2.IMREAD_COLOR)
    boxes = det.boxes_for_page(img)
    cb = det.column_boxes(boxes, (0, img.shape[1]), 9)
    print(f"  page-boxes {len(boxes)} | column_boxes(N=9) -> {len(cb)} | box[0]={cb[0] if cb else None}")
    assert len(cb) == 9
    print("  OK: wrapper ResNet34+seam tương thích align_production --reseg detector.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    _smoke() if a.smoke else print("dùng --smoke")
