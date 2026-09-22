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
                 device=None, resize: str = "linear"):
        # resize: 'linear' (mặc định, v1/STT không đổi) | 'area' (khử răng cưa; books[].detector_resize)
        self.resize = resize
        if not ckpt:
            c = _find_ckpt()
            ckpt = str(c) if c else None
        if not ckpt:
            self.det = None; self.trained = False; self.thr = float(thr)
            self.img = img; self.device = "cpu"; self._gray = None
            return
        CenterNetDetector = _load_centernet()
        # giữ mô-đun train_crop/infer_centernet vừa nạp: raw_column_boxes/enforce_count
        # (A-6) dùng _nms_vertical + enforce_count của CHÍNH nó, không nạp lại lần hai
        self._ic = sys.modules["infer_centernet"]
        self.thr = float(thr)
        self.det = CenterNetDetector(ckpt, img=img, thr=thr, split_method="seam", device=device,
                                     resize=resize)
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

    def raw_column_boxes(self, page_boxes, x_range, x_margin=0.25):
        """Hộp THÔ của 1 cột (A-6, flow N3f): lọc tâm-x ∈ [x1 − m·w, x2 + m·w] (đúng
        phép lọc của infer_centernet.column_boxes) + _nms_vertical(0,45) + sắp theo y.
        KHÔNG enforce_count — trả đúng M hộp detector thấy, mỗi hộp
        [x1, y1, x2, y2, score] (toạ độ int như column_boxes). M so với n_qn/n_ocr
        quyết nhánh gán hộp (align_production.assign_boxes)."""
        x1, x2 = x_range
        m = (x2 - x1) * x_margin
        col = [list(b[:5]) if len(b) >= 5 else list(b[:4]) + [1.0]
               for b in page_boxes if x1 - m <= (b[0] + b[2]) / 2 <= x2 + m]
        if len(col) > 1:
            col = self._ic._nms_vertical(col, iou_thr=0.45)
        col.sort(key=lambda b: (b[1] + b[3]) / 2.0)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])] for b in col]

    def enforce_count(self, boxes, n):
        """Ép M hộp thô về ĐÚNG n (M>n: giữ top-n điểm; M<n: tách hộp cao nhất theo seam
        trên ảnh xám của trang đã chạy boxes_for_page) — chính bước cũ bên trong
        column_boxes, tách riêng để nhánh 'conflict' (A-6) gọi trên hộp raw_column_boxes.
        Trả 4-int [x1,y1,x2,y2]; hộp bị bổ đôi nhận ra bằng cách so với hộp gốc."""
        out = self._ic.enforce_count([list(b) for b in boxes], n, gray_image=self._gray,
                                     split_method=self.det.split_method)
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in out]


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
