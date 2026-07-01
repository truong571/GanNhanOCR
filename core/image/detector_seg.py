"""Cắt chữ bằng CenterNet detector (train_crop/) cho PIPELINE CHÍNH (core).

Thay phương pháp valley ở bước re-segment khi OCR đếm thiếu (chữ dính). Detector
chạy 1 lần/trang -> column_boxes(page_boxes, x_range, N) cho từng cột; N = số âm
tiết QN. Tách chữ dính bằng SEAM CARVING. Audit production: N đúng -> 0% cắt dính,
99.5% crop sạch.

Tự FALLBACK None (pipeline dùng valley như cũ) nếu thiếu ckpt / lỗi import.

Ckpt tìm theo thứ tự: env NOM_DETECTOR_CKPT > train_crop/detector_r34.best.pt >
evaluation/ver_new/char_detector/detector_r34.best.pt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
_DET = None
_TRIED = False


def _find_ckpt() -> Path | None:
    cands = [os.environ.get("NOM_DETECTOR_CKPT", ""),
             REPO / "train_crop" / "detector_r34.best.pt",
             REPO / "evaluation" / "ver_new" / "char_detector" / "detector_r34.best.pt"]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def get_detector():
    """CenterNetDetector đã train (cache), hoặc None -> pipeline tự dùng valley."""
    global _DET, _TRIED
    if _TRIED:
        return _DET
    _TRIED = True
    ckpt = _find_ckpt()
    if ckpt is None:
        print("  [reseg detector] không thấy detector_r34.best.pt -> dùng valley (cũ).", flush=True)
        return None
    try:
        sys.path.insert(0, str(REPO / "train_crop"))
        for _m in ("infer_centernet", "model_centernet", "train_centernet", "data_centernet"):
            sys.modules.pop(_m, None)            # lấy bản train_crop, tránh clash tên
        from infer_centernet import CenterNetDetector
        det = CenterNetDetector(str(ckpt), thr=0.2, split_method="seam")
        if not det.trained:
            print("  [reseg detector] ckpt rỗng -> dùng valley.", flush=True)
            return None
        print(f"  [reseg detector] CenterNet v1 ({ckpt.name}, img {det.img}) -> tách chữ dính bằng seam.",
              flush=True)
        _DET = det
    except Exception as e:
        print(f"  [reseg detector] lỗi ({type(e).__name__}: {e}) -> dùng valley.", flush=True)
        _DET = None
    return _DET


def detector_column_bboxes(det, page_boxes, gray_img, col_bbox, expected):
    """N hộp [x1,y1,x2,y2] (int) cho 1 cột, hoặc [] nếu không đủ N.

    col_bbox = (x1,y1,x2,y2) của cột; expected = số âm tiết QN."""
    if det is None or page_boxes is None or not col_bbox:
        return []
    x1, _, x2, _ = col_bbox
    cb = det.column_boxes(page_boxes, (x1, x2), expected, gray_image=gray_img)
    if len(cb) != expected:
        return []
    return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in cb]
