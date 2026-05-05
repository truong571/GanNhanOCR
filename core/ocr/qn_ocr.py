"""QN text recognition using PaddleOCR (detection) + VietOCR (recognition)."""

import sys

import cv2
import numpy as np
from PIL import Image


def ocr_qn_page(image_path: str, verbose: bool = False) -> str:
    """OCR a QN text page using PaddleOCR for detection + VietOCR for recognition.

    PaddleOCR detects text line bounding boxes.
    VietOCR recognizes Vietnamese text in each detected box.
    """
    try:
        from paddleocr import PaddleOCR
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg
    except ImportError as e:
        print(f"[QN_OCR] ERROR: {e}", file=sys.stderr)
        print("[QN_OCR] Install: pip install paddleocr vietocr", file=sys.stderr)
        return ""

    # PaddleOCR for text detection only (det=True, rec=False)
    detector = PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=False, show_log=False)

    # VietOCR for Vietnamese text recognition
    vietocr_cfg = Cfg.load_config_from_name("vgg_transformer")
    vietocr_cfg["cnn"]["pretrained"] = False
    vietocr_cfg["device"] = "cpu"
    recognizer = Predictor(vietocr_cfg)

    if verbose:
        print(f"  [QN_OCR] Processing {image_path}...")

    # Detect text regions
    det_result = detector.ocr(image_path, cls=True)
    if not det_result or not det_result[0]:
        if verbose:
            print("  [QN_OCR] No text detected", file=sys.stderr)
        return ""

    # Load image for cropping
    img = cv2.imread(image_path)
    if img is None:
        return ""

    lines = []
    for line_info in det_result[0]:
        bbox_points = line_info[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        y_center = (bbox_points[0][1] + bbox_points[2][1]) / 2

        # Crop text region
        x_coords = [p[0] for p in bbox_points]
        y_coords = [p[1] for p in bbox_points]
        x1 = max(0, int(min(x_coords)))
        y1 = max(0, int(min(y_coords)))
        x2 = min(img.shape[1], int(max(x_coords)))
        y2 = min(img.shape[0], int(max(y_coords)))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Recognize with VietOCR
        try:
            text = recognizer.predict(crop_pil)
            if text and text.strip():
                lines.append((y_center, text.strip()))
        except Exception:
            # Fallback: use PaddleOCR's own recognition
            paddle_text = line_info[1][0] if line_info[1] else ""
            if paddle_text.strip():
                lines.append((y_center, paddle_text.strip()))

    # Sort by vertical position (top to bottom)
    lines.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in lines)
