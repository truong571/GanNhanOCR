"""Image denoising, binarization, and text box detection."""

import cv2
import numpy as np


def denoise_image(gray: np.ndarray) -> np.ndarray:
    """Denoise old book page images.

    Pipeline:
      GaussianBlur(3,3) -> Morph.Close(51x51) background estimate
      -> divide by background x255 -> contrast stretching (2%-98%)
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, bg_kernel)
    normalized = cv2.divide(blurred, background, scale=255)

    p2, p98 = np.percentile(normalized, (2, 98))
    if p98 > p2:
        normalized = np.clip(
            (normalized.astype(float) - p2) / (p98 - p2) * 255, 0, 255
        ).astype(np.uint8)
    return normalized


def preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    """Preprocess for Tesseract OCR: denoise -> Otsu -> morph cleanup."""
    denoised = denoise_image(gray)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    return binary


def load_and_binarize(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image and create binary mask (ink=1, background=0).

    Pipeline: GaussianBlur -> Morph.Close(51x51) -> divide -> Otsu
              -> Close(2x2) -> Open(3x3)
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, bg_kernel)
    normalized = cv2.divide(blurred, background, scale=255)

    _, binary_inv = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = (binary_inv > 0).astype(np.uint8)

    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

    return gray, binary


def detect_text_box(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Detect bounding rectangle of the main text area.

    Uses morphological line detection to find horizontal/vertical borders.
    Returns: (left, top, right, bottom)
    """
    h, w = binary.shape
    pad = 8

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    h_proj = h_lines.sum(axis=1)
    h_line_rows = np.where(h_proj > w * 0.1)[0]
    v_proj = v_lines.sum(axis=0)
    v_line_cols = np.where(v_proj > h * 0.1)[0]

    if len(h_line_rows) > 0:
        box_top = int(h_line_rows[0]) + pad
        box_bottom = int(h_line_rows[-1]) - pad
    else:
        h_proj_full = binary.sum(axis=1).astype(float)
        h_smooth = np.convolve(h_proj_full, np.ones(20) / 20, mode="same")
        ink_rows = np.where(h_smooth > h_smooth.max() * 0.03)[0]
        box_top = int(ink_rows[0]) if len(ink_rows) > 0 else 0
        box_bottom = int(ink_rows[-1]) if len(ink_rows) > 0 else h

    if len(v_line_cols) > 0:
        box_left = int(v_line_cols[0]) + pad
        box_right = int(v_line_cols[-1]) - pad
    else:
        v_proj_full = binary.sum(axis=0).astype(float)
        v_smooth = np.convolve(v_proj_full, np.ones(20) / 20, mode="same")
        ink_cols = np.where(v_smooth > v_smooth.max() * 0.03)[0]
        box_left = int(ink_cols[0]) if len(ink_cols) > 0 else 0
        box_right = int(ink_cols[-1]) if len(ink_cols) > 0 else w

    # Sanity check: box must be > 50% of image
    if (box_right - box_left) < w * 0.5 or (box_bottom - box_top) < h * 0.5:
        v_proj_full = binary.sum(axis=0).astype(float)
        h_proj_full = binary.sum(axis=1).astype(float)
        v_smooth = np.convolve(v_proj_full, np.ones(20) / 20, mode="same")
        h_smooth = np.convolve(h_proj_full, np.ones(20) / 20, mode="same")
        ink_cols = np.where(v_smooth > v_smooth.max() * 0.05)[0]
        ink_rows = np.where(h_smooth > h_smooth.max() * 0.05)[0]
        if len(ink_cols) > 0:
            box_left, box_right = int(ink_cols[0]), int(ink_cols[-1])
        if len(ink_rows) > 0:
            box_top, box_bottom = int(ink_rows[0]), int(ink_rows[-1])

    content_h = box_bottom - box_top
    box_top += int(content_h * 0.02)
    box_bottom -= int(content_h * 0.01)

    return int(box_left), int(box_top), int(box_right), int(box_bottom)
