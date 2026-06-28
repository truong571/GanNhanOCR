"""Text-line detection for the QN re-OCR path.

The original line finder was a raw horizontal-projection profile: sum dark
pixels per row, cut on the valleys. It is fast but fragile — a scan skewed by
more than ~2° (or a black scan border) smears the projection so adjacent lines
merge or vanish, corrupting the line order that the Levenshtein alignment
depends on.

This module replaces it with a pluggable detector exposing three backends:

  * ``dbnet``             — PaddleOCR's DB text detector. Finds each line as a
                            (possibly rotated) quadrilateral regardless of page
                            skew, then perspective-warps it flat. Used when
                            ``paddleocr`` is importable.
  * ``projection_deskew`` — DL-free fallback that first ESTIMATES the global
                            skew angle (projection-profile variance maximisation)
                            and DESKEWS the whole page, then runs the projection
                            line cut on the straightened image. This removes the
                            skew sensitivity that motivated the change and runs
                            with no extra dependency. This is the default when
                            PaddleOCR is not installed.
  * ``projection``        — the original raw projection profile, kept verbatim
                            for A/B comparison / reproducibility.

``detect_line_crops`` is the single entry point: it returns a list of RGB line
crops in reading order (top→bottom), already straightened, ready for VietOCR.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Backend resolution
# --------------------------------------------------------------------------- #
_PADDLE_OCR = None
_PADDLE_TRIED = False


def paddle_available() -> bool:
    """True if PaddleOCR can be imported (cheap import-spec check)."""
    import importlib.util
    return importlib.util.find_spec("paddleocr") is not None


def resolve_backend(backend: str = "auto") -> str:
    """Map ``auto`` to the best available concrete backend."""
    if backend in ("dbnet", "projection_deskew", "projection"):
        return backend
    # auto
    return "dbnet" if paddle_available() else "projection_deskew"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _binary(img_rgb: np.ndarray) -> np.ndarray:
    """Otsu binary, ink = 255 (foreground)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def _rotate(img: np.ndarray, angle_deg: float, border_value=0) -> np.ndarray:
    """Rotate about the centre, keeping the original canvas size."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def estimate_skew_angle(bw: np.ndarray, limit: float = 8.0,
                        coarse: float = 1.0, fine: float = 0.2) -> float:
    """Estimate the deskew rotation (degrees) via projection-variance search.

    Returns the angle to ROTATE the page by so text rows become horizontal.
    The horizontal-projection profile has maximum variance (sharp peaks at
    line centres, deep valleys between) exactly when the lines are horizontal.
    Two-stage (coarse then fine) over ±``limit`` degrees; image downscaled for
    speed. Returns 0.0 if the page has too little ink to judge.
    """
    h, w = bw.shape[:2]
    if int(bw.sum()) < 255 * 50:           # almost no ink
        return 0.0
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = (cv2.resize(bw, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else bw)

    def score(angle: float) -> float:
        rot = _rotate(small, angle)
        proj = rot.sum(axis=1, dtype=np.float64)
        return float(np.var(proj))

    def best_over(angles) -> float:
        best_a, best_s = 0.0, -1.0
        for a in angles:
            s = score(a)
            if s > best_s:
                best_s, best_a = s, float(a)
        return best_a

    coarse_a = best_over(np.arange(-limit, limit + coarse, coarse))
    fine_a = best_over(np.arange(coarse_a - coarse, coarse_a + coarse + fine, fine))
    return float(fine_a)


# --------------------------------------------------------------------------- #
# Backend: projection (legacy, verbatim behaviour) — returns boxes
# --------------------------------------------------------------------------- #
def detect_line_boxes(img_rgb: np.ndarray, min_height: int = 18,
                      gap: int = 8) -> list[tuple[int, int, int, int]]:
    """Horizontal-projection line cut -> list of (x1, y1, x2, y2).

    This is the original algorithm, unchanged, operating on whatever image it
    is handed (raw page, or a pre-deskewed page for ``projection_deskew``).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    proj = bw.sum(axis=1)
    h, w = bw.shape
    th = max(int(w * 0.01), 5)
    is_ink = proj > th

    runs: list[list[int]] = []
    in_run = False
    start = 0
    for y in range(h):
        if is_ink[y] and not in_run:
            in_run = True
            start = y
        elif not is_ink[y] and in_run:
            in_run = False
            if y - start >= min_height:
                runs.append([start, y])
    if in_run and h - start >= min_height:
        runs.append([start, h])

    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out: list[tuple[int, int, int, int]] = []
    for s, e in merged:
        col = bw[s:e].sum(axis=0)
        nz = np.where(col > 1)[0]
        if not len(nz):
            continue
        x1, x2 = int(nz[0]), int(nz[-1]) + 1
        pad = 5
        out.append((max(0, x1 - pad), max(0, s - pad),
                    min(w, x2 + pad), min(h, e + pad)))
    return out


# --------------------------------------------------------------------------- #
# Backend: dbnet (PaddleOCR) — returns straightened crops
# --------------------------------------------------------------------------- #
def _get_paddle():
    """Lazy DB-detector singleton (detection only, no recognition/angle-cls)."""
    global _PADDLE_OCR, _PADDLE_TRIED
    if _PADDLE_OCR is not None or _PADDLE_TRIED:
        return _PADDLE_OCR
    _PADDLE_TRIED = True
    from paddleocr import PaddleOCR  # raises ImportError if absent
    try:
        _PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang="vi",
                                use_gpu=False, show_log=False)
    except TypeError:
        # PaddleOCR >=3 dropped several of these kwargs.
        _PADDLE_OCR = PaddleOCR(lang="vi")
    return _PADDLE_OCR


def _warp_quad(img_rgb: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp a 4-point line polygon to an axis-aligned crop.

    ``quad`` is 4 (x, y) points in any order; we order them TL,TR,BR,BL and
    warp to a w×h rectangle, straightening a skewed line to horizontal.
    """
    pts = np.array(quad, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w < 2 or h < 2:
        return np.zeros((1, 1, 3), dtype=img_rgb.dtype)
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_rgb, M, (w, h))


def _polys_from_paddle(result) -> list[np.ndarray]:
    """Extract 4-point polygons from the various PaddleOCR result shapes."""
    polys: list[np.ndarray] = []
    if result is None:
        return polys
    # 2.x: ocr(img, rec=False) -> [ [poly, poly, ...] ]  (one entry per image)
    page = result[0] if (len(result) == 1 and isinstance(result[0], list)) else result
    for item in page or []:
        poly = item[0] if (isinstance(item, (list, tuple)) and len(item) and
                           isinstance(item[0], (list, tuple, np.ndarray))) else item
        arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] >= 4:
            polys.append(arr[:4])
    return polys


def _detect_dbnet(img_rgb: np.ndarray, verbose: bool = False) -> list[np.ndarray]:
    ocr = _get_paddle()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    try:
        result = ocr.ocr(img_bgr, rec=False, cls=False)
    except TypeError:
        result = ocr.ocr(img_bgr)
    polys = _polys_from_paddle(result)
    # reading order: top to bottom by polygon vertical centre
    polys.sort(key=lambda p: float(p[:, 1].mean()))
    crops = [_warp_quad(img_rgb, p) for p in polys]
    crops = [c for c in crops if c.shape[0] >= 10 and c.shape[1] >= 10]
    if verbose:
        print(f"  [line_detector/dbnet] {len(crops)} lines")
    return crops


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def detect_line_crops(img_rgb: np.ndarray, backend: str = "auto",
                      min_height: int = 18, gap: int = 8,
                      verbose: bool = False) -> list[np.ndarray]:
    """Detect text lines and return straightened RGB crops, top→bottom.

    ``backend`` ∈ {auto, dbnet, projection_deskew, projection}. ``auto`` uses
    DBNet when PaddleOCR is installed, else the deskew-aware projection. A DBNet
    runtime failure degrades gracefully to ``projection_deskew`` (the pipeline
    must never die on a detector hiccup).
    """
    resolved = resolve_backend(backend)

    if resolved == "dbnet":
        try:
            return _detect_dbnet(img_rgb, verbose=verbose)
        except Exception as e:  # noqa: BLE001 — never crash the page on detector
            print(f"  [line_detector] DBNet failed ({e}); "
                  f"falling back to projection_deskew.", file=sys.stderr)
            resolved = "projection_deskew"

    work = img_rgb
    if resolved == "projection_deskew":
        angle = estimate_skew_angle(_binary(img_rgb))
        if abs(angle) >= 0.1:
            work = _rotate(img_rgb, angle, border_value=(255, 255, 255))
            if verbose:
                print(f"  [line_detector/projection_deskew] deskew {angle:+.2f}°")

    boxes = detect_line_boxes(work, min_height=min_height, gap=gap)
    crops = [work[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
    if verbose:
        print(f"  [line_detector/{resolved}] {len(crops)} lines")
    return crops
