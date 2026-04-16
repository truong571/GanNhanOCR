"""Ablation test: toggle each improvement individually to measure impact.

Tests 4 variants against OLD baseline:
  A) OLD + border_line_removal only
  B) OLD + local_stroke_norm only
  C) OLD + line-like CC filter only
  D) ALL three combined (current NEW)

Usage: python tests/test_ablation.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sauvola_binarize(gray, k=0.2, R=128.0, window=25):
    h, w = gray.shape
    win = max(15, min(51, min(h, w) // 3)) | 1
    win = max(win, window)
    gray_f = gray.astype(np.float64)
    mean = cv2.boxFilter(gray_f, -1, (win, win))
    sqmean = cv2.boxFilter(gray_f ** 2, -1, (win, win))
    variance = np.maximum(sqmean - mean ** 2, 0)
    std = np.sqrt(variance)
    threshold = mean * (1.0 + k * (std / R - 1.0))
    binary = np.zeros_like(gray)
    binary[gray_f < threshold] = 255
    fg = np.sum(binary > 0) / binary.size
    if fg < 0.01 or fg > 0.60:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if 0.01 <= np.sum(otsu > 0) / otsu.size <= 0.60:
            binary = otsu
    return binary


def denoise_and_normalize(gray, strength=3):
    if strength > 1:
        gray = cv2.medianBlur(gray, strength | 1)
    h, w = gray.shape
    k_size = max(15, min(51, max(h, w) // 3)) | 1
    bg_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_k)
    norm = cv2.divide(gray, bg, scale=255)
    lo, hi = np.percentile(norm, (2, 98))
    if hi > lo:
        norm = np.clip((norm.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return norm


def morph_cleanup(binary):
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    r = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(r, cv2.MORPH_OPEN, open_k)


def cc_filter_old(binary):
    h, w = binary.shape
    min_area = max(10, int(h * w * 0.005))
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for i in range(1, nl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned if np.sum(cleaned) > 0 else binary


def cc_filter_lineaware(binary):
    h, w = binary.shape
    min_area = max(10, int(h * w * 0.005))
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for i in range(1, nl):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 6.0 and area < h * w * 0.02:
            continue
        cleaned[labels == i] = 255
    return cleaned if np.sum(cleaned) > 0 else binary


def stroke_norm_old(binary, min_stroke=2):
    if np.sum(binary) == 0:
        return binary
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink = dist[binary > 0]
    if len(ink) == 0:
        return binary
    avg = np.mean(ink) * 2
    if avg < min_stroke:
        k = max(2, int(min_stroke - avg) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel, iterations=1)
    elif avg > min_stroke * 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.erode(binary, kernel, iterations=1)
    return binary


def stroke_norm_local(binary, min_stroke=2):
    if np.sum(binary) == 0:
        return binary
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ink = dist[binary > 0]
    if len(ink) == 0:
        return binary
    avg = np.mean(ink) * 2
    target = float(min_stroke)
    if avg < target:
        thin_mask = np.zeros_like(binary)
        thin_mask[(binary > 0) & (dist < target / 2.0)] = 255
        k = max(2, int(target - avg) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(thin_mask, kernel, iterations=1)
        binary = cv2.bitwise_or(binary, dilated)
    elif avg > target * 3:
        thick_mask = np.zeros_like(binary)
        thick_mask[(binary > 0) & (dist > target * 1.5)] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        eroded = cv2.erode(binary, kernel, iterations=1)
        result = binary.copy()
        thick_d = cv2.dilate(thick_mask, kernel, iterations=1)
        result[thick_d > 0] = eroded[thick_d > 0]
        binary = result
    return binary


def remove_border_lines(binary):
    """Strict border removal — only fires on strong continuous ruling lines."""
    h, w = binary.shape
    if h < 20 or w < 20:
        return binary
    v_len = max(h // 2, 15)
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k)
    h_len = max(w // 2, 15)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_k)
    mx = max(4, w // 10)
    my = max(4, h // 10)
    mask = np.zeros_like(binary)
    lv = v_lines[:, :mx]
    if np.sum(lv > 0) > h * mx * 0.4:
        mask[:, :mx] = lv
    rv = v_lines[:, w - mx:]
    if np.sum(rv > 0) > h * mx * 0.4:
        mask[:, w - mx:] = rv
    th = h_lines[:my, :]
    if np.sum(th > 0) > w * my * 0.4:
        mask[:my, :] = th
    bh = h_lines[h - my:, :]
    if np.sum(bh > 0) > w * my * 0.4:
        mask[h - my:, :] = bh
    if np.sum(mask) == 0:
        return binary
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.subtract(binary, mask)


def center_resize(binary, target=64, padding=5):
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return None
    crop = binary[y:y+h, x:x+w]
    md = target - padding * 2
    r = min(md / w, md / h)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (nw, nh), interpolation=interp)
    canvas = np.zeros((target, target), dtype=np.uint8)
    xo, yo = (target - nw) // 2, (target - nh) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return cv2.bitwise_not(canvas)


# ---------------------------------------------------------------------------
# Pipeline variants
# ---------------------------------------------------------------------------

def pipeline_old(gray):
    """Original pipeline: median(3) + sauvola + morph + CC(area) + stroke_global"""
    norm = denoise_and_normalize(gray)
    binary = sauvola_binarize(norm)
    binary = morph_cleanup(binary)
    binary = cc_filter_old(binary)
    binary = stroke_norm_old(binary)
    return center_resize(binary)


def pipeline_A_border(gray):
    """OLD + border line removal only"""
    norm = denoise_and_normalize(gray)
    binary = sauvola_binarize(norm)
    binary = remove_border_lines(binary)  # <-- added
    binary = morph_cleanup(binary)
    binary = cc_filter_old(binary)
    binary = stroke_norm_old(binary)
    return center_resize(binary)


def pipeline_B_stroke(gray):
    """OLD + local stroke norm only"""
    norm = denoise_and_normalize(gray)
    binary = sauvola_binarize(norm)
    binary = morph_cleanup(binary)
    binary = cc_filter_old(binary)
    binary = stroke_norm_local(binary)  # <-- changed
    return center_resize(binary)


def pipeline_C_cc(gray):
    """OLD + line-aware CC filter only"""
    norm = denoise_and_normalize(gray)
    binary = sauvola_binarize(norm)
    binary = morph_cleanup(binary)
    binary = cc_filter_lineaware(binary)  # <-- changed
    binary = stroke_norm_old(binary)
    return center_resize(binary)


def pipeline_D_all(gray):
    """ALL three improvements combined"""
    norm = denoise_and_normalize(gray)
    binary = sauvola_binarize(norm)
    binary = remove_border_lines(binary)
    binary = morph_cleanup(binary)
    binary = cc_filter_lineaware(binary)
    binary = stroke_norm_local(binary)
    return center_resize(binary)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(img):
    if img is None:
        return None
    ink = cv2.bitwise_not(img)
    h, w = ink.shape
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    nc = nl - 1
    areas = stats[1:, cv2.CC_STAT_AREA] if nc > 0 else np.array([0])
    max_a = areas.max() if nc > 0 else 0
    total_ink = max(np.sum(ink > 0), 1)
    noise = sum(a for a in areas if a < max(10, int(h * w * 0.005)))
    lap = cv2.Laplacian(img, cv2.CV_64F).var()

    # Border detection
    mx = max(3, w // 8)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 3, 5)))
    lv = np.sum(cv2.morphologyEx(ink[:, :mx], cv2.MORPH_OPEN, vk) > 0)
    rv = np.sum(cv2.morphologyEx(ink[:, w-mx:], cv2.MORPH_OPEN, vk) > 0)
    border = lv > h * 0.3 or rv > h * 0.3

    return {
        "cc": nc, "lr": max_a / total_ink, "es": lap,
        "nr": noise / total_ink, "border": int(border),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    books = ["CacThanhTruyen2", "CacThanhTruyen4", "CacThanhTruyen11"]

    pipelines = {
        "OLD":      pipeline_old,
        "A:border": pipeline_A_border,
        "B:stroke": pipeline_B_stroke,
        "C:cc":     pipeline_C_cc,
        "D:all":    pipeline_D_all,
    }

    print("=" * 80)
    print("ABLATION TEST — isolating each improvement")
    print("=" * 80)

    # Collect all crops
    all_crops = []
    for book in books:
        crops = sorted(Path(f"prepared/{book}/detected/crops").rglob("*.png"))
        all_crops.extend(crops[:50])
    print(f"Total crops: {len(all_crops)}\n")

    # Run all pipelines
    results = {name: {"metrics": [], "time": 0, "fail": 0} for name in pipelines}

    for cp in all_crops:
        gray = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for name, fn in pipelines.items():
            t0 = time.time()
            out = fn(gray)
            results[name]["time"] += time.time() - t0
            if out is not None:
                m = metrics(out)
                if m:
                    results[name]["metrics"].append(m)
            else:
                results[name]["fail"] += 1

    # Print results
    def avg(mlist, key):
        v = [m[key] for m in mlist]
        return np.mean(v) if v else 0

    keys = [
        ("cc", "Avg CC (lower=cleaner)", True),
        ("lr", "Largest comp ratio (higher=better)", False),
        ("es", "Edge sharpness (higher=better)", False),
        ("nr", "Noise ratio (lower=better)", True),
        ("border", "Crops w/ border (lower=better)", True),
    ]

    header = f"  {'Metric':<38}"
    for name in pipelines:
        header += f" {name:>10}"
    print(header)
    print("  " + "-" * (38 + 11 * len(pipelines)))

    for key, label, lower_better in keys:
        row = f"  {label:<38}"
        old_val = avg(results["OLD"]["metrics"], key)
        for name in pipelines:
            val = avg(results[name]["metrics"], key)
            if key == "border":
                val = sum(m[key] for m in results[name]["metrics"])
                row += f" {val:>10}"
            elif key == "nr":
                row += f" {val:>10.4f}"
            elif key == "lr":
                row += f" {val:>10.3f}"
            else:
                row += f" {val:>10.1f}"
        print(row)

    # Time
    row = f"  {'Total time (s)':<38}"
    for name in pipelines:
        row += f" {results[name]['time']:>10.2f}"
    print(row)

    row = f"  {'Failures':<38}"
    for name in pipelines:
        row += f" {results[name]['fail']:>10}"
    print(row)

    # Delta from OLD
    print(f"\n  {'DELTA from OLD':<38}", end="")
    for name in pipelines:
        print(f" {name:>10}", end="")
    print()
    print("  " + "-" * (38 + 11 * len(pipelines)))

    for key, label, lower_better in keys:
        old_val = avg(results["OLD"]["metrics"], key)
        if key == "border":
            old_val = sum(m[key] for m in results["OLD"]["metrics"])
        short = label.split("(")[0].strip()
        row = f"  {short:<38}"
        for name in pipelines:
            val = avg(results[name]["metrics"], key)
            if key == "border":
                val = sum(m[key] for m in results[name]["metrics"])
            delta = val - old_val
            better = (delta < 0) if lower_better else (delta > 0)
            sign = "+" if delta > 0 else ""
            marker = " *" if better and name != "OLD" else "  " if name == "OLD" else ""
            if key == "nr":
                row += f" {sign}{delta:>7.4f}{marker}"
            elif key in ("lr",):
                row += f" {sign}{delta:>7.3f}{marker}"
            else:
                row += f" {sign}{delta:>7.1f}{marker}"
        print(row)

    print(f"\n  * = improvement over OLD")
    print("=" * 80)


if __name__ == "__main__":
    main()
