"""Compare old vs new preprocessing pipeline on real data.

Runs both the OLD pipeline (before improvements) and the NEW pipeline
on actual character crops and page images, reporting quality metrics.

Usage: python tests/test_preprocessing_compare.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# OLD pipeline functions (copy of original code before changes)
# ---------------------------------------------------------------------------

def _old_denoise_image(gray: np.ndarray) -> np.ndarray:
    """OLD: GaussianBlur(3,3) based."""
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


class OldCharacterCleaner:
    """OLD pipeline: MedianBlur + simple CC filter + global stroke norm."""

    def __init__(self, target_size=64, padding=5, sauvola_k=0.2,
                 sauvola_window=25, sauvola_R=128.0, denoise_strength=3,
                 min_stroke=2):
        self.target_size = target_size
        self.padding = padding
        self.sauvola_k = sauvola_k
        self.sauvola_window = sauvola_window | 1
        self.sauvola_R = sauvola_R
        self.denoise_strength = denoise_strength | 1
        self.min_stroke = min_stroke

    def clean(self, gray: np.ndarray) -> np.ndarray | None:
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # OLD: MedianBlur
        if self.denoise_strength > 1:
            denoised = cv2.medianBlur(gray, self.denoise_strength)
        else:
            denoised = gray

        # Background norm
        h, w = denoised.shape
        k_size = max(15, min(51, max(h, w) // 3)) | 1
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, bg_kernel)
        normalized = cv2.divide(denoised, background, scale=255)
        p_low, p_high = np.percentile(normalized, (2, 98))
        if p_high > p_low:
            normalized = np.clip(
                (normalized.astype(np.float32) - p_low) * 255 / (p_high - p_low),
                0, 255,
            ).astype(np.uint8)

        # Sauvola
        win = max(15, min(51, min(h, w) // 3)) | 1
        win = max(win, self.sauvola_window)
        gray_f = normalized.astype(np.float64)
        mean = cv2.boxFilter(gray_f, -1, (win, win))
        sqmean = cv2.boxFilter(gray_f ** 2, -1, (win, win))
        variance = np.maximum(sqmean - mean ** 2, 0)
        std = np.sqrt(variance)
        threshold = mean * (1.0 + self.sauvola_k * (std / self.sauvola_R - 1.0))
        binary = np.zeros_like(normalized)
        binary[gray_f < threshold] = 255

        fg_ratio = np.sum(binary > 0) / binary.size
        if fg_ratio < 0.01 or fg_ratio > 0.60:
            _, otsu_binary = cv2.threshold(
                normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            otsu_fg = np.sum(otsu_binary > 0) / otsu_binary.size
            if 0.01 <= otsu_fg <= 0.60:
                binary = otsu_binary

        # NO border line removal (old pipeline)
        # Morph cleanup
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

        # OLD: Simple CC removal (area only)
        min_area = max(10, int(h * w * 0.005))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        if np.sum(cleaned) > 0:
            binary = cleaned

        # OLD: Global stroke norm
        if np.sum(binary) > 0:
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            ink_dist = dist[binary > 0]
            if len(ink_dist) > 0:
                avg_t = np.mean(ink_dist) * 2
                if avg_t < self.min_stroke:
                    k_sz = max(2, int(self.min_stroke - avg_t) + 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
                    binary = cv2.dilate(binary, kernel, iterations=1)
                elif avg_t > self.min_stroke * 3:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    binary = cv2.erode(binary, kernel, iterations=1)

        # Center and resize
        coords = cv2.findNonZero(binary)
        if coords is None:
            return None
        x, y, bw, bh = cv2.boundingRect(coords)
        if bw == 0 or bh == 0:
            return None
        char_crop = binary[y:y + bh, x:x + bw]
        max_dim = self.target_size - (self.padding * 2)
        ratio = min(max_dim / bw, max_dim / bh)
        new_w = max(1, int(bw * ratio))
        new_h = max(1, int(bh * ratio))
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(char_crop, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        x_off = (self.target_size - new_w) // 2
        y_off = (self.target_size - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return cv2.bitwise_not(canvas)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(binary_image: np.ndarray) -> dict:
    """Compute quality metrics on a cleaned binary character image."""
    if binary_image is None:
        return {"error": True}

    # Invert so ink=255 for analysis (cleaned output is black-on-white)
    ink = cv2.bitwise_not(binary_image)
    h, w = ink.shape

    fg_ratio = np.sum(ink > 0) / ink.size

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    num_components = num_labels - 1  # exclude background

    # Largest component ratio
    if num_components > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = areas.max()
        total_ink = np.sum(ink > 0)
        largest_ratio = largest_area / max(total_ink, 1)
    else:
        largest_ratio = 0

    # Edge sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(binary_image, cv2.CV_64F)
    edge_sharpness = laplacian.var()

    # Noise score: ratio of small components to total ink
    noise_area = 0
    if num_components > 1:
        min_area = max(10, int(h * w * 0.005))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                noise_area += stats[i, cv2.CC_STAT_AREA]
    noise_ratio = noise_area / max(np.sum(ink > 0), 1)

    return {
        "fg_ratio": fg_ratio,
        "num_components": num_components,
        "largest_component_ratio": largest_ratio,
        "edge_sharpness": edge_sharpness,
        "noise_ratio": noise_ratio,
    }


def detect_border_lines(binary_image: np.ndarray) -> dict:
    """Detect if border lines remain in cleaned image."""
    ink = cv2.bitwise_not(binary_image)
    h, w = ink.shape

    # Check for vertical lines at edges
    margin = max(3, w // 8)
    left_strip = ink[:, :margin]
    right_strip = ink[:, w - margin:]

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 3, 5)))
    left_vlines = cv2.morphologyEx(left_strip, cv2.MORPH_OPEN, v_kernel)
    right_vlines = cv2.morphologyEx(right_strip, cv2.MORPH_OPEN, v_kernel)

    left_line_px = np.sum(left_vlines > 0)
    right_line_px = np.sum(right_vlines > 0)

    has_border = (left_line_px > h * 0.3) or (right_line_px > h * 0.3)
    return {
        "has_border_lines": has_border,
        "left_line_pixels": int(left_line_px),
        "right_line_pixels": int(right_line_px),
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_book_test(book_name: str, new_denoise_image, NewCharacterCleaner):
    """Run comparison for a single book. Returns summary dict."""
    base_dir = Path(f"prepared/{book_name}")
    crops_dir = base_dir / "detected" / "crops"
    pages_dir = base_dir / "pages"

    crop_files = sorted(crops_dir.rglob("*.png"))
    page_files = sorted(pages_dir.glob("*.png"))

    if not crop_files:
        print(f"  SKIP: No crop files for {book_name}")
        return None

    print(f"\n{'='*70}")
    print(f"  BOOK: {book_name}  ({len(crop_files)} crops, {len(page_files)} pages)")
    print(f"{'='*70}")

    result = {"book": book_name}

    # --- Page-level ---
    if page_files:
        gray = cv2.imread(str(page_files[0]), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            t0 = time.time()
            old_dn = _old_denoise_image(gray)
            old_t = time.time() - t0

            t0 = time.time()
            new_dn = new_denoise_image(gray)
            new_t = time.time() - t0

            old_lap = cv2.Laplacian(old_dn, cv2.CV_64F).var()
            new_lap = cv2.Laplacian(new_dn, cv2.CV_64F).var()

            result["page_old_edge"] = old_lap
            result["page_new_edge"] = new_lap
            result["page_old_time"] = old_t
            result["page_new_time"] = new_t

            print(f"\n  Page denoise: edge sharpness OLD={old_lap:.0f} NEW={new_lap:.0f}  "
                  f"({'NEW' if new_lap > old_lap else 'OLD'} better)")

    # --- Crop-level ---
    old_cleaner = OldCharacterCleaner()
    new_cleaner = NewCharacterCleaner()

    sample_crops = crop_files[:80]

    old_metrics, new_metrics = [], []
    old_border, new_border = 0, 0
    old_time_total, new_time_total = 0.0, 0.0

    for cp in sample_crops:
        gray = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        t0 = time.time()
        old_r = old_cleaner.clean(gray)
        old_time_total += time.time() - t0

        t0 = time.time()
        new_r, _ = new_cleaner.clean(gray)
        new_time_total += time.time() - t0

        if old_r is not None:
            old_metrics.append(compute_metrics(old_r))
            if detect_border_lines(old_r)["has_border_lines"]:
                old_border += 1
        if new_r is not None:
            new_metrics.append(compute_metrics(new_r))
            if detect_border_lines(new_r)["has_border_lines"]:
                new_border += 1

    n = len(sample_crops)

    def avg(mlist, key):
        v = [m[key] for m in mlist if key in m]
        return np.mean(v) if v else 0

    old_cc = avg(old_metrics, "num_components")
    new_cc = avg(new_metrics, "num_components")
    old_lr = avg(old_metrics, "largest_component_ratio")
    new_lr = avg(new_metrics, "largest_component_ratio")
    old_es = avg(old_metrics, "edge_sharpness")
    new_es = avg(new_metrics, "edge_sharpness")
    old_nr = avg(old_metrics, "noise_ratio")
    new_nr = avg(new_metrics, "noise_ratio")
    old_fg = avg(old_metrics, "fg_ratio")
    new_fg = avg(new_metrics, "fg_ratio")
    old_ms = old_time_total / max(n, 1) * 1000
    new_ms = new_time_total / max(n, 1) * 1000

    result.update({
        "n_samples": n,
        "old_success": len(old_metrics), "new_success": len(new_metrics),
        "old_ms": old_ms, "new_ms": new_ms,
        "old_fg": old_fg, "new_fg": new_fg,
        "old_cc": old_cc, "new_cc": new_cc,
        "old_lr": old_lr, "new_lr": new_lr,
        "old_es": old_es, "new_es": new_es,
        "old_nr": old_nr, "new_nr": new_nr,
        "old_border": old_border, "new_border": new_border,
    })

    print(f"\n  {'Metric':<35} {'OLD':<16} {'NEW':<16} {'Better'}")
    print(f"  {'-'*35} {'-'*16} {'-'*16} {'-'*8}")
    print(f"  {'Success':<35} {len(old_metrics)}/{n:<12} {len(new_metrics)}/{n:<12} {'NEW' if len(new_metrics)>=len(old_metrics) else 'OLD'}")
    print(f"  {'Avg ms/crop':<35} {old_ms:<16.1f} {new_ms:<16.1f} {'OLD' if old_ms<new_ms else 'NEW'}")
    print(f"  {'Foreground ratio':<35} {old_fg:<16.3f} {new_fg:<16.3f} =")
    print(f"  {'Connected components':<35} {old_cc:<16.1f} {new_cc:<16.1f} {'NEW' if new_cc<=old_cc else 'OLD'}")
    print(f"  {'Largest comp ratio':<35} {old_lr:<16.3f} {new_lr:<16.3f} {'NEW' if new_lr>=old_lr else 'OLD'}")
    print(f"  {'Edge sharpness':<35} {old_es:<16.1f} {new_es:<16.1f} {'NEW' if new_es>=old_es else 'OLD'}")
    print(f"  {'Noise ratio':<35} {old_nr:<16.4f} {new_nr:<16.4f} {'NEW' if new_nr<=old_nr else 'OLD'}")
    print(f"  {'Border lines remaining':<35} {old_border:<16} {new_border:<16} {'NEW' if new_border<=old_border else 'OLD'}")

    # Save visual samples
    output_dir = Path(f"tests/comparison_output/{book_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(10, len(sample_crops))):
        gray = cv2.imread(str(sample_crops[i]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        old_r = old_cleaner.clean(gray)
        new_r, _ = new_cleaner.clean(gray)
        t = 64
        orig = cv2.resize(gray, (t, t))
        parts = [orig,
                 old_r if old_r is not None else np.ones((t, t), np.uint8) * 128,
                 new_r if new_r is not None else np.ones((t, t), np.uint8) * 128]
        comp = np.hstack(parts)
        comp[:, t] = 0
        comp[:, t * 2] = 0
        cv2.imwrite(str(output_dir / f"{sample_crops[i].parent.name}_{sample_crops[i].stem}.png"), comp)
    print(f"  Saved 10 samples to {output_dir}/")

    return result


def main():
    from lib.image_processing import denoise_image as new_denoise_image
    from lib.crop_cleaner import CharacterCleaner as NewCharacterCleaner

    books = ["CacThanhTruyen2", "CacThanhTruyen4", "CacThanhTruyen11"]

    print("=" * 70)
    print("PREPROCESSING COMPARISON: OLD vs NEW — ALL BOOKS")
    print("=" * 70)

    results = []
    for book in books:
        r = run_book_test(book, new_denoise_image, NewCharacterCleaner)
        if r:
            results.append(r)

    # --- Cross-book summary ---
    print(f"\n{'='*70}")
    print("CROSS-BOOK SUMMARY")
    print(f"{'='*70}\n")

    header = f"  {'Metric':<30}"
    for r in results:
        header += f" {r['book'][-1:]:>5}old {r['book'][-1:]:>5}new"
    print(header)
    print("  " + "-" * (30 + 12 * len(results)))

    for key, label, lower_better in [
        ("cc", "Connected components", True),
        ("lr", "Largest comp ratio", False),
        ("es", "Edge sharpness", False),
        ("nr", "Noise ratio", True),
        ("border", "Border lines", True),
        ("ms", "Avg ms/crop", True),
    ]:
        row = f"  {label:<30}"
        for r in results:
            o = r.get(f"old_{key}", 0)
            n = r.get(f"new_{key}", 0)
            fmt = ".4f" if key == "nr" else (".1f" if isinstance(o, float) else "d")
            row += f" {o:>8{fmt}} {n:>8{fmt}}"
        print(row)

    # Overall score
    print(f"\n  {'Book':<25} {'Improvements':<15} {'Regressions':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    total_imp, total_reg = 0, 0
    for r in results:
        imp, reg = 0, 0
        checks = [
            r["new_es"] >= r["old_es"],
            r["new_nr"] <= r["old_nr"],
            r["new_cc"] <= r["old_cc"],
            r["new_lr"] >= r["old_lr"],
            r["new_border"] <= r["old_border"],
            r["new_success"] >= r["old_success"],
        ]
        imp = sum(checks)
        reg = len(checks) - imp
        total_imp += imp
        total_reg += reg
        print(f"  {r['book']:<25} {imp}/6{'':<11} {reg}/6")

    print(f"  {'TOTAL':<25} {total_imp}/{6*len(results):<11}    {total_reg}/{6*len(results)}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
