"""Character segmentation within columns via projection profile."""

import cv2
import numpy as np
from scipy.signal import find_peaks


def segment_characters_in_column(
    binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    min_char_height: int = 15,
    expected_char_height: float | None = None,
    expected_count: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Detect characters in one column using multiple strategies.

    1. Valley snap-to-grid (preferred for dense text)
    2. Threshold + merge/split (for clean images with clear gaps)
    3. Trimmed equal-division (safe fallback)

    Returns: [(x1, y1, x2, y2)] per character, top->bottom order
    """
    x1, y1, x2, y2 = col_bbox
    col_binary = binary[y1:y2, x1:x2]
    col_h, col_w = col_binary.shape

    if col_w == 0 or col_h == 0:
        return []

    if expected_count and expected_count > 0:
        expected_char_height = col_h / expected_count
    if expected_char_height is None:
        expected_char_height = col_h / 20

    # Approach 1: Valley snap-to-grid
    if expected_count and expected_count > 1:
        valley_chars = _segment_by_valleys(
            col_binary, col_bbox, expected_count, col_h / expected_count
        )
        if valley_chars and len(valley_chars) == expected_count:
            return valley_chars

    # Approach 2: Threshold + merge/split
    raw_chars = _extract_raw_blobs(
        col_binary, col_bbox, expected_char_height, min_char_height, 0.01
    )
    if raw_chars:
        chars = _merge_small_boxes(raw_chars, expected_char_height)
        chars = _split_large_boxes(chars, expected_char_height, binary)
        if expected_count and len(chars) < expected_count:
            chars = _force_split_to_count(chars, expected_count, expected_char_height, binary)
        if expected_count and len(chars) == expected_count:
            return chars

        best_chars = chars
        best_diff = abs(len(chars) - (expected_count or len(chars)))

        for density_scale in [0.02, 0.05, 0.005]:
            retry_raw = _extract_raw_blobs(
                col_binary, col_bbox, col_h / max(expected_count or 20, 1),
                min_char_height, density_scale,
            )
            if not retry_raw:
                continue
            retry_h = col_h / max(expected_count or 20, 1)
            retry_chars = _merge_small_boxes(retry_raw, retry_h)
            retry_chars = _split_large_boxes(retry_chars, retry_h, binary)
            if expected_count and len(retry_chars) < expected_count:
                retry_chars = _force_split_to_count(
                    retry_chars, expected_count, retry_h, binary
                )
            retry_diff = abs(len(retry_chars) - (expected_count or len(retry_chars)))
            if retry_diff < best_diff:
                best_diff = retry_diff
                best_chars = retry_chars
            if best_diff == 0:
                return best_chars

        if not expected_count:
            return best_chars

    # Approach 3: Trimmed equal-division
    if expected_count and expected_count > 0:
        cleaned = _strip_column_borders(col_binary)
        ink_top, ink_bottom = _trim_to_ink_extent(cleaned, col_h / expected_count)
        ink_h = ink_bottom - ink_top
        if ink_h > col_h * 0.3:
            step = ink_h / expected_count
            return [
                (x1, int(y1 + ink_top + i * step), x2, int(y1 + ink_top + (i + 1) * step))
                for i in range(expected_count)
            ]
        step = col_h / expected_count
        return [
            (x1, int(y1 + i * step), x2, int(y1 + (i + 1) * step))
            for i in range(expected_count)
        ]

    return raw_chars if raw_chars else []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_raw_blobs(
    col_binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    expected_char_height: float,
    min_char_height: int,
    density_threshold_scale: float = 0.01,
) -> list[tuple[int, int, int, int]]:
    """Extract raw blobs from horizontal projection."""
    x1, y1, x2, y2 = col_bbox
    col_h, col_w = col_binary.shape

    h_proj = col_binary.sum(axis=1).astype(float)
    smooth_k = max(3, int(expected_char_height * 0.1))
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    threshold = max(col_w * density_threshold_scale, 0.5)
    is_gap = h_smooth < threshold
    min_gap = max(2, int(expected_char_height * 0.03))
    gap_kernel = np.ones(min_gap)
    is_gap_clean = np.convolve(is_gap.astype(float), gap_kernel / min_gap, mode="same") > 0.8

    transitions = np.diff(is_gap_clean.astype(int))
    char_starts = np.where(transitions == -1)[0]
    char_ends = np.where(transitions == 1)[0]

    if len(char_starts) == 0 and len(char_ends) == 0:
        return []
    if len(char_starts) == 0:
        char_starts = np.array([0])
    if len(char_ends) == 0:
        char_ends = np.array([col_h])
    if char_ends[0] < char_starts[0]:
        char_starts = np.insert(char_starts, 0, 0)
    if char_starts[-1] > char_ends[-1]:
        char_ends = np.append(char_ends, col_h)

    n = min(len(char_starts), len(char_ends))
    raw_chars = []
    for i in range(n):
        ch = char_ends[i] - char_starts[i]
        if ch >= min_char_height:
            raw_chars.append((x1, int(char_starts[i] + y1), x2, int(char_ends[i] + y1)))
    return raw_chars


def _strip_column_borders(col_binary: np.ndarray) -> np.ndarray:
    """Remove ruling lines (vertical/horizontal) at column edges."""
    h, w = col_binary.shape
    if h == 0 or w == 0:
        return col_binary

    cleaned = col_binary.copy()
    margin_x = max(10, w // 4)

    # Vertical border removal via morphology
    v_kernel_len = max(30, h // 20)
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, v_k)

    left_strip = v_lines[:, :margin_x]
    if left_strip.any():
        v_proj_left = left_strip.sum(axis=0)
        for x in range(margin_x):
            if v_proj_left[x] > h * 0.05:
                cleaned[:, max(0, x - 2):min(w, x + 3)] = 0

    right_strip = v_lines[:, w - margin_x:]
    if right_strip.any():
        v_proj_right = right_strip.sum(axis=0)
        for x in range(len(v_proj_right)):
            if v_proj_right[x] > h * 0.05:
                abs_x = w - margin_x + x
                cleaned[:, max(0, abs_x - 2):min(w, abs_x + 3)] = 0

    # Projection fallback for thick ruling lines
    v_proj = cleaned.sum(axis=0).astype(float)
    v_threshold = h * 0.30

    left_trim = 0
    for x in range(min(margin_x, 40)):
        if v_proj[x] > v_threshold:
            left_trim = x + 4
    if left_trim > 0:
        cleaned[:, :left_trim] = 0

    right_trim = w
    for x in range(w - 1, max(w - margin_x, w - 40), -1):
        if v_proj[x] > v_threshold:
            right_trim = x - 4
    if right_trim < w:
        cleaned[:, right_trim:] = 0

    # Horizontal border removal
    margin_y = max(10, h // 6)
    h_kernel_len = max(20, w // 3)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, h_k)

    top_strip = h_lines[:margin_y, :]
    if top_strip.any():
        h_proj_top = top_strip.sum(axis=1)
        top_trim = 0
        for y in range(margin_y):
            if h_proj_top[y] > w * 0.15:
                top_trim = y + 4
        if top_trim > 0:
            cleaned[:top_trim, :] = 0

    bottom_strip = h_lines[h - margin_y:, :]
    if bottom_strip.any():
        h_proj_bot = bottom_strip.sum(axis=1)
        bottom_trim = h
        for y in range(len(h_proj_bot) - 1, -1, -1):
            if h_proj_bot[y] > w * 0.15:
                bottom_trim = h - margin_y + y - 4
        if bottom_trim < h:
            cleaned[bottom_trim:, :] = 0

    return cleaned


def _trim_to_ink_extent(
    col_binary: np.ndarray, expected_char_height: float
) -> tuple[int, int]:
    """Find actual ink extent in column (skip blank regions)."""
    h_proj = col_binary.sum(axis=1).astype(float)
    col_h = col_binary.shape[0]
    col_w = col_binary.shape[1]

    if col_h == 0:
        return 0, col_h

    smooth_k = max(5, int(expected_char_height * 0.5)) | 1
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    if h_smooth.max() == 0:
        return 0, col_h

    win = max(5, int(expected_char_height * 0.3)) | 1
    windowed = np.convolve(h_proj, np.ones(win) / win, mode="same")

    min_abs_threshold = col_w * 0.05
    nonzero = windowed[windowed > min_abs_threshold]
    if len(nonzero) == 0:
        nonzero = windowed[windowed > col_w * 0.02]
        if len(nonzero) == 0:
            return 0, col_h
    ink_threshold = max(min_abs_threshold, np.percentile(nonzero, 15) * 0.5)

    ink_rows = np.where(windowed > ink_threshold)[0]
    if len(ink_rows) == 0:
        return 0, col_h

    groups = []
    g_start = ink_rows[0]
    for i in range(1, len(ink_rows)):
        if ink_rows[i] - ink_rows[i - 1] > expected_char_height * 0.5:
            groups.append((g_start, ink_rows[i - 1]))
            g_start = ink_rows[i]
    groups.append((g_start, ink_rows[-1]))

    longest = max(groups, key=lambda g: g[1] - g[0])
    top = max(0, longest[0] - 3)
    bottom = min(col_h, longest[1] + 3)
    return top, bottom


def _segment_by_valleys(
    col_binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    expected_count: int,
    expected_char_height: float,
) -> list[tuple[int, int, int, int]]:
    """Segment characters using valley detection (snap-to-grid)."""
    x1, y1, x2, y2 = col_bbox
    col_h, col_w = col_binary.shape

    if col_h == 0 or col_w == 0 or expected_count < 2:
        return []

    cleaned = _strip_column_borders(col_binary)
    ink_top, ink_bottom = _trim_to_ink_extent(cleaned, expected_char_height)
    ink_region = cleaned[ink_top:ink_bottom, :]
    ink_h = ink_bottom - ink_top

    if ink_h < expected_char_height:
        return []

    real_char_h = ink_h / expected_count
    h_proj = ink_region.sum(axis=1).astype(float)
    smooth_k = max(3, int(real_char_h * 0.10)) | 1
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    min_dist = max(3, int(real_char_h * 0.2))
    valleys, _ = find_peaks(-h_smooth, distance=min_dist)

    if len(valleys) < expected_count - 1:
        min_dist2 = max(3, int(real_char_h * 0.1))
        valleys, _ = find_peaks(-h_smooth, distance=min_dist2)

    if len(valleys) < expected_count - 1:
        return []

    # Snap to grid
    split_points = []
    used_valleys: set[int] = set()
    for k in range(1, expected_count):
        target = k * real_char_h
        best_valley = None
        best_dist = float("inf")
        for vi, v in enumerate(valleys):
            if vi in used_valleys:
                continue
            dist = abs(v - target)
            if dist < best_dist and dist < real_char_h * 0.5:
                best_dist = dist
                best_valley = vi
        if best_valley is not None:
            split_points.append(int(valleys[best_valley]))
            used_valleys.add(best_valley)
        else:
            split_points.append(int(target))
    split_points.sort()

    # Local refinement
    refine_radius = max(3, int(real_char_h * 0.15))
    refined = []
    for sp in split_points:
        lo = max(0, sp - refine_radius)
        hi = min(ink_h, sp + refine_radius + 1)
        if hi > lo:
            local_proj = h_proj[lo:hi]
            refined.append(lo + int(np.argmin(local_proj)))
        else:
            refined.append(sp)

    all_points = [0] + refined + [ink_h]
    max_part = max(all_points[j + 1] - all_points[j] for j in range(len(all_points) - 1))
    if max_part > real_char_h * 3.0:
        return []

    chars = []
    for j in range(len(all_points) - 1):
        cy1 = y1 + ink_top + all_points[j]
        cy2 = y1 + ink_top + all_points[j + 1]
        if cy2 - cy1 >= 8:
            chars.append((x1, int(cy1), x2, int(cy2)))
    return chars


def _merge_small_boxes(
    chars: list[tuple[int, int, int, int]], expected_h: float,
) -> list[tuple[int, int, int, int]]:
    """Merge boxes smaller than 40% expected height with neighbors."""
    if not chars:
        return chars
    merge_threshold = expected_h * 0.4
    gap_threshold = expected_h * 0.3

    merged = [chars[0]]
    for i in range(1, len(chars)):
        curr = chars[i]
        prev = merged[-1]
        curr_h = curr[3] - curr[1]
        prev_h = prev[3] - prev[1]
        gap = curr[1] - prev[3]

        if curr_h < merge_threshold and gap < gap_threshold:
            merged[-1] = (prev[0], prev[1], prev[2], curr[3])
        elif prev_h < merge_threshold and gap < gap_threshold:
            merged[-1] = (curr[0], prev[1], curr[2], curr[3])
        else:
            merged.append(curr)
    return merged


def _split_large_boxes(
    chars: list[tuple[int, int, int, int]],
    expected_h: float,
    binary: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Split boxes larger than 1.5x expected height."""
    split_threshold = expected_h * 1.5
    result = []

    for bbox in chars:
        x1, y1, x2, y2 = bbox
        box_h = y2 - y1

        if box_h > split_threshold:
            n_parts = max(2, min(round(box_h / expected_h), 5))
            col_binary = binary[y1:y2, x1:x2]
            h_proj = col_binary.sum(axis=1).astype(float)
            smooth_k = max(3, int(expected_h * 0.05))
            h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")
            min_dist = int(expected_h * 0.35)
            valleys, _ = find_peaks(-h_smooth, distance=min_dist)

            if len(valleys) >= n_parts - 1:
                depths = h_smooth[valleys]
                best_idx = np.argsort(depths)[:n_parts - 1]
                split_points = sorted(valleys[best_idx])
            else:
                split_points = [int(box_h * i / n_parts) for i in range(1, n_parts)]

            all_points = [0] + list(split_points) + [box_h]
            for j in range(len(all_points) - 1):
                sy1 = y1 + all_points[j]
                sy2 = y1 + all_points[j + 1]
                if sy2 - sy1 >= 10:
                    result.append((x1, sy1, x2, sy2))
        else:
            result.append(bbox)
    return result


def _force_split_to_count(
    chars: list[tuple[int, int, int, int]],
    target_count: int,
    expected_h: float,
    binary: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Force-split largest blobs until reaching target count."""
    result = list(chars)
    max_iter = target_count * 2

    while len(result) < target_count and max_iter > 0:
        max_iter -= 1
        heights = [(b[3] - b[1], i) for i, b in enumerate(result)]
        heights.sort(reverse=True)

        split_done = False
        for h_val, idx in heights:
            if h_val < expected_h * 1.2:
                break

            bbox = result[idx]
            bx1, by1, bx2, by2 = bbox
            deficit = target_count - len(result)
            n_parts = max(2, min(round(h_val / expected_h), deficit + 1))

            col_bin = binary[by1:by2, bx1:bx2]
            h_proj = col_bin.sum(axis=1).astype(float)
            smooth_k = max(3, int(expected_h * 0.05))
            h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")
            min_dist = int(expected_h * 0.35)
            valleys, _ = find_peaks(-h_smooth, distance=min_dist)

            if len(valleys) >= n_parts - 1:
                depths = h_smooth[valleys]
                best_idx = np.argsort(depths)[:n_parts - 1]
                split_pts = sorted(valleys[best_idx])
            else:
                box_h = by2 - by1
                split_pts = [int(box_h * i / n_parts) for i in range(1, n_parts)]

            sub_boxes = []
            all_pts = [0] + list(split_pts) + [by2 - by1]
            for j in range(len(all_pts) - 1):
                sy1 = by1 + all_pts[j]
                sy2 = by1 + all_pts[j + 1]
                if sy2 - sy1 >= 8:
                    sub_boxes.append((bx1, sy1, bx2, sy2))

            if len(sub_boxes) > 1:
                result = result[:idx] + sub_boxes + result[idx + 1:]
                if len(result) > target_count:
                    result = result[:target_count]
                split_done = True
                break

        if not split_done:
            break
    return result
