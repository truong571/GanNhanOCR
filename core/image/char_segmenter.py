"""Character segmentation within columns via projection profile.

Primary strategy when QN syllable count is known:
    Robust valley-based segmentation that ALWAYS returns `expected_count`
    bboxes. Finds all candidate valleys with relaxed parameters, then
    greedy-selects the ones closest to grid positions k×real_char_h with a
    prominence bonus. Missing positions are filled by the grid value (NOT
    whole-column equal-division). This is the integration of the
    `test_crop/robust_segmenter.py` sandbox after empirical validation.

Fallback (no expected_count, e.g. ad-hoc tools):
    Blob extraction + merge_small / split_large.
"""

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences


def segment_characters_in_column(
    binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    min_char_height: int = 15,
    expected_char_height: float | None = None,
    expected_count: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Detect characters in one column.

    When `expected_count` is given (the normal pipeline path), uses the robust
    valley segmenter which always returns exactly `expected_count` bboxes.

    Without `expected_count`, falls back to blob extraction + merge/split.

    Returns: [(x1, y1, x2, y2)] per character, top->bottom order.
    """
    x1, y1, x2, y2 = col_bbox
    col_binary = binary[y1:y2, x1:x2]
    col_h, col_w = col_binary.shape

    if col_w == 0 or col_h == 0:
        return []

    # Primary path: robust valley segmenter (always returns expected_count)
    if expected_count and expected_count > 0:
        return _robust_segment_by_valleys(col_binary, col_bbox, expected_count)

    # Fallback for callers without expected_count
    if expected_char_height is None:
        expected_char_height = col_h / 20
    raw_chars = _extract_raw_blobs(
        col_binary, col_bbox, expected_char_height, min_char_height, 0.01,
    )
    if not raw_chars:
        return []
    chars = _merge_small_boxes(raw_chars, expected_char_height)
    chars = _split_large_boxes(chars, expected_char_height, binary)
    return chars


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


def _robust_segment_by_valleys(
    col_binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    expected_count: int,
) -> list[tuple[int, int, int, int]]:
    """Always-returns-expected_count valley segmenter.

    Strategy:
        1. Strip column ruling lines + locate ink region
        2. find_peaks with relaxed distance — get all candidate valleys
        3. For each grid position k*real_char_h, greedy-pick the closest
           unused valley within tolerance (real_char_h*0.45). Score is
           distance penalised, prominence rewarded. If no valley qualifies,
           fall back to the grid position itself (per-position fill, NOT
           whole-column equal-division).
        4. Local refinement: nudge each split to the nearest local minimum.
        5. Build exactly expected_count bboxes from the resulting splits.

    Validated on 36 samples in test_crop/results_segment/: 0% of columns
    had to fall back to grid positions; the segmenter found real ink
    boundaries in every test column.
    """
    x1, y1, x2, y2 = col_bbox
    col_h, col_w = col_binary.shape

    if col_h == 0 or col_w == 0 or expected_count < 1:
        return []

    if expected_count == 1:
        return [(x1, y1, x2, y2)]

    # 1. Strip ruling lines + find ink region
    cleaned = _strip_column_borders(col_binary)
    ink_top, ink_bottom = _trim_to_ink_extent(cleaned, col_h / expected_count)
    ink_h = ink_bottom - ink_top

    # If ink region is too small to fit expected chars, the column is
    # effectively unusable — fall back to whole-column equal division so
    # alignment can still consume the page (rare).
    if ink_h < expected_count * 5:
        step = col_h / expected_count
        return [
            (x1, int(y1 + i * step), x2, int(y1 + (i + 1) * step))
            for i in range(expected_count)
        ]

    real_char_h = ink_h / expected_count
    ink_region = cleaned[ink_top:ink_bottom, :]
    h_proj = ink_region.sum(axis=1).astype(float)

    smooth_k = max(3, int(real_char_h * 0.10)) | 1
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    # 2. All candidate valleys (relaxed distance — never reject)
    min_dist = max(3, int(real_char_h * 0.20))
    valleys, _ = find_peaks(-h_smooth, distance=min_dist)
    if len(valleys) < expected_count // 2:
        min_dist2 = max(3, int(real_char_h * 0.10))
        valleys, _ = find_peaks(-h_smooth, distance=min_dist2)

    if len(valleys) > 0:
        prominences, _, _ = peak_prominences(-h_smooth, valleys)
    else:
        prominences = np.array([])

    # 3. Greedy-select expected_count - 1 split points
    target_positions = [k * real_char_h for k in range(1, expected_count)]
    selected: list[int] = []
    used: set[int] = set()
    tolerance = real_char_h * 0.45
    h_max = max(h_smooth.max(), 1.0)

    for target in target_positions:
        best_idx = None
        best_score = float("inf")
        for vi, v in enumerate(valleys):
            if vi in used:
                continue
            dist = abs(v - target)
            if dist > tolerance:
                continue
            prom_norm = float(prominences[vi]) / h_max if vi < len(prominences) else 0.0
            score = dist - prom_norm * real_char_h * 0.5
            if score < best_score:
                best_score = score
                best_idx = vi
        if best_idx is not None:
            selected.append(int(valleys[best_idx]))
            used.add(best_idx)
        else:
            selected.append(int(target))

    selected.sort()

    # 4. Local refinement
    refine_radius = max(3, int(real_char_h * 0.12))
    refined: list[int] = []
    for sp in selected:
        lo = max(0, sp - refine_radius)
        hi = min(ink_h, sp + refine_radius + 1)
        if hi > lo:
            refined.append(lo + int(np.argmin(h_proj[lo:hi])))
        else:
            refined.append(sp)

    # 5. Build expected_count bboxes
    points = [0] + sorted(refined) + [ink_h]

    # Defensive: dedupe + ensure we have exactly expected_count + 1 points
    deduped: list[int] = []
    for p in points:
        if not deduped or p > deduped[-1]:
            deduped.append(p)
    while len(deduped) < expected_count + 1:
        max_gap_i = max(
            range(len(deduped) - 1),
            key=lambda i: deduped[i + 1] - deduped[i],
        )
        mid = (deduped[max_gap_i] + deduped[max_gap_i + 1]) // 2
        deduped = deduped[:max_gap_i + 1] + [mid] + deduped[max_gap_i + 1:]

    return [
        (x1, int(y1 + ink_top + deduped[j]), x2, int(y1 + ink_top + deduped[j + 1]))
        for j in range(expected_count)
    ]


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
