"""Column detection using vertical projection + ruling line morphology."""

import cv2
import numpy as np
from scipy.signal import find_peaks


def detect_columns(
    binary: np.ndarray,
    text_box: tuple[int, int, int, int],
    n_expected: int = 9,
) -> list[tuple[int, int, int, int]]:
    """Detect vertical text columns within text area.

    Strategy:
    1. Detect vertical ruling lines via morphology -> column boundaries
    2. For wide gaps (>1.5x expected width) -> find valleys via projection
    3. Fallback: pure vertical projection valley detection

    Returns: [(x1, y1, x2, y2)] per column, ordered right->left
    """
    left, top, right, bottom = text_box
    text_region = binary[top:bottom, left:right]
    region_h = bottom - top
    region_w = right - left

    if region_w == 0 or n_expected < 1:
        return [(left, top, right, bottom)]

    expected_col_width = region_w / n_expected

    # Vertical projection for valley detection
    v_proj = text_region.sum(axis=0).astype(float)
    kernel = max(5, int(expected_col_width / 8))
    v_smooth = np.convolve(v_proj, np.ones(kernel) / kernel, mode="same")

    # Step 1: Detect ruling lines
    ruling_positions: list[int] = []
    if region_h > 200:
        v_kernel_len = max(100, region_h // 10)
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(text_region, cv2.MORPH_OPEN, v_k)
        v_proj_lines = v_lines.sum(axis=0)
        ruling_peaks, _ = find_peaks(
            v_proj_lines.astype(float), height=region_h * 0.05, distance=30
        )
        ruling_positions = list(ruling_peaks)

    # Step 2: Build column boundaries
    if len(ruling_positions) >= 2:
        boundaries = list(ruling_positions)

        # Find sub-boundaries for wide gaps
        all_positions = sorted([0] + boundaries + [region_w])
        for i in range(len(all_positions) - 1):
            gap_l = all_positions[i]
            gap_r = all_positions[i + 1]
            gap_w = gap_r - gap_l
            n_sub = round(gap_w / expected_col_width)

            if n_sub >= 2:
                margin = max(40, int(expected_col_width * 0.2))
                sl = gap_l + margin
                sr = gap_r - margin
                if sr > sl:
                    sub_proj = v_smooth[sl:sr]
                    sub_valleys, _ = find_peaks(
                        -sub_proj, distance=int(expected_col_width * 0.3)
                    )
                    if len(sub_valleys) >= n_sub - 1:
                        depths = sub_proj[sub_valleys]
                        best = sorted(sub_valleys[np.argsort(depths)[:n_sub - 1]])
                        for v in best:
                            boundaries.append(sl + v)
                    else:
                        for j in range(1, n_sub):
                            boundaries.append(int(gap_l + j * gap_w / n_sub))

        boundaries = sorted(set(boundaries))
        boundaries = [b for b in boundaries if 30 < b < region_w - 30]

        # Too many -> remove boundary creating narrowest column
        while len(boundaries) > n_expected - 1:
            all_b = [0] + boundaries + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            min_idx = int(np.argmin(widths))
            if min_idx == 0:
                boundaries.pop(0)
            elif min_idx == len(widths) - 1:
                boundaries.pop(-1)
            else:
                if widths[min_idx - 1] < widths[min_idx + 1]:
                    boundaries.pop(min_idx - 1)
                else:
                    boundaries.pop(min_idx)

        # Too few -> split widest gap
        while len(boundaries) < n_expected - 1:
            all_b = [0] + boundaries + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            widest = int(np.argmax(widths))
            mid = (all_b[widest] + all_b[widest + 1]) // 2
            boundaries.append(mid)
            boundaries.sort()

        # Remove columns narrower than 40% expected width
        min_col_w = int(expected_col_width * 0.4)
        fixed = True
        while fixed:
            fixed = False
            all_b = [0] + boundaries + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            for idx, w in enumerate(widths):
                if w < min_col_w:
                    if idx == 0:
                        boundaries.pop(0)
                    elif idx == len(widths) - 1:
                        boundaries.pop(-1)
                    else:
                        boundaries.pop(idx)
                    # Re-add in widest column
                    all_b2 = [0] + boundaries + [region_w]
                    widths2 = [all_b2[i + 1] - all_b2[i] for i in range(len(all_b2) - 1)]
                    widest = int(np.argmax(widths2))
                    wl, wr = all_b2[widest], all_b2[widest + 1]
                    m = max(40, int(expected_col_width * 0.2))
                    if wr - wl > 2 * m:
                        sub = v_smooth[wl + m:wr - m]
                        sv, _ = find_peaks(-sub, distance=int(expected_col_width * 0.3))
                        if len(sv) > 0:
                            best_v = sv[np.argmin(sub[sv])]
                            boundaries.append(wl + m + best_v)
                        else:
                            boundaries.append((wl + wr) // 2)
                    else:
                        boundaries.append((wl + wr) // 2)
                    boundaries.sort()
                    fixed = True
                    break

        best_valleys = boundaries
    else:
        # Fallback: valley-based approach
        min_gap_distance = int(expected_col_width * 0.5)
        valleys, _ = find_peaks(-v_smooth, distance=min_gap_distance)
        valley_depths = v_smooth[valleys]

        if len(valleys) >= n_expected - 1:
            sorted_idx = np.argsort(valley_depths)
            best_valleys = sorted(valleys[sorted_idx[:n_expected - 1]])
        elif len(valleys) > 0:
            best_valleys = sorted(valleys)
        else:
            best_valleys = [int(region_w * i / n_expected) for i in range(1, n_expected)]

        while len(best_valleys) < n_expected - 1:
            all_b = [0] + list(best_valleys) + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            widest = int(np.argmax(widths))
            mid = (all_b[widest] + all_b[widest + 1]) // 2
            best_valleys.append(mid)
            best_valleys.sort()

    # Build column bboxes
    all_bounds = [0] + list(best_valleys) + [region_w]
    columns = []
    for i in range(len(all_bounds) - 1):
        x1 = all_bounds[i] + left
        x2 = all_bounds[i + 1] + left
        columns.append((x1, top, x2, bottom))

    # Right->left order (column 1 = rightmost)
    columns.reverse()
    return columns


def auto_detect_n_columns(
    binary: np.ndarray,
    text_box: tuple[int, int, int, int],
) -> int:
    """Auto-detect column count from vertical projection peaks."""
    left, top, right, bottom = text_box
    text_region = binary[top:bottom, left:right]
    region_w = right - left

    if region_w == 0:
        return 1

    v_proj = text_region.sum(axis=0).astype(float)
    kernel = max(5, region_w // 50)
    v_smooth = np.convolve(v_proj, np.ones(kernel) / kernel, mode="same")

    if v_smooth.max() == 0:
        return 1

    min_dist = max(10, int(region_w * 0.03))
    threshold = v_smooth.max() * 0.15
    valleys, _ = find_peaks(-v_smooth, distance=min_dist, height=-threshold)

    significant = [v for v in valleys if v_smooth[v] < v_smooth.max() * 0.2]
    n_cols = max(1, min(len(significant) + 1, 50))
    return n_cols
