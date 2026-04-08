#!/usr/bin/env python3
"""
detect_characters.py - Phát hiện và cắt từng ký tự Nôm từ ảnh trang sách

Phương pháp: Classical CV cải tiến (B+)
- Adaptive thresholding + border removal
- Column detection bằng vertical projection + auto-detect số cột từ transcription
- Character segmentation bằng horizontal projection + merge/split
- Cross-check với transcription để validate

Usage:
  # Xử lý 1 thư mục prepared data:
  python detect_characters.py data/prepared/CacThanhTruyen2

  # Xử lý với debug images:
  python detect_characters.py data/prepared/CacThanhTruyen2 --debug

  # Chỉ xử lý 1 trang:
  python detect_characters.py data/prepared/CacThanhTruyen2 --page 12
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def load_and_binarize(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ảnh và tạo binary mask (Bước 2.1 — Binarization).

    Pipeline:
      Ảnh gốc → GaussianBlur(3,3)
      → Background estimation: Morphological Closing (51×51)
      → Background removal: pixel ÷ background × 255
      → Otsu thresholding → ảnh nhị phân (1=mực, 0=nền)
      → Close (2×2): nối nét bị đứt nhỏ
      → Open (3×3): xóa chấm nhiễu nhỏ

    Returns:
        (gray_image, binary_mask) - binary_mask: 1=ink, 0=background
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # 1. Denoise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Estimate background to remove uneven illumination
    # Large kernel morphological closing = background estimate
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, bg_kernel)

    # Normalize: remove background variation
    # Result: foreground dark on white background
    normalized = cv2.divide(blurred, background, scale=255)

    # 3. Otsu on normalized image
    _, binary_inv = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary_inv > 0).astype(np.uint8)

    # 4. Morphological operations: close tiny gaps, remove tiny noise
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # Remove small noise blobs
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    return gray, binary


def detect_text_box(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Phát hiện hình chữ nhật bao quanh vùng text chính.

    Chiến lược:
    1. Tìm đường viền (border lines) bằng morphological line detection
    2. Xác định innermost rectangle từ 4 đường thẳng
    3. Cắt bỏ border + annotation

    Returns:
        (left, top, right, bottom) - tọa độ vùng text chính
    """
    h, w = binary.shape
    pad = 8  # Padding bên trong border

    # Detect horizontal lines (dài > 40% chiều rộng)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect vertical lines (dài > 40% chiều cao)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Find horizontal line positions (top and bottom)
    h_proj = h_lines.sum(axis=1)
    h_line_rows = np.where(h_proj > w * 0.1)[0]

    # Find vertical line positions (left and right)
    v_proj = v_lines.sum(axis=0)
    v_line_cols = np.where(v_proj > h * 0.1)[0]

    # Determine box from detected lines
    if len(h_line_rows) > 0:
        box_top = int(h_line_rows[0]) + pad
        box_bottom = int(h_line_rows[-1]) - pad
    else:
        # Fallback: use ink extent
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
        # Fallback: use full ink extent
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

    # Cắt bỏ annotation ngoài text box (số cột, số trang)
    content_h = box_bottom - box_top
    box_top += int(content_h * 0.02)
    box_bottom -= int(content_h * 0.01)

    return int(box_left), int(box_top), int(box_right), int(box_bottom)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_columns(
    binary: np.ndarray, text_box: tuple[int, int, int, int], n_expected: int = 9
) -> list[tuple[int, int, int, int]]:
    """Phát hiện cột trong vùng text.

    Chiến lược 2 bước:
    1. Phát hiện đường kẻ dọc (ruling lines) bằng morphology → ranh giới cột chính xác
    2. Nếu khoảng cách giữa 2 ruling lines > 1.5x expected width → tìm valley bổ sung
    3. Fallback: dùng vertical projection valleys nếu không tìm đủ ruling lines

    Args:
        binary: Binary image (1=ink)
        text_box: (left, top, right, bottom) vùng text
        n_expected: Số cột kỳ vọng (mặc định 9)

    Returns:
        List[(x1, y1, x2, y2)] cho mỗi cột, thứ tự phải→trái (cột 1 ở phải)
    """
    left, top, right, bottom = text_box
    text_region = binary[top:bottom, left:right]
    region_h = bottom - top
    region_w = right - left

    if region_w == 0 or n_expected < 1:
        return [(left, top, right, bottom)]

    expected_col_width = region_w / n_expected

    # Vertical projection (dùng cho valley detection)
    v_proj = text_region.sum(axis=0).astype(float)
    kernel = max(5, int(expected_col_width / 8))
    v_smooth = np.convolve(v_proj, np.ones(kernel) / kernel, mode="same")

    # --- Bước 1: Phát hiện ruling lines (đường kẻ dọc) ---
    ruling_positions = []
    if region_h > 200:
        v_kernel_len = max(100, region_h // 10)
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(text_region, cv2.MORPH_OPEN, v_k)
        v_proj_lines = v_lines.sum(axis=0)
        ruling_peaks, _ = find_peaks(
            v_proj_lines.astype(float), height=region_h * 0.05, distance=30
        )
        ruling_positions = list(ruling_peaks)

    # --- Bước 2: Xây dựng ranh giới cột ---
    if len(ruling_positions) >= 2:
        # Dùng ruling lines làm ranh giới chính
        boundaries = list(ruling_positions)

        # Tìm sub-boundaries cho khoảng rộng (> 1.5x expected width)
        all_positions = sorted([0] + boundaries + [region_w])
        for i in range(len(all_positions) - 1):
            gap_l = all_positions[i]
            gap_r = all_positions[i + 1]
            gap_w = gap_r - gap_l
            n_sub = round(gap_w / expected_col_width)

            if n_sub >= 2:
                # Tìm valleys trong khoảng, bỏ qua vùng sát ruling line
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
                        best = sorted(sub_valleys[np.argsort(depths)[: n_sub - 1]])
                        for v in best:
                            boundaries.append(sl + v)
                    else:
                        # Chia đều khoảng rộng
                        for j in range(1, n_sub):
                            boundaries.append(int(gap_l + j * gap_w / n_sub))

        boundaries = sorted(set(boundaries))

        # Loại bỏ boundaries sát mép (< 30px từ edge) → thường là border line
        boundaries = [b for b in boundaries if 30 < b < region_w - 30]

        # Nếu quá nhiều → loại bỏ boundary tạo cột hẹp nhất
        while len(boundaries) > n_expected - 1:
            all_b = [0] + boundaries + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            min_w_idx = int(np.argmin(widths))
            # Gộp cột hẹp nhất vào cột lân cận nhỏ hơn
            if min_w_idx == 0:
                boundaries.pop(0)
            elif min_w_idx == len(widths) - 1:
                boundaries.pop(-1)
            else:
                if widths[min_w_idx - 1] < widths[min_w_idx + 1]:
                    boundaries.pop(min_w_idx - 1)
                else:
                    boundaries.pop(min_w_idx)

        # Nếu thiếu → thêm bằng cách chia đều khoảng rộng nhất
        while len(boundaries) < n_expected - 1:
            all_b = [0] + boundaries + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            widest_idx = int(np.argmax(widths))
            mid = (all_b[widest_idx] + all_b[widest_idx + 1]) // 2
            boundaries.append(mid)
            boundaries.sort()

        # Post-processing: sửa cột quá hẹp (< 40% expected width)
        min_col_w = int(expected_col_width * 0.4)
        fixed = True
        while fixed:
            fixed = False
            all_b = [0] + boundaries + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            for idx, w in enumerate(widths):
                if w < min_col_w:
                    # Cột quá hẹp → xóa boundary tạo ra nó, rồi chia lại cột rộng
                    if idx == 0:
                        boundaries.pop(0)
                    elif idx == len(widths) - 1:
                        boundaries.pop(-1)
                    else:
                        # Xóa boundary tạo cột hẹp, gộp vào cột lân cận
                        boundaries.pop(idx)
                    # Thêm lại boundary ở cột rộng nhất
                    all_b2 = [0] + boundaries + [region_w]
                    widths2 = [all_b2[i + 1] - all_b2[i] for i in range(len(all_b2) - 1)]
                    widest = int(np.argmax(widths2))
                    # Tìm valley trong cột rộng nhất
                    wl, wr = all_b2[widest], all_b2[widest + 1]
                    m = max(40, int(expected_col_width * 0.2))
                    if wr - wl > 2 * m:
                        sub = v_smooth[wl + m : wr - m]
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
        # --- Fallback: Valley-based approach (khi không có ruling lines) ---
        min_gap_distance = int(expected_col_width * 0.5)
        valleys, props = find_peaks(-v_smooth, distance=min_gap_distance)
        valley_depths = v_smooth[valleys]

        if len(valleys) >= n_expected - 1:
            sorted_idx = np.argsort(valley_depths)
            best_valleys = sorted(valleys[sorted_idx[: n_expected - 1]])
        elif len(valleys) > 0:
            best_valleys = sorted(valleys)
        else:
            best_valleys = [int(region_w * i / n_expected) for i in range(1, n_expected)]

        while len(best_valleys) < n_expected - 1:
            all_b = [0] + list(best_valleys) + [region_w]
            widths = [all_b[i + 1] - all_b[i] for i in range(len(all_b) - 1)]
            widest_idx = int(np.argmax(widths))
            mid = (all_b[widest_idx] + all_b[widest_idx + 1]) // 2
            best_valleys.append(mid)
            best_valleys.sort()

    # Build column bboxes
    all_bounds = [0] + list(best_valleys) + [region_w]
    columns = []
    for i in range(len(all_bounds) - 1):
        x1 = all_bounds[i] + left
        x2 = all_bounds[i + 1] + left
        columns.append((x1, top, x2, bottom))

    # Thứ tự phải→trái (cột 1 = bên phải nhất)
    columns.reverse()

    return columns


def _auto_detect_n_columns(binary: np.ndarray, text_box: tuple[int, int, int, int]) -> int:
    """Tự động phát hiện số cột khi không có transcription.

    Dùng vertical projection + find_peaks để đếm số vùng có mực (peaks),
    không ràng buộc số cột cố định.

    Returns:
        Số cột phát hiện được (tối thiểu 1)
    """
    left, top, right, bottom = text_box
    text_region = binary[top:bottom, left:right]
    region_w = right - left

    if region_w == 0:
        return 1

    # Vertical projection
    v_proj = text_region.sum(axis=0).astype(float)

    # Smooth rộng hơn để tìm cột lớn (không phải nét nhỏ)
    kernel = max(5, region_w // 50)
    v_smooth = np.convolve(v_proj, np.ones(kernel) / kernel, mode="same")

    if v_smooth.max() == 0:
        return 1

    # Tìm valleys (khoảng trống giữa cột)
    # min_distance: cột tối thiểu rộng 3% ảnh
    min_dist = max(10, int(region_w * 0.03))
    # Ngưỡng valley: dưới 20% max density
    threshold = v_smooth.max() * 0.15

    valleys, _ = find_peaks(-v_smooth, distance=min_dist, height=-threshold)

    # Số cột = số valley + 1, nhưng lọc valley quá nông
    significant_valleys = []
    for v in valleys:
        if v_smooth[v] < v_smooth.max() * 0.2:
            significant_valleys.append(v)

    n_cols = len(significant_valleys) + 1

    # Sanity check: tối thiểu 1, tối đa 50
    n_cols = max(1, min(n_cols, 50))

    return n_cols


# ---------------------------------------------------------------------------
# Character segmentation
# ---------------------------------------------------------------------------

def _extract_raw_blobs(
    col_binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    expected_char_height: float,
    min_char_height: int,
    density_threshold_scale: float = 0.01,
) -> list[tuple[int, int, int, int]]:
    """Trích xuất raw blobs từ horizontal projection (absolute threshold).

    Tách riêng để có thể gọi lại với tham số khác (multi-pass).
    """
    x1, y1, x2, y2 = col_bbox
    col_h, col_w = col_binary.shape

    # Horizontal projection
    h_proj = col_binary.sum(axis=1).astype(float)

    # Smooth: nối nét gần nhau trong cùng 1 ký tự
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
    """Xóa vùng border lines (dọc + ngang) ở mép cột.

    Sách Hán Nôm có đường kẻ dọc ngăn cách cột và đường kẻ ngang ở
    đầu/cuối vùng text. Đường kẻ tạo ink giả → horizontal projection
    không tìm được gap giữa ký tự.

    Phương pháp kết hợp:
    1. Morphological line detection (kernel ngắn hơn → bắt line đứt đoạn)
    2. Projection-based fallback (bắt ruling line dày nhưng không liên tục)
    """
    h, w = col_binary.shape
    if h == 0 or w == 0:
        return col_binary

    cleaned = col_binary.copy()

    # --- Xóa border DỌC ---
    margin_x = max(10, w // 4)

    # Phương pháp 1: Morphological detection (kernel ngắn hơn để bắt line đứt đoạn)
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

    # Phương pháp 2: Projection fallback (bắt ruling line dày/đứt đoạn)
    # Scan mép trái: nếu v_proj > 30% height → đó là ruling line
    v_proj = cleaned.sum(axis=0).astype(float)
    v_line_threshold = h * 0.30

    left_trim = 0
    for x in range(min(margin_x, 40)):
        if v_proj[x] > v_line_threshold:
            left_trim = x + 4
    if left_trim > 0:
        cleaned[:, :left_trim] = 0

    right_trim = w
    for x in range(w - 1, max(w - margin_x, w - 40), -1):
        if v_proj[x] > v_line_threshold:
            right_trim = x - 4
    if right_trim < w:
        cleaned[:, right_trim:] = 0

    # --- Xóa border NGANG ---
    margin_y = max(10, h // 6)

    # Morphological detection
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

    # Projection fallback cho border ngang
    h_proj = cleaned.sum(axis=1).astype(float)
    h_line_threshold = w * 0.40
    top_trim2 = 0
    for y in range(min(margin_y, 60)):
        if h_proj[y] > h_line_threshold:
            top_trim2 = y + 4
    if top_trim2 > 0:
        cleaned[:top_trim2, :] = 0

    bottom_trim2 = h
    for y in range(h - 1, max(h - margin_y, h - 60), -1):
        if h_proj[y] > h_line_threshold:
            bottom_trim2 = y - 4
    if bottom_trim2 < h:
        cleaned[bottom_trim2:, :] = 0

    return cleaned


def _trim_to_ink_extent(
    col_binary: np.ndarray, expected_char_height: float
) -> tuple[int, int]:
    """Tìm vùng chứa chữ thật trong cột (bỏ vùng trống trên/dưới).

    Nhiều cột có chiều cao = full page nhưng chữ chỉ chiếm phần trên.
    Nếu dùng full height, expected_char_height bị sai → chia sai.

    Cải tiến: dùng sliding window để phân biệt vùng có chữ thật (nhiều ink
    liên tục) vs vùng nhiễu (ink rời rạc). Tránh trim quá sát ký tự đầu/cuối.

    Returns:
        (top_offset, bottom_offset) relative to column top
    """
    h_proj = col_binary.sum(axis=1).astype(float)
    col_h = col_binary.shape[0]
    col_w = col_binary.shape[1]

    if col_h == 0:
        return 0, col_h

    # Smooth mạnh (1 expected char height) để tìm vùng chữ liên tục
    smooth_k = max(5, int(expected_char_height * 0.5)) | 1
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    if h_smooth.max() == 0:
        return 0, col_h

    # Sliding window: tính density trung bình trên mỗi cửa sổ = 1 char height
    # Vùng có chữ thật: density cao liên tục, vùng trống: density ~0
    win = max(5, int(expected_char_height * 0.3)) | 1
    windowed = np.convolve(h_proj, np.ones(win) / win, mode="same")

    # Ngưỡng kép:
    # 1. Minimum absolute: ít nhất 5% chiều rộng cột phải có ink
    #    (loại bỏ noise/residue từ ruling lines)
    # 2. Relative: percentile-based cho vùng có chữ thật
    min_abs_threshold = col_w * 0.05
    nonzero = windowed[windowed > min_abs_threshold]
    if len(nonzero) == 0:
        # Thử threshold thấp hơn
        nonzero = windowed[windowed > col_w * 0.02]
        if len(nonzero) == 0:
            return 0, col_h
    ink_threshold = max(min_abs_threshold, np.percentile(nonzero, 15) * 0.5)

    ink_rows = np.where(windowed > ink_threshold)[0]
    if len(ink_rows) == 0:
        return 0, col_h

    # Tìm vùng ink liên tục dài nhất (loại bỏ nhiễu rải rác)
    groups = []
    g_start = ink_rows[0]
    for i in range(1, len(ink_rows)):
        if ink_rows[i] - ink_rows[i - 1] > expected_char_height * 0.5:
            groups.append((g_start, ink_rows[i - 1]))
            g_start = ink_rows[i]
    groups.append((g_start, ink_rows[-1]))

    # Chọn group dài nhất
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
    """Tách ký tự bằng valley detection (local minima) trong horizontal projection.

    Dùng khi threshold approach thất bại (chữ viết dày, không có gap rõ ràng).

    Pipeline:
    1. Xóa border lines (đường kẻ dọc) → loại ink giả
    2. Trim vùng trống → chỉ giữ vùng có chữ
    3. Tính expected_char_height chính xác từ vùng có chữ thật
    4. Tìm ALL valleys, rồi "snap to grid": chọn valley gần nhất
       với mỗi vị trí kỳ vọng (equal-spaced grid)
    """
    x1, y1, x2, y2 = col_bbox
    col_h, col_w = col_binary.shape

    if col_h == 0 or col_w == 0 or expected_count < 2:
        return []

    # 1. Xóa border lines
    cleaned = _strip_column_borders(col_binary)

    # 2. Trim vùng trống trên/dưới
    ink_top, ink_bottom = _trim_to_ink_extent(cleaned, expected_char_height)
    ink_region = cleaned[ink_top:ink_bottom, :]
    ink_h = ink_bottom - ink_top

    if ink_h < expected_char_height:
        return []

    # 3. Tính lại expected char height từ vùng ink thật
    real_char_h = ink_h / expected_count

    # 4. Horizontal projection trên vùng đã clean
    h_proj = ink_region.sum(axis=1).astype(float)

    # Smooth: ~10% expected height
    smooth_k = max(3, int(real_char_h * 0.10)) | 1
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    # Tìm TẤT CẢ valleys (min_dist nhỏ để không bỏ sót)
    min_dist = max(3, int(real_char_h * 0.2))
    valleys, _ = find_peaks(-h_smooth, distance=min_dist)

    if len(valleys) < expected_count - 1:
        # Thử min_dist nhỏ hơn nữa
        min_dist2 = max(3, int(real_char_h * 0.1))
        valleys, _ = find_peaks(-h_smooth, distance=min_dist2)

    if len(valleys) < expected_count - 1:
        return []

    # 5. "Snap to grid": chọn valley gần nhất với mỗi vị trí kỳ vọng
    # Vị trí kỳ vọng: k * real_char_h cho k = 1, ..., N-1
    n_needed = expected_count - 1
    split_points = []
    used_valleys = set()

    for k in range(1, expected_count):
        target = k * real_char_h
        # Tìm valley gần nhất chưa dùng
        best_valley = None
        best_dist = float("inf")
        for vi, v in enumerate(valleys):
            if vi in used_valleys:
                continue
            dist = abs(v - target)
            # Cho phép lệch tối đa 50% real_char_h
            if dist < best_dist and dist < real_char_h * 0.5:
                best_dist = dist
                best_valley = vi
        if best_valley is not None:
            split_points.append(int(valleys[best_valley]))
            used_valleys.add(best_valley)
        else:
            # Không có valley gần → dùng vị trí kỳ vọng (equal-div)
            split_points.append(int(target))

    split_points.sort()

    # 6. Local refinement: tinh chỉnh mỗi split point bằng cách tìm
    # minimum thực sự trong projection gốc (chưa smooth) trong vùng lân cận
    # Điều này sửa lỗi smooth làm lệch vị trí valley
    refine_radius = max(3, int(real_char_h * 0.15))
    refined_points = []
    for sp in split_points:
        lo = max(0, sp - refine_radius)
        hi = min(ink_h, sp + refine_radius + 1)
        if hi > lo:
            local_proj = h_proj[lo:hi]
            # Tìm vị trí có density thấp nhất (gap rõ nhất)
            best_local = int(np.argmin(local_proj))
            refined_points.append(lo + best_local)
        else:
            refined_points.append(sp)
    split_points = refined_points

    # Giữ nguyên split points, chỉ đảm bảo start=0 và end=ink_h
    all_points = [0] + split_points + [ink_h]

    # Validate: không có phần nào > 3x expected → reject nếu quá tệ
    max_part = max(all_points[j + 1] - all_points[j] for j in range(len(all_points) - 1))
    if max_part > real_char_h * 3.0:
        return []

    # Tạo bboxes (chuyển offset về tọa độ gốc)
    chars = []
    for j in range(len(all_points) - 1):
        cy1 = y1 + ink_top + all_points[j]
        cy2 = y1 + ink_top + all_points[j + 1]
        if cy2 - cy1 >= 8:
            chars.append((x1, int(cy1), x2, int(cy2)))

    return chars


def _quality_score(chars: list[tuple[int, int, int, int]], expected_h: float,
                   expected_count: int | None) -> float:
    """Đánh giá chất lượng segmentation: height đồng đều → score thấp (tốt).

    Score = count_diff_penalty + height_variance_penalty.
    """
    if not chars:
        return float("inf")

    heights = [c[3] - c[1] for c in chars]
    mean_h = sum(heights) / len(heights)

    # Penalty cho count mismatch
    count_penalty = 0
    if expected_count:
        count_penalty = abs(len(chars) - expected_count) * 100

    # Penalty cho height variance (CV = std/mean)
    if mean_h > 0:
        variance = sum((h - mean_h) ** 2 for h in heights) / len(heights)
        cv = (variance ** 0.5) / mean_h
    else:
        cv = 1.0

    # Penalty cho height quá khác expected
    h_deviation = abs(mean_h - expected_h) / expected_h if expected_h > 0 else 0

    return count_penalty + cv * 50 + h_deviation * 20


def detect_chars_in_column(
    binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    min_char_height: int = 15,
    expected_char_height: float | None = None,
    expected_count: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Phát hiện ký tự trong 1 cột.

    Chiến lược cải tiến: thử nhiều approach, chọn kết quả tốt nhất.
    1. Valley approach (snap-to-grid): ưu tiên cho chữ viết dày đặc
    2. Threshold approach + merge/split: cho ảnh sạch có gap rõ
    3. Trimmed equal-division: fallback an toàn
    4. So sánh quality score → chọn kết quả đồng đều nhất

    Returns:
        List[(x1, y1, x2, y2)] cho mỗi ký tự, thứ tự trên→dưới
    """
    x1, y1, x2, y2 = col_bbox
    col_binary = binary[y1:y2, x1:x2]
    col_h, col_w = col_binary.shape

    if col_w == 0 or col_h == 0:
        return []

    # Tính expected char height từ expected_count nếu có
    if expected_count and expected_count > 0:
        expected_char_height = col_h / expected_count

    if expected_char_height is None:
        expected_char_height = col_h / 20  # Fallback

    # --- Approach 1 (ưu tiên): Valley snap-to-grid ---
    # Valley dùng ranh giới thực tế (local minima) → chính xác hơn chia đều
    if expected_count and expected_count > 1:
        valley_chars = _segment_by_valleys(
            col_binary, col_bbox, expected_count, col_h / expected_count
        )
        if valley_chars and len(valley_chars) == expected_count:
            return valley_chars

    # --- Approach 2: Threshold + merge/split ---
    raw_chars = _extract_raw_blobs(
        col_binary, col_bbox, expected_char_height, min_char_height, 0.01
    )
    if raw_chars:
        chars = _merge_small_boxes(raw_chars, expected_char_height)
        chars = _split_large_boxes(chars, expected_char_height, binary)
        if expected_count and len(chars) < expected_count:
            chars = _force_split_to_count(chars, expected_count, expected_char_height, binary)

        # Nếu threshold cho đúng count → dùng (vì dựa trên gap thực tế)
        if expected_count and len(chars) == expected_count:
            return chars

        # Thử thêm density thresholds
        best_chars = chars
        best_diff = abs(len(chars) - (expected_count or len(chars)))

        for density_scale in [0.02, 0.05, 0.005]:
            retry_raw = _extract_raw_blobs(
                col_binary, col_bbox, col_h / max(expected_count or 20, 1),
                min_char_height, density_scale
            )
            if not retry_raw:
                continue
            retry_h = col_h / max(expected_count or 20, 1)
            retry_chars = _merge_small_boxes(retry_raw, retry_h)
            retry_chars = _split_large_boxes(retry_chars, retry_h, binary)
            if expected_count and len(retry_chars) < expected_count:
                retry_chars = _force_split_to_count(
                    retry_chars, expected_count, retry_h, binary)
            retry_diff = abs(len(retry_chars) - (expected_count or len(retry_chars)))
            if retry_diff < best_diff:
                best_diff = retry_diff
                best_chars = retry_chars
            if best_diff == 0:
                return best_chars

        if not expected_count:
            return best_chars

    # --- Approach 3 (fallback): Trimmed equal-division ---
    if expected_count and expected_count > 0:
        cleaned = _strip_column_borders(col_binary)
        ink_top, ink_bottom = _trim_to_ink_extent(cleaned, col_h / expected_count)
        ink_h = ink_bottom - ink_top
        if ink_h > col_h * 0.3:
            step = ink_h / expected_count
            return [(x1, int(y1 + ink_top + i * step), x2, int(y1 + ink_top + (i + 1) * step))
                    for i in range(expected_count)]
        step = col_h / expected_count
        return [(x1, int(y1 + i * step), x2, int(y1 + (i + 1) * step))
                for i in range(expected_count)]

    return raw_chars if raw_chars else []


def _merge_small_boxes(
    chars: list[tuple[int, int, int, int]], expected_h: float
) -> list[tuple[int, int, int, int]]:
    """Merge box quá nhỏ (< 40% expected height) vào box gần nhất."""
    if not chars:
        return chars

    merge_threshold = expected_h * 0.4
    gap_threshold = expected_h * 0.3  # Max gap để merge

    merged = [chars[0]]
    for i in range(1, len(chars)):
        curr = chars[i]
        prev = merged[-1]

        curr_h = curr[3] - curr[1]
        prev_h = prev[3] - prev[1]
        gap = curr[1] - prev[3]

        # Merge nếu box hiện tại nhỏ VÀ gần box trước
        if curr_h < merge_threshold and gap < gap_threshold:
            merged[-1] = (prev[0], prev[1], prev[2], curr[3])
        # Merge nếu box trước nhỏ VÀ gần box hiện tại
        elif prev_h < merge_threshold and gap < gap_threshold:
            merged[-1] = (curr[0], prev[1], curr[2], curr[3])
        else:
            merged.append(curr)

    return merged


def _force_split_to_count(
    chars: list[tuple[int, int, int, int]],
    target_count: int,
    expected_h: float,
    binary: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Khi vẫn thiếu ký tự, force-split các blob lớn nhất cho đến khi đạt target.

    Chỉ split blob có height > 1.2x expected_h.
    Sử dụng projection valleys ưu tiên, fallback connected components, rồi chia đều.
    """
    result = list(chars)
    max_iterations = target_count * 2  # Safety limit

    while len(result) < target_count and max_iterations > 0:
        max_iterations -= 1

        # Tìm blob lớn nhất có thể split
        heights = [(b[3] - b[1], i) for i, b in enumerate(result)]
        heights.sort(reverse=True)

        split_done = False
        for h_val, idx in heights:
            if h_val < expected_h * 1.2:
                break  # Không có blob đủ lớn để split

            bbox = result[idx]
            x1, y1, x2, y2 = bbox
            n_parts = min(round(h_val / expected_h), target_count - len(result) + 1)
            n_parts = max(2, n_parts)

            col_binary = binary[y1:y2, x1:x2]
            h_proj = col_binary.sum(axis=1).astype(float)

            # Smooth nhẹ hơn để phát hiện valley mịn
            smooth_k = max(3, int(expected_h * 0.05))
            h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

            min_dist = int(expected_h * 0.35)
            valleys, _ = find_peaks(-h_smooth, distance=min_dist)

            split_points = None

            if len(valleys) >= n_parts - 1:
                depths = h_smooth[valleys]
                best_idx = np.argsort(depths)[: n_parts - 1]
                split_points = sorted(valleys[best_idx])
            else:
                # Thử connected components
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    col_binary, connectivity=8
                )
                if num_labels > 2:
                    centers = []
                    for lbl in range(1, num_labels):
                        area = stats[lbl, cv2.CC_STAT_AREA]
                        if area > col_binary.size * 0.003:
                            cy = stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT] // 2
                            centers.append(cy)
                    if len(centers) >= n_parts:
                        centers.sort()
                        gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
                        gaps.sort(reverse=True)
                        split_points = sorted(
                            [(centers[i] + centers[i + 1]) // 2
                             for _, i in gaps[: n_parts - 1]]
                        )

            if split_points is None:
                box_h = y2 - y1
                split_points = [int(box_h * i / n_parts) for i in range(1, n_parts)]

            sub_boxes = []
            all_points = [0] + list(split_points) + [y2 - y1]
            for j in range(len(all_points) - 1):
                sy1 = y1 + all_points[j]
                sy2 = y1 + all_points[j + 1]
                if sy2 - sy1 >= 8:
                    sub_boxes.append((x1, sy1, x2, sy2))

            if len(sub_boxes) > 1:
                result = result[:idx] + sub_boxes + result[idx + 1:]
                split_done = True
                break

        if not split_done:
            break

    return result


def _split_large_boxes(
    chars: list[tuple[int, int, int, int]], expected_h: float, binary: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """Split box quá lớn (> 1.5x expected height) thành 2+ ký tự.

    Sử dụng 2 chiến lược:
      1. Horizontal projection valleys (tìm khoảng trống ngang)
      2. Connected component separation (nếu projection không đủ valleys)
    """
    split_threshold = expected_h * 1.5

    result = []
    for bbox in chars:
        x1, y1, x2, y2 = bbox
        box_h = y2 - y1

        if box_h > split_threshold:
            n_parts = round(box_h / expected_h)
            n_parts = max(2, min(n_parts, 5))

            col_binary = binary[y1:y2, x1:x2]
            h_proj = col_binary.sum(axis=1).astype(float)

            # Smooth nhẹ hơn để giữ chi tiết valley
            smooth_k = max(3, int(expected_h * 0.05))
            h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

            min_dist = int(expected_h * 0.35)
            valleys, _ = find_peaks(-h_smooth, distance=min_dist)

            split_points = None

            if len(valleys) >= n_parts - 1:
                # Chọn valleys sâu nhất
                depths = h_smooth[valleys]
                best_idx = np.argsort(depths)[: n_parts - 1]
                split_points = sorted(valleys[best_idx])
            else:
                # Thử connected components: tìm bounding boxes dọc của từng component
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    col_binary, connectivity=8
                )
                if num_labels > 2:  # > 1 component (+ background)
                    # Gom components theo vị trí dọc trung tâm
                    centers = []
                    for lbl in range(1, num_labels):
                        cy = stats[lbl, cv2.CC_STAT_TOP] + stats[lbl, cv2.CC_STAT_HEIGHT] // 2
                        area = stats[lbl, cv2.CC_STAT_AREA]
                        if area > col_binary.size * 0.005:  # Bỏ nhiễu nhỏ
                            centers.append(cy)

                    if len(centers) >= n_parts:
                        # Cluster centers thành n_parts nhóm bằng khoảng cách
                        centers.sort()
                        # Tìm n_parts-1 gaps lớn nhất giữa các centers
                        gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
                        gaps.sort(reverse=True)
                        cc_splits = sorted(
                            [(centers[i] + centers[i + 1]) // 2
                             for _, i in gaps[: n_parts - 1]]
                        )
                        # Kiểm tra split points có hợp lý không
                        valid = all(
                            expected_h * 0.3 <= cc_splits[j] - (cc_splits[j - 1] if j > 0 else 0)
                            for j in range(len(cc_splits))
                        )
                        if valid:
                            split_points = cc_splits

            if split_points is None:
                # Fallback: chia đều
                split_points = [int(box_h * i / n_parts) for i in range(1, n_parts)]

            # Tạo sub-boxes
            all_points = [0] + list(split_points) + [box_h]
            for j in range(len(all_points) - 1):
                sub_y1 = y1 + all_points[j]
                sub_y2 = y1 + all_points[j + 1]
                if sub_y2 - sub_y1 >= 10:
                    result.append((x1, sub_y1, x2, sub_y2))
        else:
            result.append(bbox)

    return result


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def detect_page(
    image_path: str,
    n_columns: int | None = None,
    expected_counts: list[int] | None = None,
) -> dict:
    """Pipeline đầy đủ: load ảnh → detect columns → detect chars.

    Args:
        n_columns: Số cột kỳ vọng. Nếu None → suy ra từ expected_counts
                   hoặc auto-detect bằng vertical projection.
        expected_counts: Số ký tự kỳ vọng mỗi cột (từ transcription).
                         Dùng để cải thiện split/merge VÀ suy ra số cột.

    Returns:
        dict với columns, chars, metadata
    """
    gray, binary = load_and_binarize(image_path)
    h, w = gray.shape

    # Detect text box
    text_box = detect_text_box(binary)

    # Xác định số cột: ưu tiên n_columns > len(expected_counts) > auto-detect
    if n_columns is None and expected_counts is not None:
        n_columns = len(expected_counts)
    if n_columns is None:
        n_columns = _auto_detect_n_columns(binary, text_box)

    # Detect columns
    columns = detect_columns(binary, text_box, n_expected=n_columns)

    # Estimate expected char height
    col_height = text_box[3] - text_box[1]
    expected_char_h = col_height / 22

    # Detect chars in each column
    all_chars = []
    column_results = []

    for col_idx, col_bbox in enumerate(columns):
        exp_count = None
        if expected_counts and col_idx < len(expected_counts):
            exp_count = expected_counts[col_idx]

        chars = detect_chars_in_column(
            binary, col_bbox,
            min_char_height=int(expected_char_h * 0.3),
            expected_char_height=expected_char_h,
            expected_count=exp_count,
        )

        col_result = {
            "column": col_idx + 1,
            "bbox": list(col_bbox),
            "num_chars": len(chars),
            "chars": [],
        }

        for char_idx, char_bbox in enumerate(chars):
            cx1, cy1, cx2, cy2 = char_bbox
            char_info = {
                "char_idx": char_idx,
                "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "width": int(cx2 - cx1),
                "height": int(cy2 - cy1),
            }
            col_result["chars"].append(char_info)
            all_chars.append((col_idx + 1, char_idx, char_bbox))

        column_results.append(col_result)

    return {
        "image_size": [w, h],
        "text_box": list(text_box),
        "expected_char_height": round(expected_char_h, 1),
        "num_columns": len(columns),
        "total_chars": len(all_chars),
        "columns": column_results,
    }


# ---------------------------------------------------------------------------
# Validation & output
# ---------------------------------------------------------------------------

def validate_with_transcription(
    detection: dict, transcription_path: str
) -> dict:
    """So sánh số ký tự detected vs số âm tiết trong transcription."""
    with open(transcription_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    validation = {
        "total_detected": detection["total_chars"],
        "total_expected": 0,
        "column_comparison": [],
        "accuracy": 0,
    }

    for col in detection["columns"]:
        col_idx = col["column"] - 1
        expected = len(lines[col_idx].split()) if col_idx < len(lines) else 0
        detected = col["num_chars"]
        diff = detected - expected

        validation["total_expected"] += expected
        validation["column_comparison"].append({
            "column": col["column"],
            "detected": detected,
            "expected": expected,
            "diff": diff,
            "match": abs(diff) <= max(1, int(expected * 0.1)),
        })

    total_exp = validation["total_expected"]
    total_det = validation["total_detected"]
    if total_exp > 0:
        validation["accuracy"] = round(
            1 - abs(total_det - total_exp) / total_exp, 4
        )

    return validation


def _tighten_char_bbox(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    col_center_x: float,
    expected_h: float = 0,
    padding: int = 5,
) -> tuple[int, int, int, int]:
    """Tighten bbox quanh ink thực tế của ký tự.

    Xử lý cả X-axis (loại bỏ chữ cột bên cạnh, ruling lines)
    và Y-axis (loại bỏ phần chữ trên/dưới khi bbox quá cao).

    Args:
        binary: Binary image toàn trang (INV: ink=255, bg=0)
        bbox: (x1, y1, x2, y2) full-column-width bbox
        col_center_x: Tâm x của cột
        expected_h: Chiều cao kỳ vọng 1 ký tự (từ detection JSON)
        padding: Pixels padding sau khi tighten

    Returns:
        (x1, y1, x2, y2) tight-fit bbox
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = binary.shape
    col_w = x2 - x1
    char_h = y2 - y1

    if col_w <= 0 or char_h <= 0:
        return bbox

    # Crop region
    region = binary[y1:y2, x1:x2].copy()
    rh, rw = region.shape
    if rh == 0 or rw == 0:
        return bbox

    # --- Bước 1: Loại bỏ ruling lines (đường kẻ dọc mỏng ở biên cột) ---
    edge_margin = max(6, int(rw * 0.08))

    for cx in range(edge_margin):
        col_ink = np.count_nonzero(region[:, cx])
        if col_ink > rh * 0.5:
            region[:, cx] = 0

    for cx in range(edge_margin):
        col_ink = np.count_nonzero(region[:, rw - 1 - cx])
        if col_ink > rh * 0.5:
            region[:, rw - 1 - cx] = 0

    # --- Bước 2: Connected Components → chỉ giữ ink gần tâm cột (X-axis) ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        region, connectivity=8
    )

    if num_labels <= 1:
        return bbox

    keep_mask = np.zeros_like(region)
    center_margin = rw * 0.30

    for lbl in range(1, num_labels):
        cx = centroids[lbl][0]
        area = stats[lbl, cv2.CC_STAT_AREA]

        if center_margin <= cx <= rw - center_margin:
            keep_mask[labels == lbl] = 255
        elif area > region.size * 0.02 and center_margin * 0.5 <= cx <= rw - center_margin * 0.5:
            keep_mask[labels == lbl] = 255
        else:
            cc_x1 = stats[lbl, cv2.CC_STAT_LEFT]
            cc_x2 = cc_x1 + stats[lbl, cv2.CC_STAT_WIDTH]
            if cc_x1 < rw - center_margin and cc_x2 > center_margin:
                keep_mask[labels == lbl] = 255

    # --- Bước 3: Y-axis trim — loại bỏ ink của chữ trên/dưới ---
    if expected_h > 0:
        coords_check = cv2.findNonZero(keep_mask)
        if coords_check is not None:
            _, ry_c, _, rh_c = cv2.boundingRect(coords_check)
            ink_h = rh_c

            # Nếu ink cao > 1.4× expected → có thể chứa phần chữ trên/dưới
            if ink_h > expected_h * 1.4:
                # Horizontal projection (sum ink per row) trong keep_mask
                h_proj = keep_mask.sum(axis=1).astype(float)

                # Smooth projection
                k = max(3, int(expected_h * 0.06)) | 1
                h_smooth = np.convolve(h_proj, np.ones(k) / k, mode="same")

                # Tìm vùng ink chính: vùng liên tục có projection > threshold
                thresh = h_smooth.max() * 0.08
                above = h_smooth > thresh

                # Tìm tất cả các runs of True
                runs = []
                in_run = False
                run_start = 0
                for ry in range(len(above)):
                    if above[ry] and not in_run:
                        run_start = ry
                        in_run = True
                    elif not above[ry] and in_run:
                        runs.append((run_start, ry))
                        in_run = False
                if in_run:
                    runs.append((run_start, len(above)))

                if len(runs) >= 2:
                    # Nhiều runs → chọn run có tổng ink lớn nhất
                    best_run = max(runs, key=lambda r: h_smooth[r[0]:r[1]].sum())
                    trim_y1, trim_y2 = best_run

                    # Mở rộng nhẹ để không cắt sát nét
                    expand = max(3, int(expected_h * 0.05))
                    trim_y1 = max(0, trim_y1 - expand)
                    trim_y2 = min(rh, trim_y2 + expand)

                    # Chỉ áp dụng nếu trim đáng kể (> 15% height)
                    if (rh - (trim_y2 - trim_y1)) > rh * 0.12:
                        # Xoá ink ngoài vùng chính
                        keep_mask[:trim_y1, :] = 0
                        keep_mask[trim_y2:, :] = 0

                elif len(runs) == 1:
                    # Một run nhưng vẫn quá cao → trim dựa trên tâm
                    run_h = runs[0][1] - runs[0][0]
                    if run_h > expected_h * 1.5:
                        # Tìm valley sâu nhất gần tâm để split
                        center_y = rh // 2
                        search_start = max(0, int(center_y - expected_h * 0.7))
                        search_end = min(rh, int(center_y + expected_h * 0.7))
                        search_range = h_smooth[search_start:search_end]

                        if len(search_range) > 10:
                            valley_y = search_start + np.argmin(search_range)
                            valley_depth = h_smooth[valley_y]

                            # Chỉ split nếu valley đủ sâu (< 30% peak)
                            if valley_depth < h_smooth.max() * 0.30:
                                # Giữ phần có nhiều ink hơn
                                upper_ink = h_smooth[:valley_y].sum()
                                lower_ink = h_smooth[valley_y:].sum()

                                expand = max(3, int(expected_h * 0.05))
                                if upper_ink >= lower_ink:
                                    keep_mask[valley_y + expand:, :] = 0
                                else:
                                    keep_mask[:max(0, valley_y - expand), :] = 0

    # --- Bước 4: Final tighten bbox ---
    coords = cv2.findNonZero(keep_mask)
    if coords is None:
        coords = cv2.findNonZero(region)
        if coords is None:
            return bbox

    rx, ry, rw_tight, rh_tight = cv2.boundingRect(coords)

    new_x1 = max(0, x1 + rx - padding)
    new_x2 = min(img_w, x1 + rx + rw_tight + padding)
    new_y1 = max(0, y1 + ry - padding)
    new_y2 = min(img_h, y1 + ry + rh_tight + padding)

    # Safety: bbox mới không nhỏ hơn 15% diện tích bbox cũ
    orig_area = col_w * char_h
    new_area = (new_x2 - new_x1) * (new_y2 - new_y1)
    if new_area < orig_area * 0.15:
        return bbox

    return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))


def save_char_crops(
    image_path: str, detection: dict, output_dir: Path, page_num: int
) -> list[str]:
    """Cắt và lưu ảnh từng ký tự với tight-fit bbox."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binary cho tightening (INV: ink=white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    crops_dir = output_dir / "crops" / f"page_{page_num:04d}"
    crops_dir.mkdir(parents=True, exist_ok=True)

    expected_h = detection.get("expected_char_height", 0)

    crop_paths = []
    for col in detection["columns"]:
        # Tâm x của cột
        col_bbox = col["bbox"]
        col_center_x = (col_bbox[0] + col_bbox[2]) / 2.0

        for char_info in col["chars"]:
            x1, y1, x2, y2 = char_info["bbox"]

            # Tighten bbox: loại bỏ ruling lines + chữ cột bên cạnh + chữ trên/dưới
            tx1, ty1, tx2, ty2 = _tighten_char_bbox(
                binary, (x1, y1, x2, y2), col_center_x, expected_h
            )
            char_info["bbox_tight"] = [int(tx1), int(ty1), int(tx2), int(ty2)]

            # Crop từ tight bbox
            crop = gray[ty1:ty2, tx1:tx2]

            filename = f"col{col['column']:02d}_char{char_info['char_idx']:03d}.png"
            crop_path = crops_dir / filename
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(str(crop_path))

            char_info["crop_file"] = f"crops/page_{page_num:04d}/{filename}"

    return crop_paths


def save_debug_image(
    image_path: str, detection: dict, output_path: str
):
    """Lưu ảnh debug với bbox columns và chars vẽ lên."""
    img = cv2.imread(image_path)

    colors = [
        (0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 128, 255), (200, 0, 200),
        (200, 200, 0), (0, 200, 200), (128, 0, 200), (200, 0, 128),
    ]

    # Vẽ text box
    tb = detection["text_box"]
    cv2.rectangle(img, (tb[0], tb[1]), (tb[2], tb[3]), (128, 128, 128), 2)

    # Vẽ columns và chars
    for col in detection["columns"]:
        color = colors[(col["column"] - 1) % len(colors)]
        bx = col["bbox"]
        cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)

        # Label column number
        cv2.putText(
            img, str(col["column"]),
            (bx[0] + 5, bx[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3,
        )

        for char_info in col["chars"]:
            cb = char_info["bbox"]
            cv2.rectangle(img, (cb[0], cb[1]), (cb[2], cb[3]), color, 1)

    cv2.imwrite(output_path, img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_prepared_dir(
    prepared_dir: Path,
    output_dir: Path | None = None,
    debug: bool = False,
    page_filter: int | None = None,
    verbose: bool = True,
    use_paddle: bool = False,
):
    """Xử lý toàn bộ thư mục prepared data."""
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Không tìm thấy manifest: {manifest_path}", file=sys.stderr)
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if output_dir is None:
        output_dir = prepared_dir / "detected"
    output_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    results = []
    total_detected = 0
    total_expected = 0
    pages_matched = 0

    # PaddleOCR hybrid detector
    paddle_detector = None
    if use_paddle:
        from paddle_detector import PaddleHybridDetector
        paddle_detector = PaddleHybridDetector()
        if not paddle_detector.is_available():
            print("[WARNING] PaddleOCR chưa cài đặt, fallback về classical CV")
            print("  pip install paddlepaddle paddleocr")
            paddle_detector = None

    if verbose:
        method_name = "PaddleOCR Hybrid" if paddle_detector else "Classical CV"
        print(f"\nChar Detection: {prepared_dir.name} [{method_name}]")
        print(f"Output: {output_dir}/")
        print("-" * 70)

    for page_info in manifest["pages"]:
        book_page = page_info["book_page"]

        if page_filter is not None and book_page != page_filter:
            continue

        original_image_path = str(prepared_dir / page_info["image_file"])
        trans_path = str(prepared_dir / page_info["transcription_file"])

        if not Path(original_image_path).exists():
            if verbose:
                print(f"  [SKIP] Trang {book_page}: ảnh không tồn tại")
            continue

        # Ưu tiên ảnh đã khử nhiễu cho detection (từ prepare_data.py --denoise)
        # Crop ký tự vẫn dùng ảnh gốc (giữ texture tự nhiên cho embedding)
        detect_image_path = original_image_path
        denoised_path = str(
            prepared_dir / "pages_denoised" / Path(page_info["image_file"]).name
        )
        if Path(denoised_path).exists():
            detect_image_path = denoised_path

        # Load expected counts from transcription (for better split/merge)
        # Số dòng trong transcription = số cột trong ảnh (auto-detect)
        expected_counts = None
        n_columns = None
        if Path(trans_path).exists():
            with open(trans_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            expected_counts = [len(line.split()) for line in lines]
            n_columns = len(lines)  # Số cột = số dòng text QN

        # Fallback: đọc num_columns từ manifest nếu transcription không có
        if n_columns is None:
            n_columns = page_info.get("num_columns")

        # Detect (dùng ảnh denoised nếu có → binarize tốt hơn)
        if paddle_detector:
            detection = paddle_detector.detect_page(
                detect_image_path, n_columns=n_columns,
                expected_counts=expected_counts,
            )
        else:
            detection = detect_page(detect_image_path, n_columns=n_columns, expected_counts=expected_counts)

        # Validate
        validation = {}
        if Path(trans_path).exists():
            validation = validate_with_transcription(detection, trans_path)
            total_detected += validation["total_detected"]
            total_expected += validation["total_expected"]
            if validation["accuracy"] >= 0.9:
                pages_matched += 1

        # Save char crops (dùng ảnh GỐC — giữ texture tự nhiên cho embedding)
        save_char_crops(original_image_path, detection, output_dir, book_page)

        # Save debug image (dùng ảnh gốc để thấy bbox trên ảnh thực)
        if debug:
            debug_path = str(debug_dir / f"page_{book_page:04d}_debug.png")
            save_debug_image(original_image_path, detection, debug_path)

        # Save detection JSON
        det_json_path = output_dir / f"page_{book_page:04d}_detection.json"
        result = {
            "book_page": book_page,
            "detection": detection,
            "validation": validation,
        }
        with open(det_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        results.append(result)

        if verbose:
            acc_str = ""
            warn_str = ""
            if validation:
                acc = validation["accuracy"]
                col_match = sum(1 for c in validation["column_comparison"] if c["match"])
                n_cols = detection['num_columns']
                acc_str = (
                    f"  det={validation['total_detected']:>3} "
                    f"exp={validation['total_expected']:>3} "
                    f"acc={acc:.0%} "
                    f"col_match={col_match}/{n_cols}"
                )
                # Cảnh báo khi N detected ≠ M expected
                diff = abs(validation['total_detected'] - validation['total_expected'])
                if diff > max(3, int(validation['total_expected'] * 0.1)):
                    warn_str = f" ⚠ lệch {diff} ký tự"
                # Cảnh báo từng cột lệch nhiều
                for cc in validation["column_comparison"]:
                    if abs(cc["diff"]) > max(2, int(cc["expected"] * 0.2)):
                        warn_str += f" [cột {cc['column']}: {cc['detected']}≠{cc['expected']}]"
            print(f"  Trang {book_page:4d}: {detection['total_chars']:>3} chars{acc_str}{warn_str}")

    # Summary
    if verbose and results:
        print("\n" + "=" * 70)
        print("TỔNG KẾT")
        print("=" * 70)
        print(f"  Tổng trang xử lý    : {len(results)}")
        print(f"  Tổng ký tự detected : {total_detected}")
        print(f"  Tổng ký tự expected : {total_expected}")
        if total_expected > 0:
            overall_acc = 1 - abs(total_detected - total_expected) / total_expected
            print(f"  Accuracy tổng       : {overall_acc:.1%}")
        print(f"  Trang accuracy ≥90% : {pages_matched}/{len(results)}")

    # Save summary
    summary = {
        "source_dir": str(prepared_dir),
        "total_pages": len(results),
        "total_detected": total_detected,
        "total_expected": total_expected,
        "pages_matched_90pct": pages_matched,
        "pages": [
            {
                "book_page": r["book_page"],
                "total_chars": r["detection"]["total_chars"],
                "validation": r.get("validation", {}),
            }
            for r in results
        ],
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    if verbose:
        print(f"\n  Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phát hiện và cắt từng ký tự Nôm từ ảnh trang sách",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prepared_dir", type=Path, help="Thư mục prepared data (từ prepare_data.py)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Thư mục output")
    parser.add_argument("--debug", action="store_true", help="Lưu ảnh debug với bbox")
    parser.add_argument("--page", type=int, default=None, help="Chỉ xử lý 1 trang")
    parser.add_argument("--quiet", action="store_true", help="Không hiển thị chi tiết")
    parser.add_argument(
        "--paddle", action="store_true",
        help="Dùng PaddleOCR hybrid detection (cần cài paddlepaddle + paddleocr)"
    )

    args = parser.parse_args()

    process_prepared_dir(
        args.prepared_dir,
        output_dir=args.output_dir,
        debug=args.debug,
        page_filter=args.page,
        verbose=not args.quiet,
        use_paddle=args.paddle,
    )


if __name__ == "__main__":
    main()
