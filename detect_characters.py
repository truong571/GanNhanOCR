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

    Sử dụng vertical projection + find_peaks với constraint n_expected.

    Args:
        binary: Binary image (1=ink)
        text_box: (left, top, right, bottom) vùng text
        n_expected: Số cột kỳ vọng (mặc định 9)

    Returns:
        List[(x1, y1, x2, y2)] cho mỗi cột, thứ tự phải→trái (cột 1 ở phải)
    """
    left, top, right, bottom = text_box
    text_region = binary[top:bottom, left:right]
    region_w = right - left

    # Vertical projection
    v_proj = text_region.sum(axis=0).astype(float)

    # Smooth: kernel size phải đủ lớn để merge noise nhưng nhỏ hơn gap giữa cột
    expected_col_width = region_w / n_expected
    kernel = max(5, int(expected_col_width / 8))
    v_smooth = np.convolve(v_proj, np.ones(kernel) / kernel, mode="same")

    # Tìm valleys (gaps giữa cột)
    min_gap_distance = int(expected_col_width * 0.5)
    valleys, props = find_peaks(-v_smooth, distance=min_gap_distance)
    valley_depths = v_smooth[valleys]

    # Cần đúng n_expected-1 gaps
    if len(valleys) >= n_expected - 1:
        # Chọn n-1 valley sâu nhất (lowest density)
        sorted_idx = np.argsort(valley_depths)
        best_valleys = sorted(valleys[sorted_idx[: n_expected - 1]])
    elif len(valleys) > 0:
        # Ít hơn kỳ vọng → dùng tất cả + chia đều phần còn thiếu
        best_valleys = sorted(valleys)
    else:
        # Không tìm được gap → chia đều
        best_valleys = [int(region_w * i / n_expected) for i in range(1, n_expected)]

    # Bổ sung thêm nếu thiếu (chia đều các cột lớn)
    while len(best_valleys) < n_expected - 1:
        # Tìm khoảng rộng nhất chưa chia
        boundaries = [0] + list(best_valleys) + [region_w]
        widths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        widest_idx = np.argmax(widths)
        mid = (boundaries[widest_idx] + boundaries[widest_idx + 1]) // 2
        best_valleys.append(mid)
        best_valleys.sort()

    # Build column bboxes
    boundaries = [0] + list(best_valleys) + [region_w]
    columns = []
    for i in range(len(boundaries) - 1):
        x1 = boundaries[i] + left
        x2 = boundaries[i + 1] + left
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
    """Xóa vùng border lines bên trái/phải cột.

    Sách Hán Nôm có đường kẻ dọc ngăn cách các cột. Đường kẻ tạo ink giả
    suốt chiều dài cột → horizontal projection không tìm được gap.

    Phương pháp: dùng vertical projection để tìm vùng có ink liên tục
    (> 50% chiều cao) ở 2 bên mép → mask = 0.
    """
    h, w = col_binary.shape
    if h == 0 or w == 0:
        return col_binary

    v_proj = col_binary.sum(axis=0).astype(float)
    # Ngưỡng: cột pixel nào có ink > 40% chiều cao → có thể là border line
    line_threshold = h * 0.40

    cleaned = col_binary.copy()

    # Scan từ trái vào: xóa đến khi hết border
    left_trim = 0
    for x in range(min(w // 4, 40)):
        if v_proj[x] > line_threshold:
            left_trim = x + 4  # Thêm 4px buffer
    if left_trim > 0:
        cleaned[:, :left_trim] = 0

    # Scan từ phải vào
    right_trim = w
    for x in range(w - 1, max(w - w // 4, w - 40), -1):
        if v_proj[x] > line_threshold:
            right_trim = x - 4
    if right_trim < w:
        cleaned[:, right_trim:] = 0

    return cleaned


def _trim_to_ink_extent(
    col_binary: np.ndarray, expected_char_height: float
) -> tuple[int, int]:
    """Tìm vùng chứa chữ thật trong cột (bỏ vùng trống trên/dưới).

    Nhiều cột có chiều cao = full page nhưng chữ chỉ chiếm phần trên.
    Nếu dùng full height, expected_char_height bị sai → chia sai.

    Returns:
        (top_offset, bottom_offset) relative to column top
    """
    h_proj = col_binary.sum(axis=1).astype(float)
    col_w = col_binary.shape[1]

    # Smooth mạnh hơn (1 expected char height) để phân biệt
    # vùng có chữ (density cao) và vùng trống (density thấp)
    smooth_k = max(5, int(expected_char_height * 0.5)) | 1
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    if h_smooth.max() == 0:
        return 0, col_binary.shape[0]

    # Ngưỡng: 10% của density trung bình vùng có ink
    mean_ink = np.mean(h_smooth[h_smooth > 0])
    ink_threshold = mean_ink * 0.10

    ink_rows = np.where(h_smooth > ink_threshold)[0]
    if len(ink_rows) == 0:
        return 0, col_binary.shape[0]

    top = max(0, int(ink_rows[0]) - 5)
    bottom = min(col_binary.shape[0], int(ink_rows[-1]) + 5)

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
    4. find_peaks tìm N-1 valleys sâu nhất
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

    # Tìm valleys
    min_dist = max(5, int(real_char_h * 0.5))
    valleys, _ = find_peaks(-h_smooth, distance=min_dist)

    if len(valleys) < expected_count - 1:
        # Thử min_dist nhỏ hơn
        min_dist2 = max(5, int(real_char_h * 0.35))
        valleys, _ = find_peaks(-h_smooth, distance=min_dist2)

    if len(valleys) < expected_count - 1:
        return []

    # Chọn N-1 valleys sâu nhất
    depths = h_smooth[valleys]
    n_needed = expected_count - 1
    best_idx = np.argsort(depths)[:n_needed]
    split_points = sorted(valleys[best_idx])

    # Giữ nguyên split points, chỉ đảm bảo start=0 và end=ink_h
    all_points = [0] + list(split_points) + [ink_h]

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


def detect_chars_in_column(
    binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    min_char_height: int = 15,
    expected_char_height: float | None = None,
    expected_count: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Phát hiện ký tự trong 1 cột.

    Chiến lược 3 bước:
    1. Threshold approach: tìm gap tuyệt đối trong horizontal projection
    2. Valley approach: nếu threshold thất bại, tìm local minima (chữ dày đặc)
    3. Multi-pass retry với nhiều mức threshold

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

    # --- Pass 1: Standard density threshold ---
    raw_chars = _extract_raw_blobs(
        col_binary, col_bbox, expected_char_height, min_char_height, 0.01
    )

    if not raw_chars:
        # Không tìm được gap nào → thử valley-based segmentation
        if expected_count and expected_count > 1:
            valley_chars = _segment_by_valleys(
                col_binary, col_bbox, expected_count, col_h / expected_count
            )
            if valley_chars and abs(len(valley_chars) - expected_count) <= max(1, expected_count // 5):
                return valley_chars
            # Fallback: chia đều trên vùng có chữ (trim border + trống)
            cleaned = _strip_column_borders(col_binary)
            ink_top, ink_bottom = _trim_to_ink_extent(cleaned, col_h / expected_count)
            ink_h = ink_bottom - ink_top
            if ink_h > col_h * 0.3:  # Trimmed phải > 30% column
                step = ink_h / expected_count
                return [(x1, int(y1 + ink_top + i * step), x2, int(y1 + ink_top + (i + 1) * step))
                        for i in range(expected_count)]
            # Chia đều full column nếu trim không hợp lý
            step = col_h / expected_count
            return [(x1, int(y1 + i * step), x2, int(y1 + (i + 1) * step))
                    for i in range(expected_count)]
        return []

    # Post-processing: merge small, then split large
    chars = _merge_small_boxes(raw_chars, expected_char_height)
    chars = _split_large_boxes(chars, expected_char_height, binary)

    # Final adjustment: force-split largest blobs nếu vẫn thiếu
    if expected_count and len(chars) < expected_count:
        chars = _force_split_to_count(chars, expected_count, expected_char_height, binary)

    # --- Pass 2: Nếu kết quả vẫn lệch quá nhiều, thử valley approach ---
    if expected_count and expected_count > 1:
        diff = abs(len(chars) - expected_count)
        tolerance = max(2, int(expected_count * 0.15))

        if diff > tolerance:
            best_chars = chars
            best_diff = diff

            # Thử valley-based segmentation (hiệu quả với chữ dày đặc)
            valley_chars = _segment_by_valleys(
                col_binary, col_bbox, expected_count, col_h / expected_count
            )
            if valley_chars:
                valley_diff = abs(len(valley_chars) - expected_count)
                if valley_diff < best_diff:
                    best_diff = valley_diff
                    best_chars = valley_chars

            # Thử nhiều density thresholds
            if best_diff > tolerance:
                for density_scale in [0.02, 0.05, 0.005]:
                    retry_raw = _extract_raw_blobs(
                        col_binary, col_bbox, col_h / expected_count,
                        min_char_height, density_scale
                    )
                    if not retry_raw:
                        continue
                    retry_h = col_h / expected_count
                    retry_chars = _merge_small_boxes(retry_raw, retry_h)
                    retry_chars = _split_large_boxes(retry_chars, retry_h, binary)
                    if len(retry_chars) < expected_count:
                        retry_chars = _force_split_to_count(
                            retry_chars, expected_count, retry_h, binary)
                    retry_diff = abs(len(retry_chars) - expected_count)
                    if retry_diff < best_diff:
                        best_diff = retry_diff
                        best_chars = retry_chars
                    if best_diff <= 1:
                        break

            chars = best_chars

    return chars


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


def save_char_crops(
    image_path: str, detection: dict, output_dir: Path, page_num: int
) -> list[str]:
    """Cắt và lưu ảnh từng ký tự."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    crops_dir = output_dir / "crops" / f"page_{page_num:04d}"
    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = []
    for col in detection["columns"]:
        for char_info in col["chars"]:
            x1, y1, x2, y2 = char_info["bbox"]

            # Thêm padding nhỏ
            pad = 3
            y1p = max(0, y1 - pad)
            y2p = min(gray.shape[0], y2 + pad)
            x1p = max(0, x1 - pad)
            x2p = min(gray.shape[1], x2 + pad)

            crop = gray[y1p:y2p, x1p:x2p]

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

    if verbose:
        print(f"\nChar Detection: {prepared_dir.name}")
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

    args = parser.parse_args()

    process_prepared_dir(
        args.prepared_dir,
        output_dir=args.output_dir,
        debug=args.debug,
        page_filter=args.page,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
