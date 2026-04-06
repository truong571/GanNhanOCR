#!/usr/bin/env python3
"""
detect_characters.py - Phát hiện và cắt từng ký tự Nôm từ ảnh trang sách

Phương pháp: Classical CV cải tiến (B+)
- Adaptive thresholding + border removal
- Column detection bằng vertical projection + constraint 9 cột
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
    """Load ảnh và tạo binary mask.

    Sử dụng Sauvola-like local thresholding để xử lý tốt cả
    ảnh sạch (SachThanhTruyen) và ảnh có background noise (CacThanhTruyen).

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


# ---------------------------------------------------------------------------
# Character segmentation
# ---------------------------------------------------------------------------

def detect_chars_in_column(
    binary: np.ndarray,
    col_bbox: tuple[int, int, int, int],
    min_char_height: int = 15,
    expected_char_height: float | None = None,
    expected_count: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Phát hiện ký tự trong 1 cột.

    Chiến lược kết hợp:
    1. Horizontal projection để tìm character blobs
    2. Dùng expected_count (từ transcription) để tune split/merge

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

    # Horizontal projection
    h_proj = col_binary.sum(axis=1).astype(float)

    # Smooth: nối nét gần nhau trong cùng 1 ký tự
    # Kernel ~ 10% chiều cao ký tự
    smooth_k = max(3, int(expected_char_height * 0.1))
    h_smooth = np.convolve(h_proj, np.ones(smooth_k) / smooth_k, mode="same")

    # Threshold rất thấp: bất kỳ dòng nào có ink đều "not gap"
    threshold = max(col_w * 0.01, 0.5)
    is_gap = h_smooth < threshold

    # Dilate gaps: mở rộng gap nhẹ để tách ký tự dính nhau
    # Nhưng chỉ tại vị trí density thấp
    min_gap = max(2, int(expected_char_height * 0.03))
    # Erode is_gap to remove tiny gaps (noise)
    gap_kernel = np.ones(min_gap)
    is_gap_clean = np.convolve(is_gap.astype(float), gap_kernel / min_gap, mode="same") > 0.8

    # Find connected regions
    transitions = np.diff(is_gap_clean.astype(int))
    char_starts = np.where(transitions == -1)[0]
    char_ends = np.where(transitions == 1)[0]

    if len(char_starts) == 0 and len(char_ends) == 0:
        # Toàn bộ cột là 1 blob → chia đều
        if expected_count and expected_count > 0:
            step = col_h / expected_count
            return [(x1, int(y1 + i * step), x2, int(y1 + (i + 1) * step))
                    for i in range(expected_count)]
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

    if not raw_chars:
        return raw_chars

    # Post-processing: merge small, then split large
    chars = _merge_small_boxes(raw_chars, expected_char_height)
    chars = _split_large_boxes(chars, expected_char_height, binary)

    # Final adjustment: if we still have too few chars and know expected count,
    # force-split the largest remaining blobs
    if expected_count and len(chars) < expected_count:
        chars = _force_split_to_count(chars, expected_count, expected_char_height, binary)

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

    Chỉ split blob có height > 1.3x expected_h.
    """
    result = list(chars)
    max_iterations = target_count  # Safety limit

    while len(result) < target_count and max_iterations > 0:
        max_iterations -= 1

        # Tìm blob lớn nhất có thể split
        heights = [(b[3] - b[1], i) for i, b in enumerate(result)]
        heights.sort(reverse=True)

        split_done = False
        for h_val, idx in heights:
            if h_val < expected_h * 1.3:
                break  # Không có blob đủ lớn để split

            bbox = result[idx]
            x1, y1, x2, y2 = bbox
            n_parts = min(round(h_val / expected_h), target_count - len(result) + 1)
            n_parts = max(2, n_parts)

            # Tìm split point bằng projection valley
            col_binary = binary[y1:y2, x1:x2]
            h_proj = col_binary.sum(axis=1).astype(float)
            h_smooth = np.convolve(h_proj, np.ones(3) / 3, mode="same")

            min_dist = int(expected_h * 0.4)
            valleys, _ = find_peaks(-h_smooth, distance=min_dist)

            if len(valleys) >= n_parts - 1:
                depths = h_smooth[valleys]
                best_idx = np.argsort(depths)[: n_parts - 1]
                split_points = sorted(valleys[best_idx])
            else:
                # Chia đều
                box_h = y2 - y1
                split_points = [int(box_h * i / n_parts) for i in range(1, n_parts)]

            # Replace blob with sub-blobs
            sub_boxes = []
            all_points = [0] + list(split_points) + [y2 - y1]
            for j in range(len(all_points) - 1):
                sy1 = y1 + all_points[j]
                sy2 = y1 + all_points[j + 1]
                if sy2 - sy1 >= 10:
                    sub_boxes.append((x1, sy1, x2, sy2))

            if len(sub_boxes) > 1:
                result = result[:idx] + sub_boxes + result[idx + 1 :]
                split_done = True
                break

        if not split_done:
            break  # Không thể split thêm

    return result


def _split_large_boxes(
    chars: list[tuple[int, int, int, int]], expected_h: float, binary: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """Split box quá lớn (> 1.6x expected height) thành 2+ ký tự."""
    split_threshold = expected_h * 1.6

    result = []
    for bbox in chars:
        x1, y1, x2, y2 = bbox
        box_h = y2 - y1

        if box_h > split_threshold:
            n_parts = round(box_h / expected_h)
            n_parts = max(2, min(n_parts, 4))  # 2-4 phần

            # Tìm split point tốt nhất (gap nhỏ nhất trong box)
            col_binary = binary[y1:y2, x1:x2]
            h_proj = col_binary.sum(axis=1).astype(float)
            h_smooth = np.convolve(h_proj, np.ones(3) / 3, mode="same")

            # Tìm n_parts-1 valleys trong box
            min_dist = int(expected_h * 0.4)
            valleys, _ = find_peaks(-h_smooth, distance=min_dist)

            if len(valleys) >= n_parts - 1:
                # Chọn valleys sâu nhất
                depths = h_smooth[valleys]
                best_idx = np.argsort(depths)[: n_parts - 1]
                split_points = sorted(valleys[best_idx])
            else:
                # Chia đều
                split_points = [int(box_h * i / n_parts) for i in range(1, n_parts)]

            # Tạo sub-boxes
            all_points = [0] + list(split_points) + [box_h]
            for j in range(len(all_points) - 1):
                sub_y1 = y1 + all_points[j]
                sub_y2 = y1 + all_points[j + 1]
                if sub_y2 - sub_y1 >= 10:  # Min size
                    result.append((x1, sub_y1, x2, sub_y2))
        else:
            result.append(bbox)

    return result


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

def detect_page(
    image_path: str,
    n_columns: int = 9,
    expected_counts: list[int] | None = None,
) -> dict:
    """Pipeline đầy đủ: load ảnh → detect columns → detect chars.

    Args:
        expected_counts: Số ký tự kỳ vọng mỗi cột (từ transcription).
                         Dùng để cải thiện split/merge.

    Returns:
        dict với columns, chars, metadata
    """
    gray, binary = load_and_binarize(image_path)
    h, w = gray.shape

    # Detect text box
    text_box = detect_text_box(binary)

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

        image_path = str(prepared_dir / page_info["image_file"])
        trans_path = str(prepared_dir / page_info["transcription_file"])

        if not Path(image_path).exists():
            if verbose:
                print(f"  [SKIP] Trang {book_page}: ảnh không tồn tại")
            continue

        # Load expected counts from transcription (for better split/merge)
        expected_counts = None
        if Path(trans_path).exists():
            with open(trans_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            expected_counts = [len(line.split()) for line in lines]

        # Detect
        detection = detect_page(image_path, expected_counts=expected_counts)

        # Validate
        validation = {}
        if Path(trans_path).exists():
            validation = validate_with_transcription(detection, trans_path)
            total_detected += validation["total_detected"]
            total_expected += validation["total_expected"]
            if validation["accuracy"] >= 0.9:
                pages_matched += 1

        # Save char crops
        save_char_crops(image_path, detection, output_dir, book_page)

        # Save debug image
        if debug:
            debug_path = str(debug_dir / f"page_{book_page:04d}_debug.png")
            save_debug_image(image_path, detection, debug_path)

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
            if validation:
                acc = validation["accuracy"]
                col_match = sum(1 for c in validation["column_comparison"] if c["match"])
                acc_str = (
                    f"  det={validation['total_detected']:>3} "
                    f"exp={validation['total_expected']:>3} "
                    f"acc={acc:.0%} "
                    f"col_match={col_match}/9"
                )
            print(f"  Trang {book_page:4d}: {detection['total_chars']:>3} chars{acc_str}")

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
