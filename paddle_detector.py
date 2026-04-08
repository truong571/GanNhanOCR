#!/usr/bin/env python3
"""
paddle_detector.py - PaddleOCR hybrid character detection

PaddleOCR v3.4 detect TEXT LINES (cột), không phải từng ký tự.
→ Hybrid approach:
  1. PaddleOCR: detect column polygons (tight-fit, better than projection)
     + recognition → đếm số ký tự recognized mỗi cột để cross-validate
  2. Classical CV: character segmentation within each column
     (horizontal projection + merge/split — giữ nguyên)
  3. Validation: so sánh expected_count (QN) vs PaddleOCR rec count
     → flag columns có thể bị sai

Ưu điểm so với pure classical CV:
  - Column boundaries chính xác hơn (polygon tight-fit vs projection valleys)
  - PaddleOCR recognition cho thêm signal để validate alignment
  - Fallback graceful khi PaddleOCR fail

Usage:
  python paddle_detector.py page.png --transcription page.txt --compare
  python detect_characters.py data/prepared/Book --paddle
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Suppress PaddleOCR connectivity check
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_paddleocr = None


def _get_paddleocr():
    """Lazy-load PaddleOCR v3.4."""
    global _paddleocr
    if _paddleocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddleocr = PaddleOCR(
                use_textline_orientation=False,
                lang="chinese_cht",
            )
        except ImportError:
            raise ImportError(
                "PaddleOCR chưa được cài đặt. Chạy:\n"
                "  pip install paddlepaddle paddleocr\n"
                "Hoặc với GPU:\n"
                "  pip install paddlepaddle-gpu paddleocr"
            )
    return _paddleocr


# ---------------------------------------------------------------------------
# PaddleOCR column detection
# ---------------------------------------------------------------------------

def paddle_detect_columns(
    image_path: str,
    min_aspect_ratio: float = 2.0,
    min_height_ratio: float = 0.3,
) -> list[dict]:
    """Dùng PaddleOCR detect columns (text lines dọc) và recognize text.

    PaddleOCR v3.4 detect text lines, không phải individual chars.
    Với sách Hán Nôm dọc, mỗi text line = 1 cột.

    Args:
        image_path: Path to page image
        min_aspect_ratio: Tỷ lệ h/w tối thiểu để coi là cột text (loại noise)
        min_height_ratio: Chiều cao tối thiểu so với ảnh (loại annotation nhỏ)

    Returns:
        List[dict] mỗi item:
          - polygon: np.ndarray (4 corners)
          - bbox: (x1, y1, x2, y2)
          - rec_text: recognized text string
          - rec_score: recognition confidence
          - rec_char_count: số ký tự recognized
    """
    ocr = _get_paddleocr()
    results = list(ocr.predict(image_path))

    if not results or not results[0]:
        return []

    r = results[0]
    polys = r.get("dt_polys", [])
    rec_texts = r.get("rec_texts", [])
    rec_scores = r.get("rec_scores", [])

    # Lấy kích thước ảnh
    img = cv2.imread(image_path)
    if img is None:
        return []
    img_h, img_w = img.shape[:2]

    columns = []
    for i, poly in enumerate(polys):
        pts = np.array(poly, dtype=np.float32)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        aspect = h / w

        # Filter: chỉ giữ text regions đủ lớn và dọc (cột)
        if aspect < min_aspect_ratio:
            continue
        if h < img_h * min_height_ratio:
            continue

        text = rec_texts[i] if i < len(rec_texts) else ""
        score = rec_scores[i] if i < len(rec_scores) else 0.0

        columns.append({
            "polygon": pts,
            "bbox": (x1, y1, x2, y2),
            "rec_text": text,
            "rec_score": float(score),
            "rec_char_count": len(text),
        })

    # Sort phải → trái (cột 1 = phải nhất)
    columns.sort(key=lambda c: -c["bbox"][0])

    return columns


def _match_paddle_columns_with_classical(
    paddle_cols: list[dict],
    classical_cols: list[tuple[int, int, int, int]],
    max_x_offset: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Match PaddleOCR columns với classical columns, tạo merged column list.

    Strategy:
    - Mỗi classical column tìm PaddleOCR column gần nhất (theo center_x)
    - Nếu match → dùng PaddleOCR x boundaries (tighter), giữ classical y boundaries
    - Nếu không match → giữ nguyên classical column

    Args:
        paddle_cols: PaddleOCR detected columns (sorted R→L)
        classical_cols: Classical CV columns (sorted R→L)
        max_x_offset: Max offset ratio (so với column width) để match

    Returns:
        Merged column bboxes (same order as classical_cols)
    """
    if not paddle_cols:
        return list(classical_cols)

    merged = []
    used_paddle = set()

    for cl_bbox in classical_cols:
        cl_cx = (cl_bbox[0] + cl_bbox[2]) / 2.0
        cl_w = cl_bbox[2] - cl_bbox[0]

        best_pi = None
        best_dist = float("inf")

        for pi, pcol in enumerate(paddle_cols):
            if pi in used_paddle:
                continue
            p_cx = (pcol["bbox"][0] + pcol["bbox"][2]) / 2.0
            dist = abs(p_cx - cl_cx)
            if dist < best_dist and dist < cl_w * max_x_offset:
                best_dist = dist
                best_pi = pi

        if best_pi is not None:
            used_paddle.add(best_pi)
            pb = paddle_cols[best_pi]["bbox"]
            # Dùng PaddleOCR x (tighter), classical y (consistent)
            merged.append((pb[0], cl_bbox[1], pb[2], cl_bbox[3]))
        else:
            merged.append(cl_bbox)

    return merged


# ---------------------------------------------------------------------------
# Hybrid detector class
# ---------------------------------------------------------------------------

class PaddleHybridDetector:
    """Hybrid detector: PaddleOCR columns + classical CV char segmentation.

    PaddleOCR cải thiện:
    1. Column boundaries (tight-fit polygon thay vì projection valleys)
    2. Validation signal (rec_text count vs expected_count)
    """

    def __init__(self):
        self._paddle_available = None

    def is_available(self) -> bool:
        """Check xem PaddleOCR có cài đặt không."""
        if self._paddle_available is None:
            try:
                import paddleocr  # noqa: F401
                self._paddle_available = True
            except ImportError:
                self._paddle_available = False
        return self._paddle_available

    def detect_page(
        self,
        image_path: str,
        n_columns: int | None = None,
        expected_counts: list[int] | None = None,
    ) -> dict:
        """Detect characters using hybrid PaddleOCR + classical CV.

        Pipeline:
        1. Classical CV: binarize + detect text_box
        2. PaddleOCR: detect column polygons + recognize text
        3. Merge column boundaries (PaddleOCR x + classical y)
        4. Classical CV: character segmentation within merged columns
        5. Cross-validate: expected_count vs PaddleOCR rec count

        Falls back to pure classical CV if PaddleOCR fails.
        """
        from detect_characters import (
            load_and_binarize,
            detect_text_box,
            detect_columns,
            detect_chars_in_column,
            _auto_detect_n_columns,
        )

        gray, binary = load_and_binarize(image_path)
        h, w = gray.shape

        text_box = detect_text_box(binary)

        if n_columns is None and expected_counts is not None:
            n_columns = len(expected_counts)
        if n_columns is None:
            n_columns = _auto_detect_n_columns(binary, text_box)

        # Classical column detection (always — serves as baseline)
        classical_columns = detect_columns(binary, text_box, n_expected=n_columns)

        col_height = text_box[3] - text_box[1]
        expected_char_h = col_height / 22

        # --- PaddleOCR column detection ---
        paddle_cols = []
        method = "classical"

        if self.is_available():
            try:
                paddle_cols = paddle_detect_columns(image_path)
                if paddle_cols:
                    method = "paddle_hybrid"
                    logger.info(
                        f"PaddleOCR detected {len(paddle_cols)} columns "
                        f"(classical: {len(classical_columns)})"
                    )
            except Exception as e:
                logger.warning(f"PaddleOCR failed: {e}")

        # Merge columns: PaddleOCR boundaries + classical alignment
        if paddle_cols:
            columns = _match_paddle_columns_with_classical(
                paddle_cols, classical_columns
            )
        else:
            columns = list(classical_columns)

        # --- Character segmentation (classical CV within columns) ---
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

            # Cross-validation with PaddleOCR recognition
            paddle_rec_count = None
            paddle_rec_text = None
            if paddle_cols and col_idx < len(paddle_cols):
                # Tìm paddle column match gần nhất
                col_cx = (col_bbox[0] + col_bbox[2]) / 2.0
                for pcol in paddle_cols:
                    p_cx = (pcol["bbox"][0] + pcol["bbox"][2]) / 2.0
                    col_w = col_bbox[2] - col_bbox[0]
                    if abs(p_cx - col_cx) < col_w * 0.5:
                        paddle_rec_count = pcol["rec_char_count"]
                        paddle_rec_text = pcol["rec_text"]
                        break

            col_result = {
                "column": col_idx + 1,
                "bbox": list(col_bbox),
                "num_chars": len(chars),
                "method": method if paddle_cols else "classical",
                "chars": [],
            }

            # Add validation info
            if paddle_rec_count is not None:
                col_result["paddle_rec_count"] = paddle_rec_count
                col_result["paddle_rec_text"] = paddle_rec_text
                if exp_count:
                    col_result["paddle_count_match"] = (
                        abs(paddle_rec_count - exp_count) <= max(2, exp_count * 0.1)
                    )

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
            "method": method,
            "paddle_columns_detected": len(paddle_cols),
            "columns": column_results,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    from detect_characters import NumpyEncoder

    parser = argparse.ArgumentParser(description="PaddleOCR hybrid character detection")
    parser.add_argument("image", type=str, help="Path to page image")
    parser.add_argument("--n-columns", type=int, default=None)
    parser.add_argument("--expected-counts", type=str, default=None,
                        help="Comma-separated expected char counts per column")
    parser.add_argument("--transcription", type=str, default=None,
                        help="Path to transcription file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--compare", action="store_true",
                        help="So sánh PaddleOCR hybrid vs Classical CV")
    parser.add_argument("--debug-image", type=str, default=None,
                        help="Save debug image with column boundaries")

    args = parser.parse_args()

    expected_counts = None
    n_columns = args.n_columns

    if args.transcription:
        with open(args.transcription, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        expected_counts = [len(line.split()) for line in lines]
        if n_columns is None:
            n_columns = len(lines)
    elif args.expected_counts:
        expected_counts = [int(x) for x in args.expected_counts.split(",")]

    detector = PaddleHybridDetector()

    if not detector.is_available():
        print("[ERROR] PaddleOCR chưa cài đặt!")
        print("  pip install paddlepaddle paddleocr")
        return

    print(f"Detecting: {args.image}")

    # PaddleOCR raw column detection
    paddle_cols = paddle_detect_columns(args.image)
    print(f"\n--- PaddleOCR Column Detection ---")
    print(f"  Detected {len(paddle_cols)} columns")
    for i, col in enumerate(paddle_cols):
        bbox = col["bbox"]
        print(
            f"  Col {i+1}: bbox=({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) "
            f"rec={col['rec_char_count']} chars "
            f"score={col['rec_score']:.2f} "
            f"text='{col['rec_text'][:20]}'"
        )

    # Hybrid detection
    detection = detector.detect_page(args.image, n_columns=n_columns,
                                     expected_counts=expected_counts)

    print(f"\n--- Hybrid Detection ---")
    print(f"  Method: {detection['method']}")
    print(f"  Columns: {detection['num_columns']}")
    print(f"  Total chars: {detection['total_chars']}")

    for col in detection["columns"]:
        exp = ""
        if expected_counts and col["column"] - 1 < len(expected_counts):
            exp = f" (exp={expected_counts[col['column']-1]})"
        paddle_info = ""
        if "paddle_rec_count" in col:
            paddle_info = f" paddle_rec={col['paddle_rec_count']}"
        print(
            f"    Col {col['column']}: {col['num_chars']} chars{exp}{paddle_info}"
        )

    if args.compare:
        print(f"\n--- Classical CV ---")
        from detect_characters import detect_page
        classical = detect_page(args.image, n_columns=n_columns,
                                expected_counts=expected_counts)
        print(f"  Total chars: {classical['total_chars']}")
        for col in classical["columns"]:
            exp = ""
            if expected_counts and col["column"] - 1 < len(expected_counts):
                exp = f" (exp={expected_counts[col['column']-1]})"
            print(f"    Col {col['column']}: {col['num_chars']} chars{exp}")

        if expected_counts:
            total_exp = sum(expected_counts)
            hybrid_diff = abs(detection["total_chars"] - total_exp)
            classical_diff = abs(classical["total_chars"] - total_exp)
            print(f"\n  Expected total: {total_exp}")
            print(f"  Hybrid diff:   {hybrid_diff} {'(better)' if hybrid_diff < classical_diff else '(worse)' if hybrid_diff > classical_diff else '(same)'}")
            print(f"  Classical diff: {classical_diff} {'(better)' if classical_diff < hybrid_diff else '(worse)' if classical_diff > hybrid_diff else '(same)'}")

    if args.debug_image:
        _save_debug_image(args.image, detection, paddle_cols, args.debug_image)
        print(f"\n  Debug image: {args.debug_image}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(detection, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"\n  Saved: {args.output}")


def _save_debug_image(
    image_path: str,
    detection: dict,
    paddle_cols: list[dict],
    output_path: str,
):
    """Save debug image with both PaddleOCR and hybrid column boundaries."""
    img = cv2.imread(image_path)
    if img is None:
        return

    # PaddleOCR polygons (xanh lá)
    for pcol in paddle_cols:
        pts = pcol["polygon"].astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    # Hybrid column bboxes (đỏ)
    for col in detection["columns"]:
        bx = col["bbox"]
        cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
        cv2.putText(
            img, f"C{col['column']}: {col['num_chars']}",
            (bx[0] + 5, bx[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

    # Character bboxes (xanh dương nhạt)
    for col in detection["columns"]:
        for char_info in col["chars"]:
            cb = char_info["bbox"]
            cv2.rectangle(img, (cb[0], cb[1]), (cb[2], cb[3]), (255, 128, 0), 1)

    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    main()
