#!/usr/bin/env python3
"""
visualize_results.py - Tạo hình minh họa kết quả detection + recognition

Tạo 3 ảnh cho mỗi trang:
  1. Detection: bbox đỏ quanh từng ký tự (giống hình 1)
  2. Recognition: chữ Nôm xanh dương overlay lên trang (giống hình 2)
  3. Pipeline: Page → Detection → Recognition (giống hình 3)

Usage:
  python visualize_results.py data/prepared/CacThanhTruyen2
  python visualize_results.py data/prepared/CacThanhTruyen2 --page 12
  python visualize_results.py data/prepared/CacThanhTruyen2 --page 12 --pipeline
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_font(size: int):
    """Load font Nôm cho rendering chữ."""
    font_paths = [
        Path(__file__).parent.parent / "FontDiffusion/fonts/NomNaTong-Regular.ttf",
    ]
    for fp in font_paths:
        if fp.exists():
            return ImageFont.truetype(str(fp), size)
    # Fallback: system CJK font
    system_fonts = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in system_fonts:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_detection(page_img: np.ndarray, detection: dict) -> np.ndarray:
    """Vẽ bbox đỏ quanh từng ký tự detected."""
    vis = page_img.copy()
    thickness = 3

    for col in detection["columns"]:
        for char in col["chars"]:
            x1, y1, x2, y2 = char["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), thickness)

    return vis


def draw_recognition(page_img: np.ndarray, detection: dict,
                     labels: dict) -> np.ndarray:
    """Vẽ chữ Nôm xanh dương overlay lên trang."""
    if not HAS_PIL:
        print("[WARN] PIL not available, skipping recognition overlay")
        return page_img

    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for col in detection["columns"]:
        for char in col["chars"]:
            x1, y1, x2, y2 = char["bbox"]
            crop_file = char["crop_file"]

            # Look up label
            label_info = labels.get(crop_file)
            if not label_info:
                continue

            nom_char = label_info["nom_char"]
            if not nom_char:
                continue

            # Font size based on bbox
            char_h = y2 - y1
            char_w = x2 - x1
            font_size = int(min(char_h, char_w) * 0.85)
            font = load_font(max(16, font_size))

            # Draw character centered in bbox
            bbox_text = font.getbbox(nom_char)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            tx = x1 + (char_w - tw) // 2
            ty = y1 + (char_h - th) // 2

            # Color based on confidence
            conf = label_info.get("confidence", "medium")
            if conf == "high":
                color = (0, 0, 255)       # Blue
            elif conf == "medium":
                color = (70, 70, 220)     # Lighter blue
            else:
                color = (180, 0, 0)       # Red for low

            draw.text((tx, ty), nom_char, fill=color, font=font)

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result


def create_pipeline_image(page_img: np.ndarray, det_img: np.ndarray,
                          rec_img: np.ndarray, page_name: str) -> np.ndarray:
    """Tạo hình pipeline: Page → Detection → Recognition."""
    h, w = page_img.shape[:2]

    # Scale down to fit 3 panels
    scale = 500 / h
    sw = int(w * scale)
    sh = int(h * scale)

    page_small = cv2.resize(page_img, (sw, sh))
    det_small = cv2.resize(det_img, (sw, sh))
    rec_small = cv2.resize(rec_img, (sw, sh))

    # Canvas
    arrow_w = 80
    padding = 30
    label_h = 50
    canvas_w = sw * 3 + arrow_w * 2 + padding * 2
    canvas_h = sh + label_h + padding * 2

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Draw 3 panels
    panels = [
        (page_small, "Page"),
        (det_small, "Text Detection"),
        (rec_small, "Text Recognition"),
    ]

    x_offsets = []
    x = padding
    for i, (img, label) in enumerate(panels):
        y = padding
        canvas[y:y + sh, x:x + sw] = img
        # Border
        cv2.rectangle(canvas, (x, y), (x + sw, y + sh), (0, 0, 0), 2)
        x_offsets.append(x)

        # Label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = x + (sw - text_size[0]) // 2
        ty = y + sh + 35
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        x += sw

        # Arrow between panels
        if i < 2:
            arrow_y = padding + sh // 2
            ax1 = x + 10
            ax2 = x + arrow_w - 10
            # Arrow body
            cv2.arrowedLine(canvas, (ax1, arrow_y), (ax2, arrow_y),
                            (0, 180, 240), 3, tipLength=0.3)
            x += arrow_w

    return canvas


def load_labels(labeled_dir: Path, page_num: int) -> dict:
    """Load labels từ CSV cho 1 trang."""
    csv_path = labeled_dir / "labels.csv"
    if not csv_path.exists():
        return {}

    labels = {}
    page_prefix = f"page_{page_num:04d}"

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("image", "")
            if page_prefix not in img_path:
                continue

            # Map crop_file → use the original crop path
            # labels.csv has: crops_cleaned/page_0012/col01_char000.png
            # detection JSON has: crops/page_0012/col01_char000.png
            crop_key = img_path.replace("crops_cleaned/", "crops/")

            labels[crop_key] = {
                "nom_char": row.get("nom_char", ""),
                "label": row.get("label", ""),
                "reading": row.get("reading", ""),
                "confidence": row.get("confidence", "medium"),
            }

    return labels


def process_page(book_dir: Path, page_num: int, output_dir: Path,
                 do_pipeline: bool = False):
    """Xử lý 1 trang: tạo detection + recognition images."""
    page_name = f"page_{page_num:04d}"
    page_img_path = book_dir / "pages" / f"{page_name}.png"
    det_json_path = book_dir / "detected" / f"{page_name}_detection.json"

    if not page_img_path.exists():
        print(f"  [SKIP] {page_name}: không tìm thấy ảnh")
        return
    if not det_json_path.exists():
        print(f"  [SKIP] {page_name}: không tìm thấy detection JSON")
        return

    # Load
    page_img = cv2.imread(str(page_img_path))
    with open(det_json_path) as f:
        det_data = json.load(f)
    detection = det_data["detection"]
    labels = load_labels(book_dir / "labeled", page_num)

    # 1. Detection image (red bboxes)
    det_img = draw_detection(page_img, detection)
    det_path = output_dir / f"{page_name}_detection.png"
    cv2.imwrite(str(det_path), det_img)
    print(f"  {page_name}_detection.png")

    # 2. Recognition image (blue text overlay)
    rec_img = draw_recognition(page_img, detection, labels)
    rec_path = output_dir / f"{page_name}_recognition.png"
    cv2.imwrite(str(rec_path), rec_img)
    print(f"  {page_name}_recognition.png")

    # 3. Pipeline image (3 panels)
    if do_pipeline:
        pip_img = create_pipeline_image(page_img, det_img, rec_img, page_name)
        pip_path = output_dir / f"{page_name}_pipeline.png"
        cv2.imwrite(str(pip_path), pip_img)
        print(f"  {page_name}_pipeline.png")


def main():
    parser = argparse.ArgumentParser(
        description="Tạo hình minh họa kết quả detection + recognition")
    parser.add_argument("book_dir", help="Thư mục prepared data (vd: data/prepared/CacThanhTruyen2)")
    parser.add_argument("--page", type=int, help="Chỉ xử lý 1 trang")
    parser.add_argument("--pipeline", action="store_true",
                        help="Tạo thêm hình pipeline (3 panels)")
    parser.add_argument("--output", help="Thư mục output (mặc định: book_dir/visualization/)")
    args = parser.parse_args()

    book_dir = Path(args.book_dir)
    if not book_dir.exists():
        print(f"[ERROR] Không tìm thấy: {book_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else book_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    book_name = book_dir.name
    print(f"Visualize Results: {book_name}")
    print(f"Output: {output_dir}/")
    print("-" * 60)

    if args.page:
        process_page(book_dir, args.page, output_dir, args.pipeline)
    else:
        # Process all pages
        det_files = sorted((book_dir / "detected").glob("*_detection.json"))
        for det_file in det_files:
            page_num = int(det_file.stem.replace("page_", "").replace("_detection", ""))
            process_page(book_dir, page_num, output_dir, args.pipeline)

    print("-" * 60)
    print(f"Done! Output: {output_dir}/")


if __name__ == "__main__":
    main()
