#!/usr/bin/env python3
"""
visualize_labels.py - Vẽ bbox + nom_char + QN reading lên ảnh trang Nôm gốc

Giúp kiểm tra trực quan xem nom_char có khớp với chữ viết tay trong bbox không.

Màu theo confidence:
  - Xanh lá (green)  = high confidence
  - Vàng (yellow)     = medium confidence
  - Đỏ (red)          = low confidence

Usage:
  # 1 trang:
  python visualize_labels.py Data/prepared/CacThanhTruyen2 --page 12

  # Tất cả trang:
  python visualize_labels.py Data/prepared/CacThanhTruyen2

  # Chỉ hiện medium/low (chỗ cần review):
  python visualize_labels.py Data/prepared/CacThanhTruyen2 --page 12 --ambiguous-only

  # Output PDF gộp tất cả trang:
  python visualize_labels.py Data/prepared/CacThanhTruyen2 --pdf output.pdf
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
except ImportError:
    print("[ERROR] Pillow required: pip install Pillow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Font + rendering
# ---------------------------------------------------------------------------

_NOM_FONT_PATH = "FontDiffusion/fonts/NomNaTong-Regular.ttf"


def _find_font(prepared_dir: Path) -> str:
    """Tìm font NomNaTong."""
    candidates = [
        prepared_dir / "../../FontDiffusion/fonts/NomNaTong-Regular.ttf",
        Path(_NOM_FONT_PATH),
        Path(__file__).parent / _NOM_FONT_PATH,
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return ""


def _draw_text_pil(
    cv_img: np.ndarray,
    text: str,
    position: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill=(255, 0, 0),
    bg_fill=(255, 255, 255, 200),
) -> np.ndarray:
    """Vẽ text Unicode (Nôm) lên ảnh OpenCV bằng PIL (hỗ trợ CJK)."""
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = position
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Background rectangle
    pad = 2
    draw.rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        fill=bg_fill,
    )
    draw.text((x - bbox[0], y - bbox[1]), text, fill=fill, font=font)

    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Confidence colors
# ---------------------------------------------------------------------------

COLORS = {
    "high": (0, 200, 0),      # Green (BGR)
    "medium": (0, 200, 255),   # Yellow/Orange
    "low": (0, 0, 255),        # Red
}

COLORS_PIL = {
    "high": (0, 200, 0),       # Green (RGB)
    "medium": (255, 200, 0),   # Yellow/Orange
    "low": (255, 0, 0),        # Red
}

BG_COLORS_PIL = {
    "high": (255, 255, 255, 180),
    "medium": (255, 255, 200, 220),
    "low": (255, 200, 200, 220),
}


# ---------------------------------------------------------------------------
# Core visualization
# ---------------------------------------------------------------------------

def visualize_page(
    image_path: str,
    labels: list[dict],
    font_path: str,
    ambiguous_only: bool = False,
    font_size: int = 28,
) -> np.ndarray:
    """Vẽ bbox + nom_char + QN reading lên ảnh trang gốc.

    Args:
        image_path: Path to original page image
        labels: List of label dicts (from labels.csv) for this page
        font_path: Path to NomNaTong font
        ambiguous_only: Chỉ vẽ medium/low (bỏ high)
        font_size: Font size cho text overlay

    Returns:
        Annotated image (BGR numpy array)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    # Load PIL font
    if font_path and Path(font_path).exists():
        pil_font = ImageFont.truetype(font_path, font_size)
        pil_font_small = ImageFont.truetype(font_path, max(14, font_size // 2))
    else:
        pil_font = ImageFont.load_default()
        pil_font_small = pil_font

    for lab in labels:
        confidence = lab.get("confidence", "low")

        if ambiguous_only and confidence == "high":
            continue

        bbox_str = lab.get("bbox", "")
        if not bbox_str:
            continue

        coords = [int(x) for x in bbox_str.split(",")]
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords

        nom_char = lab.get("nom_char", "?")
        reading = lab.get("reading", "")
        color_bgr = COLORS.get(confidence, (128, 128, 128))
        color_rgb = COLORS_PIL.get(confidence, (128, 128, 128))
        bg_rgba = BG_COLORS_PIL.get(confidence, (255, 255, 255, 180))

        # Vẽ bbox rectangle
        thickness = 2 if confidence != "high" else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

        # Vẽ nom_char bên trái bbox
        text_x = max(0, x1 - font_size - 8)
        text_y = y1 + 2
        if nom_char:
            img = _draw_text_pil(
                img, nom_char, (text_x, text_y),
                pil_font, fill=color_rgb, bg_fill=bg_rgba,
            )

        # Vẽ QN reading nhỏ bên dưới nom_char
        if reading:
            img = _draw_text_pil(
                img, reading, (text_x, text_y + font_size + 2),
                pil_font_small, fill=color_rgb, bg_fill=bg_rgba,
            )

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vẽ bbox + nom_char lên ảnh trang Nôm gốc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prepared_dir", type=Path,
                        help="Thư mục prepared data")
    parser.add_argument("--page", type=int, default=None,
                        help="Chỉ xử lý 1 trang")
    parser.add_argument("--ambiguous-only", action="store_true",
                        help="Chỉ vẽ medium/low confidence (chỗ cần review)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Thư mục output (mặc định: labeled/visualize/)")
    parser.add_argument("--font-size", type=int, default=28,
                        help="Font size cho text overlay")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Xuất tất cả trang thành 1 file PDF")

    args = parser.parse_args()

    prepared_dir = args.prepared_dir
    labeled_dir = prepared_dir / "labeled"
    labels_csv = labeled_dir / "labels.csv"

    if not labels_csv.exists():
        print(f"[ERROR] Không tìm thấy {labels_csv}")
        print("  Chạy label_characters.py trước")
        sys.exit(1)

    # Load labels
    with open(labels_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_labels = list(reader)

    # Group by page
    page_labels: dict[int, list[dict]] = {}
    for lab in all_labels:
        p = int(lab.get("page", 0))
        page_labels.setdefault(p, []).append(lab)

    # Load manifest for image paths
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Không tìm thấy {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    page_images = {}
    for pi in manifest["pages"]:
        page_images[pi["book_page"]] = str(prepared_dir / pi["image_file"])

    # Font
    font_path = _find_font(prepared_dir)
    if not font_path:
        print("[WARNING] Không tìm thấy NomNaTong font, text sẽ dùng font mặc định")

    # Output dir
    output_dir = args.output_dir or labeled_dir / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter pages
    pages_to_process = sorted(page_labels.keys())
    if args.page is not None:
        pages_to_process = [p for p in pages_to_process if p == args.page]
        if not pages_to_process:
            print(f"[ERROR] Trang {args.page} không có trong labels")
            sys.exit(1)

    print(f"Visualize Labels: {prepared_dir.name}")
    print(f"  Labels: {len(all_labels)} entries")
    print(f"  Pages: {len(pages_to_process)}")
    if args.ambiguous_only:
        print(f"  Mode: ambiguous-only (medium/low)")
    print(f"  Output: {output_dir}/")
    print("-" * 60)

    output_images = []

    for page_num in pages_to_process:
        if page_num not in page_images:
            print(f"  [SKIP] Trang {page_num}: ảnh không tìm thấy")
            continue

        labels = page_labels[page_num]

        # Count confidence
        n_high = sum(1 for l in labels if l.get("confidence") == "high")
        n_med = sum(1 for l in labels if l.get("confidence") == "medium")
        n_low = sum(1 for l in labels if l.get("confidence") == "low")

        if args.ambiguous_only and n_med + n_low == 0:
            print(f"  Trang {page_num:4d}: skip (tất cả high)")
            continue

        annotated = visualize_page(
            page_images[page_num],
            labels,
            font_path,
            ambiguous_only=args.ambiguous_only,
            font_size=args.font_size,
        )

        out_path = output_dir / f"page_{page_num:04d}_labels.png"
        cv2.imwrite(str(out_path), annotated)
        output_images.append(str(out_path))

        suffix = ""
        if args.ambiguous_only:
            suffix = f" (showing {n_med + n_low} ambiguous)"
        print(f"  Trang {page_num:4d}: H={n_high} M={n_med} L={n_low}{suffix} -> {out_path.name}")

    # Export PDF if requested
    if args.pdf and output_images:
        _export_pdf(output_images, args.pdf)
        print(f"\n  PDF: {args.pdf} ({len(output_images)} pages)")

    print(f"\nDone! Output: {output_dir}/")


def _export_pdf(image_paths: list[str], pdf_path: str):
    """Gộp nhiều ảnh thành 1 file PDF."""
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)

    if images:
        images[0].save(
            pdf_path, save_all=True, append_images=images[1:],
            resolution=150,
        )


if __name__ == "__main__":
    main()
