#!/usr/bin/env python3
"""
visualize_labels.py - Draw bbox + nom_char on original Nom page images.

Colors:
  - Black = matched (correct)
  - Red   = unmatched (incorrect/unconfirmed)

Usage:
  python tools/visualize_labels.py Data/prepared/CacThanhTruyen2 --page 12
  python tools/visualize_labels.py Data/prepared/CacThanhTruyen2
  python tools/visualize_labels.py Data/prepared/CacThanhTruyen2 --pdf output.pdf
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

def _find_font(prepared_dir: Path) -> str:
    """Find NomNaTong font."""
    candidates = [
        prepared_dir / "../../font_diffusion/fonts/NomNaTong-Regular.ttf",
        Path("font_diffusion/fonts/NomNaTong-Regular.ttf"),
        Path(__file__).parent.parent / "font_diffusion/fonts/NomNaTong-Regular.ttf",
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
    fill=(0, 0, 0),
    bg_fill=(255, 255, 255, 200),
) -> np.ndarray:
    """Draw Unicode text (Nom) on OpenCV image via PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x, y = position
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    pad = 2
    draw.rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        fill=bg_fill,
    )
    draw.text((x - bbox[0], y - bbox[1]), text, fill=fill, font=font)

    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Colors: matched=black, unmatched=red
# ---------------------------------------------------------------------------

COLOR_MATCHED_BGR = (0, 0, 0)        # Black (BGR)
COLOR_UNMATCHED_BGR = (0, 0, 255)    # Red (BGR)
COLOR_MATCHED_RGB = (0, 0, 0)        # Black (RGB)
COLOR_UNMATCHED_RGB = (255, 0, 0)    # Red (RGB)
BG_MATCHED = (255, 255, 255, 180)
BG_UNMATCHED = (255, 200, 200, 220)


# ---------------------------------------------------------------------------
# Core visualization
# ---------------------------------------------------------------------------

def visualize_page(
    image_path: str,
    labels: list[dict],
    font_path: str,
    font_size: int = 28,
) -> np.ndarray:
    """Draw bbox + nom_char on original page image.

    Matched labels -> black bbox + black text
    Unmatched labels -> red bbox + red text
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    if font_path and Path(font_path).exists():
        pil_font = ImageFont.truetype(font_path, font_size)
        pil_font_small = ImageFont.truetype(font_path, max(14, font_size // 2))
    else:
        pil_font = ImageFont.load_default()
        pil_font_small = pil_font

    for lab in labels:
        matched = lab.get("matched", "False")
        is_matched = matched in (True, "True", "true", "1")

        bbox_str = lab.get("bbox", "")
        if not bbox_str:
            continue

        # Parse bbox (may be "[x1, y1, x2, y2]" or "x1,y1,x2,y2")
        bbox_str = str(bbox_str).strip("[]")
        coords = [int(float(x.strip())) for x in bbox_str.split(",")]
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords

        nom_char = lab.get("nom_char", "?")
        reading = lab.get("syllable", "") or lab.get("reading", "")

        if is_matched:
            color_bgr = COLOR_MATCHED_BGR
            color_rgb = COLOR_MATCHED_RGB
            bg_rgba = BG_MATCHED
            thickness = 1
        else:
            color_bgr = COLOR_UNMATCHED_BGR
            color_rgb = COLOR_UNMATCHED_RGB
            bg_rgba = BG_UNMATCHED
            thickness = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

        text_x = max(0, x1 - font_size - 8)
        text_y = y1 + 2
        if nom_char:
            img = _draw_text_pil(
                img, nom_char, (text_x, text_y),
                pil_font, fill=color_rgb, bg_fill=bg_rgba,
            )

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
        description="Visualize labels on Nom page images (black=matched, red=unmatched)",
    )
    parser.add_argument("prepared_dir", type=Path, help="Prepared data directory")
    parser.add_argument("--page", type=int, default=None, help="Single page number")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--pdf", type=str, default=None, help="Export all pages to PDF")

    args = parser.parse_args()
    prepared_dir = args.prepared_dir
    labels_csv = prepared_dir / "labeled" / "labels.csv"

    if not labels_csv.exists():
        print(f"[ERROR] Not found: {labels_csv}")
        print("  Run step3_label.py first")
        sys.exit(1)

    with open(labels_csv, "r", encoding="utf-8-sig") as f:
        all_labels = list(csv.DictReader(f))

    page_labels: dict[str, list[dict]] = {}
    for lab in all_labels:
        p = lab.get("page", "")
        page_labels.setdefault(p, []).append(lab)

    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERROR] Not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    page_images = {}
    for pi in manifest["pages"]:
        page_name = f"page_{pi['book_page']:04d}"
        page_images[page_name] = str(prepared_dir / pi["image_file"])

    font_path = _find_font(prepared_dir)
    if not font_path:
        print("[WARNING] NomNaTong font not found, using default")

    output_dir = args.output_dir or prepared_dir / "labeled" / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)

    pages_to_process = sorted(page_labels.keys())
    if args.page is not None:
        target = f"page_{args.page:04d}"
        pages_to_process = [p for p in pages_to_process if p == target]

    print(f"Visualize Labels: {prepared_dir.name}")
    print(f"  Labels: {len(all_labels)} entries")
    print(f"  Pages: {len(pages_to_process)}")
    print(f"  Output: {output_dir}/")
    print("-" * 60)

    output_images = []
    for page_name in pages_to_process:
        if page_name not in page_images:
            print(f"  [SKIP] {page_name}: image not found")
            continue

        labels = page_labels[page_name]
        n_matched = sum(1 for l in labels if l.get("matched") in (True, "True", "true", "1"))
        n_unmatched = len(labels) - n_matched

        annotated = visualize_page(
            page_images[page_name], labels, font_path, font_size=args.font_size,
        )

        out_path = output_dir / f"{page_name}_labels.png"
        cv2.imwrite(str(out_path), annotated)
        output_images.append(str(out_path))

        print(f"  {page_name}: {n_matched} matched (black), {n_unmatched} unmatched (red)")

    if args.pdf and output_images:
        images = [Image.open(p).convert("RGB") for p in output_images]
        if images:
            images[0].save(args.pdf, save_all=True, append_images=images[1:], resolution=150)
            print(f"\n  PDF: {args.pdf} ({len(output_images)} pages)")

    print(f"\nDone! Output: {output_dir}/")


if __name__ == "__main__":
    main()
