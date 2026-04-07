#!/usr/bin/env python3
"""
prepare_data.py - Chuẩn bị dữ liệu training cho embedding model

Công việc:
  1. Tải New-SinoNom_Dataset từ Kaggle (nếu chưa có)
  2. Render gallery ảnh font từ NomNaTong cho tất cả ký tự trong từ điển
  3. Tạo manifest CSV: (ký tự, class_id, scan_image_path, font_image_path)

Usage:
  # Tải dataset + render gallery + tạo manifest:
  python embedding/prepare_data.py --download

  # Chỉ render gallery (đã tải dataset rồi):
  python embedding/prepare_data.py --render-only

  # Chỉ tạo manifest (đã có gallery rồi):
  python embedding/prepare_data.py --manifest-only
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
ROOT = Path(__file__).resolve().parent.parent
DICT_PATH = ROOT / "Alignment" / "Code" / "dict" / "QuocNgu_SinoNom_Merged.csv"
FONT_PATH = ROOT / "FontDiffusion" / "fonts" / "NomNaTong-Regular.ttf"
DATA_DIR = ROOT / "embedding" / "data"
GALLERY_DIR = DATA_DIR / "gallery"  # Ảnh font render
DATASET_DIR = DATA_DIR / "sinonom_dataset"  # New-SinoNom_Dataset
MANIFEST_PATH = DATA_DIR / "manifest.csv"


def download_dataset(output_dir: Path):
    """Tải New-SinoNom_Dataset từ Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)

    kaggle_dataset = "5c09041f61f1bd528a0281281a55ed4ddb6b4aa1c83bdb0c0e21a1553339ad32"
    print(f"Tải New-SinoNom_Dataset từ Kaggle...")
    print(f"Output: {output_dir}")
    print()
    print("Nếu chưa cài kaggle CLI:")
    print("  pip install kaggle")
    print("  Đặt kaggle.json vào ~/.kaggle/kaggle.json")
    print()

    cmd = [
        "kaggle", "datasets", "download",
        "-d", kaggle_dataset,
        "-p", str(output_dir),
        "--unzip",
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\nTải xong: {output_dir}")
    except FileNotFoundError:
        print("[ERROR] Không tìm thấy kaggle CLI. Cài đặt: pip install kaggle")
        print(f"Hoặc tải thủ công từ:")
        print(f"  https://kaggle.com/datasets/{kaggle_dataset}")
        print(f"Giải nén vào: {output_dir}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Tải thất bại: {e}")
        print(f"Tải thủ công từ:")
        print(f"  https://kaggle.com/datasets/{kaggle_dataset}")
        print(f"Giải nén vào: {output_dir}")
        sys.exit(1)


def load_dict_characters(dict_path: Path) -> set[str]:
    """Load tất cả ký tự Nôm unique từ từ điển."""
    chars = set()
    with open(dict_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                nom_char = row[1].strip()
                if nom_char:
                    chars.add(nom_char)
    print(f"Loaded {len(chars)} ký tự unique từ từ điển")
    return chars


def get_font_supported_chars(font_path: str) -> set[str]:
    """Lấy danh sách ký tự mà font hỗ trợ."""
    try:
        from fontTools.ttLib import TTFont
        chars = set()
        with TTFont(str(font_path), 0, ignoreDecompileErrors=True) as ttf:
            for table in ttf["cmap"].tables:
                for code in table.cmap.keys():
                    chars.add(chr(code))
        return chars
    except ImportError:
        print("[WARN] fontTools chưa cài, bỏ qua kiểm tra font support")
        return None


def render_gallery(
    chars: set[str],
    font_path: Path,
    output_dir: Path,
    img_size: int = 96,
    font_size: int = 72,
):
    """Render ảnh font cho tất cả ký tự Nôm.

    Output: output_dir/{hex_unicode}.png cho mỗi ký tự
    Ảnh: grayscale, nét đen trên nền trắng, img_size × img_size
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.truetype(str(font_path), font_size)
    supported = get_font_supported_chars(font_path)

    rendered = 0
    skipped = 0

    for char in sorted(chars):
        # Kiểm tra font hỗ trợ ký tự này
        if supported is not None and char not in supported:
            skipped += 1
            continue

        code = f"{ord(char):04X}"
        out_path = output_dir / f"{code}.png"

        if out_path.exists():
            rendered += 1
            continue

        # Render
        img = Image.new("L", (img_size, img_size), 255)  # Grayscale, nền trắng
        draw = ImageDraw.Draw(img)

        # Tính vị trí căn giữa
        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (img_size - text_w) / 2 - bbox[0]
        y = (img_size - text_h) / 2 - bbox[1]

        draw.text((x, y), char, font=font, fill=0)  # Nét đen

        # Kiểm tra ảnh có nội dung (không phải toàn trắng = font không có glyph)
        arr = np.array(img)
        if arr.min() > 200:  # Toàn trắng → font không render được
            skipped += 1
            continue

        img.save(str(out_path))
        rendered += 1

    print(f"Gallery: {rendered} rendered, {skipped} skipped (font không hỗ trợ)")
    return rendered


def find_scan_images(dataset_dir: Path) -> dict[str, list[Path]]:
    """Tìm ảnh scan trong New-SinoNom_Dataset (cấu trúc ImageFolder).

    Cấu trúc: dataset_dir/NewSinoNomData/NewSinoNomData/{class_name}/image.png
    class_name = mã Unicode hex (VD: "4E00")

    Returns: {unicode_hex: [path1, path2, ...]}
    """
    scan_map = {}

    # Tìm thư mục gốc dataset (có thể nested)
    candidates = [
        dataset_dir,
        dataset_dir / "NewSinoNomData",
        dataset_dir / "NewSinoNomData" / "NewSinoNomData",
    ]

    data_root = None
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            # Kiểm tra có subdirectory là hex code không
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if subdirs:
                try:
                    int(subdirs[0].name, 16)
                    data_root = c
                    break
                except ValueError:
                    continue

    if data_root is None:
        print(f"[WARN] Không tìm thấy cấu trúc ImageFolder trong {dataset_dir}")
        return scan_map

    print(f"Dataset root: {data_root}")

    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue

        try:
            int(class_dir.name, 16)  # Validate hex
        except ValueError:
            continue

        images = sorted(class_dir.glob("*.[pP][nN][gG]"))
        images += sorted(class_dir.glob("*.[jJ][pP][gG]"))
        images += sorted(class_dir.glob("*.[jJ][pP][eE][gG]"))

        if images:
            scan_map[class_dir.name.upper()] = images

    print(f"Scan images: {len(scan_map)} classes, "
          f"{sum(len(v) for v in scan_map.values())} total images")
    return scan_map


def create_manifest(
    gallery_dir: Path,
    scan_map: dict[str, list[Path]],
    output_path: Path,
):
    """Tạo manifest CSV cho training.

    Columns: char, unicode_hex, class_id, scan_path, font_path
    Mỗi scan image tạo 1 row, font_path là gallery image tương ứng.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build class_id mapping
    all_codes = sorted(set(scan_map.keys()))
    code_to_id = {code: idx for idx, code in enumerate(all_codes)}

    rows = []
    for code, scan_paths in scan_map.items():
        font_path = gallery_dir / f"{code}.png"
        if not font_path.exists():
            continue  # Không có ảnh font → bỏ qua

        char = chr(int(code, 16))
        class_id = code_to_id[code]

        for sp in scan_paths:
            rows.append({
                "char": char,
                "unicode_hex": code,
                "class_id": class_id,
                "scan_path": str(sp),
                "font_path": str(font_path),
            })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["char", "unicode_hex", "class_id", "scan_path", "font_path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    n_classes = len(set(r["class_id"] for r in rows))
    print(f"Manifest: {len(rows)} rows, {n_classes} classes → {output_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dữ liệu training cho embedding model",
    )
    parser.add_argument("--download", action="store_true",
                        help="Tải New-SinoNom_Dataset từ Kaggle")
    parser.add_argument("--render-only", action="store_true",
                        help="Chỉ render gallery (đã tải dataset)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Chỉ tạo manifest (đã có gallery)")
    parser.add_argument("--img-size", type=int, default=96,
                        help="Kích thước ảnh gallery (mặc định: 96)")
    parser.add_argument("--font-size", type=int, default=72,
                        help="Cỡ font render (mặc định: 72)")

    args = parser.parse_args()

    # Step 1: Download
    if args.download:
        download_dataset(DATASET_DIR)

    # Step 2: Render gallery
    if not args.manifest_only:
        print("\n--- Render gallery ---")
        chars = load_dict_characters(DICT_PATH)
        render_gallery(chars, FONT_PATH, GALLERY_DIR,
                       img_size=args.img_size, font_size=args.font_size)

    # Step 3: Create manifest
    print("\n--- Tạo manifest ---")
    scan_map = find_scan_images(DATASET_DIR)
    if scan_map:
        create_manifest(GALLERY_DIR, scan_map, MANIFEST_PATH)
    else:
        print("[WARN] Không có scan images. Chạy --download trước.")
        if not args.download:
            print("  python embedding/prepare_data.py --download")


if __name__ == "__main__":
    main()
