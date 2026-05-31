"""Gọi kinhhannom OCR cho 1 ảnh đơn lẻ, render overlay + in thống kê.

Usage:
    python3 evaluation/run_single_image.py <image_path> [--out <dir>]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ocr.ocr_api import upload_image, recognize, boxes_to_columns
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_adhoc")
    args = ap.parse_args()

    img_path = Path(args.image).resolve()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    print(f"[1/3] Upload {img_path.name} ...")
    file_name = upload_image(str(img_path))
    if not file_name:
        sys.exit("Upload failed.")
    print(f"      file_name = {file_name}")

    print(f"[2/3] OCR ...")
    boxes = recognize(file_name)
    if boxes is None:
        sys.exit("OCR failed.")
    print(f"      raw boxes = {len(boxes)}")

    cols = boxes_to_columns(boxes)
    print(f"      columns   = {len(cols)} (R->L)")

    # Save raw OCR + columns
    raw_path = out_dir / f"{stem}_ocr_raw.json"
    raw_path.write_text(
        json.dumps({"image": str(img_path), "boxes": boxes, "columns": cols},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # In thống kê cột
    print("\n=== Cột (right -> left) ===")
    for i, col in enumerate(cols):
        chars = "".join(c["char"] for c in col)
        x_left = col[0]["bbox"][0] if col else "-"
        print(f"  col{i+1:>2}  n={len(col):>2}  x_left={x_left:>5}  | {chars}")

    total_chars = sum(len(c) for c in cols)
    suspicious = [(i + 1, len(c)) for i, c in enumerate(cols) if len(c) <= 2]
    print(f"\n  Tổng chữ: {total_chars}")
    print(f"  Cột nghi 'ma' (n<=2): {len(suspicious)} -> {suspicious}")

    # GAP detection
    gaps = []
    for ci, col in enumerate(cols):
        if len(col) < 2:
            continue
        heights = [b["bbox"][3] - b["bbox"][1] for b in col if b.get("bbox")]
        if not heights:
            continue
        med_h = sorted(heights)[len(heights) // 2]
        for a, b in zip(col, col[1:]):
            d = b["y_center"] - a["y_center"]
            if med_h > 0 and d > 2.0 * med_h:
                est = int(round(d / med_h)) - 1
                gaps.append((ci + 1, a["bbox"], b["bbox"], d, med_h, est))

    if gaps:
        print(f"\n  GAP nghi bỏ chữ: {len(gaps)} điểm, ước tính ~{sum(g[5] for g in gaps)} chữ bị bỏ")
        for g in gaps:
            print(f"    col{g[0]} y={g[1][3]}→{g[2][1]} gap={g[3]:.0f}px med_h={g[4]:.0f}px est={g[5]}")
    else:
        print("\n  Không phát hiện GAP dọc bất thường.")

    # Render overlay
    print(f"\n[3/3] Render overlay ...")
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    for ci, col in enumerate(cols):
        color = (255, 0, 0, 200) if len(col) <= 2 else (0, 180, 0, 200)
        for c in col:
            b = c.get("bbox")
            if b:
                draw.rectangle(b, outline=color, width=3)
        if col:
            b0 = col[0]["bbox"]
            draw.text((b0[0], max(0, b0[1] - 26)),
                      f"c{ci+1}×{len(col)}",
                      fill=(200, 0, 0) if len(col) <= 2 else (0, 130, 0),
                      font=font)

    for ci, _a, _b, _d, _h, est in gaps:
        a, b = _a, _b
        x0 = min(a[0], b[0]) - 8
        x1 = max(a[2], b[2]) + 8
        draw.rectangle([x0, a[3], x1, b[1]],
                       outline=(255, 140, 0, 255), width=4)
        draw.text((x1 + 4, (a[3] + b[1]) // 2 - 12),
                  f"GAP ~{est}", fill=(255, 100, 0), font=font)

    overlay = out_dir / f"{stem}_overlay.png"
    im.save(overlay)
    print(f"      saved: {overlay}")
    print(f"      raw:   {raw_path}")


if __name__ == "__main__":
    main()
