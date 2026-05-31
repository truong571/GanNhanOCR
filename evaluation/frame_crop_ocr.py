"""Detect khung viền in của trang Hán-Nôm, crop tới khung, rồi OCR.

Hai chiến lược:
  --strategy frame  : crop NGUYÊN khung 1 lần, gọi OCR (1 API call/trang)
  --strategy cols   : crop khung -> chia N cột đều, OCR từng cột (N API call/trang,
                                                                  chính xác cao nhất)

Cách detect khung:
  1) Grayscale + Otsu inverse
  2) Morphology mở rộng đường ngang/dọc dày
  3) Tìm contour lớn nhất với 4 đỉnh -> bounding rect
  Fallback: bbox của vùng có pixel đen (loại 5% mép)

Usage:
    python3 evaluation/frame_crop_ocr.py <image> [--strategy frame|cols] [--ncols 9]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.ocr.ocr_api import upload_image, recognize, boxes_to_columns


def detect_frame(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Trả về (x0, y0, x1, y1) của khung viền in chính."""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Đường viền khung thường là nét dọc/ngang DÀI nhất
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, W // 20), 1)),
    )
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, H // 20))),
    )
    frame_mask = cv2.bitwise_or(horiz, vert)
    frame_mask = cv2.dilate(frame_mask, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Chọn contour có diện tích bbox lớn nhất nhưng < 95% ảnh
        best = None
        best_area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            a = w * h
            if a > 0.20 * H * W and a < 0.97 * H * W and a > best_area:
                best = (x, y, x + w, y + h)
                best_area = a
        if best:
            return best

    # Fallback: bbox của vùng đen, bỏ mép 3%
    ys, xs = np.where(bw > 0)
    if len(xs) == 0:
        return (0, 0, W, H)
    return (
        max(0, int(np.percentile(xs, 1))),
        max(0, int(np.percentile(ys, 1))),
        min(W, int(np.percentile(xs, 99))),
        min(H, int(np.percentile(ys, 99))),
    )


def crop_columns(crop_bgr: np.ndarray, n: int, pad: int = 8) -> list[tuple[int, np.ndarray]]:
    """Chia ảnh khung thành n cột đều (R->L), trả [(col_index_1..n, ảnh_BGR)]."""
    H, W = crop_bgr.shape[:2]
    w_col = W / n
    out = []
    for k in range(n):  # k=0 là cột trái nhất của ảnh
        x0 = max(0, int(round(k * w_col)) - pad)
        x1 = min(W, int(round((k + 1) * w_col)) + pad)
        sub = crop_bgr[:, x0:x1]
        # Cột R->L: k=0 là cột trái -> index = n-k
        out.append((n - k, sub))
    out.sort(key=lambda t: t[0])  # về thứ tự 1..n
    return out


def ocr_image_file(path: Path) -> list[dict]:
    fname = upload_image(str(path))
    if not fname:
        return []
    boxes = recognize(fname) or []
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--strategy", choices=["frame", "cols"], default="frame")
    ap.add_argument("--ncols", type=int, default=9)
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_frame")
    args = ap.parse_args()

    img_path = Path(args.image).resolve()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = detect_frame(bgr)
    print(f"[frame] image={W}x{H}  frame=({x0},{y0})-({x1},{y1})  size={x1-x0}x{y1-y0}")

    # Visualize frame
    vis = bgr.copy()
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 4)
    cv2.imwrite(str(out_dir / f"{stem}_frame.png"), vis)

    crop = bgr[y0:y1, x0:x1].copy()
    crop_path = out_dir / f"{stem}_crop.png"
    cv2.imwrite(str(crop_path), crop)

    if args.strategy == "frame":
        print(f"[OCR] strategy=frame, 1 API call ...")
        boxes = ocr_image_file(crop_path)
        cols = boxes_to_columns(boxes)
        print(f"  raw boxes={len(boxes)}  cols={len(cols)}")
        for i, c in enumerate(cols):
            chs = "".join(b["char"] for b in c)
            print(f"    col{i+1:>2} n={len(c):>2} | {chs}")
        result = {"strategy": "frame", "n_cols": len(cols),
                  "columns": [[b["char"] for b in c] for c in cols]}

    else:  # cols
        print(f"[OCR] strategy=cols, {args.ncols} API calls ...")
        strips = crop_columns(crop, args.ncols, pad=10)
        columns_text = {}
        for idx, sub in strips:
            tmp = out_dir / f"{stem}_col{idx}.png"
            cv2.imwrite(str(tmp), sub)
            boxes = ocr_image_file(tmp)
            sub_cols = boxes_to_columns(boxes)
            chars = []
            for c in sub_cols:
                chars.extend(b["char"] for b in c)
            columns_text[idx] = chars
            print(f"  col{idx:>2}: n={len(chars):>2} | {''.join(chars)}")
        result = {"strategy": "cols", "n_cols": args.ncols,
                  "columns": columns_text}

    (out_dir / f"{stem}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Overlay tổng hợp: frame xanh + nhãn cột (chỉ với strategy=frame)
    pil = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([x0, y0, x1, y1], outline=(0, 180, 0, 220), width=5)
    if args.strategy == "frame":
        w_col = (x1 - x0) / max(1, len(cols))
        for i, c in enumerate(cols):
            cx = x0 + int(w_col * (len(cols) - i - 0.5))  # R->L
            draw.text((cx - 20, y0 - 50), f"#{i+1}", fill=(0, 120, 0), font=font)
    pil.save(out_dir / f"{stem}_overlay.png")
    print(f"\n[OK] {out_dir / f'{stem}_overlay.png'}")


if __name__ == "__main__":
    main()
