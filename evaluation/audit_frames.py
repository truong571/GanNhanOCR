"""Kiểm toàn bộ trang đã crop: tìm ảnh có chữ bị cắt mất khỏi khung.

Cách: với mỗi trang, so sánh tổng mực BÊN TRONG khung vs TỔNG mực toàn ảnh.
Nếu phần mực ngoài khung > MISS_THRESHOLD * tổng → flag (frame thiếu chữ).
Cũng phân biệt mực ngoài là 'page number' (nhỏ, gần mép) vs chữ thật.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.detect_frame_v2 import detect_frame_hybrid


def audit_page(img_path: Path, miss_th: float = 0.04) -> dict:
    bgr = cv2.imread(str(img_path))
    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = detect_frame_hybrid(bgr)
    # Pad consistent with batch script
    px, py = 25, 5
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(W, x1 + px); y1 = min(H, y1 + py)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Loại viền giấy mép
    m = max(int(0.02 * min(H, W)), 8)
    bw[:m, :] = 0; bw[-m:, :] = 0; bw[:, :m] = 0; bw[:, -m:] = 0

    total_ink = int(bw.sum() / 255)
    if total_ink == 0:
        return {"page": img_path.stem, "status": "EMPTY"}

    inside = bw[y0:y1, x0:x1]
    inside_ink = int(inside.sum() / 255)
    outside_ink = total_ink - inside_ink

    # Đo mực ở các vùng ngoài khung (4 dải)
    top_strip = bw[:y0, :].sum() / 255 if y0 > 0 else 0
    bot_strip = bw[y1:, :].sum() / 255 if y1 < H else 0
    left_strip = bw[y0:y1, :x0].sum() / 255 if x0 > 0 else 0
    right_strip = bw[y0:y1, x1:].sum() / 255 if x1 < W else 0

    miss_ratio = outside_ink / total_ink

    # Phân loại ngoài
    side_max = max(top_strip, bot_strip, left_strip, right_strip)
    if side_max == 0:
        side = "-"
    elif side_max == top_strip:
        side = "TOP"
    elif side_max == bot_strip:
        side = "BOT"
    elif side_max == left_strip:
        side = "LEFT"
    else:
        side = "RIGHT"

    # Phán xét: bỏ qua nếu mực dưới (BOT) và rất ít (chỉ là số trang)
    is_just_pagenum = (side == "BOT" and bot_strip < 0.015 * total_ink)
    is_just_marker = (side == "TOP" and top_strip < 0.015 * total_ink)
    significant_miss = (miss_ratio > miss_th) and not (is_just_pagenum or is_just_marker)

    return {
        "page": img_path.stem,
        "W": W, "H": H,
        "bbox": (x0, y0, x1, y1),
        "frame_w": x1 - x0, "frame_h": y1 - y0,
        "total_ink": total_ink,
        "inside_ink": inside_ink,
        "outside_ink": outside_ink,
        "miss_ratio": round(miss_ratio, 4),
        "side_max": side,
        "top": int(top_strip), "bot": int(bot_strip),
        "left": int(left_strip), "right": int(right_strip),
        "flagged": significant_miss,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--miss-th", type=float, default=0.04,
                    help="Tỷ lệ mực ngoài/tổng trên ngưỡng này thì flag (default 4%%)")
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_audit")
    args = ap.parse_args()

    pages_dir = ROOT / "prepared" / args.book / "pages"
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(pages_dir.glob("*.png"))
    print(f"Auditing {len(pages)} pages, miss_th={args.miss_th}\n")

    rows = []
    for i, p in enumerate(pages, 1):
        r = audit_page(p, args.miss_th)
        rows.append(r)
        if i % 30 == 0 or i == len(pages):
            print(f"  {i}/{len(pages)} done")

    flagged = [r for r in rows if r.get("flagged")]
    print(f"\n[RESULT] flagged {len(flagged)}/{len(rows)} pages "
          f"({100*len(flagged)/len(rows):.1f}%)")
    print("\nFlagged pages (mực bị cắt mất):")
    for r in sorted(flagged, key=lambda x: -x["miss_ratio"]):
        print(f"  {r['page']}: miss={r['miss_ratio']:.2%}  side={r['side_max']}  "
              f"T={r['top']} B={r['bot']} L={r['left']} R={r['right']}")

    # CSV all
    fields = ["page", "miss_ratio", "side_max", "flagged",
              "top", "bot", "left", "right",
              "frame_w", "frame_h", "W", "H", "total_ink", "outside_ink"]
    with (out_dir / f"{args.book}_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV: {out_dir / f'{args.book}_audit.csv'}")


if __name__ == "__main__":
    main()
