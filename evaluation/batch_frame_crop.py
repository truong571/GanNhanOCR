"""Batch detect + crop khung viền cho toàn bộ trang 1 sách.

Sinh 2 file/trang trong out_dir:
  page_XXXX_frame.png  : ảnh gốc có vẽ khung xanh (để review có cắt mất chữ không)
  page_XXXX_crop.png   : ảnh đã crop khung (sẵn sàng feed kinhhannom)

Sinh thêm summary CSV liệt kê kích thước khung / tỷ lệ so với ảnh gốc.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.detect_frame_v2 import detect_frame_hybrid as detect_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="SachThanhTruyen2")
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_frame_batch")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pad", type=int, default=12,
                    help="Pad mặc định cho cả 4 cạnh (bị override bởi --pad-x/--pad-y nếu có)")
    ap.add_argument("--pad-x", type=int, default=None,
                    help="Pad riêng trái/phải (mặc định lấy --pad)")
    ap.add_argument("--pad-y", type=int, default=None,
                    help="Pad riêng trên/dưới (mặc định lấy --pad)")
    args = ap.parse_args()
    pad_x = args.pad_x if args.pad_x is not None else args.pad
    pad_y = args.pad_y if args.pad_y is not None else args.pad

    pages_dir = ROOT / "prepared" / args.book / "pages"
    out_dir = ROOT / args.out / args.book
    (out_dir / "frame").mkdir(parents=True, exist_ok=True)
    (out_dir / "crop").mkdir(parents=True, exist_ok=True)

    pages = sorted(pages_dir.glob("*.png"))
    if args.limit:
        pages = pages[: args.limit]

    rows = []
    for i, p in enumerate(pages, 1):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = detect_frame(bgr)
        # Nở khung: pad_x cho L/R, pad_y cho T/B, kẹp trong ảnh
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(W, x1 + pad_x)
        y1 = min(H, y1 + pad_y)
        fw, fh = x1 - x0, y1 - y0
        area_ratio = (fw * fh) / (W * H)

        # frame.png : vẽ khung xanh lên ảnh gốc
        vis = bgr.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 5)
        cv2.imwrite(str(out_dir / "frame" / p.name), vis)

        # crop.png : crop chỉ phần khung
        crop = bgr[y0:y1, x0:x1].copy()
        cv2.imwrite(str(out_dir / "crop" / p.name), crop)

        # cờ bất thường để dễ review
        flag = ""
        if area_ratio < 0.35:
            flag = "TOO_SMALL"
        elif area_ratio > 0.92:
            flag = "TOO_BIG"
        elif fw < 0.5 * W:
            flag = "NARROW"
        elif fh < 0.5 * H:
            flag = "SHORT"

        rows.append({
            "page": p.stem, "W": W, "H": H,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "frame_w": fw, "frame_h": fh,
            "area_ratio": round(area_ratio, 3),
            "flag": flag,
        })
        if i % 20 == 0 or i == len(pages):
            print(f"  {i}/{len(pages)} pages processed")

    # CSV summary
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_flag = sum(1 for r in rows if r["flag"])
    print(f"\n[OK] {n} trang. Khung trung bình {sum(r['frame_w'] for r in rows)/n:.0f} x "
          f"{sum(r['frame_h'] for r in rows)/n:.0f}  "
          f"(area ratio avg {sum(r['area_ratio'] for r in rows)/n:.2%})")
    print(f"     Trang có cờ nghi vấn (xem ưu tiên): {n_flag}")
    if n_flag:
        print("     →", [r["page"] for r in rows if r["flag"]][:15])
    print(f"\n  Frame review:  {out_dir / 'frame'}")
    print(f"  Crop input  :  {out_dir / 'crop'}")
    print(f"  Summary CSV :  {csv_path}")


if __name__ == "__main__":
    main()
