"""So sánh các thuật toán detect khung trang Hán-Nôm (công cụ DEV).

Thuật toán CHUẨN dùng cho pipeline nằm ở: core/image/frame_detector.py
(detect_frame_hybrid, refine_to_text, ...). File này CHỈ re-export chúng để
so sánh side-by-side + 2 baseline tham khảo (hough, layout).

Usage:
    python3 evaluation/detect_frame_v2.py <image> [--method hough|contour|textbbox|all]
    python3 evaluation/detect_frame_v2.py --book SachThanhTruyen2 --pages page_0024,...
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Thuật toán khung CHUẨN (nguồn duy nhất) — dùng chung với pipeline + QA.
from core.image.frame_detector import (  # noqa: E402,F401
    detect_frame_contour,
    detect_frame_textbbox,
    detect_frame_hybrid,
    refine_to_text,
    _split_runs,
    _trim_marker_band,
    _remove_scan_border,
)


# -------------------- Baseline tham khảo: Hough --------------------

def detect_frame_hough(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Tìm khung bằng 4 đường viền dài nhất (chỉ để so sánh)."""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, W // 8), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(50, H // 8)))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    h_lines = cv2.HoughLinesP(horiz, 1, np.pi / 180, threshold=120,
                              minLineLength=W // 3, maxLineGap=50)
    v_lines = cv2.HoughLinesP(vert, 1, np.pi / 180, threshold=120,
                              minLineLength=H // 3, maxLineGap=50)
    ys, xs = [], []
    if h_lines is not None:
        for x1, y1, x2, y2 in h_lines[:, 0]:
            if abs(y2 - y1) < 10:
                ys.append((y1 + y2) / 2)
    if v_lines is not None:
        for x1, y1, x2, y2 in v_lines[:, 0]:
            if abs(x2 - x1) < 10:
                xs.append((x1 + x2) / 2)
    margin_y, margin_x = 0.01 * H, 0.01 * W
    ys = [y for y in ys if margin_y < y < H - margin_y]
    xs = [x for x in xs if margin_x < x < W - margin_x]
    if len(ys) < 2 or len(xs) < 2:
        return detect_frame_contour(img_bgr)
    y0, y1 = int(min(ys)), int(max(ys))
    x0, x1 = int(min(xs)), int(max(xs))
    if (x1 - x0) < 0.3 * W or (y1 - y0) < 0.3 * H:
        return detect_frame_contour(img_bgr)
    return x0, y0, x1, y1


# -------------------- Baseline tham khảo: LayoutParser --------------------
_lp_model = None

def detect_frame_layout(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """LayoutParser HJDataset (chỉ chạy nếu đã cài deps; để so sánh)."""
    global _lp_model
    if _lp_model is None:
        import layoutparser as lp
        _lp_model = lp.Detectron2LayoutModel(
            "lp://HJDataset/faster_rcnn_R_50_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3])
    layout = _lp_model.detect(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    page_frames = [b for b in layout if b.type == "Page Frame"]
    if page_frames:
        b = max(page_frames, key=lambda b: b.block.area).block
        return int(b.x_1), int(b.y_1), int(b.x_2), int(b.y_2)
    regions = [b for b in layout if b.type in ("Text Region", "Row")]
    if not regions:
        return detect_frame_contour(img_bgr)
    return (int(min(b.block.x_1 for b in regions)), int(min(b.block.y_1 for b in regions)),
            int(max(b.block.x_2 for b in regions)), int(max(b.block.y_2 for b in regions)))


METHODS = {
    "contour": detect_frame_contour,
    "hough": detect_frame_hough,
    "textbbox": detect_frame_textbbox,
    "hybrid": detect_frame_hybrid,
    "layout": detect_frame_layout,
}


def run_on_image(path: Path, methods: list[str], out_dir: Path, pad: int = 30) -> dict:
    bgr = cv2.imread(str(path))
    H, W = bgr.shape[:2]
    rows = {}
    for m in methods:
        try:
            x0, y0, x1, y1 = METHODS[m](bgr)
        except Exception as e:
            rows[m] = {"error": f"{type(e).__name__}: {e}"}
            continue
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
        ratio = (x1 - x0) * (y1 - y0) / (W * H)
        rows[m] = {"bbox": (x0, y0, x1, y1), "w": x1 - x0, "h": y1 - y0,
                   "area_ratio": round(ratio, 3)}
    colors = {"contour": (0, 0, 255), "hough": (0, 200, 0),
              "textbbox": (255, 100, 0), "hybrid": (0, 200, 0), "layout": (200, 0, 200)}
    vis = bgr.copy()
    for m, r in rows.items():
        if "bbox" not in r:
            continue
        x0, y0, x1, y1 = r["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), colors.get(m, (0, 200, 0)), 5)
        cv2.putText(vis, m.upper(), (x0 + 10, y0 - 12 if y0 > 30 else y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors.get(m, (0, 200, 0)), 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_compare.png"), vis)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--book")
    ap.add_argument("--pages", help="comma-separated page stems")
    ap.add_argument("--method", default="all",
                    choices=["contour", "hough", "textbbox", "hybrid", "layout", "all"])
    ap.add_argument("--out", default="evaluation/_kinhhannom_debug/_frame_compare")
    ap.add_argument("--pad", type=int, default=30)
    args = ap.parse_args()
    out_dir = ROOT / args.out
    methods = ["contour", "hybrid", "textbbox"] if args.method == "all" else [args.method]
    if args.image:
        targets = [Path(args.image).resolve()]
    elif args.book and args.pages:
        targets = [ROOT / "prepared" / args.book / "pages" / f"{p}.png"
                   for p in args.pages.split(",")]
    else:
        ap.error("Provide image OR --book+--pages")
    print(f"Methods: {methods}\nTargets: {len(targets)}\nOut: {out_dir}\n")
    for p in targets:
        print(f"== {p.name} ==")
        rows = run_on_image(p, methods, out_dir, args.pad)
        for m, r in rows.items():
            if "error" in r:
                print(f"  {m:8s}: ERROR {r['error']}")
            else:
                print(f"  {m:8s}: bbox={r['bbox']}  {r['w']}x{r['h']}  ratio={r['area_ratio']}")
        print()


if __name__ == "__main__":
    main()
