"""Test PP-OCRv5_server_det on a single page.

Usage:
    python scripts/test_ppocrv5_det.py <image_path> [out_dir]
"""
import sys
import os
import time
import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "PP-OCRv5")
MODEL_DIR = os.path.abspath(MODEL_DIR)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else \
        "prepared/SachThanhTruyen11/pages/page_0010.png"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "scratch_ppocrv5_out"
    os.makedirs(out_dir, exist_ok=True)

    from paddleocr import TextDetection

    print(f"[load] model_dir={MODEL_DIR}")
    model = TextDetection(model_name="PP-OCRv5_server_det", model_dir=MODEL_DIR)

    print(f"[predict] {img_path}")
    t0 = time.time()
    results = model.predict(img_path, batch_size=1)
    dt = time.time() - t0

    img = cv2.imread(img_path)
    n = 0
    for res in results:
        polys = res["dt_polys"]
        n = len(polys)
        for poly in polys:
            pts = np.array(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        res.save_to_json(out_dir)

    base = os.path.splitext(os.path.basename(img_path))[0]
    vis = os.path.join(out_dir, f"{base}_det_vis.png")
    cv2.imwrite(vis, img)
    print(f"[done] {n} boxes in {dt:.2f}s  ->  {vis}")


if __name__ == "__main__":
    main()
