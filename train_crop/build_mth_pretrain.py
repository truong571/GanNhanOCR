"""Đóng gói MTH/TKH (đã tải) thành pretrain-pack GỌN cho Kaggle.

Đọc nhãn label_char (parser tự nhận diện), downscale ảnh + scale box, ghi ra
<out>/mth_images/*.jpg + <out>/mth_manifest.json (đường dẫn tương đối). Nhờ vậy
4.7GB raw -> vài trăm MB, train đọc thẳng qua --manifest mth_manifest.json (nhanh,
khỏi parse lúc train).

Chạy:
  .venv/bin/python test/build_mth_pretrain.py            # -> test/kaggle_pkg/mth_*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from data_centernet import read_mth_items                # noqa: E402

# MTH raw mặc định: train_crop/MTH/... nếu có, không thì <repo>/MTH/... (kho dữ liệu)
_MTH_DEFAULT = HERE / "MTH" / "TKHMTH2200"
if not _MTH_DEFAULT.exists():
    _MTH_DEFAULT = REPO / "MTH" / "TKHMTH2200"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mth-root", default=str(_MTH_DEFAULT))
    ap.add_argument("--out", default=str(HERE / "kaggle_pkg"))
    ap.add_argument("--max", type=int, default=1280, help="cạnh dài tối đa sau downscale")
    ap.add_argument("--quality", type=int, default=88, help="chất lượng JPEG")
    a = ap.parse_args()

    items = read_mth_items(a.mth_root)
    if not items:
        raise SystemExit("Không đọc được MTH (xem log [MTH]).")
    out = Path(a.out); img_dir = out / "mth_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    man, n_box, skipped = [], 0, 0
    for i, it in enumerate(items):
        im = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        if im is None:
            skipped += 1; continue
        H, W = im.shape[:2]
        s = min(1.0, a.max / max(H, W))
        if s < 1.0:
            im = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        fn = f"mth_{i:05d}.jpg"
        cv2.imwrite(str(img_dir / fn), im, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        boxes = [[round(x1 * s, 1), round(y1 * s, 1), round(x2 * s, 1), round(y2 * s, 1)]
                 for x1, y1, x2, y2 in it["boxes"]]
        man.append({"image": f"mth_images/{fn}", "boxes": boxes, "n_boxes": len(boxes)})
        n_box += len(boxes)
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(items)}", flush=True)
    json.dump(man, open(out / "mth_manifest.json", "w", encoding="utf-8"), ensure_ascii=False)
    mb = sum(p.stat().st_size for p in img_dir.glob("*.jpg")) / 1e6
    print(f"[mth-pack] {len(man)} trang | {n_box} box | bỏ {skipped} | ảnh {mb:.0f} MB "
          f"-> {out}/mth_manifest.json")


if __name__ == "__main__":
    main()
