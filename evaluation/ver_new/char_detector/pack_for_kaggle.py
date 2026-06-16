"""Pack the character-detection training set into a Kaggle-uploadable folder.

detect_manifest.json references absolute local page paths at full resolution. For
Kaggle we (1) downscale each page to --max px (chars stay ~40-60px, plenty for
detection; keeps the dataset small), (2) scale the boxes by the same factor, and
(3) rewrite the manifest with RELATIVE image paths. The trainer + count_constrained
are copied in so the Kaggle notebook is one `!python train_centernet.py` away.

Output: char_detector/kaggle_det_pkg/
  images/<book>_<page>.png        downscaled pages
  detect_manifest.json            relative paths + scaled boxes
  train_centernet.py, count_constrained.py

Run:
  .venv/bin/python evaluation/ver_new/char_detector/bootstrap_boxes.py   # if not done
  .venv/bin/python evaluation/ver_new/char_detector/pack_for_kaggle.py   # --max 1280
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "detect_manifest.json"))
    ap.add_argument("--out", default=str(HERE / "kaggle_det_pkg"))
    ap.add_argument("--max", type=int, default=1280, help="max page dimension after downscale")
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding="utf-8"))
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
    new_man = []
    skipped = 0
    for i, it in enumerate(man):
        im = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        if im is None:
            skipped += 1; continue
        H, W = im.shape[:2]
        s = min(1.0, args.max / max(H, W))
        if s < 1.0:
            im = cv2.resize(im, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        fn = f"{it['book']}_{it['page']}.png"
        cv2.imwrite(str(out / "images" / fn), im)
        boxes = [[round(x1 * s, 1), round(y1 * s, 1), round(x2 * s, 1), round(y2 * s, 1)]
                 for x1, y1, x2, y2 in it["boxes"]]
        new_man.append({"image": f"images/{fn}", "book": it["book"], "page": it["page"],
                        "boxes": boxes, "labels": it.get("labels", []), "n_boxes": len(boxes)})
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(man)}", flush=True)

    json.dump(new_man, open(out / "detect_manifest.json", "w", encoding="utf-8"), ensure_ascii=False)
    for f in ("train_centernet.py", "count_constrained.py"):
        shutil.copy(HERE / f, out / f)

    tot_boxes = sum(m["n_boxes"] for m in new_man)
    size_mb = sum(p.stat().st_size for p in (out / "images").glob("*.png")) / 1e6
    print(f"\npacked -> {out}")
    print(f"  pages {len(new_man)} (skipped {skipped}) | boxes {tot_boxes} | images {size_mb:.0f} MB")
    print(f"  upload {out} as a Kaggle Dataset, then in a GPU notebook:")
    print(f"    !python train_centernet.py --manifest detect_manifest.json --img 768 --epochs 40 --batch 8")
    print(f"  (see char_detector/KAGGLE.md)")


if __name__ == "__main__":
    main()
