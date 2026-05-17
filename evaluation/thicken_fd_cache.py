"""Post-process fd_cache: thicken strokes via morphological dilate.

Why: FontDiffusion-generated images have ~44% less ink than the source font
(measured: 7.6% vs 13.5% ink ratio on 96x96). Strokes survive but minor radicals
get dropped. A 1-px dilate restores most of the lost ink and improves DINOv2
similarity to real handwritten crops.

Output goes to a sibling directory so you can A/B test before committing:

    prepared/_universal_fd_cache/           (original, untouched)
    prepared/_universal_fd_cache_dilated/   (this script's output)

Switch the ranker to the dilated cache by either:
  (a) editing config/pipeline.yaml `fd_cache_universal:` to point at it, or
  (b) renaming the directories (back up first).

Usage:
    .venv/bin/python evaluation/thicken_fd_cache.py
    .venv/bin/python evaluation/thicken_fd_cache.py --kernel 3 --iters 1
    .venv/bin/python evaluation/thicken_fd_cache.py --limit 200       # sample run for QA
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "prepared" / "_universal_fd_cache"
DST = REPO / "prepared" / "_universal_fd_cache_dilated"


def measure_ink(img: np.ndarray) -> float:
    """Ratio of dark pixels (ink) to total."""
    _, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return float(bw.sum()) / (img.shape[0] * img.shape[1])


def thicken(img: np.ndarray, kernel_size: int, iters: int) -> np.ndarray:
    """Dilate dark strokes (== erode the white background)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    # Strokes are dark on light bg → erode the image (min filter) to grow dark.
    return cv2.erode(img, kernel, iterations=iters)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", type=int, default=3,
                   help="Structuring element size (odd). 3 = ~1px growth.")
    p.add_argument("--iters", type=int, default=1,
                   help="Number of dilate iterations.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N files (for quick QA).")
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--dst", default=str(DST))
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"Source dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.iterdir())
    if args.limit:
        files = files[: args.limit]

    print(f"Source: {src}  ({len(files):,} files)")
    print(f"Output: {dst}")
    print(f"Kernel: {args.kernel}x{args.kernel} ellipse, iters={args.iters}")

    ink_before, ink_after = [], []
    t0 = time.time()
    written = 0
    skipped = 0
    for f in files:
        if not f.name.endswith(".png"):
            continue
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        ink_before.append(measure_ink(img))
        out = thicken(img, args.kernel, args.iters)
        ink_after.append(measure_ink(out))

        cv2.imwrite(str(dst / f.name), out)
        written += 1

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s   ({written:,} written, {skipped} skipped)")
    if ink_before:
        b = float(np.mean(ink_before)) * 100
        a = float(np.mean(ink_after)) * 100
        print(f"Ink ratio: {b:.1f}% -> {a:.1f}%  (Δ +{a-b:.1f}pp)")
        # Goal: match FONT-render ~13.5%. Report if we overshot.
        if a > 16:
            print(f"  ! Overshoot vs font-render baseline 13.5% — strokes may be TOO thick.")
            print(f"  ! Re-run with --kernel 2 to soften, or revert and keep original cache.")
        elif a < 10:
            print(f"  ! Still thinner than font-render 13.5% — consider --iters 2.")
        else:
            print(f"  Within target band (10-16% vs font baseline 13.5%).")
    print()
    print("Next steps:")
    print(f"  1. Inspect samples:  open {dst}/U+4E00.png (compare with {src}/U+4E00.png)")
    print(f"  2. To activate, edit config/pipeline.yaml:")
    try:
        rel = dst.relative_to(REPO)
        print(f"       fd_cache_universal: {rel}")
    except ValueError:
        print(f"       fd_cache_universal: {dst}")
    print(f"  3. Re-run step 3 to use the dilated cache for tier-3 visual matching.")


if __name__ == "__main__":
    main()
