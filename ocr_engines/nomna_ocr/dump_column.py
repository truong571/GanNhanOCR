"""Dump one column's crop + preprocessed input + predictions for
visual inspection. Useful when probe agreement looks low — the image
tells you whether NomNaOCR is seeing sensible input.

Usage:
    python -m ocr_engines.nomna_ocr.dump_column --page page_0016 --col 1
    python -m ocr_engines.nomna_ocr.dump_column --page page_0020 --col 5 --book CacThanhTruyen2

Outputs to `debug_nomna/`:
    <page>_col<N>_raw.png       # raw column crop from the page
    <page>_col<N>_model.png     # what CRNN actually sees (432x48)
    <page>_col<N>_info.txt      # kimhannom vs nomna_ocr sequences
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ocr_engines.nomna_ocr.engine import NomNaOCREngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", default="CacThanhTruyen2")
    parser.add_argument("--data-dir", default="prepared")
    parser.add_argument("--page", required=True, help="e.g. page_0016")
    parser.add_argument("--col", type=int, required=True)
    parser.add_argument("--out-dir", default="debug_nomna")
    args = parser.parse_args()

    book_dir = Path(args.data_dir) / args.book
    det_path = book_dir / "detected" / f"{args.page}_detection.json"
    if not det_path.exists():
        print(f"[dump] not found: {det_path}", file=sys.stderr)
        return 1

    with open(det_path, "r", encoding="utf-8") as f:
        detection = json.load(f)

    col = next(
        (c for c in detection["columns"] if c["column"] == args.col),
        None,
    )
    if col is None:
        print(f"[dump] column {args.col} not in {args.page}", file=sys.stderr)
        return 1

    page_path = book_dir / "pages" / f"{args.page}.png"
    img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[dump] cannot read {page_path}", file=sys.stderr)
        return 1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    eng = NomNaOCREngine()
    eng._ensure_loaded()

    strip = eng._crop_column(img_rgb, col["chars"])
    if strip is None:
        print(f"[dump] failed to crop column", file=sys.stderr)
        return 1

    # What the model actually sees
    import tensorflow as tf
    processed = eng._crnn.process_image(strip).numpy()
    processed_uint = (processed * 255).astype(np.uint8)

    decoded = eng._predict(strip)
    kim_seq = "".join(ch.get("ocr_char") or "?" for ch in col["chars"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.page}_col{args.col:02d}"

    # Raw crop: RGB → BGR for cv2.imwrite
    cv2.imwrite(str(out_dir / f"{stem}_raw.png"),
                cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_model.png"),
                cv2.cvtColor(processed_uint, cv2.COLOR_RGB2BGR))

    info_path = out_dir / f"{stem}_info.txt"
    info_path.write_text(
        f"page:      {args.page}\n"
        f"column:    {args.col}\n"
        f"num_chars: {len(col['chars'])}\n"
        f"crop_size: {strip.shape[1]}×{strip.shape[0]}  (W×H)\n"
        f"model_in:  {processed_uint.shape[1]}×{processed_uint.shape[0]}\n"
        f"\n"
        f"kimhannom ({len(kim_seq)} chars):\n  {kim_seq}\n"
        f"\nnomna_ocr ({len(decoded)} chars):\n  {decoded or '<empty>'}\n",
        encoding="utf-8",
    )

    print(f"[dump] saved: {out_dir}/{stem}_*")
    print(f"  raw:       {strip.shape[1]}x{strip.shape[0]}")
    print(f"  model in:  {processed_uint.shape[1]}x{processed_uint.shape[0]}")
    print(f"  kimhannom: {kim_seq}")
    print(f"  nomna_ocr: {decoded or '<empty>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
