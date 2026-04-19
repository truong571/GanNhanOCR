"""Probe NomNaOCR on real columns to verify handwritten performance.

Because NomNaOCR operates at column level, we sample N columns (not
single crops), run CRNN, and compare the decoded sequence against
Kimhannom's per-char predictions in the same column.

Usage:
    python -m ocr_engines.nomna_ocr.probe                   # default: 5 columns
    python -m ocr_engines.nomna_ocr.probe --book CacThanhTruyen4 --n 8

Agreement is computed per character position (sequence[i] vs
kimhannom_char[i]). NomNaOCR decoded sequences of different length are
truncated/padded so positional comparison is still defined.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from ocr_engines.nomna_ocr.engine import NomNaOCREngine


def sample_columns(book_dir: Path, n: int, seed: int = 42) -> list[dict]:
    """Return up to N columns from detection.json files."""
    random.seed(seed)
    detection_files = sorted((book_dir / "detected").glob("page_*_detection.json"))
    pool: list[dict] = []
    for det in detection_files:
        with open(det, "r", encoding="utf-8") as f:
            data = json.load(f)
        page_name = data.get("book_page") or det.stem.replace("_detection", "")
        for col in data.get("columns", []):
            if col.get("chars"):
                pool.append({"page": page_name, "column": col})
    random.shuffle(pool)
    return pool[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe NomNaOCR on real columns")
    parser.add_argument("--book", default="CacThanhTruyen2")
    parser.add_argument("--data-dir", default="prepared")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    book_dir = Path(args.data_dir) / args.book
    if not book_dir.exists():
        print(f"[probe] Book dir not found: {book_dir}", file=sys.stderr)
        return 1

    samples = sample_columns(book_dir, args.n, args.seed)
    if not samples:
        print("[probe] No columns available. Run Step 2 first.", file=sys.stderr)
        return 1

    print(f"[probe] Book:   {args.book}")
    print(f"[probe] Sample: {len(samples)} columns")
    print("[probe] Loading NomNaOCR CRNN (first run may be slow)...")

    engine = NomNaOCREngine()
    t0 = time.time()
    engine._ensure_loaded()
    print(f"[probe] Loaded in {time.time() - t0:.1f}s")
    print()

    import cv2

    total_chars = 0
    agree_chars = 0
    t_start = time.time()

    for i, s in enumerate(samples, 1):
        page_name = s["page"]
        col = s["column"]
        chars = col["chars"]
        kim_seq = "".join(ch.get("ocr_char") or "?" for ch in chars)

        # Load page image and crop column
        page_path = book_dir / "pages" / f"{page_name}.png"
        img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] {page_name} image missing")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        strip = engine._crop_column(img, chars)
        if strip is None:
            continue
        decoded = engine._predict(strip)

        n = len(chars)
        nom_seq = (decoded + " " * n)[:n]

        matches = sum(
            1 for a, b in zip(kim_seq, nom_seq)
            if a != "?" and b != " " and a == b
        )
        total_chars += n
        agree_chars += matches

        print(f"#{i}  {page_name}  col{col['column']:02d}  ({n} chars)")
        print(f"    kimhannom: {kim_seq}")
        print(f"    nomna_ocr: {nom_seq.strip() or '∅'}  "
              f"(decoded {len(decoded)} chars)")
        print(f"    match:     {matches}/{n}")
        print()

    dt = time.time() - t_start
    if total_chars == 0:
        print("[probe] No characters compared.")
        return 1

    pct = 100.0 * agree_chars / total_chars
    avg = dt / max(1, len(samples))
    print("-" * 60)
    print(f"Agreement: {agree_chars}/{total_chars} ({pct:.0f}%)   "
          f"avg: {avg:.1f} s/column   total: {dt:.1f}s")

    if pct >= 40:
        verdict = "USE — NomNaOCR is producing useful handwritten decodings."
    elif pct >= 20:
        verdict = ("INSPECT — Partial agreement. Examine a few columns "
                   "visually before deciding.")
    else:
        verdict = ("REJECT — Very low agreement. Check image preprocessing "
                   "or try a different variant.")
    print(f"Verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
