"""Probe TrOCR-nom on real crops to verify handwritten suitability.

Samples N crops from a prepared book, runs TrOCRNomEngine, compares
with Kimhannom's ocr_char (as a sanity signal — NOT ground truth), and
prints a verdict.

Usage:
    python -m ocr_engines.trocr_nom.probe                            # default: 10 crops, CacThanhTruyen2
    python -m ocr_engines.trocr_nom.probe --book CacThanhTruyen4 --n 20
    python -m ocr_engines.trocr_nom.probe --model tt1225/finetuned-trocr-base-vietnamese-nom

Interpretation:
    - Agreement >= 60%    → TrOCR-nom is likely trained on handwritten Nom; use it.
    - Agreement 30-60%    → Inconclusive; inspect outputs manually.
    - Agreement < 30%     → Likely printed-only or wrong domain; switch to NomNaOCR.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from ocr_engines.trocr_nom.engine import TrOCRNomEngine


def sample_crops(book_dir: Path, n: int, seed: int = 42) -> list[tuple[Path, str]]:
    """Return up to N (crop_path, kimhannom_char) pairs from detection.json files."""
    random.seed(seed)
    detection_files = sorted((book_dir / "detected").glob("page_*_detection.json"))
    pool: list[tuple[Path, str]] = []

    for det in detection_files:
        with open(det, "r", encoding="utf-8") as f:
            data = json.load(f)
        for col in data.get("columns", []):
            for ch in col.get("chars", []):
                ocr_char = ch.get("ocr_char")
                crop_file = ch.get("crop_file")
                if not ocr_char or not crop_file:
                    continue
                full = book_dir / "detected" / crop_file
                if full.exists():
                    pool.append((full, ocr_char))

    if not pool:
        return []
    random.shuffle(pool)
    return pool[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe TrOCR-nom on real crops")
    parser.add_argument("--book", default="CacThanhTruyen2")
    parser.add_argument("--data-dir", default="prepared")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--model", default=None,
                        help="Override HF model id")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    book_dir = Path(args.data_dir) / args.book
    if not book_dir.exists():
        print(f"[probe] Book dir not found: {book_dir}", file=sys.stderr)
        return 1

    samples = sample_crops(book_dir, args.n, args.seed)
    if not samples:
        print("[probe] No crops available. Run Step 2 first.", file=sys.stderr)
        return 1

    print(f"[probe] Book:   {args.book}")
    print(f"[probe] Sample: {len(samples)} crops")
    print(f"[probe] Loading TrOCR-nom (first run downloads ~60-550 MB)...")

    engine = TrOCRNomEngine(model_id=args.model)
    t0 = time.time()
    engine._ensure_loaded()
    print(f"[probe] Loaded in {time.time() - t0:.1f}s on {engine.device}")
    print(f"[probe] Model:  {engine.model_id}")
    print()
    print(f"{'#':>3}  {'crop':40s}  {'kimhannom':>10s}  {'trocr_nom':>10s}  {'match':>6s}  {'conf':>5s}")
    print("-" * 90)

    agree = 0
    miss = 0
    t_start = time.time()

    for i, (crop_path, kim_char) in enumerate(samples, 1):
        res = engine.recognize_crop(str(crop_path))
        trocr_char = res.char or "∅"
        match = "YES" if res.char == kim_char else "no"
        if res.char is None:
            miss += 1
        elif res.char == kim_char:
            agree += 1
        rel = crop_path.relative_to(book_dir / "detected")
        print(f"{i:>3}  {str(rel):40s}  {kim_char:>10s}  {trocr_char:>10s}  "
              f"{match:>6s}  {res.confidence:>5.2f}")

    dt = time.time() - t_start
    avg_ms = 1000 * dt / len(samples)
    pct = 100.0 * agree / len(samples)

    print("-" * 90)
    print(f"Agreement: {agree}/{len(samples)} ({pct:.0f}%)   "
          f"misses: {miss}   avg: {avg_ms:.0f} ms/crop   total: {dt:.1f}s")

    print()
    if pct >= 60:
        verdict = "USE — TrOCR-nom looks trained on handwritten Nom."
    elif pct >= 30:
        verdict = ("INSPECT — Agreement is ambiguous. Open a few crops manually "
                   "and compare visually before deciding.")
    else:
        verdict = ("REJECT — Likely printed-only or wrong domain. "
                   "Wrap NomNaOCR (ds4v) instead.")
    print(f"Verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
