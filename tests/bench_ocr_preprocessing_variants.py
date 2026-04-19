"""Benchmark: OCR accuracy vs image-preprocessing variant.

Tests whether the HCMUS SinoNom OCR API works better with:
  A. raw page (current pipeline behaviour)
  B. page cropped to the inner text box (outer frame removed)
  C. variant B + vertical ruling lines inpainted out

Scoring
-------
For each page × variant, the OCR result is aligned against the page's
Quốc-Ngữ transcription (per column, Levenshtein). The primary metric is
`coverage` = (# OCR chars whose transcription is in qn_to_nom[aligned_syllable])
             / (# total QN syllables on the page).

Output
------
  tests/bench_cache/<variant>_<hash>.json   — cached OCR responses
  tests/bench_images/<variant>_<hash>.png   — exact image uploaded to API
  tests/bench_results/ocr_variants_<ts>.json — aggregated scores

Run
---
  python tests/bench_ocr_preprocessing_variants.py \
      --book CacThanhTruyen2 --pages 3

Each variant that is not already cached issues 1 upload + 1 OCR call per page.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.bench_utils import (  # noqa: E402
    REPO_ROOT,
    list_available_pages,
    load_page,
    ocr_variant,
    score_ocr_vs_transcription,
    fmt_score,
    variant_raw,
    variant_crop_frame,
    variant_crop_and_erase_columns,
)


VARIANTS = [
    ("raw", variant_raw),
    ("crop_frame", variant_crop_frame),
    ("crop_erase_cols", variant_crop_and_erase_columns),
]


def run(book: str, n_pages: int, verbose: bool = True) -> dict:
    pages = list_available_pages(book, limit=n_pages)
    if not pages:
        print(f"[ERR] No pages with transcription for book: {book}")
        return {}

    print(f"\n{'='*78}")
    print(f"OCR PREPROCESSING VARIANT BENCHMARK — book={book}, pages={len(pages)}")
    print(f"{'='*78}")

    per_page: list[dict] = []

    for page_stem in pages:
        print(f"\n— {page_stem} —")
        gray, qn_cols = load_page(book, page_stem)
        page_row = {"page": page_stem, "variants": {}}

        for label, fn in VARIANTS:
            t0 = time.time()
            preprocessed = fn(gray)
            prep_ms = (time.time() - t0) * 1000

            full_label = f"{book}_{page_stem}_{label}"
            t0 = time.time()
            res = ocr_variant(preprocessed, full_label, verbose=verbose)
            ocr_ms = (time.time() - t0) * 1000

            if res is None:
                print(f"  {label:<18} OCR failed")
                page_row["variants"][label] = {"error": True}
                continue

            score = score_ocr_vs_transcription(res.columns, qn_cols)
            score["prep_ms"] = round(prep_ms, 1)
            score["ocr_ms"] = round(ocr_ms, 1)
            score["shape"] = list(preprocessed.shape)
            page_row["variants"][label] = score
            print(f"  {label:<18} {fmt_score(score)}  prep={prep_ms:.0f}ms")

        per_page.append(page_row)

    # ---------------- aggregate ----------------
    print(f"\n{'='*78}")
    print("AGGREGATE (mean across pages)")
    print(f"{'='*78}")
    header = (f"  {'variant':<18} {'coverage':>9} {'hit_rate':>9} "
              f"{'aligned':>8} {'cols Δ':>7} {'chars Δ':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary: dict[str, dict] = {}
    for label, _ in VARIANTS:
        vs = [pr["variants"].get(label) for pr in per_page
              if pr["variants"].get(label) and not pr["variants"][label].get("error")]
        if not vs:
            continue
        n = len(vs)
        s = {
            "coverage": sum(v["coverage"] for v in vs) / n,
            "dict_hit_rate": sum(v["dict_hit_rate"] for v in vs) / n,
            "aligned": sum(v["aligned_pairs"] for v in vs) / n,
            "cols_delta": sum(v["ocr_cols"] - v["qn_cols"] for v in vs) / n,
            "chars_delta": sum(v["ocr_chars"] - v["qn_syls"] for v in vs) / n,
            "pages": n,
        }
        summary[label] = s
        print(f"  {label:<18} {s['coverage']:>9.3f} {s['dict_hit_rate']:>9.3f} "
              f"{s['aligned']:>8.1f} {s['cols_delta']:>+7.1f} {s['chars_delta']:>+8.1f}")

    if summary:
        winner = max(summary.items(), key=lambda kv: kv[1]["coverage"])
        print(f"\n  Winner by coverage: {winner[0]}  "
              f"(coverage={winner[1]['coverage']:.3f})")

    # ---------------- persist ----------------
    out_dir = REPO_ROOT / "tests" / "bench_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"ocr_variants_{book}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"book": book, "pages": per_page, "aggregate": summary},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nSaved: {out_file}")
    return {"book": book, "pages": per_page, "aggregate": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="CacThanhTruyen2",
                    help="Book name under prepared/")
    ap.add_argument("--pages", type=int, default=3,
                    help="Number of pages to test (default 3)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(args.book, args.pages, verbose=not args.quiet)


if __name__ == "__main__":
    main()
