"""Benchmark: OCR accuracy vs page-level denoise.

Built-in labels:
  - raw          original page (no denoise)
  - current      lib.image_processing.denoise_image (existing pipeline)

Optional external models — drop pre-generated PNGs at
    tests/bench_denoised/<label>/<book>/<page_stem>.png
and the script will auto-detect them.

Output
------
  tests/bench_cache/<label>_<hash>.json        — cached OCR per image
  tests/bench_results/denoise_<book>_<ts>.json — aggregated scores

Run
---
  python tests/bench_denoise_models.py \
      --book CacThanhTruyen2 --pages 10 --preprocess raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.image_processing import denoise_image as current_denoise  # noqa: E402
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


DENOISED_DIR = REPO_ROOT / "tests" / "bench_denoised"

PREPROCESS_FNS = {
    "raw": variant_raw,
    "crop_frame": variant_crop_frame,
    "crop_erase_cols": variant_crop_and_erase_columns,
}

# Built-in labels (don't need external model outputs)
BUILTIN_LABELS = ["raw", "current"]


def available_external_models() -> list[str]:
    """Sub-directories under tests/bench_denoised/ that contain PNGs."""
    if not DENOISED_DIR.exists():
        return []
    return sorted(
        p.name for p in DENOISED_DIR.iterdir()
        if p.is_dir() and any(p.rglob("*.png"))
    )


def load_denoised(
    model: str, book: str, page_stem: str, gray_raw: np.ndarray
) -> np.ndarray | None:
    """Return denoised page for the given model label, or None if missing."""
    if model == "raw":
        return gray_raw
    if model == "current":
        return current_denoise(gray_raw)

    # External model — look for a pre-generated PNG on disk
    candidates = [
        DENOISED_DIR / model / book / f"{page_stem}.png",
        DENOISED_DIR / model / f"{book}_{page_stem}.png",
        DENOISED_DIR / model / f"{page_stem}.png",
    ]
    for p in candidates:
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    return None


def run(
    book: str,
    n_pages: int,
    preprocess: str,
    extra_models: list[str] | None,
    verbose: bool = True,
) -> dict:
    pages = list_available_pages(book, limit=n_pages)
    if not pages:
        print(f"[ERR] No pages with transcription for book: {book}")
        return {}

    prep_fn = PREPROCESS_FNS[preprocess]

    models: list[str] = list(BUILTIN_LABELS)
    detected = available_external_models()
    models.extend(m for m in detected if m not in models)
    if extra_models:
        for m in extra_models:
            if m not in models:
                models.append(m)

    print(f"\n{'='*78}")
    print(f"DENOISE MODEL BENCHMARK")
    print(f"  book={book}  pages={len(pages)}  preprocess={preprocess}")
    print(f"  models: {', '.join(models)}")
    if detected:
        print(f"  detected external outputs under {DENOISED_DIR}: "
              f"{', '.join(detected)}")
    else:
        print(f"  (no external model outputs found in {DENOISED_DIR})")
    print(f"{'='*78}")

    per_page: list[dict] = []

    for page_stem in pages:
        print(f"\n— {page_stem} —")
        gray_raw, qn_cols = load_page(book, page_stem)
        page_row = {"page": page_stem, "models": {}}

        for model in models:
            denoised = load_denoised(model, book, page_stem, gray_raw)
            if denoised is None:
                print(f"  {model:<14} MISSING  (drop PNG into "
                      f"{DENOISED_DIR}/{model}/{book}/{page_stem}.png)")
                page_row["models"][model] = {"missing": True}
                continue

            t0 = time.time()
            processed = prep_fn(denoised)
            prep_ms = (time.time() - t0) * 1000

            label = f"{book}_{page_stem}_denoise-{model}_prep-{preprocess}"
            res = ocr_variant(processed, label, verbose=verbose)
            if res is None:
                print(f"  {model:<14} OCR failed")
                page_row["models"][model] = {"error": True}
                continue

            score = score_ocr_vs_transcription(res.columns, qn_cols)
            score["prep_ms"] = round(prep_ms, 1)
            score["shape"] = list(processed.shape)
            page_row["models"][model] = score
            print(f"  {model:<14} {fmt_score(score)}")

        per_page.append(page_row)

    # ---------------- aggregate ----------------
    print(f"\n{'='*78}")
    print(f"AGGREGATE (mean across pages) — preprocess={preprocess}")
    print(f"{'='*78}")
    header = (f"  {'model':<14} {'coverage':>9} {'hit_rate':>9} "
              f"{'aligned':>8} {'cols Δ':>7} {'chars Δ':>8} {'pages':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary: dict[str, dict] = {}
    for model in models:
        vs = [pr["models"].get(model) for pr in per_page
              if pr["models"].get(model)
              and not pr["models"][model].get("error")
              and not pr["models"][model].get("missing")]
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
        summary[model] = s
        print(f"  {model:<14} {s['coverage']:>9.3f} {s['dict_hit_rate']:>9.3f} "
              f"{s['aligned']:>8.1f} {s['cols_delta']:>+7.1f} "
              f"{s['chars_delta']:>+8.1f} {n:>6d}")

    if summary:
        winner = max(summary.items(), key=lambda kv: kv[1]["coverage"])
        print(f"\n  Winner by coverage: {winner[0]}  "
              f"(coverage={winner[1]['coverage']:.3f})")

    # ---------------- persist ----------------
    out_dir = REPO_ROOT / "tests" / "bench_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"denoise_{book}_{preprocess}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"book": book, "preprocess": preprocess,
             "pages": per_page, "aggregate": summary},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nSaved: {out_file}")
    return {"book": book, "preprocess": preprocess,
            "pages": per_page, "aggregate": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="CacThanhTruyen2")
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument(
        "--preprocess", choices=list(PREPROCESS_FNS.keys()), default="raw",
        help="Preprocess step applied AFTER denoise, before OCR. "
             "Set to the winner of bench_ocr_preprocessing_variants.py.",
    )
    ap.add_argument(
        "--models", nargs="*", default=None,
        help="Force extra model labels (must have PNGs under "
             "tests/bench_denoised/<label>/<book>/<page>.png).",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(args.book, args.pages, args.preprocess, args.models,
        verbose=not args.quiet)


if __name__ == "__main__":
    main()
