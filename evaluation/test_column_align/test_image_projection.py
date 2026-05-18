"""Test: image-projection column detection (Final Plan Step 2 PRIMARY).

Final Plan spec:
  Primary  : image projection → 9 cols, right→left.
  Fallback : Kimhannom filter `len > 3` → if 9 cols, use it.
  Else     : flag nom_column_suspect.

This test measures the projection-PRIMARY approach standalone on 3 books
to verify it produces 9 cols reliably, and reports the fallback chain
coverage.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.image.column_detector import detect_columns
from core.image.image_processing import load_and_binarize, detect_text_box


def kimhannom_filter_9(ocr_columns, min_len=4):
    """Return number of cols after filter len >= min_len."""
    return sum(1 for c in ocr_columns if c and len(c) >= min_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+",
                    default=["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    repo = Path(args.repo)
    grand = {}

    for book in args.books:
        book_dir = repo / "prepared" / book
        pages = sorted((book_dir / "detected").glob("page_*_ocr_cache.json"))
        proj_9 = filter_9 = both_9 = neither = 0
        proj_only = filter_only = 0
        n = 0
        for p in pages:
            page = p.stem.replace("_ocr_cache", "")
            img_path = book_dir / "pages_denoised" / f"{page}.png"
            if not img_path.exists():
                img_path = book_dir / "pages" / f"{page}.png"
            if not img_path.exists():
                continue
            n += 1
            _, binary = load_and_binarize(str(img_path))
            try:
                tb = detect_text_box(binary)
                cols = detect_columns(binary, tb, n_expected=9)
                n_proj = len(cols)
            except Exception:
                n_proj = 0
            ocr_data = json.load(open(p))
            n_filter = kimhannom_filter_9(ocr_data.get("columns", []), 4)

            p_ok = (n_proj == 9)
            f_ok = (n_filter == 9)
            if p_ok and f_ok:
                both_9 += 1
            elif p_ok and not f_ok:
                proj_only += 1
            elif f_ok and not p_ok:
                filter_only += 1
            else:
                neither += 1
            if p_ok: proj_9 += 1
            if f_ok: filter_9 += 1

        grand[book] = dict(n=n, proj_9=proj_9, filter_9=filter_9,
                           both=both_9, proj_only=proj_only,
                           filter_only=filter_only, neither=neither)
        print(f"{book}: n={n} proj_9={proj_9} filter_9={filter_9} "
              f"both={both_9} proj_only={proj_only} "
              f"filter_only={filter_only} neither={neither}")

    md = ["# Image-projection PRIMARY vs Kimhannom-filter FALLBACK\n",
          "Coverage of `detect_columns(binary, text_box, n_expected=9) == 9` "
          "and `Kimhannom filter len≥4 → 9 cols`.\n"]
    md.append("| Book | Pages | Proj=9 | Filter=9 | Both | Proj-only "
              "| Filter-only | Neither |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    tot = dict(n=0, proj_9=0, filter_9=0, both=0,
               proj_only=0, filter_only=0, neither=0)
    for book, r in grand.items():
        md.append(f"| {book} | {r['n']} | {r['proj_9']} | {r['filter_9']} "
                  f"| {r['both']} | {r['proj_only']} | {r['filter_only']} "
                  f"| {r['neither']} |")
        for k in tot:
            tot[k] += r[k]
    md.append(f"| **TOTAL** | **{tot['n']}** | **{tot['proj_9']}** "
              f"({tot['proj_9']*100/max(1,tot['n']):.1f}%) "
              f"| **{tot['filter_9']}** "
              f"({tot['filter_9']*100/max(1,tot['n']):.1f}%) "
              f"| **{tot['both']}** | **{tot['proj_only']}** "
              f"| **{tot['filter_only']}** | **{tot['neither']}** |")
    md.append(f"\nFallback chain coverage (Proj → Filter):")
    chain = tot["proj_9"] + tot["filter_only"]
    md.append(f"- Proj primary: {tot['proj_9']}/{tot['n']} "
              f"({tot['proj_9']*100/max(1,tot['n']):.1f}%)")
    md.append(f"- Plus filter fallback: +{tot['filter_only']} → "
              f"**{chain}/{tot['n']} ({chain*100/max(1,tot['n']):.1f}%)**")
    md.append(f"- Neither (must flag suspect): {tot['neither']}")

    out = Path(__file__).parent / "out" / "RESULTS_image_projection.md"
    out.write_text("\n".join(md) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
