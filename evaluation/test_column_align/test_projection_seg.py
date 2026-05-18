"""Test: image-projection segmentation with expected_count from QN.

This is Final Plan Step 2B. For each cluster col, we have a body x-range
(from hybrid method). The Final Plan says: ignore Kimhannom char count;
segment the Nôm image inside that col x-range to exactly `expected_count`
glyphs (where expected_count = len(qn_lines[k])).

Test verifies:
  - segment_characters_in_column actually returns expected_count bboxes.
  - Each bbox has sane area (width × height > 100 px², ink_ratio in valid range).

Reports per-book + global stats:
  pages_seg_ok        : pages where all cols segmented to exact expected count
  cols_seg_ok         : cols where seg returned expected count
  cols_seg_lowqual    : cols where seg returned count but mean ink_ratio < 0.03
                        (likely empty/noise bboxes)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.image.char_segmenter import segment_characters_in_column
from core.image.image_processing import load_and_binarize

from parser_v3 import parse_v3
from parser_v2 import load_v1_transcription
from probe import load_qn_to_nom
from run_full import nom_cols_hybrid, get_qn_lines


def col_bbox_from_cluster(cluster: dict, img_h: int) -> tuple[int, int, int, int]:
    """Derive a column bbox from cluster chars.

    x: min/max of cluster char bboxes.
    y: span the FULL image height (segmenter will trim with projection).
    """
    chars = cluster["chars"]
    x_lo = min(c["bbox"][0] for c in chars)
    x_hi = max(c["bbox"][2] for c in chars)
    y_lo = min(c["bbox"][1] for c in chars)
    y_hi = max(c["bbox"][3] for c in chars)
    # Pad y slightly so descenders/ascenders aren't clipped.
    pad = 12
    return (x_lo, max(0, y_lo - pad), x_hi, min(img_h, y_hi + pad))


def ink_ratio(binary_crop: np.ndarray) -> float:
    if binary_crop.size == 0:
        return 0.0
    return float((binary_crop > 0).sum()) / binary_crop.size


def test_page(page_name: str, book_dir: Path, qn_lines: dict) -> dict:
    img_path = book_dir / "pages_denoised" / f"{page_name}.png"
    if not img_path.exists():
        img_path = book_dir / "pages" / f"{page_name}.png"
    if not img_path.exists():
        return {"page": page_name, "error": "no_image"}

    _, binary = load_and_binarize(str(img_path))
    H, W = binary.shape

    ocr_data = json.load(
        open(book_dir / "detected" / f"{page_name}_ocr_cache.json"))
    cols = nom_cols_hybrid(ocr_data.get("columns", []), min_len=4)

    qn_keys = sorted(qn_lines.keys())
    n_align = min(len(cols), len(qn_keys))

    col_stats = []
    page_seg_ok = (len(cols) == len(qn_keys))

    for i in range(n_align):
        expected = len(qn_lines[qn_keys[i]])
        bbox = col_bbox_from_cluster(cols[i], H)
        try:
            bboxes = segment_characters_in_column(
                binary, bbox, expected_count=expected,
            )
        except Exception as e:
            bboxes = []
            col_stats.append({
                "col": qn_keys[i], "expected": expected, "returned": 0,
                "ok": False, "error": str(e)[:80],
                "mean_ink": 0.0, "low_ink_cnt": 0,
            })
            page_seg_ok = False
            continue

        # Validate each bbox
        inks = []
        low_ink = 0
        widths = []
        heights = []
        for (x1, y1, x2, y2) in bboxes:
            crop = binary[y1:y2, x1:x2]
            ir = ink_ratio(crop)
            inks.append(ir)
            widths.append(x2 - x1)
            heights.append(y2 - y1)
            if ir < 0.03:
                low_ink += 1

        mean_ink = float(np.mean(inks)) if inks else 0.0
        mean_h = float(np.mean(heights)) if heights else 0.0
        mean_w = float(np.mean(widths)) if widths else 0.0
        returned_ok = (len(bboxes) == expected)
        # A column is "OK" if exact count AND no >25% of bboxes are low-ink.
        col_ok = returned_ok and (low_ink / max(1, len(bboxes)) <= 0.25)
        if not col_ok:
            page_seg_ok = False

        col_stats.append({
            "col": qn_keys[i], "expected": expected,
            "returned": len(bboxes), "ok": col_ok,
            "mean_ink": round(mean_ink, 4),
            "mean_w": round(mean_w, 1), "mean_h": round(mean_h, 1),
            "low_ink_cnt": low_ink,
        })

    return {
        "page": page_name,
        "page_seg_ok": page_seg_ok and (len(cols) == len(qn_keys)),
        "n_cols": len(cols),
        "n_qn": len(qn_keys),
        "n_cols_seg_ok": sum(1 for c in col_stats if c["ok"]),
        "col_stats": col_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+",
                    default=["SachThanhTruyen2", "SachThanhTruyen4", "SachThanhTruyen11"])
    ap.add_argument("--max_pages_per_book", type=int, default=None,
                    help="Cap for quick smoke test.")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    repo = Path(args.repo)
    qn_to_nom = load_qn_to_nom(str(repo / "Dict" / "QuocNgu_SinoNom_TongHop3.csv"))
    qn_dict = set(qn_to_nom.keys())

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    grand = {}
    for book in args.books:
        book_dir = repo / "prepared" / book
        aligned = sorted((book_dir / "aligned").glob("page_*_aligned.json"))
        if args.max_pages_per_book:
            aligned = aligned[:args.max_pages_per_book]

        results = []
        n_seg_ok = 0
        n_cols_total = 0
        n_cols_seg_ok = 0
        n_low_ink = 0
        n_low_ink_pages = 0

        for af in aligned:
            page = af.stem.replace("_aligned", "")
            qn_lines, _ = get_qn_lines(book_dir, page, qn_dict)
            if not qn_lines:
                continue
            r = test_page(page, book_dir, qn_lines)
            if "error" in r:
                continue
            results.append(r)
            if r["page_seg_ok"]:
                n_seg_ok += 1
            n_cols_total += len(r["col_stats"])
            n_cols_seg_ok += r["n_cols_seg_ok"]
            page_low = sum(1 for c in r["col_stats"] if c["low_ink_cnt"] > 0)
            n_low_ink += page_low
            if page_low > 0:
                n_low_ink_pages += 1

        grand[book] = {
            "n_pages": len(results),
            "pages_seg_ok": n_seg_ok,
            "cols_total": n_cols_total,
            "cols_seg_ok": n_cols_seg_ok,
            "pages_with_low_ink": n_low_ink_pages,
        }
        print(f"{book}: pages={len(results)} "
              f"page_seg_ok={n_seg_ok}/{len(results)} "
              f"cols_seg_ok={n_cols_seg_ok}/{n_cols_total} "
              f"pages_with_lowink={n_low_ink_pages}")

        json.dump(results, open(out_dir / f"projection_seg_{book}.json", "w"),
                  ensure_ascii=False, indent=2)

    # Write summary
    md = []
    md.append("# Image-projection segmentation test (Final Plan Step 2B)\n")
    md.append("For each col, run `segment_characters_in_column(binary, bbox, "
              "expected_count=len(qn_line))` and check:\n")
    md.append("- count returned == expected (segmenter guarantees this when given expected_count).")
    md.append("- ≤25% of bboxes have ink_ratio < 0.03 (rules out empty/noise bboxes).\n")
    md.append("## Per book\n")
    md.append("| Book | Pages | Page seg ok | Cols total | Cols seg ok | "
              "Pages w/ low-ink |")
    md.append("|---|---:|---:|---:|---:|---:|")
    tot = {"n_pages": 0, "pages_seg_ok": 0, "cols_total": 0,
           "cols_seg_ok": 0, "pages_with_low_ink": 0}
    for book, r in grand.items():
        md.append(f"| {book} | {r['n_pages']} | {r['pages_seg_ok']} "
                  f"| {r['cols_total']} | {r['cols_seg_ok']} "
                  f"| {r['pages_with_low_ink']} |")
        for k in tot:
            tot[k] += r[k]
    md.append(f"| **TOTAL** | **{tot['n_pages']}** "
              f"| **{tot['pages_seg_ok']}** "
              f"({tot['pages_seg_ok']*100/max(1,tot['n_pages']):.1f}%) "
              f"| **{tot['cols_total']}** | **{tot['cols_seg_ok']}** "
              f"({tot['cols_seg_ok']*100/max(1,tot['cols_total']):.1f}%) "
              f"| **{tot['pages_with_low_ink']}** |")

    (out_dir / "RESULTS_projection_seg.md").write_text("\n".join(md) + "\n")
    print(f"\n[done] wrote {out_dir/'RESULTS_projection_seg.md'}")


if __name__ == "__main__":
    main()
