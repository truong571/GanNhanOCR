"""Roadmap #5 — bootstrap a character-DETECTION training set for free from the
already-confirmed labels (no new annotation).

Each confirmed GOLD/SILVER pair in dataset_out/labels.csv carries a `bbox` in
ORIGINAL full-page coordinates (the frame-offset was already corrected upstream in
align_production._detect, so these are NOT the ~1.7-column-shifted OCR boxes — they
are page-space). Grouped per page they are exactly the (box, Unicode-label) pairs a
detector needs.

THE COMPLETENESS TRAP: a detector trained with MISSING boxes learns to suppress
real characters (an unboxed glyph becomes a false negative). A column that still
has any REVIEW row is only PARTIALLY confirmed -> its boxes are incomplete. So by
default we emit boxes only from COMPLETE columns (every pair in the column is
GOLD/SILVER); `--all-columns` keeps everything (more boxes, noisier labels).

Output: char_detector/detect_manifest.json
  [{ "image": "<abs page png>", "book","page",
     "boxes": [[x1,y1,x2,y2], ...], "labels": ["U+XXXX", ...],
     "n_boxes": k }]
A CenterNet/HRCenterNet trainer (see README) consumes this directly: centre =
box centre, size = (w,h); the class head is optional (detection only needs
"character vs background"), the labels are kept for analysis/recognition reuse.

Run:
  .venv/bin/python evaluation/ver_new/char_detector/bootstrap_boxes.py
  .venv/bin/python evaluation/ver_new/char_detector/bootstrap_boxes.py --all-columns
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE.parent / "dataset_out"))
    ap.add_argument("--out", default=str(HERE / "detect_manifest.json"))
    ap.add_argument("--complete-only", action="store_true",
                    help="keep ONLY columns with no REVIEW row (cleanest, far fewer boxes)")
    args = ap.parse_args()
    # GOLD+SILVER+SYLLABLE are all well-positioned boxes from matched/anchored
    # columns (SYLLABLE has a reliable position even if the char is unsure); REVIEW
    # boxes come from diverged columns (the wrong-position case) -> excluded.
    DETECT_TIERS = {"GOLD", "SILVER", "SYLLABLE"}

    cfg = load_config(args.config)
    name_map = {b["name"][12:]: b["name"] for b in cfg["books"]}   # short -> full book name
    data_root = REPO / cfg["paths"]["data_dir"]

    rows = list(csv.DictReader(open(Path(args.dataset) / "labels.csv", encoding="utf-8")))

    # per-column tier set -> a column is COMPLETE iff it has no REVIEW row
    col_tiers = defaultdict(set)
    for r in rows:
        col_tiers[(r["book"], r["page"], r["column"])].add(r["tier"])
    def complete(r):
        return "REVIEW" not in col_tiers[(r["book"], r["page"], r["column"])]

    by_page = defaultdict(list)
    kept = dropped = 0
    for r in rows:
        if r["tier"] not in DETECT_TIERS:
            continue
        if not r["bbox"] or r["bbox"] == "null":
            continue
        if args.complete_only and not complete(r):
            dropped += 1
            continue
        try:
            bb = [int(v) for v in json.loads(r["bbox"])]
        except Exception:
            continue
        if bb[2] - bb[0] < 4 or bb[3] - bb[1] < 4:
            continue
        by_page[(r["book"], r["page"])].append((bb, r["unicode"] or ""))
        kept += 1

    manifest = []
    missing_pages = 0
    for (book, page), items in sorted(by_page.items()):
        full = name_map.get(book, book)
        png = data_root / full / "pages" / f"{page}.png"
        if not png.exists():
            missing_pages += 1
            continue
        manifest.append({
            "image": str(png), "book": book, "page": page,
            "boxes": [it[0] for it in items],
            "labels": [it[1] for it in items],
            "n_boxes": len(items),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)

    nb = [m["n_boxes"] for m in manifest]
    nb.sort()
    print(f"detection manifest -> {args.out}")
    print(f"  pages: {len(manifest)}  (missing page PNGs skipped: {missing_pages})")
    print(f"  boxes kept: {kept}   dropped (partial columns): {dropped}"
          + ("" if args.complete_only else "   [all columns; --complete-only for the clean subset]"))
    if nb:
        print(f"  boxes/page: min {nb[0]}  median {nb[len(nb)//2]}  max {nb[-1]}")
    print("  -> train a CenterNet/HRCenterNet on these (README); at inference, on a")
    print("     diverged column, reconcile its proposals with N via count_constrained.")


if __name__ == "__main__":
    main()
