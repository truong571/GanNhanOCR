"""Roadmap #1 — explain the REVIEW pile instead of dumping it silently.

The thesis-critical question for pain-point A (segmentation): *how much of REVIEW
is actually a segmentation problem?* REVIEW lumps together three very different
failures, and only ONE of them is fixable by a better character segmenter:

  diverged_column        the Nôm-OCR MIScounted the column (touching/missed glyphs)
                         -> the alignment could not anchor the pair -> this is the
                         SEGMENTATION-ADDRESSABLE slice (a count-constrained detector
                         can re-segment it; roadmap #5/A1).
  alignment_gap (ins)    a QN syllable with NO Nôm character box at all — the Nôm
                         glyph is missing/merged so the banded-DP left it unmatched.
                         These never become rows in labels.csv (no bbox), so they are
                         INVISIBLE in the manifest; we recover them by re-running the
                         alignment and counting `op=='ins'`. PARTIALLY segmentation-
                         addressable (a detector may surface a glyph the OCR missed).
  below_visual_threshold the column counted fine, the pair is anchored, but neither
                         the dictionary nor S3 could confirm a character. This is a
                         LABELING-COVERAGE / S3 problem, NOT segmentation — a better
                         segmenter changes nothing here.
  unconfirmed_no_s3      same as above but S3 was OFF for that pair.

OUTPUT (evaluation/ver_new/results/review_breakdown.json + a printed table):
  - REVIEW rows by reason, read straight from the live dataset_out/labels.csv
    (so the numbers match the released dataset exactly), and
  - with --with-gaps: a fresh alignment pass (NO S3 needed; gaps & divergence are
    decided by the banded DP alone) that adds the invisible alignment-gap count and
    per-column divergence stats.

The headline it produces — "segmentation-addressable ceiling = diverged rows +
gaps, vs. the S3/coverage remainder" — both justifies (or de-prioritises) the A1
detector AND is a stand-alone table for the thesis.

Run:
  .venv/bin/python evaluation/ver_new/review_breakdown.py                 # fast, from labels.csv
  .venv/bin/python evaluation/ver_new/review_breakdown.py --with-gaps     # + re-align for gaps (~3-5 min)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent

# Reason -> coarse class used for the segmentation-vs-coverage headline.
SEG_ADDRESSABLE = {"diverged_column"}                       # rows a re-segmenter can rescue
COVERAGE_BOUND = {"below_visual_threshold", "unconfirmed_no_s3"}  # S3/dictionary problem, not seg


def from_labels(labels_csv: Path) -> dict:
    """Exact REVIEW breakdown of the RELEASED manifest (rows that carry a bbox)."""
    tiers = Counter()
    review_reason = Counter()
    diverged_cols = set()        # (book,page,column) of any diverged REVIEW row
    review_cols = set()
    rows = 0
    for r in csv.DictReader(open(labels_csv, encoding="utf-8")):
        rows += 1
        tiers[r["tier"]] += 1
        if r["tier"] == "REVIEW":
            review_reason[r["rule"]] += 1
            key = (r["book"], r["page"], r["column"])
            review_cols.add(key)
            if r["rule"] == "diverged_column":
                diverged_cols.add(key)
    return {
        "total_rows": rows,
        "tiers": dict(tiers),
        "review_total": sum(review_reason.values()),
        "review_by_reason": dict(review_reason.most_common()),
        "review_columns": len(review_cols),
        "diverged_columns_with_rows": len(diverged_cols),
    }


def realign_gaps(limit: int = 0) -> dict:
    """Fresh alignment pass to count the INVISIBLE failures (no S3 required).

    `align_page(..., mode='new')` returns `n_review_gap` (= number of `ins` ops =
    QN syllables left with no Nôm box) and per-pair `matched` flags. Gaps and column
    divergence are produced by the banded DP + count comparison ALONE, so this is
    faithful to the released dataset without needing the encoder checkpoint.
    """
    from pipeline.step0_setup import load_config
    from core.text.dictionary import load_qn_to_nom, load_similarity_dict
    from evaluation.ver_new.align_production import align_page

    cfg = load_config(str(REPO / "config" / "pipeline.yaml"))
    paths = cfg["paths"]
    qn_to_nom = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    qn_dict_set = set(qn_to_nom.keys())
    similar = load_similarity_dict(str(REPO / paths["similar_dict"]))
    data_root = REPO / paths["data_dir"]

    n_gap_total = 0
    pages = 0
    emitted_pairs = 0
    diverged_pairs = 0          # emitted pairs whose column count did NOT match
    diverged_columns = set()
    total_columns = set()
    for b in cfg["books"]:
        book = b["name"]
        data_dir = data_root / book
        trans = sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
        trans = [t for t in trans if not t.endswith("_qn_ocr_cache.json")]
        if limit:
            trans = trans[:limit]
        for tf in trans:
            page = Path(tf).stem
            try:
                rec = align_page(page, data_dir, qn_dict_set, qn_to_nom, similar, "new")
            except Exception:
                continue
            if rec is None:
                continue
            pages += 1
            n_gap_total += rec.get("n_review_gap", 0)
            for p in rec["pairs"]:
                emitted_pairs += 1
                col_key = (book, page, p["column"])
                total_columns.add(col_key)
                if not p.get("matched", False):
                    diverged_pairs += 1
                    diverged_columns.add(col_key)
        print(f"  [realign] {book}: {pages} pages cumulative, gaps so far {n_gap_total}", flush=True)
    return {
        "pages": pages,
        "emitted_pairs": emitted_pairs,
        "alignment_gap_ins": n_gap_total,        # invisible: QN syllables with no Nôm box
        "diverged_pairs_emitted": diverged_pairs,
        "diverged_columns": len(diverged_columns),
        "total_columns": len(total_columns),
        "diverged_column_share": round(len(diverged_columns) / max(len(total_columns), 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--out", default=str(HERE / "results" / "review_breakdown.json"))
    ap.add_argument("--with-gaps", action="store_true",
                    help="re-run alignment to also count invisible alignment gaps (~3-5 min)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lab = Path(args.dataset) / "labels.csv"
    if not lab.exists():
        sys.exit(f"labels.csv not found at {lab} — run build_dataset.py first.")
    rep = {"source_manifest": str(lab), **from_labels(lab)}

    print("=" * 68)
    print(" REVIEW BREAKDOWN — what is actually in the discard pile")
    print("=" * 68)
    print(f" manifest rows: {rep['total_rows']}   tiers: {rep['tiers']}")
    print(f"\n REVIEW rows (carry a bbox): {rep['review_total']}")
    seg = cov = other = 0
    for reason, n in rep["review_by_reason"].items():
        bucket = ("SEGMENTATION-addressable" if reason in SEG_ADDRESSABLE
                  else "S3/coverage (seg won't help)" if reason in COVERAGE_BOUND
                  else "other")
        if reason in SEG_ADDRESSABLE: seg += n
        elif reason in COVERAGE_BOUND: cov += n
        else: other += n
        print(f"   {reason:24s} {n:7d}   [{bucket}]")
    rt = max(rep["review_total"], 1)
    print(f"\n   -> SEGMENTATION-addressable rows : {seg:7d}  ({seg/rt:.1%} of REVIEW)")
    print(f"   -> S3/coverage-bound rows        : {cov:7d}  ({cov/rt:.1%} of REVIEW)")
    rep["segmentation_addressable_rows"] = seg
    rep["coverage_bound_rows"] = cov

    if args.with_gaps:
        print("\n re-aligning to count the INVISIBLE alignment gaps (no S3) ...", flush=True)
        g = realign_gaps(args.limit)
        rep["realign"] = g
        seg_ceiling = seg + g["alignment_gap_ins"]
        rep["segmentation_addressable_ceiling"] = seg_ceiling
        print(f"\n alignment gaps (QN syllables with NO Nôm box, invisible in manifest): "
              f"{g['alignment_gap_ins']}")
        print(f" diverged columns: {g['diverged_columns']}/{g['total_columns']} "
              f"({g['diverged_column_share']:.1%} of columns)")
        print(f"\n >>> SEGMENTATION-ADDRESSABLE CEILING (diverged rows + gaps) = {seg_ceiling}")
        print(f"     i.e. the MOST a perfect count-constrained detector (#5/A1) could add,")
        print(f"     vs {cov} rows that are an S3/coverage problem a detector cannot touch.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n wrote {args.out}")


if __name__ == "__main__":
    main()
