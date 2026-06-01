"""FULL production-path A/B evaluation.

Unlike run_eval.py (a quick index-vs-DP testbed on raw caches), this runs the
REAL labeling front end for every page — identical column detection
(detect_nom_columns_v3) and QN parsing (parse_v5) to pipeline/step2_align.py —
and compares the two pairing strategies end-to-end:

    OLD = current production pairing (count force-equalize + index emit)
    NEW = banded dictionary-anchored DP  (evaluation/ver_new/anchor_align.py)

It then applies the 3-signal consensus tiering (S3 absent -> dictionary floor)
and writes the resulting GOLD/SILVER/REVIEW dataset so you can inspect the
final labels.

Run (takes a few minutes — it loads & binarizes every page image):
  cd /Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR
  .venv/bin/python evaluation/ver_new/run_full_eval.py
Optional: limit pages per book for a quick smoke test:
  .venv/bin/python evaluation/ver_new/run_full_eval.py --limit 10
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

from pipeline.step0_setup import load_config            # noqa: E402
from core.text.dictionary import load_qn_to_nom, load_similarity_dict  # noqa: E402
from evaluation.ver_new.anchor_align import is_confirmed  # noqa: E402
from evaluation.ver_new.consensus import decide_label    # noqa: E402
from evaluation.ver_new.align_production import align_page  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def tier_of(p, qn_to_nom, similar, s3=None):
    """Tier a single pair. `p` carries matched/anchored flags."""
    dec = decide_label(p.get("ocr_char"), p["syllable"], p.get("matched", False),
                       qn_to_nom, similar, s3=s3, anchored=p.get("anchored", False))
    return dec.tier, dec.rule_id, dec.label


def maybe_s3(p, page_png, qn_to_nom, vs3, _unused=False):
    """Compute S3 exactly where it can change the tier = every anchored pair
    that is NOT already dictionary-confirmed (S1∩S2). That covers both the
    'syllable-in-dict but OCR disagrees' case (vision corrects) and the
    out-of-dict Ext-B case. Dict-confirmed (GOLD) pairs skip S3 -> S3 cost is
    only the ~non-confirmed fraction. Returns S3 or None."""
    if vs3 is None or not p.get("ocr_char"):
        return None
    if not (p.get("matched") or p.get("anchored")):
        return None
    cands = qn_to_nom.get((p["syllable"] or "").lower(), [])
    if p["ocr_char"] in cands:                  # already dict-confirmed -> GOLD, no S3
        return None
    return vs3.compute(page_png, p.get("bbox"), p["ocr_char"], cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="max pages per book (0=all)")
    ap.add_argument("--use-s3", action="store_true",
                    help="enable the visual S3 signal (DINOv2+FD) -> populates SILVER")
    ap.add_argument("--s3-gold-sanity", action="store_true",
                    help="also run S3 on dict-confirmed GOLD pairs as a sanity gate (slower)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    paths = config["paths"]
    qn_to_nom = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    qn_dict_set = set(qn_to_nom.keys())
    similar = load_similarity_dict(str(REPO / paths["similar_dict"]))
    data_root = REPO / paths["data_dir"]

    vs3 = None
    if args.use_s3:
        from evaluation.ver_new.visual_signal import VisualS3
        print("Loading S3 visual signal (DINOv2 + FontDiffusion cache)...", flush=True)
        vs3 = VisualS3(REPO, font_path=str(REPO / paths["font_path"]),
                       fd_dir=str(REPO / paths["fd_cache_universal"]),
                       cache_dir=str(OUT / "emb_cache"))
        print(f"  FD glyph index: {len(vs3.fd_index)} chars", flush=True)
    n_s3 = 0

    agg = {m: {"pairs": 0, "confirmed": 0, "tiers": Counter()} for m in ("old", "new")}
    split = {m: {"matched": [0, 0], "diverged": [0, 0]} for m in ("old", "new")}  # [pairs, conf]
    col_better = col_worse = col_tie = 0
    pages_done = pages_fail = 0
    pages_ok = 0
    review_gap_syllables = 0
    fixed_examples = []
    tier_rows = {"GOLD": [], "SILVER": [], "REVIEW": []}

    for b in config["books"]:
        book = b["name"]
        data_dir = data_root / book
        trans = sorted(glob.glob(str(data_dir / "transcriptions" / "page_*.json")))
        trans = [t for t in trans if not t.endswith("_qn_ocr_cache.json")]
        if args.limit:
            trans = trans[:args.limit]
        print(f"[{book}] {len(trans)} pages ...", flush=True)

        for ti, tf in enumerate(trans):
            page = Path(tf).stem
            try:
                rec_old = align_page(page, data_dir, qn_dict_set, qn_to_nom, similar, "old")
                rec_new = align_page(page, data_dir, qn_dict_set, qn_to_nom, similar, "new")
            except Exception as e:
                pages_fail += 1
                if pages_fail <= 5:
                    print(f"   [warn] {book}/{page}: {type(e).__name__}: {e}", flush=True)
                continue
            if rec_old is None or rec_new is None:
                pages_fail += 1
                continue
            pages_done += 1
            if rec_new["page_ok"]:
                pages_ok += 1
            review_gap_syllables += rec_new["n_review_gap"]

            page_png = str(data_dir / "pages" / f"{page}.png")
            # per-method aggregation + per-column confirmed counts for head-to-head
            col_conf = {"old": defaultdict(int), "new": defaultdict(int)}
            for m, rec in (("old", rec_old), ("new", rec_new)):
                for p in rec["pairs"]:
                    conf = p["confirmed"] if "confirmed" in p else \
                        is_confirmed(p.get("ocr_char"), p["syllable"], qn_to_nom, similar)
                    p["confirmed"] = conf
                    agg[m]["pairs"] += 1
                    agg[m]["confirmed"] += conf
                    bucket = "matched" if p.get("matched") else "diverged"
                    split[m][bucket][0] += 1
                    split[m][bucket][1] += conf
                    col_conf[m][p["column"]] += conf
                    s3 = None
                    if m == "new" and vs3 is not None:
                        s3 = maybe_s3(p, page_png, qn_to_nom, vs3, args.s3_gold_sanity)
                        if s3 is not None:
                            n_s3 += 1
                    tier, rule, label = tier_of(p, qn_to_nom, similar, s3=s3)
                    agg[m]["tiers"][tier] += 1
                    if m == "new":
                        tier_rows[tier].append({
                            "book": book[12:], "page": page, "column": p["column"],
                            "ocr_char": p.get("ocr_char") or "", "syllable": p["syllable"],
                            "label": label or "", "confirmed": int(conf), "rule": rule,
                            "s3_cosine": round(s3.cosine, 3) if s3 else "",
                            "s3_top": s3.top_char if s3 else "",
                            "bbox": json.dumps(p.get("bbox")),
                        })

            # head-to-head per column
            for c in set(col_conf["old"]) | set(col_conf["new"]):
                o, n = col_conf["old"][c], col_conf["new"][c]
                if n > o:
                    col_better += 1
                    if len(fixed_examples) < 50:
                        fixed_examples.append((book[12:], page, c, o, n))
                elif n < o:
                    col_worse += 1
                else:
                    col_tie += 1

    # ---- artefacts --------------------------------------------------------
    def rate(d): return d["confirmed"] / max(d["pairs"], 1) * 100
    for t, rows in tier_rows.items():
        with open(OUT / f"dataset_{t.lower()}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["book", "page", "column", "ocr_char",
                                              "syllable", "label", "confirmed", "rule",
                                              "s3_cosine", "s3_top", "bbox"])
            w.writeheader(); w.writerows(rows)
    summary = {
        "pages_done": pages_done, "pages_fail": pages_fail, "pages_ok": pages_ok,
        "review_gap_syllables": review_gap_syllables,
        "old": {"pairs": agg["old"]["pairs"], "confirmed": agg["old"]["confirmed"],
                "confirm_rate": round(rate(agg["old"]) / 100, 4),
                "tiers": dict(agg["old"]["tiers"])},
        "new": {"pairs": agg["new"]["pairs"], "confirmed": agg["new"]["confirmed"],
                "confirm_rate": round(rate(agg["new"]) / 100, 4),
                "tiers": dict(agg["new"]["tiers"])},
        "by_column_type": split,
        "head_to_head_columns": {"new_better": col_better, "new_worse": col_worse, "tie": col_tie},
    }
    json.dump(summary, open(OUT / "summary_full.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(OUT / "fixed_examples_full.txt", "w", encoding="utf-8") as f:
        for bk, pg, c, o, n in fixed_examples:
            f.write(f"{bk} {pg} col{c}: confirmed {o} -> {n}\n")

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 72)
    print(" FULL production-path A/B — current pairing (OLD) vs banded-DP (NEW)")
    print(f" pages processed {pages_done} (fail {pages_fail}), page_ok {pages_ok}/{pages_done}")
    print("=" * 72)
    print(f"\nOVERALL emitted pairs / dict-confirmed:")
    print(f"  OLD : {agg['old']['pairs']:6d} pairs | {agg['old']['confirmed']:6d} confirmed = {rate(agg['old']):.1f}%")
    print(f"  NEW : {agg['new']['pairs']:6d} pairs | {agg['new']['confirmed']:6d} confirmed = {rate(agg['new']):.1f}%")
    print(f"  Δ   : {agg['new']['pairs']-agg['old']['pairs']:+d} pairs, "
          f"{agg['new']['confirmed']-agg['old']['confirmed']:+d} confirmed, "
          f"{rate(agg['new'])-rate(agg['old']):+.1f} pp")
    for bucket in ("matched", "diverged"):
        o = split["old"][bucket]; n = split["new"][bucket]
        orr = o[1] / max(o[0], 1) * 100; nrr = n[1] / max(n[0], 1) * 100
        print(f"\n{bucket.upper()}-count columns:")
        print(f"  OLD {o[1]}/{o[0]} = {orr:.1f}%   NEW {n[1]}/{n[0]} = {nrr:.1f}%   ({nrr-orr:+.1f} pp)")
    print(f"\nPer-column head-to-head: NEW better {col_better} | tie {col_tie} | NEW worse {col_worse}")
    s3_tag = "WITH S3 visual" if vs3 is not None else "S3-independent floor"
    print(f"\nDataset tiers (emitted labels, {s3_tag}):")
    for t in ("GOLD", "SILVER", "REVIEW"):
        print(f"  {t:7s}  OLD {agg['old']['tiers'].get(t,0):6d}   NEW {agg['new']['tiers'].get(t,0):6d}")
    print(f"  (+{review_gap_syllables} syllables with no Nôm box -> REVIEW recovery queue)")
    if vs3 is not None:
        print(f"  S3 computed on {n_s3} crops  (FD-glyph refs {vs3.n_fd}, font refs {vs3.n_font})")
    print(f"\nArtefacts in {OUT}/: summary_full.json, dataset_gold/silver/review.csv, fixed_examples_full.txt")


if __name__ == "__main__":
    main()
