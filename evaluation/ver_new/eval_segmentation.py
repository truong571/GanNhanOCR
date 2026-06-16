"""TEST for Problem A (segmentation) — establish the BASELINE the count-constrained
anchorless detector (HRCenterNet, roadmap #5) must beat, measurable NOW without GT.

Two honest, ground-truth-free metrics on the real corpus:

  (1) COUNT ACCURACY — fraction of columns whose segmentation yields exactly N
      boxes, where N = the known QN-syllable count. The current midpoint/OCR
      segmenter is correct exactly when the column count matched (else it is a
      "diverged" column). A count-constrained detector forces N by construction, so
      this is the headroom it would close. (Read from results/review_breakdown.json.)

  (2) CROP RECOGNIZABILITY (quality proxy) — feed each GOLD crop to the encoder's
      1,591-way head; a WELL-cut single glyph is recognised (top-1 == its label),
      a merged/clipped crop is not. top-1 accuracy is thus a segmentation-quality
      proxy that needs no manual boxes. Reported overall and by seg_flag ('tall' =
      suspected merged 2-glyph). When the detector ships, re-cut crops and re-run
      this harness: a better segmenter should RAISE recognizability, especially on
      the 'tall' bucket and the recovered diverged columns.

Run:
  .venv/bin/python evaluation/ver_new/eval_segmentation.py --n 4000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from evaluation.ver_new.visual_signal import VisualS3        # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--n", type=int, default=4000, help="GOLD crops to score for recognizability")
    args = ap.parse_args()

    D = Path(args.dataset)
    rows = list(csv.DictReader(open(D / "labels.csv", encoding="utf-8")))

    # (1) count accuracy from the alignment divergence stats
    rb = {}
    rbp = HERE / "results" / "review_breakdown.json"
    if rbp.exists():
        rb = json.load(open(rbp, encoding="utf-8")).get("realign", {})
    div_cols = rb.get("diverged_columns")
    tot_cols = rb.get("total_columns")
    count_acc = (1 - div_cols / tot_cols) if (div_cols and tot_cols) else None

    # (2) crop recognizability via the 1591-way head
    vs3 = VisualS3(REPO, fd_dir="")     # only need the encoder+head
    enc = vs3.enc
    if not enc.has_head:
        sys.exit("ckpt has no head; cannot run the recognizability proxy.")

    gold = [r for r in rows if r["tier"] == "GOLD" and r["image"] and r["label"]
            and r["label"] in {lab for lab in enc.classes.values()}]
    import random
    random.seed(0); random.shuffle(gold)
    gold = gold[: args.n]

    hit = n = 0
    by_flag = defaultdict(lambda: [0, 0])      # seg_flag -> [hit, n]
    by_inkbucket = defaultdict(lambda: [0, 0])
    for r in gold:
        emb = enc.embed_path(str(D / r["image"]))
        if emb is None:
            continue
        top = enc.predict_topk(emb, 1)
        if not top:
            continue
        pred = top[0][0]
        ok = int(pred == r["label"])
        n += 1; hit += ok
        f = r.get("seg_flag", "ok") or "ok"
        by_flag[f][0] += ok; by_flag[f][1] += 1
        try:
            ink = float(r.get("ink_pct") or 0)
        except ValueError:
            ink = 0
        bucket = "<15%" if ink < 0.15 else ("15-25%" if ink < 0.25 else ">=25%")
        by_inkbucket[bucket][0] += ok; by_inkbucket[bucket][1] += 1

    print("=" * 64)
    print(" PROBLEM A — segmentation baseline (to beat with the detector)")
    print("=" * 64)
    if count_acc is not None:
        print(f"  (1) COUNT accuracy (cols with exactly N boxes): {count_acc:.1%}  "
              f"(diverged {div_cols}/{tot_cols})")
        print(f"      -> a count-constrained detector targets the remaining {1-count_acc:.1%}.")
    else:
        print("  (1) COUNT accuracy: run review_breakdown.py --with-gaps first.")
    print(f"\n  (2) CROP recognizability (head top-1 == label) on {n} GOLD crops: {hit/max(n,1):.1%}")
    print(f"      by seg_flag:")
    for f, (h, nn) in sorted(by_flag.items()):
        print(f"        {f:5s}: {h/max(nn,1):.1%}  (n={nn})")
    print(f"      by ink%:")
    for b in ("<15%", "15-25%", ">=25%"):
        if b in by_inkbucket:
            h, nn = by_inkbucket[b]; print(f"        {b:7s}: {h/max(nn,1):.1%}  (n={nn})")
    print("\n  Interpretation: this top-1 rate is a no-GT segmentation-quality proxy "
          "(well-cut glyph -> recognised). 'tall' crops recognised worse = merged-glyph "
          "signature. Re-run after the detector to show the lift.")

    out = {"count_accuracy": count_acc, "diverged_columns": div_cols, "total_columns": tot_cols,
           "recognizability": {"n": n, "top1_acc": round(hit / max(n, 1), 4),
                               "by_seg_flag": {f: {"acc": round(h/max(nn,1), 4), "n": nn}
                                               for f, (h, nn) in by_flag.items()},
                               "by_ink": {b: {"acc": round(by_inkbucket[b][0]/max(by_inkbucket[b][1],1), 4),
                                              "n": by_inkbucket[b][1]} for b in by_inkbucket}}}
    p = HERE / "results" / "eval_segmentation.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {p}")


if __name__ == "__main__":
    main()
