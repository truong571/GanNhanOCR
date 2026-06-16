"""Roadmap #7 — which REJECT rule should gate SILVER? An apples-to-apples AURC
comparison of confidence estimators on the held-out retrieval decision.

The synthesis hypothesis: the per-tier ISOTONIC calibration overfits on a thin
GOLD-val split, which is plausibly why the operating-point sweep collapses to the
most permissive threshold. The fix is a post-hoc, retrain-free confidence rule. We
FIX the system's decision (VisualS3.decide picks the char) and then score how well
each candidate confidence estimator separates correct from wrong retrievals, by
AURC (lower = the rule that knows best when it is right):

  isotonic  the current calibrated per-tier P(match)            [status quo]
  cosine    raw cosine of the winner to its reference bank      [= the kNN reject:
            abstain when the crop is far from every prototype; Sun et al. ICML'22]
  margin    winner cosine - runner-up cosine                    [contrastive gate]
  mls       Max-Logit-Score: crop's max cosine over ALL 1591    [open-set gate,
            trained classes, candidate-INDEPENDENT               Vaze et al. ICLR'22]

`mls` needs the ArcFace head — now shipped in the checkpoint and loaded by
NomEncoder (infer.py). If absent, the mls row is skipped.

Reported on GOLD test crops (held out). Like every GOLD-based S3 number this is an
optimistic proxy (the true figure is the human SILVER audit); the COMPARISON
between rules is the point, not the absolute precision.

Run:
  .venv/bin/python evaluation/ver_new/s3_reject_ablation.py            # GOLD test
  .venv/bin/python evaluation/ver_new/s3_reject_ablation.py --n 1500
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402
from evaluation.ver_new.s3_risk_coverage import risk_coverage      # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--tier", default="GOLD")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths.get("fd_cache_similar", "")) if paths.get("fd_cache_similar") else "")
    enc = vs3.enc
    print(f"  ArcFace head for MLS: {'available' if enc.has_head else 'ABSENT (mls skipped)'}", flush=True)

    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == args.tier and r["split"] == args.split and r["image"] and r["syllable"] and r["label"]]
    if args.n:
        rows = rows[:args.n]
    print(f"{args.tier}/{args.split} crops: {len(rows)}", flush=True)

    scorers = ["isotonic", "cosine", "margin"] + (["mls"] if enc.has_head else [])
    recs = {s: [] for s in scorers}
    n = 0
    for i, r in enumerate(rows):
        ce = enc.embed_path(str(D / r["image"]))
        if ce is None:
            continue
        true = r["label"]
        cands = []
        for c in ([r["ocr_char"]] if r["ocr_char"] else []) + qn.get((r["syllable"] or "").lower(), []):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if true and _is_cjk(true) and true not in cands:
            cands.append(true)
        if len(cands) < 2:
            continue
        # fixed system decision
        dec = vs3.decide(ce, cands)
        winner = dec["top_char"]
        correct = int(winner == true)
        # raw best-tier cosine per candidate -> cosine/margin confidences
        best_raw = {c: max(vs3.tier_cosines(ce, c).values()) for c in cands}
        wcos = best_raw.get(winner, -1.0)
        others = [v for c, v in best_raw.items() if c != winner]
        margin = wcos - (max(others) if others else -1.0)
        n += 1
        recs["isotonic"].append((float(dec["p_match"]), 0.0, correct))
        recs["cosine"].append((float(wcos), 0.0, correct))
        recs["margin"].append((float(margin), 0.0, correct))
        if enc.has_head:
            recs["mls"].append((float(enc.mls(ce)), 0.0, correct))
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(rows)}", flush=True)

    base_acc = float(np.mean([c for _, _, c in recs["isotonic"]])) if n else 0.0
    print(f"\n=== reject-rule comparison on {n} decisions (full-coverage acc {base_acc:.3f}) ===")
    print(f"  {'reject rule':10s} {'AURC':>8s} {'risk@cov=0.8':>13s}   (lower AURC = better gate)")
    res = {"tier": args.tier, "split": args.split, "n": n,
           "full_coverage_acc": round(base_acc, 4), "rules": {}}
    for s in scorers:
        cov, sel, gen, aurc, augrc = risk_coverage(recs[s])
        # selective risk at 80% coverage
        k = int(0.8 * len(cov))
        risk80 = float(sel[k]) if k < len(sel) else float("nan")
        res["rules"][s] = {"AURC": round(float(aurc), 4), "AUGRC": round(float(augrc), 4),
                           "risk_at_cov0.8": round(risk80, 4)}
        print(f"  {s:10s} {aurc:>8.4f} {risk80:>13.4f}")

    best = min(res["rules"], key=lambda s: res["rules"][s]["AURC"])
    print(f"\n  best gate by AURC: {best}")
    print("  'cosine' = the kNN/distance reject; 'mls' = open-set max-logit (head-based).")
    print("  If a post-hoc rule beats 'isotonic', adopt it in visual_signal.decide and re-pick (tau,delta).")
    out_p = HERE / "results" / "s3_reject_ablation.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {out_p}")


if __name__ == "__main__":
    main()
