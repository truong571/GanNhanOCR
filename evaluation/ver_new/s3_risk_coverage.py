"""Roadmap #2 — turn the single S3 operating point into a RISK-COVERAGE curve.

The reviewer-bait in the current write-up is the lone operating point
`(tau_p=0.5, delta_p=0)`: it *looks* untuned. It is not — it is the coverage-
maximising point on a selective-classification curve at the P>=0.90 constraint.
The fix recommended by the literature (Traub et al., "Overcoming Common Flaws in
the Evaluation of Selective Classification," NeurIPS 2024 Spotlight, arXiv
2407.01032) is to report the WHOLE curve and integrate it, then place (tau,delta)
as ONE labelled point on it.

This script reconstructs the exact per-decision array S3 produces at label time —
for each held-out VAL crop, the candidate set R = {ocr_char} u dict-readings, the
true char as the positive, scored by the live (calibrated) VisualS3.decide — and
from `(confidence, correct)` computes:

  * the risk-coverage curve (selective risk vs coverage),
  * AURC  = area under risk-coverage           (Geifman & El-Yaniv, NeurIPS 2017),
  * AUGRC = area under GENERALISED risk-cov.    (Traub et al., NeurIPS 2024),
  * coverage & selective risk AT the calibrated operating point (the labelled dot).

Outputs to evaluation/ver_new/results/:
  s3_risk_coverage.csv   (coverage, selective_risk, generalised_risk) per cut
  s3_risk_coverage.json  (AURC, AUGRC, operating-point row, n)
  s3_risk_coverage.png   (only if matplotlib is installed; otherwise skipped)

HONESTY (state in the thesis): this curve is computed on VAL GOLD-tier crops — the
EASY regime where S1∩S2 already agree. It proves the operating point is principled
and shows the selective behaviour, but the precision NUMBER is still an optimistic
proxy; the non-circular figure must come from the human SILVER audit (measure_
precision.py). Pass --tier SILVER to draw the curve on a different stratum once you
have human labels for it.

Run:
  .venv/bin/python evaluation/ver_new/s3_risk_coverage.py
  .venv/bin/python evaluation/ver_new/s3_risk_coverage.py --tier GOLD --split val --n 4000
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

HERE = Path(__file__).resolve().parent


def decisions(vs3, qn, rows):
    """-> list of (confidence, margin, correct) over the real candidate set.

    Mirrors visual_signal.compute's candidate construction so the curve reflects
    the SAME decision the labeler makes; the crop PNGs in dataset_out are already
    ink-tightened, so embed_path matches the production framing.
    """
    out = []
    for i, r in enumerate(rows):
        cp = Path(r["_dataset"]) / r["image"]
        ce = vs3.enc.embed_path(str(cp))
        if ce is None:
            continue
        true = r["label"]
        syl = (r["syllable"] or "").lower()
        cands = []
        for c in ([r["ocr_char"]] if r["ocr_char"] else []) + qn.get(syl, []):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if true and _is_cjk(true) and true not in cands:
            cands.append(true)          # ensure the positive is scorable
        if len(cands) < 2 or not true:
            continue
        dec = vs3.decide(ce, cands)
        out.append((float(dec["p_match"]), float(dec["p_margin"]), int(dec["top_char"] == true)))
        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(rows)}", flush=True)
    return out


def risk_coverage(recs):
    """recs = [(conf, margin, correct)]. Sort by confidence desc; sweep accept-top-k.

    selective_risk(k)  = wrong_in_topk / k              (the classic risk-coverage)
    generalised_risk(k)= wrong_in_topk / N              (Traub: errors over ALL samples)
    AURC  = mean_k selective_risk(k)
    AUGRC = mean_k generalised_risk(k)
    """
    n = len(recs)
    order = sorted(recs, key=lambda t: (t[0], t[1]), reverse=True)   # most-confident first
    correct = np.array([c for _, _, c in order], float)
    wrong_cum = np.cumsum(1.0 - correct)                              # wrong among top-k
    k = np.arange(1, n + 1)
    sel_risk = wrong_cum / k
    gen_risk = wrong_cum / n
    cov = k / n
    aurc = float(sel_risk.mean())
    augrc = float(gen_risk.mean())
    return cov, sel_risk, gen_risk, aurc, augrc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--tier", default="GOLD")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=0, help="cap rows (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths.get("fd_cache_similar", "")) if paths.get("fd_cache_similar") else "")

    D = Path(args.dataset)
    rows = [dict(r, _dataset=str(D)) for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == args.tier and r["split"] == args.split and r["image"] and r["syllable"] and r["label"]]
    if args.n:
        rows = rows[:args.n]
    print(f"{args.tier}/{args.split} crops with a candidate decision: {len(rows)}", flush=True)

    recs = decisions(vs3, qn, rows)
    if len(recs) < 50:
        sys.exit(f"too few scorable decisions ({len(recs)}).")
    cov, sel, gen, aurc, augrc = risk_coverage(recs)
    base_acc = float(np.mean([c for _, _, c in recs]))

    # operating point from the live calibration -> its coverage & selective risk
    op = None
    cal = vs3.calib
    if cal:
        tau, delta = cal.get("tau_p", 0.5), cal.get("delta_p", 0.0)
        acc = [(p, m, c) for p, m, c in recs if p >= tau and m >= delta]
        if acc:
            op = {"tau_p": tau, "delta_p": delta,
                  "coverage": round(len(acc) / len(recs), 4),
                  "selective_risk": round(1 - sum(c for _, _, c in acc) / len(acc), 4),
                  "precision": round(sum(c for _, _, c in acc) / len(acc), 4),
                  "accepted": len(acc)}

    res = {"tier": args.tier, "split": args.split, "n_decisions": len(recs),
           "full_coverage_accuracy": round(base_acc, 4),
           "AURC": round(aurc, 5), "AUGRC": round(augrc, 5), "operating_point": op,
           "note": "curve on dictionary-confirmed crops (easy/optimistic); the non-circular "
                   "number is the human SILVER audit (measure_precision.py)."}

    out_dir = HERE / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "s3_risk_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["coverage", "selective_risk", "generalised_risk"])
        for i in range(0, len(cov), max(1, len(cov) // 400)):     # ~400 points is plenty
            w.writerow([round(float(cov[i]), 5), round(float(sel[i]), 5), round(float(gen[i]), 5)])
    json.dump(res, open(out_dir / "s3_risk_coverage.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n=== S3 selective-classification (risk-coverage) ===")
    print(f"  decisions        : {len(recs)}   full-coverage acc: {base_acc:.3f}")
    print(f"  AURC  (lower=better): {aurc:.4f}")
    print(f"  AUGRC (Traub 2024)  : {augrc:.4f}")
    if op:
        print(f"  operating point  τ={op['tau_p']} δ={op['delta_p']} -> coverage {op['coverage']:.3f}, "
              f"precision {op['precision']:.3f} (selective risk {op['selective_risk']:.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.plot(cov, sel, lw=2, label="selective risk")
        ax.plot(cov, gen, lw=1.4, ls="--", color="#888", label="generalised risk (AUGRC)")
        if op:
            ax.scatter([op["coverage"]], [op["selective_risk"]], color="crimson", zorder=5,
                       label=f"operating point (τ={op['tau_p']},δ={op['delta_p']})")
        ax.set_xlabel("coverage"); ax.set_ylabel("risk (error rate)")
        ax.set_title(f"S3 risk-coverage [{args.tier}/{args.split}]  AURC={aurc:.3f} AUGRC={augrc:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out_dir / "s3_risk_coverage.png", dpi=140)
        print(f"  figure -> {out_dir / 's3_risk_coverage.png'}")
    except ImportError:
        print("  (matplotlib not installed -> PNG skipped; CSV written. `pip install matplotlib` for the figure.)")
    print(f"  -> {out_dir / 's3_risk_coverage.json'} , s3_risk_coverage.csv")


if __name__ == "__main__":
    main()
