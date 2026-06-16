"""TEST for Problem B (evaluation) — a Cleanlab / Confident-Learning style RESIDUAL
LABEL-ERROR estimate, the headline number a weakly-labeled dataset release needs
(Northcutt et al., "Confident Learning," JAIR 2021 / "Pervasive Label Errors,"
NeurIPS D&B 2021).

Idea: held-out GOLD (val+test) crops are OUT-OF-SAMPLE for the encoder (it trained
on the train split only), so the 1,591-way head's predictions on them are honest
self-confidence. A label is a likely error when the head CONFIDENTLY disagrees with
the assigned label: argmax class != assigned AND softmax(argmax) - softmax(assigned)
> margin. The flagged fraction is an upper-bound residual-error estimate; per
Northcutt, only ~half of flags are真 errors, so the TRUE rate is lower and the flags
are an audit QUEUE, not auto-corrections.

Outputs: results/eval_label_errors.json + results/label_error_candidates.csv
(crop, assigned label, head's confident alternative) for human review.

Also provides `cohen_kappa()` for the 2-annotator agreement the human audit needs.

Run:
  .venv/bin/python evaluation/ver_new/eval_label_errors.py --margin 0.5
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

from evaluation.ver_new.visual_signal import VisualS3        # noqa: E402

HERE = Path(__file__).resolve().parent
ARC_S = 30.0      # ArcMargin scale used at training (kaggle_train ArcMargin default)


def cohen_kappa(a, b):
    """Cohen's κ for two annotators' binary/categorical judgements (lists, equal len)."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--margin", type=float, default=0.5, help="confident-disagreement softmax margin")
    ap.add_argument("--n", type=int, default=0, help="cap held-out crops (0 = all)")
    args = ap.parse_args()

    vs3 = VisualS3(REPO, fd_dir="")
    enc = vs3.enc
    if not enc.has_head:
        sys.exit("ckpt has no head.")
    lab2idx = {lab: i for i, lab in enc.classes.items()}

    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] in ("GOLD", "SILVER") and r["split"] in ("val", "test")
            and r["image"] and r["label"] in lab2idx]
    if args.n:
        rows = rows[: args.n]
    print(f"held-out (val+test) char crops with in-vocab label: {len(rows)}", flush=True)

    flagged = []
    confs = []
    n = 0
    for r in rows:
        emb = enc.embed_path(str(D / r["image"]))
        if emb is None:
            continue
        lg = enc.logits(emb) * ARC_S          # cosine logits -> scaled
        p = np.exp(lg - lg.max()); p /= p.sum()
        gi = lab2idx[r["label"]]
        ai = int(p.argmax())
        n += 1
        confs.append(float(p[gi]))
        if ai != gi and (p[ai] - p[gi]) > args.margin:
            flagged.append({"image": r["image"], "tier": r["tier"], "book": r["book"],
                            "page": r["page"], "assigned": r["label"], "syllable": r["syllable"],
                            "head_alt": enc.classes.get(ai, str(ai)),
                            "p_assigned": round(float(p[gi]), 3), "p_alt": round(float(p[ai]), 3)})

    rate = len(flagged) / max(n, 1)
    print("=" * 64)
    print(" PROBLEM B — residual label-error estimate (Confident-Learning style)")
    print("=" * 64)
    print(f"  held-out crops scored        : {n}")
    print(f"  mean self-confidence p(label): {np.mean(confs):.3f}")
    print(f"  flagged (confident disagree, margin>{args.margin}): {len(flagged)}  "
          f"= estimated residual-error UPPER BOUND {rate:.2%}")
    print(f"  (Northcutt: ~half of flags are true errors -> true rate likely ~{rate/2:.2%}; "
          f"flags are an AUDIT QUEUE, not auto-fixes.)")
    if flagged:
        print("  sample flags (assigned -> head's confident alternative):")
        for r in sorted(flagged, key=lambda x: -x["p_alt"])[:6]:
            print(f"    {r['book']} {r['page']}: {r['assigned']} (p={r['p_assigned']}) "
                  f"-> {r['head_alt']} (p={r['p_alt']})  [{r['tier']}, âm {r['syllable']}]")

    out_csv = HERE / "results" / "label_error_candidates.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if flagged:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flagged[0].keys()))
            w.writeheader(); w.writerows(flagged)
    out = {"n_scored": n, "mean_self_confidence": round(float(np.mean(confs)), 4),
           "margin": args.margin, "flagged": len(flagged),
           "residual_error_upper_bound": round(rate, 4),
           "note": "held-out GOLD/SILVER is OOF for the encoder; flags = audit queue (Northcutt ~51% real)."}
    json.dump(out, open(HERE / "results" / "eval_label_errors.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  -> results/eval_label_errors.json , label_error_candidates.csv ({len(flagged)} rows)")


if __name__ == "__main__":
    main()
