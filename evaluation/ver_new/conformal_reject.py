"""S3 design #1 — set the SILVER reject threshold with a DISTRIBUTION-FREE PRECISION
GUARANTEE, replacing the grid-search in calibrate_s3.py.

Method: Learn-Then-Test (Angelopoulos, Bates, Candès, Jordan, Lei; arXiv 2110.01052)
with a Hoeffding–Bentkus p-value and FIXED-SEQUENCE testing — valid because the
False-Selection-Rate FSR(τ) = P(top_char ≠ true | accepted) = 1 − precision is
monotone in τ (Zhao & Su, arXiv 2311.03811). Result: the lowest threshold τ
(= max coverage) for which

    P( true SILVER precision  ≥  1 − α )  ≥  1 − δ.

Each calibration decision i contributes confidence p_i and correctness c_i. We sweep
τ high→low; accept while the HB p-value for H0: FSR(τ) > α stays ≤ δ; emit the last
surviving τ. ~CPU, sub-second.

HONESTY: run on VAL GOLD this bounds precision on the GOLD-like (easy) population —
an OPTIMISTIC proxy. The non-circular thesis number requires calibrating on a
human-audited SILVER-eligible set: fill eval_sample/verify.csv (human_correct) and
pass --audit eval_sample/verify.csv.

Run:
  .venv/bin/python evaluation/ver_new/conformal_reject.py --alpha 0.10 --delta 0.05
  .venv/bin/python evaluation/ver_new/conformal_reject.py --audit eval_sample/verify.csv  # honest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------- Hoeffding–Bentkus
def _h1(a: float, b: float) -> float:
    t1 = a * math.log(a / b) if a > 0 else 0.0
    t2 = (1 - a) * math.log((1 - a) / (1 - b)) if a < 1 else 0.0
    return t1 + t2


def hb_pvalue(r_hat: float, alpha: float, n: int) -> float:
    """Valid p-value for H0: risk > alpha (small p => risk <= alpha)."""
    if n == 0 or r_hat >= alpha:
        return 1.0
    from scipy.stats import binom
    hoeff = math.exp(-n * _h1(r_hat, alpha))
    bentkus = math.e * binom.cdf(math.ceil(n * r_hat), n, alpha)
    return float(min(hoeff, bentkus, 1.0))


def ltt_fixed_sequence(conf, correct, alpha, delta):
    """conf, correct: arrays over calibration decisions. Returns the lowest τ
    (max coverage) with P(precision ≥ 1−α) ≥ 1−δ, or None if even the safest τ fails."""
    order = np.argsort(-np.asarray(conf))            # high confidence first
    conf, correct = np.asarray(conf)[order], np.asarray(correct)[order]
    n = len(conf)
    best = None
    # candidate thresholds = each accepted-prefix boundary (accept top-k)
    for k in range(1, n + 1):
        tau = float(conf[k - 1])
        acc_correct = correct[:k]
        r_hat = 1.0 - acc_correct.mean()             # FSR = 1 − precision on accepted
        p = hb_pvalue(r_hat, alpha, k)
        if p <= delta:
            best = {"tau": tau, "k": k, "coverage": k / n,
                    "emp_precision": float(acc_correct.mean()), "emp_fsr": float(r_hat), "p": p}
        else:
            if best is not None:
                break                                # fixed-sequence: stop at first failure past a success
    return best, n


def load_gold_decisions(args):
    """Reconstruct VAL GOLD (conf, correct) like s3_risk_coverage."""
    from pipeline.step0_setup import load_config
    from core.text.dictionary import load_qn_to_nom
    from evaluation.ver_new.visual_signal import VisualS3
    from evaluation.ver_new.s3_risk_coverage import decisions
    cfg = load_config(args.config); paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths.get("fd_cache_similar", "")) if paths.get("fd_cache_similar") else "")
    D = Path(args.dataset)
    rows = [dict(r, _dataset=str(D)) for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == args.tier and r["split"] == args.split and r["image"] and r["syllable"] and r["label"]]
    recs = decisions(vs3, qn, rows)                  # [(conf, margin, correct)]
    return [r[0] for r in recs], [r[2] for r in recs], f"{args.tier}/{args.split} (GOLD proxy — optimistic)"


def load_audit_decisions(path):
    """Human-audited set: verify.csv with s3_cosine (confidence) + human_correct."""
    conf, correct, n_skip = [], [], 0
    for r in csv.DictReader(open(path, encoding="utf-8")):
        hc = (r.get("human_correct") or "").strip()
        if hc not in ("0", "1"):
            n_skip += 1; continue
        try:
            c = float(r.get("s3_cosine") or r.get("p_match") or "")
        except ValueError:
            continue
        conf.append(c); correct.append(int(hc))
    return conf, correct, f"human-audited ({len(conf)} labels, {n_skip} unfilled) — NON-CIRCULAR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(HERE / "dataset_out"))
    ap.add_argument("--tier", default="GOLD"); ap.add_argument("--split", default="val")
    ap.add_argument("--alpha", type=float, default=0.10, help="target FSR (precision >= 1-alpha)")
    ap.add_argument("--delta", type=float, default=0.05, help="confidence (guarantee holds w.p. >= 1-delta)")
    ap.add_argument("--audit", default="", help="human-audited verify.csv (the honest, non-circular set)")
    args = ap.parse_args()

    if args.audit:
        p = Path(args.audit)
        if not p.is_absolute():
            p = HERE / args.audit
        conf, correct, src = load_audit_decisions(p)
        if len(conf) < 30:
            sys.exit(f"only {len(conf)} human-labeled rows in {p} — fill verify.csv (human_correct) first.")
    else:
        conf, correct, src = load_gold_decisions(args)

    best, n = ltt_fixed_sequence(conf, correct, args.alpha, args.delta)
    print("=" * 70)
    print(" CONFORMAL REJECT (Learn-Then-Test, Hoeffding–Bentkus, fixed-sequence)")
    print("=" * 70)
    print(f"  calibration set : {src}   (n={n})")
    print(f"  target          : precision ≥ {1-args.alpha:.0%}  (α={args.alpha})  w.p. ≥ {1-args.delta:.0%}  (δ={args.delta})")
    if best is None:
        print(f"\n  ✗ NO threshold achieves the guarantee at n={n} — need more calibration data,")
        print(f"    a looser α, or a stronger signal. (At small n the HB bound is conservative.)")
        out = {"guaranteed": False, "n": n, "alpha": args.alpha, "delta": args.delta, "source": src}
    else:
        print(f"\n  ✓ GUARANTEED threshold τ = {best['tau']:.4f}")
        print(f"     coverage        : {best['coverage']:.1%}  ({best['k']}/{n} accepted)")
        print(f"     empirical prec. : {best['emp_precision']:.3f}  (FSR {best['emp_fsr']:.3f})")
        print(f"\n  >>> CLAIM: with probability ≥ {1-args.delta:.0%}, true precision of accepted "
              f"SILVER ≥ {1-args.alpha:.0%}.")
        out = {"guaranteed": True, "tau": best["tau"], "coverage": best["coverage"],
               "empirical_precision": best["emp_precision"], "empirical_fsr": best["emp_fsr"],
               "accepted": best["k"], "n": n, "alpha": args.alpha, "delta": args.delta, "source": src}
    if not args.audit:
        print("\n  ⚠️  This is the GOLD-calibrated (optimistic) number. The non-circular thesis figure")
        print("      needs the human-audited SILVER set: fill eval_sample/verify.csv then")
        print("      run with --audit eval_sample/verify.csv.")
    outp = HERE / "results" / ("conformal_reject_audit.json" if args.audit else "conformal_reject.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {outp}")


if __name__ == "__main__":
    main()
