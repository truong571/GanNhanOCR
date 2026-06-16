"""Bước 3 (2/2) — turn the human-filled verify.csv into MEASURED precision.

Reads eval_sample/verify.csv (human_correct in {0,1}; human_label = true char when
wrong) and reports, per TIER and per RULE:
  precision = mean(human_correct), n, and a Wilson 95% confidence interval,
plus the overall char-level (GOLD+SILVER) precision and a confusion list of the
disagreements (proposed label/syllable vs the human's true label).

This is the NON-CIRCULAR number for the thesis: a human (not the dictionary) judged
whether the crop depicts the proposed char/syllable. Use it to (a) report real
GOLD/SILVER/SYLLABLE precision with CIs, and (b) re-pick the SILVER operating point
in calibrate_s3.py if SILVER precision is below target.

  .venv/bin/python evaluation/ver_new/measure_precision.py                 # real
  .venv/bin/python evaluation/ver_new/measure_precision.py --self-test     # demo math
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _report(rows):
    def block(title, keyfn):
        groups = defaultdict(lambda: [0, 0])  # key -> [k_correct, n]
        for r in rows:
            g = groups[keyfn(r)]
            g[0] += int(r["_hc"]); g[1] += 1
        print(f"\n=== precision by {title} ===")
        print(f"  {'group':34s} {'prec':>6s} {'95% CI':>16s} {'n':>5s}")
        for key in sorted(groups):
            k, n = groups[key]
            p, lo, hi = wilson(k, n)
            print(f"  {key:34s} {p:6.1%} [{lo:5.1%},{hi:5.1%}] {n:5d}")
        return groups

    block("TIER", lambda r: r["tier"])
    block("RULE", lambda r: f"{r['tier']}/{r['rule']}")

    # overall char-level usable (GOLD + SILVER)
    cl = [r for r in rows if r["tier"] in ("GOLD", "SILVER")]
    k = sum(int(r["_hc"]) for r in cl)
    p, lo, hi = wilson(k, len(cl))
    print(f"\n=== OVERALL char-level (GOLD+SILVER) ===")
    print(f"  precision = {p:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  (n={len(cl)})")

    wrong = [r for r in rows if not int(r["_hc"]) and r["tier"] in ("GOLD", "SILVER", "SYLLABLE")]
    if wrong:
        print(f"\n=== disagreements ({len(wrong)}) — proposed -> human ===")
        for r in wrong[:40]:
            prop = r["label"] or f"[âm:{r['syllable']}]"
            print(f"  {r['sample_id']} {r['tier']:8s} {r['rule']:22s} proposed={prop} "
                  f"true={r.get('human_label','') or '?'}  (P={r.get('s3_cosine','')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "eval_sample" / "verify.csv"))
    ap.add_argument("--self-test", action="store_true",
                    help="auto-fill human_correct with a synthetic pattern to demo the math")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.self_test:
        # synthetic verdicts ONLY to demonstrate the report (NOT real precision)
        rate = {"GOLD/s1_inter_s2_direct": 0.99, "GOLD/s1_inter_s2_similar": 0.92,
                "SILVER/s2_inter_s3_corrected": 0.90, "SILVER/s1_inter_s3_out_of_dict": 0.85,
                "SYLLABLE/nghia_consensus": 0.93}
        random.seed(0)
        for r in rows:
            base = rate.get(f"{r['tier']}/{r['rule']}", 0.2)  # REVIEW control ~0.2
            r["_hc"] = 1 if random.random() < base else 0
        print("### SELF-TEST: synthetic verdicts (demonstrates the math, NOT real precision) ###")
    else:
        filled = [r for r in rows if r.get("human_correct", "").strip() in ("0", "1")]
        if not filled:
            print(f"No filled rows in {args.csv}. Fill `human_correct` (1/0) first, "
                  "or run with --self-test to preview the report.")
            return
        for r in filled:
            r["_hc"] = int(r["human_correct"])
        rows = filled
        print(f"### MEASURED precision from {len(rows)} human-verified rows ###")

    _report(rows)


if __name__ == "__main__":
    main()
