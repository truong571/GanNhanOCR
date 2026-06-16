"""Bước 2 — calibrate the S3 visual signal into a P(match) at a target precision.

WHAT IT DOES
  For every held-out VAL-split GOLD crop (real woodblock, known char + syllable),
  reconstruct the REAL S3 decision: candidate set R = {ocr_char} ∪ dict-readings of
  the syllable; the true char is the positive, the other readings are real negatives.
  For each candidate we take its RAW cosine to each reference tier (crop-prototype /
  similar-font / FD). We then:
    (1) fit a PER-TIER isotonic map  cosine -> P(match)  (PAVA, no sklearn needed),
        so a crop-cosine and a glyph-cosine become COMPARABLE probabilities; and
    (2) pick an operating point (tau_p, delta_p) on P(match) at a TARGET PRECISION,
        replacing the placeholder TAU_SILVER=0.62 / DELTA_SILVER=0.06.
  Output: nom-embed/s3_calibration.json  (loaded automatically by visual_signal.VisualS3).

HONEST CAVEATS (state these in the thesis)
  - The calibration set is GOLD (dict-confirmed) crops — the EASY regime. SILVER fires
    on HARDER unconfirmed crops, so the reported precision is an OPTIMISTIC proxy; a
    small HUMAN-verified set on SILVER-eligible crops (Bước 3) is the final word.
  - Crop-prototype references come from TRAIN crops of the SAME books, so within-book
    near-duplicates make the crop tier optimistic for *cross-book* generalization
    (fine for labeling THIS corpus; report same/cross-book separately for the thesis).

Run:
  .venv/bin/python evaluation/ver_new/calibrate_s3.py            # --target 0.90
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.step0_setup import load_config                       # noqa: E402
from core.text.dictionary import load_qn_to_nom                    # noqa: E402
from evaluation.ver_new.visual_signal import VisualS3, _is_cjk     # noqa: E402


def _pava(vals, wts):
    """Pool-Adjacent-Violators -> non-decreasing fit. `vals` sorted by x ascending."""
    blocks = []  # [value, weight, size]
    for v, w in zip(vals, wts):
        blocks.append([float(v), float(w), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, s2 = blocks.pop()
            v1, w1, s1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, s1 + s2])
    out = []
    for v, _w, s in blocks:
        out += [v] * s
    return out


def calibrate_tier(cos, match, nbins=15):
    """Isotonic cosine->P(match) as knots (x = bin-mean cosine, p = monotone rate)."""
    cos = np.asarray(cos, float)
    match = np.asarray(match, float)
    if len(cos) < 40:
        return None
    order = np.argsort(cos)
    cos, match = cos[order], match[order]
    edges = np.linspace(0, len(cos), min(nbins, len(cos) // 8 or 1) + 1).astype(int)
    xs, ps, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            xs.append(float(cos[a:b].mean()))
            ps.append(float(match[a:b].mean()))
            ws.append(int(b - a))
    return {"x": xs, "p": _pava(ps, ws), "n": int(len(cos))}


def _interp(cal, cos):
    return float(np.interp(cos, cal["x"], cal["p"])) if cal else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "config" / "pipeline.yaml"))
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "dataset_out"))
    ap.add_argument("--target", type=float, default=0.90, help="target SILVER precision")
    ap.add_argument("--glyph-guard", type=float, default=0.10,
                    help="open-set guard: abstain if a candidate beats the winner on the "
                         "glyph tier by > this raw-cosine margin (crop-bias protection)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    qn = load_qn_to_nom(str(REPO / paths["qn_to_nom_dict"]))
    vs3 = VisualS3(REPO, fd_dir=str(REPO / paths["fd_cache_universal"]),
                   simfont_dir=str(REPO / paths.get("fd_cache_similar", "")) if paths.get("fd_cache_similar") else "")

    D = Path(args.dataset)
    rows = [r for r in csv.DictReader(open(D / "labels.csv", encoding="utf-8"))
            if r["tier"] == "GOLD" and r["split"] == "val" and r["image"] and r["syllable"]]
    if args.limit:
        rows = rows[:args.limit]
    print(f"calibration crops (VAL GOLD): {len(rows)}", flush=True)

    samples = defaultdict(lambda: ([], []))   # tier -> (cosines, matches)
    decisions = []                            # (true_char, {cand: {tier: cos}})
    for i, r in enumerate(rows):
        cp = D / r["image"]
        ce = vs3.enc.embed_path(str(cp))
        if ce is None:
            continue
        true = r["label"]
        syl = (r["syllable"] or "").lower()
        cands = []
        for c in ([r["ocr_char"]] if r["ocr_char"] else []) + qn.get(syl, []):
            if _is_cjk(c) and c not in cands:
                cands.append(c)
        if true not in cands:
            cands.append(true)
        if len(cands) < 2:
            continue
        perc = {}
        for c in cands:
            tc = vs3.tier_cosines(ce, c)
            perc[c] = tc
            for t, v in tc.items():
                samples[t][0].append(v)
                samples[t][1].append(1 if c == true else 0)
        decisions.append((true, perc))
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(rows)}", flush=True)

    # --- per-tier isotonic calibration ---
    calib = {}
    print("\n=== per-tier calibration (cosine -> P(match)) ===")
    for t, (cs, ms) in samples.items():
        cal = calibrate_tier(cs, ms)
        if cal:
            calib[t] = cal
            same = np.mean([c for c, m in zip(cs, ms) if m == 1]) if any(ms) else float("nan")
            diff = np.mean([c for c, m in zip(cs, ms) if m == 0]) if not all(ms) else float("nan")
            print(f"  {t:8s} n={cal['n']:6d} | mean cosine same={same:+.3f} diff={diff:+.3f} "
                  f"sep={same - diff:+.3f} | P-range [{cal['p'][0]:.2f}..{cal['p'][-1]:.2f}]")

    # --- decision sweep: choose (tau_p, delta_p) at target precision ---
    recs = []   # (P_win, margin, correct)
    for true, perc in decisions:
        P = {}
        for c, tc in perc.items():
            ps = [_interp(calib.get(t), v) for t in tc for v in [tc[t]]]
            ps = [p for p in ps if p is not None]
            P[c] = max(ps) if ps else 0.0
        rk = sorted(P.values(), reverse=True)
        win_c = max(P, key=P.get)
        pw = rk[0]
        pr = rk[1] if len(rk) > 1 else 0.0
        recs.append((pw, pw - pr, win_c == true))

    base_r1 = np.mean([c for _, _, c in recs]) if recs else 0.0
    print(f"\nretrieval@1 over the real candidate set (uncalibrated decision): {base_r1:.1%} "
          f"on {len(recs)} VAL decisions (median |R| varies)")

    best = None
    for delta in (0.0, 0.03, 0.05, 0.08, 0.12):
        for tau in np.linspace(0.50, 0.97, 48):
            acc = [c for pw, mg, c in recs if pw >= tau and mg >= delta]
            if len(acc) >= 50:
                prec = sum(acc) / len(acc)
                cov = len(acc) / len(recs)
                if prec >= args.target and (best is None or cov > best["coverage"]):
                    best = {"tau_p": round(float(tau), 3), "delta_p": float(delta),
                            "measured_precision": round(float(prec), 4),
                            "coverage": round(float(cov), 4), "accepted": len(acc)}
    if best is None:                       # target unreachable -> most precise point
        tau = 0.97
        acc = [c for pw, mg, c in recs if pw >= tau]
        prec = (sum(acc) / len(acc)) if acc else 0.0
        best = {"tau_p": tau, "delta_p": 0.0, "measured_precision": round(prec, 4),
                "coverage": round(len(acc) / max(len(recs), 1), 4), "accepted": len(acc),
                "note": "target precision unreachable; using most-precise point"}

    out = {"target_precision": args.target, **best, "n_decisions": len(recs),
           "retrieval_at_1": round(float(base_r1), 4),
           "glyph_guard_margin": args.glyph_guard, "tiers": calib}
    p = Path(__file__).resolve().parent / "s3_calibration.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n=== chosen operating point (target precision "
          f"{args.target:.0%}) ===")
    print(f"  tau_p={best['tau_p']} delta_p={best['delta_p']} -> "
          f"precision={best['measured_precision']:.3f} coverage={best['coverage']:.3f} "
          f"({best['accepted']} accepted / {len(recs)} decisions)")
    print(f"  -> {p}")
    print("\n  ⚠️  precision is measured on GOLD (easy) crops — an OPTIMISTIC proxy for "
          "SILVER; human-verify a SILVER-eligible sample (Bước 3) for the final number.")


if __name__ == "__main__":
    main()
